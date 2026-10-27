"""
Generate exact 3-tier physical map for Qwen3.8-Flash-Next IQ3_XXS.
Extracts exact GGUF chunk byte offsets for all 48 layers x 512 experts (24,576 experts).
Ranks experts empirically from decoded activation traces:
  - Tier 0 VRAM: Top 1,870 experts (~3.5 GB VRAM)
  - Tier 1 RAM: Next 3,250 experts (~6.0 GB RAM pinned via VirtualLock)
  - Tier 2 Cold: Remaining 19,456 experts on NVMe
Outputs format 100% compatible with atlas-engine.cpp parse_json_map and autotune.py.
"""
import os
import sys
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))
from gguf_utils import parse_gguf


def generate_qwen38_physical_map(
    model_path: str,
    trace_path: str,
    output_path: str,
    tier0_count: int = 1870,
    tier1_count: int = 3250,
):
    print(f"[MAP-GEN] Parsing GGUF header and tensor index: {model_path} ...")
    parsed = parse_gguf(model_path)
    meta = parsed["metadata"]

    n_layers = int(meta.get("qwen4exp.block_count", meta.get("qwen35moe.block_count", 48)))
    n_experts = int(meta.get("qwen4exp.expert_count", meta.get("qwen35moe.expert_count", 512)))
    top_k = int(meta.get("qwen4exp.expert_used_count", meta.get("qwen35moe.expert_used_count", 10)))
    arch = str(meta.get("general.architecture", "qwen4exp"))

    data_start = parsed["data_start"]
    file_size = parsed["file_size"]
    alignment = parsed.get("alignment", 4096)

    tensor_map = {t.name: t for t in parsed["tensors"]}
    print(f"[MAP-GEN] Architecture: {arch}, {n_layers}L x {n_experts}E (Top-{top_k}), data_start={data_start}")

    # Parse trace for empirical hotness ranking
    freq_counts = Counter()
    if trace_path and os.path.exists(trace_path):
        print(f"[MAP-GEN] Parsing empirical trace: {trace_path} ...")
        with open(trace_path, "r", encoding="utf-8") as tf:
            for line in tf:
                line = line.strip()
                if not line.startswith("ATLAS_MOE"):
                    continue
                m = re.search(r"layer=(\d+).*vals=([\d,]+)", line)
                if m:
                    layer = int(m.group(1))
                    vals = [int(x) for x in m.group(2).split(",")]
                    for e in vals:
                        freq_counts[(layer, e)] += 1
        print(f"[MAP-GEN] Found {len(freq_counts)} uniquely activated experts in trace.")

    # Build all (layer, expert) keys and rank them by frequency descending, then layer, expert ascending
    all_keys = []
    for l in range(n_layers):
        for e in range(n_experts):
            all_keys.append((l, e))

    all_keys.sort(key=lambda k: (-freq_counts[k], k[0], k[1]))

    tier_assignments = {}
    for idx, key in enumerate(all_keys):
        if idx < tier0_count:
            tier_assignments[key] = ("tier0_vram", idx < 64)  # Pin only core top-64 anchors
        elif idx < tier0_count + tier1_count:
            tier_assignments[key] = ("tier1_ram", False)
        else:
            tier_assignments[key] = ("tier2_cold", False)

    resolved_model_path = str(Path(model_path).resolve())

    # Build expert map
    expert_map = {}
    total_exp_bytes = 0

    print(f"[MAP-GEN] Calculating chunk offsets for {len(all_keys)} experts ...")
    for l in range(n_layers):
        gate_name = f"blk.{l}.ffn_gate_exps.weight"
        up_name = f"blk.{l}.ffn_up_exps.weight"
        down_name = f"blk.{l}.ffn_down_exps.weight"

        gate_t = tensor_map.get(gate_name)
        up_t = tensor_map.get(up_name)
        down_t = tensor_map.get(down_name)

        if not (gate_t and up_t and down_t):
            raise RuntimeError(f"Missing expert tensors for layer {l}")

        gate_slice = gate_t.storage_extent_bytes // n_experts
        up_slice = up_t.storage_extent_bytes // n_experts
        down_slice = down_t.storage_extent_bytes // n_experts
        exp_total_bytes = gate_slice + up_slice + down_slice
        total_exp_bytes = exp_total_bytes

        for e in range(n_experts):
            key_str = f"{l}:{e}"
            tier_name, pinned = tier_assignments.get((l, e), ("tier2_cold", False))
            gate_off = data_start + gate_t.offset + e * gate_slice
            up_off = data_start + up_t.offset + e * up_slice
            down_off = data_start + down_t.offset + e * down_slice

            # Ensure 'kind' is first key in chunk dict for parse_json_map compatibility
            expert_map[key_str] = {
                "layer": l,
                "expert_id": e,
                "tier": tier_name,
                "pinned": pinned,
                "total_bytes": exp_total_bytes,
                "chunks": [
                    {
                        "kind": "gate",
                        "tensor": gate_name,
                        "blob_path": resolved_model_path,
                        "file_offset": gate_off,
                        "size_bytes": gate_slice,
                        "type": "IQ3_S",
                    },
                    {
                        "kind": "up",
                        "tensor": up_name,
                        "blob_path": resolved_model_path,
                        "file_offset": up_off,
                        "size_bytes": up_slice,
                        "type": "IQ3_S",
                    },
                    {
                        "kind": "down",
                        "tensor": down_name,
                        "blob_path": resolved_model_path,
                        "file_offset": down_off,
                        "size_bytes": down_slice,
                        "type": "IQ3_XXS",
                    },
                ],
            }

    print(f"[MAP-GEN] Generated {len(expert_map)} experts (each {total_exp_bytes} B = {total_exp_bytes / (1024**2):.3f} MB).")

    payload = {
        "format": "atlas-physical-map-v1",
        "source": os.path.basename(model_path),
        "mode": "3tier_dynamic_cascading",
        "file": {
            "path": resolved_model_path,
            "file_size": file_size,
            "data_start": data_start,
            "alignment": alignment,
        },
        "model": {
            "name": meta.get("general.name", "Qwen3.8-Flash-Next"),
            "architecture": arch,
            "block_count": n_layers,
            "layers": n_layers,
            "expert_count": n_experts,
            "experts": n_experts,
            "expert_used_count": top_k,
            "top_k": top_k,
            "expert_feed_forward_length": meta.get("qwen4exp.expert_feed_forward_length", 640),
            "total_physical_size_bytes": file_size,
            "total_physical_size_gb": round(file_size / (1024**3), 3),
        },
        "3tier_configuration": {
            "tier0_vram_experts": tier0_count,
            "tier1_ram_experts": tier1_count,
            "tier2_cold_experts": len(expert_map) - tier0_count - tier1_count,
            "expert_count_total": len(expert_map),
            "expert_slice_bytes": total_exp_bytes,
            "direct_io": True,
            "dynamic_cascading_working_set": True,
        },
        "notes": [
            "Exact GGUF chunk byte offsets for all 48L x 512E experts.",
            f"Calibrated expert slice bytes: {total_exp_bytes} B.",
            f"3-Tier ranking: {tier0_count} VRAM, {tier1_count} RAM, {len(expert_map) - tier0_count - tier1_count} NVMe.",
        ],
        "expert_count_total": len(expert_map),
        "expert_size_bytes": total_exp_bytes,
        "experts": expert_map,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MAP-GEN] Serializing map to {output_path} ...")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[MAP-GEN] Successfully saved physical map ({out_p.stat().st_size / (1024**2):.2f} MB)")


if __name__ == "__main__":
    model = r"models\qwen3.8-gsq-rco-iq3_xxs\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
    trace = r"config\trace_qwen38_128.jsonl"
    out = r"config\atlas_physical_map_qwen38_iq3xxs.json"
    generate_qwen38_physical_map(model, trace, out)
