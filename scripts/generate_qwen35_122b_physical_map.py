"""
Generate exact 3-tier physical map for Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive (Q6_K_P).
Extracts exact GGUF chunk byte offsets for all 48 layers x 256 experts (12,288 experts).

Architecture Details:
  - Architecture: qwen35moe (Qwen 3.5 MoE)
  - 48 Layers, 256 Experts per layer, Top-8 routed + 1 Shared Expert per layer
  - Embedding length: 3072, Expert FFN length: 1024, Shared FFN: 1024
  - Gate slice: 3,342,336 B (3.19 MiB) [Q8_0]
  - Up slice:   3,342,336 B (3.19 MiB) [Q8_0]
  - Down slice: 2,580,480 B (2.46 MiB) [Q6_K]
  - Total per expert: 9,265,152 B (8.836 MiB)
  - Total routed experts: 12,288 (108.55 GiB / 92.23 GiB uncompressed slices)
  - File size: 104,576,886,368 bytes (97.39 GiB)
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

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    23: "IQ4_XS",
    28: "IQ3_XXS",
    29: "IQ3_S",
}


def generate_qwen35_122b_physical_map(
    header_source: str,
    output_path: str,
    target_model_path: str = None,
    trace_path: str = None,
    tier0_count: int = 180,
    tier1_count: int = 600,
):
    print(f"[MAP-GEN] Parsing GGUF metadata from {header_source} ...")
    parsed = parse_gguf(header_source)
    meta = parsed["metadata"]

    arch = str(meta.get("general.architecture", "qwen35moe"))
    n_layers = int(meta.get("qwen35moe.block_count", 48))
    n_experts = int(meta.get("qwen35moe.expert_count", 256))
    top_k = int(meta.get("qwen35moe.expert_used_count", 8))
    emb_len = int(meta.get("qwen35moe.embedding_length", 3072))
    ffn_exp_len = int(meta.get("qwen35moe.expert_feed_forward_length", 1024))
    shared_ffn_len = int(meta.get("qwen35moe.expert_shared_feed_forward_length", 1024))

    data_start = parsed["data_start"]
    file_size = 104576886368  # Exact final GGUF file size
    alignment = parsed.get("alignment", 32)
    tensor_map = {t.name: t for t in parsed["tensors"]}

    print(f"[MAP-GEN] Model: {meta.get('general.name', 'Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive')}")
    print(f"[MAP-GEN] Architecture: {arch}, {n_layers}L x {n_experts}E (Top-{top_k}), data_start={data_start}")
    print(f"[MAP-GEN] Embedding: {emb_len}, Expert FFN: {ffn_exp_len}, Shared FFN: {shared_ffn_len}")

    if target_model_path is None:
        target_model_path = str((ROOT / "models" / "Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf").resolve())

    # Parse trace for empirical hotness ranking if available
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

    # Build all (layer, expert) keys and rank
    all_keys = []
    for l in range(n_layers):
        for e in range(n_experts):
            all_keys.append((l, e))

    if freq_counts:
        all_keys.sort(key=lambda k: (-freq_counts[k], k[0], k[1]))
    else:
        # Default prior: prioritize lower-layer core experts and anchor layer 0-1
        all_keys.sort(key=lambda k: (0 if k[0] < 2 else 1, k[0], k[1]))

    tier_assignments = {}
    for idx, key in enumerate(all_keys):
        if idx < tier0_count:
            tier_assignments[key] = ("tier0_vram", idx < 32)
        elif idx < tier0_count + tier1_count:
            tier_assignments[key] = ("tier1_ram", False)
        else:
            tier_assignments[key] = ("tier2_cold", False)

    expert_map = {}
    sample_exp_bytes = 0

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
        sample_exp_bytes = exp_total_bytes

        gate_type = GGML_TYPE_NAMES.get(gate_t.type_id, "Q8_0")
        up_type = GGML_TYPE_NAMES.get(up_t.type_id, "Q8_0")
        down_type = GGML_TYPE_NAMES.get(down_t.type_id, "Q6_K")

        for e in range(n_experts):
            key_str = f"{l}:{e}"
            tier_name, pinned = tier_assignments.get((l, e), ("tier2_cold", False))
            gate_off = data_start + gate_t.offset + e * gate_slice
            up_off = data_start + up_t.offset + e * up_slice
            down_off = data_start + down_t.offset + e * down_slice

            # Ensure 'kind' is first key in chunk dict for atlas-engine.cpp line parser compatibility
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
                        "blob_path": target_model_path,
                        "file_offset": gate_off,
                        "size_bytes": gate_slice,
                        "type": gate_type,
                    },
                    {
                        "kind": "up",
                        "tensor": up_name,
                        "blob_path": target_model_path,
                        "file_offset": up_off,
                        "size_bytes": up_slice,
                        "type": up_type,
                    },
                    {
                        "kind": "down",
                        "tensor": down_name,
                        "blob_path": target_model_path,
                        "file_offset": down_off,
                        "size_bytes": down_slice,
                        "type": down_type,
                    },
                ],
            }

    print(f"[MAP-GEN] Generated {len(expert_map)} experts (each {sample_exp_bytes} B = {sample_exp_bytes / (1024**2):.3f} MB).")

    payload = {
        "format": "atlas-physical-map-v1",
        "source": os.path.basename(target_model_path),
        "mode": "3tier_dynamic_cascading",
        "file": {
            "path": target_model_path,
            "file_size": file_size,
            "data_start": data_start,
            "alignment": alignment,
        },
        "model": {
            "name": meta.get("general.name", "Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive"),
            "architecture": arch,
            "block_count": n_layers,
            "layers": n_layers,
            "expert_count": n_experts,
            "experts": n_experts,
            "expert_used_count": top_k,
            "top_k": top_k,
            "expert_feed_forward_length": ffn_exp_len,
            "expert_shared_feed_forward_length": shared_ffn_len,
            "total_physical_size_bytes": file_size,
            "total_physical_size_gb": round(file_size / (1024**3), 3),
        },
        "3tier_configuration": {
            "tier0_vram_experts": tier0_count,
            "tier1_ram_experts": tier1_count,
            "tier2_cold_experts": len(expert_map) - tier0_count - tier1_count,
            "expert_count_total": len(expert_map),
            "expert_slice_bytes": sample_exp_bytes,
            "direct_io": True,
            "dynamic_cascading_working_set": True,
        },
        "notes": [
            "Tendou Launcher 3-Tier Physical Map for Qwen3.5-122B-A10B (Q6_K_P)",
            "48 layers x 256 experts = 12,288 total routed experts + 48 shared experts",
            "Each expert slice: 9,265,152 bytes (Gate Q8_0 3.19 MB + Up Q8_0 3.19 MB + Down Q6_K 2.46 MB)",
            "Tier 0 VRAM: ~180 hot experts (~1.6 GB VRAM)",
            "Tier 1 RAM: ~600 warm experts (~5.3 GB host RAM VirtualLock pinned)",
            "Tier 2 Cold NVMe: remaining 11,508 experts (~101.6 GB on disk)",
            "Dense weights (4.79 GB) + Shared experts (0.37 GB) + KV (0.38 GB) fit into 8 GB Laptop VRAM",
        ],
        "experts": expert_map,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MAP-GEN] Writing JSON physical map to {output_path} ...")
    with open(out_file, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)

    written_size = out_file.stat().st_size
    print(f"[MAP-GEN] SUCCESS! Written {written_size:,} bytes ({written_size / 1024**2:.2f} MB) to {output_path}")


if __name__ == "__main__":
    incomplete_candidates = list((ROOT / "models" / ".cache" / "huggingface" / "download").glob("*.incomplete"))
    target = ROOT / "models" / "Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
    
    if target.is_file():
        header_file = str(target)
    elif incomplete_candidates:
        header_file = str(incomplete_candidates[0])
    else:
        raise FileNotFoundError("Could not find downloaded or incomplete model file to read GGUF header from.")

    out_json = ROOT / "config" / "atlas_physical_map_122b_q6k.json"
    generate_qwen35_122b_physical_map(header_file, str(out_json), target_model_path=str(target))
