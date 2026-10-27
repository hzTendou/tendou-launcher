"""
Generate exact 3-tier physical map for Llama-4-Scout-17B-16E (Q6_K).
Extracts exact GGUF chunk byte offsets for all 48 layers x 16 experts (768 experts).

Architecture Details:
  - Architecture: llama4 (Meta Llama 4 Scout)
  - 48 Layers, 16 Experts per layer, Top-1 routed + 1 Shared Expert per layer
  - Embedding length: 5120, Expert FFN length: 8192
  - Expert tensor type: GGML_TYPE_Q6_K (14)
  - Gate slice: 34,406,400 B (32.81 MiB)
  - Up slice:   34,406,400 B (32.81 MiB)
  - Down slice: 34,406,400 B (32.81 MiB)
  - Total per expert: 103,219,200 B (98.44 MiB)
  - Total routed experts: 768 (75.60 GiB / 79.27 GB)

Split Architecture:
  - Part 1 (45,097,156,608 B): Layers 0 to 23 (384 experts)
  - Part 2 (43,334,989,408 B): Layers 24 to 47 (384 experts)
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))
from gguf_utils import parse_gguf

PART1_SIZE = 45097156608
PART2_SIZE = 43334989408
TOTAL_SIZE = PART1_SIZE + PART2_SIZE


def generate_llama4_physical_map(
    header_source: str,
    output_path: str,
    part1_path: str = None,
    part2_path: str = None,
    tier0_count: int = 32,
    tier1_count: int = 64,
):
    print(f"[MAP-GEN] Parsing GGUF metadata from {header_source} ...")
    parsed = parse_gguf(header_source)
    meta = parsed["metadata"]

    arch = str(meta.get("general.architecture", "llama4"))
    n_layers = int(meta.get("llama4.block_count", 48))
    n_experts = int(meta.get("llama4.expert_count", 16))
    top_k = int(meta.get("llama4.expert_used_count", 1))
    emb_len = int(meta.get("llama4.embedding_length", 5120))
    ffn_exp_len = int(meta.get("llama4.expert_feed_forward_length", 8192))

    data_start = parsed["data_start"]
    alignment = parsed.get("alignment", 32)
    tensor_map = {t.name: t for t in parsed["tensors"]}

    print(f"[MAP-GEN] Model: {meta.get('general.name', 'Llama 4 Scout 17B 16E')}")
    print(f"[MAP-GEN] Architecture: {arch}, {n_layers}L x {n_experts}E (Top-{top_k}), data_start={data_start}")
    print(f"[MAP-GEN] Embedding: {emb_len}, Expert FFN: {ffn_exp_len}")

    if part1_path is None:
        part1_path = str((ROOT / "models" / "Llama-4-Scout-17B-16E.i1-Q6_K.gguf.part1of2").resolve())
    if part2_path is None:
        part2_path = str((ROOT / "models" / "Llama-4-Scout-17B-16E.i1-Q6_K.gguf.part2of2").resolve())

    # Build expert list
    all_keys = []
    for l in range(n_layers):
        for e in range(n_experts):
            all_keys.append((l, e))

    tier_assignments = {}
    for idx, key in enumerate(all_keys):
        if idx < tier0_count:
            tier_assignments[key] = ("tier0_vram", idx < 8)
        elif idx < tier0_count + tier1_count:
            tier_assignments[key] = ("tier1_ram", False)
        else:
            tier_assignments[key] = ("tier2_cold", False)

    # Calculate exact tensor offsets & chunk definitions
    # In Q6_K: 256 elements take 210 bytes
    gate_slice = (emb_len * ffn_exp_len * 210) // 256
    up_slice   = (emb_len * ffn_exp_len * 210) // 256
    down_slice = (ffn_exp_len * emb_len * 210) // 256
    exp_total_bytes = gate_slice + up_slice + down_slice

    assert gate_slice == 34406400, f"Expected 34406400, got {gate_slice}"
    assert exp_total_bytes == 103219200, f"Expected 103219200, got {exp_total_bytes}"

    expert_map = {}
    print(f"[MAP-GEN] Processing {len(all_keys)} experts ({n_layers}L x {n_experts}E) ...")

    for l in range(n_layers):
        gate_name = f"blk.{l}.ffn_gate_exps.weight"
        up_name   = f"blk.{l}.ffn_up_exps.weight"
        down_name = f"blk.{l}.ffn_down_exps.weight"

        gate_t = tensor_map.get(gate_name)
        up_t   = tensor_map.get(up_name)
        down_t = tensor_map.get(down_name)

        if not (gate_t and up_t and down_t):
            raise RuntimeError(f"Missing expert tensors for layer {l} in GGUF index")

        # Global file offsets from beginning of continuous model stream
        gate_global = data_start + gate_t.offset
        up_global   = data_start + up_t.offset
        down_global = data_start + down_t.offset

        # Determine split file assignment
        # Layers 0..23 in Part 1, Layers 24..47 in Part 2
        for e in range(n_experts):
            key_str = f"{l}:{e}"
            tier_name, pinned = tier_assignments[(l, e)]

            g_off_glob = gate_global + e * gate_slice
            u_off_glob = up_global + e * up_slice
            d_off_glob = down_global + e * down_slice

            # Check blob location
            if g_off_glob < PART1_SIZE:
                blob_path = part1_path
                g_off = g_off_glob
                u_off = u_off_glob
                d_off = d_off_glob
            else:
                blob_path = part2_path
                g_off = g_off_glob - PART1_SIZE
                u_off = u_off_glob - PART1_SIZE
                d_off = d_off_glob - PART1_SIZE

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
                        "blob_path": blob_path,
                        "file_offset": g_off,
                        "size_bytes": gate_slice,
                        "type": "Q6_K",
                    },
                    {
                        "kind": "up",
                        "tensor": up_name,
                        "blob_path": blob_path,
                        "file_offset": u_off,
                        "size_bytes": up_slice,
                        "type": "Q6_K",
                    },
                    {
                        "kind": "down",
                        "tensor": down_name,
                        "blob_path": blob_path,
                        "file_offset": d_off,
                        "size_bytes": down_slice,
                        "type": "Q6_K",
                    },
                ],
            }

    print(f"[MAP-GEN] Generated {len(expert_map)} experts.")
    print(f"[MAP-GEN] Expert slice: {exp_total_bytes:,} B ({exp_total_bytes / (1024**2):.2f} MiB).")
    print(f"[MAP-GEN] Tier 0 (VRAM): {tier0_count} experts ({tier0_count * exp_total_bytes / (1024**3):.2f} GiB)")
    print(f"[MAP-GEN] Tier 1 (RAM):  {tier1_count} experts ({tier1_count * exp_total_bytes / (1024**3):.2f} GiB)")
    print(f"[MAP-GEN] Tier 2 (Cold): {len(expert_map) - tier0_count - tier1_count} experts ({(len(expert_map) - tier0_count - tier1_count) * exp_total_bytes / (1024**3):.2f} GiB)")

    payload = {
        "format": "atlas-physical-map-v1",
        "source": "Llama-4-Scout-17B-16E.i1-Q6_K.gguf",
        "mode": "3tier_dynamic_cascading",
        "files": {
            "part1": {
                "path": part1_path,
                "size_bytes": PART1_SIZE,
                "layers": "0-23",
                "experts_count": 384,
            },
            "part2": {
                "path": part2_path,
                "size_bytes": PART2_SIZE,
                "layers": "24-47",
                "experts_count": 384,
            },
            "data_start": data_start,
            "alignment": alignment,
        },
        "model": {
            "name": meta.get("general.name", "Llama 4 Scout 17B 16E"),
            "architecture": arch,
            "block_count": n_layers,
            "layers": n_layers,
            "expert_count": n_experts,
            "experts": n_experts,
            "expert_used_count": top_k,
            "top_k": top_k,
            "embedding_length": emb_len,
            "expert_feed_forward_length": ffn_exp_len,
            "has_shared_expert": True,
            "shared_expert_feed_forward_length": ffn_exp_len,
            "total_physical_size_bytes": TOTAL_SIZE,
            "total_physical_size_gb": round(TOTAL_SIZE / (1024**3), 3),
        },
        "3tier_configuration": {
            "tier0_vram_experts": tier0_count,
            "tier0_vram_bytes": tier0_count * exp_total_bytes,
            "tier0_vram_gib": round(tier0_count * exp_total_bytes / (1024**3), 3),
            "tier1_ram_experts": tier1_count,
            "tier1_ram_bytes": tier1_count * exp_total_bytes,
            "tier1_ram_gib": round(tier1_count * exp_total_bytes / (1024**3), 3),
            "tier2_cold_experts": len(expert_map) - tier0_count - tier1_count,
            "tier2_cold_bytes": (len(expert_map) - tier0_count - tier1_count) * exp_total_bytes,
            "tier2_cold_gib": round((len(expert_map) - tier0_count - tier1_count) * exp_total_bytes / (1024**3), 3),
            "expert_count_total": len(expert_map),
            "expert_slice_bytes": exp_total_bytes,
            "direct_io": True,
            "dynamic_cascading_working_set": True,
        },
        "notes": [
            "Exact physical GGUF chunk byte offsets for Meta Llama 4 Scout 17B 16E (Q6_K).",
            "48 layers x 16 experts = 768 routed experts + 48 shared experts.",
            f"Calibrated expert slice: {exp_total_bytes:,} bytes (~98.44 MiB).",
            f"3-Tier distribution: {tier0_count} VRAM (~{tier0_count * exp_total_bytes / 1024**3:.2f} GiB), {tier1_count} RAM (~{tier1_count * exp_total_bytes / 1024**3:.2f} GiB), {len(expert_map) - tier0_count - tier1_count} NVMe (~{(len(expert_map) - tier0_count - tier1_count) * exp_total_bytes / 1024**3:.2f} GiB).",
        ],
        "expert_count_total": len(expert_map),
        "expert_size_bytes": exp_total_bytes,
        "experts": expert_map,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MAP-GEN] Serializing map to {output_path} ...")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[MAP-GEN] Successfully saved physical map ({out_p.stat().st_size / (1024**2):.2f} MB)")


if __name__ == "__main__":
    header = r"scratch\test_head.bin"
    output = r"config\atlas_physical_map_llama4_scout_q6k.json"
    generate_llama4_physical_map(header, output)
