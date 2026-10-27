"""
Generate exact 3-tier physical map for Qwen3.5-122B-A10B MXFP4_MOE (3-shard GGUF).
Extracts exact GGUF chunk byte offsets for all 48 layers x 256 experts (12,288 experts)
across Shard 2 (Layers 0..30) and Shard 3 (Layers 31..47).

Architecture Details:
  - Architecture: qwen35moe (Qwen 3.5 MoE)
  - 48 Layers, 256 Experts per layer, Top-8 routed + 1 Shared Expert per layer
  - Embedding length: 3072, Expert FFN length: 1024, Shared FFN: 1024
  - Gate slice: 1,671,168 B (1.594 MiB) [MXFP4]
  - Up slice:   1,671,168 B (1.594 MiB) [MXFP4]
  - Down slice: 2,162,688 B (2.063 MiB) [Q5_K]
  - Total per expert: 5,505,024 B (5.250 MiB)
  - Total routed experts: 12,288 (64.51 GiB / 69.27 GB)
  - Total model size across 3 shards: 74,664,408,608 bytes (74.66 GB / 69.54 GiB)
"""
import os
import sys
import json
from pathlib import Path
import gguf

ROOT = Path(__file__).resolve().parents[1]

def generate_122b_mxfp4_physical_map(
    shard1_path: Path,
    shard2_path: Path,
    shard3_path: Path,
    output_path: Path,
    tier0_count: int = 0,
    tier1_count: int = 600,
):
    print(f"[MAP-GEN] Parsing GGUF headers...")
    r1 = gguf.GGUFReader(shard1_path)
    r2 = gguf.GGUFReader(shard2_path)
    r3 = gguf.GGUFReader(shard3_path)

    # Read model architecture from Shard 1 metadata
    def get_field_val(reader, key, default=None):
        f = reader.get_field(key)
        if f is None:
            return default
        parts = f.parts
        if not parts:
            return default
        val = parts[-1]
        if hasattr(val, "tolist"):
            val = val.tolist()
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            elif isinstance(val, list) and isinstance(val[0], int) and len(val) > 1:
                # String bytes
                try:
                    return bytes(val).decode("utf-8")
                except Exception:
                    pass
        return val

    arch = "qwen35moe"
    name = "Qwen3.5-122B-A10B-MXFP4_MOE"
    n_layers = 48
    n_experts = 256
    top_k = 8
    emb_len = 3072
    ffn_exp_len = 1024
    shared_ffn_len = 1024

    for k, v in r1.fields.items():
        if "general.name" in k:
            name = str(r1.get_field(k).parts[-1].tobytes().decode("utf-8", errors="ignore"))
        elif "general.architecture" in k:
            arch = str(r1.get_field(k).parts[-1].tobytes().decode("utf-8", errors="ignore"))

    total_file_size = shard1_path.stat().st_size + shard2_path.stat().st_size + shard3_path.stat().st_size
    print(f"[MAP-GEN] Model: {name}, Arch: {arch}, Layers: {n_layers}, Experts: {n_experts}, Top-{top_k}")
    print(f"[MAP-GEN] Total 3-shard file size: {total_file_size:,} bytes ({total_file_size / 1024**3:.2f} GiB)")

    # Build tensor maps for Shard 2 and Shard 3
    t_map = {}
    for t in r2.tensors:
        if "_exps" in t.name:
            t_map[t.name] = (shard2_path.resolve(), t)
    for t in r3.tensors:
        if "_exps" in t.name:
            t_map[t.name] = (shard3_path.resolve(), t)

    print(f"[MAP-GEN] Total MoE tensors indexed: {len(t_map)}")
    assert len(t_map) == n_layers * 3, f"Expected {n_layers * 3} tensors, got {len(t_map)}"

    # Tier assignment: anchor lower layers into tier1 RAM, rest into tier2 cold NVMe
    tier_assignments = {}
    all_keys = [(l, e) for l in range(n_layers) for e in range(n_experts)]
    # Natural ordering: anchor layer 0..2 first
    all_keys.sort(key=lambda k: (0 if k[0] < 2 else 1, k[0], k[1]))

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

        gate_path, gate_t = t_map[gate_name]
        up_path, up_t = t_map[up_name]
        down_path, down_t = t_map[down_name]

        gate_slice = gate_t.n_bytes // n_experts
        up_slice = up_t.n_bytes // n_experts
        down_slice = down_t.n_bytes // n_experts
        exp_total_bytes = gate_slice + up_slice + down_slice
        sample_exp_bytes = exp_total_bytes

        gate_type = gate_t.tensor_type.name
        up_type = up_t.tensor_type.name
        down_type = down_t.tensor_type.name

        for e in range(n_experts):
            key_str = f"{l}:{e}"
            tier_name, pinned = tier_assignments.get((l, e), ("tier2_cold", False))
            gate_off = gate_t.data_offset + e * gate_slice
            up_off = up_t.data_offset + e * up_slice
            down_off = down_t.data_offset + e * down_slice

            # Strict key order for C++ line-parser compatibility
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
                        "blob_path": str(gate_path),
                        "file_offset": gate_off,
                        "size_bytes": gate_slice,
                        "type": gate_type,
                    },
                    {
                        "kind": "up",
                        "tensor": up_name,
                        "blob_path": str(up_path),
                        "file_offset": up_off,
                        "size_bytes": up_slice,
                        "type": up_type,
                    },
                    {
                        "kind": "down",
                        "tensor": down_name,
                        "blob_path": str(down_path),
                        "file_offset": down_off,
                        "size_bytes": down_slice,
                        "type": down_type,
                    },
                ],
            }

    print(f"[MAP-GEN] Generated {len(expert_map)} experts (each {sample_exp_bytes} B = {sample_exp_bytes / (1024**2):.3f} MB).")

    payload = {
        "format": "atlas-physical-map-v1",
        "source": shard1_path.name,
        "mode": "3tier_dynamic_cascading",
        "file": {
            "path": str(shard1_path.resolve()),
            "file_size": total_file_size,
            "data_start": 0,
            "alignment": 32,
        },
        "model": {
            "name": name,
            "architecture": arch,
            "block_count": n_layers,
            "layers": n_layers,
            "expert_count": n_experts,
            "experts": n_experts,
            "expert_used_count": top_k,
            "top_k": top_k,
            "expert_feed_forward_length": ffn_exp_len,
            "expert_shared_feed_forward_length": shared_ffn_len,
            "total_physical_size_bytes": total_file_size,
            "total_physical_size_gb": round(total_file_size / (1024**3), 3),
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
            "Tendou Launcher 3-Tier Physical Map for Qwen3.5-122B-A10B (MXFP4_MOE 3-Shard)",
            "48 layers x 256 experts = 12,288 total routed experts + 48 shared experts",
            "Each expert slice: 5,505,024 bytes (Gate MXFP4 1.594 MB + Up MXFP4 1.594 MB + Down Q5_K 2.063 MB)",
            "32% smaller expert slices vs Q6_K_P (5.25 MB vs 7.74 MB), saving 1.0 GB per-token memory traffic",
            "Dense weights (6.19 GB) + Shared experts + KV (0.38 GB) offloaded to 8 GB Laptop VRAM via -ngl 99",
            "Tier 1 RAM: ~600 warm experts (~3.15 GB host RAM VirtualLock pinned)",
            "Tier 2 Cold NVMe: remaining 11,688 experts (~61.3 GB on disk)",
            "DirectStorage / BypassIO streaming active across Shard 2 and Shard 3",
        ],
        "experts": expert_map,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MAP-GEN] Writing JSON physical map to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)

    written_size = output_path.stat().st_size
    print(f"[MAP-GEN] SUCCESS! Written {written_size:,} bytes ({written_size / 1024**2:.2f} MB) to {output_path}")


if __name__ == "__main__":
    shard1 = ROOT / "models" / "MXFP4_MOE" / "Qwen3.5-122B-A10B-MXFP4_MOE-00001-of-00003.gguf"
    shard2 = ROOT / "models" / "MXFP4_MOE" / "Qwen3.5-122B-A10B-MXFP4_MOE-00002-of-00003.gguf"
    shard3 = ROOT / "models" / "MXFP4_MOE" / "Qwen3.5-122B-A10B-MXFP4_MOE-00003-of-00003.gguf"

    if not (shard1.is_file() and shard2.is_file() and shard3.is_file()):
        raise FileNotFoundError(f"Missing one or more shards in {ROOT / 'models' / 'MXFP4_MOE'}")

    out_json = ROOT / "config" / "atlas_physical_map_122b_mxfp4.json"
    generate_122b_mxfp4_physical_map(shard1, shard2, shard3, out_json)
