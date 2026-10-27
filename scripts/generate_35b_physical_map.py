"""
Generate the physical map for Qwen3.6-35B-A3B GGUF models (e.g. Q8_K_XL, IQ4_XS).
Computes exact per-expert chunk offsets for all 40 layers x 256 experts.
Outputs format compatible with atlas-engine.cpp, atlas_physical_simulator.py, and build_physical_map.py.
"""
import os
import sys
import json
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
}

def generate_physical_map(
    model_path: str,
    output_path: str,
    logical_map_path: str = None
):
    print(f"Parsing GGUF: {model_path} ...")
    parsed = parse_gguf(model_path)
    meta = parsed["metadata"]
    
    n_layers = int(meta.get("qwen35moe.block_count", 40))
    n_experts = int(meta.get("qwen35moe.expert_count", 256))
    top_k = int(meta.get("qwen35moe.expert_used_count", 8))
    arch = str(meta.get("general.architecture", "qwen35moe"))
    
    data_start = parsed["data_start"]
    file_size = parsed["file_size"]
    
    tensor_map = {t.name: t for t in parsed["tensors"]}
    
    tensor_index = []
    for t in parsed["tensors"]:
        extent = t.storage_extent_bytes
        tensor_index.append({
            "name": t.name,
            "dims": t.dims,
            "type_id": t.type_id,
            "offset": data_start + t.offset,
            "offset_relative": t.offset,
            "storage_bytes": extent,
            "storage_extent_bytes": extent,
            "size_exact": True
        })
    
    expert_map = {}
    
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
        
        gate_type = GGML_TYPE_NAMES.get(gate_t.type_id, "Q8_0")
        up_type = GGML_TYPE_NAMES.get(up_t.type_id, "Q8_0")
        down_type = GGML_TYPE_NAMES.get(down_t.type_id, "Q8_0")
        
        for e in range(n_experts):
            key = f"{l}:{e}"
            gate_off = data_start + gate_t.offset + e * gate_slice
            up_off = data_start + up_t.offset + e * up_slice
            down_off = data_start + down_t.offset + e * down_slice
            
            expert_map[key] = {
                "layer": l,
                "expert_id": e,
                "total_bytes": exp_total_bytes,
                "chunks": [
                    {
                        "kind": "gate",
                        "tensor": gate_name,
                        "file_index": 0,
                        "blob_path": str(Path(model_path).resolve()),
                        "file_offset": gate_off,
                        "offset": gate_off,
                        "size_bytes": gate_slice,
                        "type": gate_type,
                        "contiguous_expert_slice": True
                    },
                    {
                        "kind": "up",
                        "tensor": up_name,
                        "file_index": 0,
                        "blob_path": str(Path(model_path).resolve()),
                        "file_offset": up_off,
                        "offset": up_off,
                        "size_bytes": up_slice,
                        "type": up_type,
                        "contiguous_expert_slice": True
                    },
                    {
                        "kind": "down",
                        "tensor": down_name,
                        "file_index": 0,
                        "blob_path": str(Path(model_path).resolve()),
                        "file_offset": down_off,
                        "offset": down_off,
                        "size_bytes": down_slice,
                        "type": down_type,
                        "contiguous_expert_slice": True
                    }
                ]
            }
            
    print(f"Generated {len(expert_map)} experts across {n_layers} layers (each {exp_total_bytes} bytes).")
    
    logical = {}
    join_stats = {}
    if logical_map_path and os.path.exists(logical_map_path):
        with open(logical_map_path, "r", encoding="utf-8") as f:
            logical = json.load(f)
        observed = logical.get("experts", {})
        exact_joined = sum(1 for k in observed if k in expert_map)
        join_stats = {
            "logical_experts": len(observed),
            "physical_exact_joined": exact_joined,
            "not_exact_or_missing": len(observed) - exact_joined
        }
        print(f"Logical join: {exact_joined}/{len(observed)} matched.")

    payload = {
        "format": "atlas-physical-map-v1",
        "source": os.path.basename(model_path),
        "file": {
            "path": str(Path(model_path).resolve()),
            "file_size": file_size,
            "data_start": data_start,
            "alignment": parsed["alignment"]
        },
        "model": {
            "name": meta.get("general.name", "Qwen3.6-35B-A3B"),
            "architecture": arch,
            "block_count": n_layers,
            "layers": n_layers,
            "expert_count": n_experts,
            "experts": n_experts,
            "expert_used_count": top_k,
            "top_k": top_k,
            "context_length": meta.get("qwen35moe.context_length", 262144),
            "embedding_length": meta.get("qwen35moe.embedding_length", 2048),
            "intermediate_size": meta.get("qwen35moe.expert_feed_forward_length", 512),
            "shared_intermediate_size": meta.get("qwen35moe.expert_shared_feed_forward_length", 512),
            "total_physical_size_bytes": file_size,
            "total_physical_size_gb": round(file_size / (1024**3), 3)
        },
        "notes": [
            "Physical offsets are read from the actual GGUF tensor index.",
            f"Expert-level exact slices across {n_layers} layers x {n_experts} experts."
        ],
        "logical_map_joined": bool(logical_map_path),
        "expert_count_total": len(expert_map),
        "expert_size_bytes": exp_total_bytes,
        "join_stats": join_stats,
        "tensor_index": tensor_index,
        "experts": expert_map
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved physical map to {output_path} ({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    model = r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    out1 = r"config\atlas_physical_map_35b.json"
    out2 = r"config\atlas_physical_map.json"
    log_map = r"config\atlas_map_35b.json"
    generate_physical_map(model, out1, log_map)
    generate_physical_map(model, out2, log_map)
