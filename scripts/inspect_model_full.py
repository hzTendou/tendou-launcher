import os
import sys
import json
import glob
from pathlib import Path
import gguf

def resolve_path(p):
    path = Path(p)
    if path.is_symlink():
        target = path.resolve()
        return str(path), str(target), True
    return str(path), str(path), False

def inspect_all():
    snapshot_dir = r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d"
    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.gguf")))
    
    parts_info = []
    total_physical_size = 0
    for f in files:
        snap_path, blob_path, is_symlink = resolve_path(f)
        size = os.path.getsize(blob_path)
        total_physical_size += size
        parts_info.append({
            "snapshot_path": snap_path,
            "filename": os.path.basename(f),
            "blob_path": blob_path,
            "is_reparse_or_symlink": is_symlink,
            "size_bytes": size,
            "size_gb": round(size / (1024**3), 3)
        })
    
    print(f"Discovered {len(files)} GGUF parts, total size: {round(total_physical_size / (1024**3), 3)} GB")
    
    # Read part 1 for metadata
    reader1 = gguf.GGUFReader(files[0])
    meta = {}
    for k in reader1.fields:
        f = reader1.get_field(k)
        if f.types and f.types[0] == gguf.GGUFValueType.STRING:
            meta[k] = str(bytes(f.parts[-1]), encoding="utf-8", errors="replace")
        elif f.types and f.types[0] in (gguf.GGUFValueType.INT32, gguf.GGUFValueType.INT64,
                                        gguf.GGUFValueType.UINT32, gguf.GGUFValueType.UINT64,
                                        gguf.GGUFValueType.INT16, gguf.GGUFValueType.UINT16,
                                        gguf.GGUFValueType.INT8, gguf.GGUFValueType.UINT8):
            meta[k] = int(f.parts[-1][0])
        elif f.types and f.types[0] in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            meta[k] = float(f.parts[-1][0])
        elif f.types and f.types[0] == gguf.GGUFValueType.BOOL:
            meta[k] = bool(f.parts[-1][0])
        elif f.types and f.types[0] == gguf.GGUFValueType.ARRAY:
            meta[k] = f"Array(len={len(f.parts)-2})"

    # Now inspect all tensors across all parts
    all_tensors = []
    file_tensor_map = {}
    for idx, f in enumerate(files):
        snap_path, blob_path, _ = resolve_path(f)
        r = gguf.GGUFReader(f)
        for t in r.tensors:
            t_info = {
                "name": t.name,
                "shape": t.shape.tolist(),
                "type": t.tensor_type.name,
                "file_index": idx,
                "blob_path": blob_path,
                "offset": int(t.data_offset),
                "nbytes": int(t.n_bytes)
            }
            all_tensors.append(t_info)
            file_tensor_map[t.name] = t_info

    # Inspect MTP file
    mtp_snap = r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\38bb39ee97821de2c9009abb7e93950eec396e66\MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    mtp_snap_path, mtp_blob_path, mtp_is_symlink = resolve_path(mtp_snap)
    mtp_size = os.path.getsize(mtp_blob_path) if os.path.exists(mtp_blob_path) else 0
    mtp_reader = gguf.GGUFReader(mtp_snap)
    mtp_meta = {}
    for k in mtp_reader.fields:
        f = mtp_reader.get_field(k)
        if f.types and f.types[0] == gguf.GGUFValueType.STRING:
            mtp_meta[k] = str(bytes(f.parts[-1]), encoding="utf-8", errors="replace")
        elif f.types and f.types[0] in (gguf.GGUFValueType.INT32, gguf.GGUFValueType.INT64,
                                        gguf.GGUFValueType.UINT32, gguf.GGUFValueType.UINT64):
            mtp_meta[k] = int(f.parts[-1][0])
        elif f.types and f.types[0] in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            mtp_meta[k] = float(f.parts[-1][0])
        elif f.types and f.types[0] == gguf.GGUFValueType.BOOL:
            mtp_meta[k] = bool(f.parts[-1][0])

    mtp_tensors = [{"name": t.name, "shape": t.shape.tolist(), "type": t.tensor_type.name, "nbytes": int(t.n_bytes)} for t in mtp_reader.tensors]

    audit_result = {
        "model": {
            "name": meta.get("general.name", "Qwen3.8-Flash-Next"),
            "architecture": meta.get("general.architecture", "qwen4exp"),
            "layers": meta.get("qwen4exp.block_count", 48),
            "hidden_size": meta.get("qwen4exp.embedding_length", 2560),
            "intermediate_size": meta.get("qwen4exp.expert_feed_forward_length", 640),
            "shared_intermediate_size": meta.get("qwen4exp.expert_shared_feed_forward_length", 640),
            "attention_heads": meta.get("qwen4exp.attention.head_count", 24),
            "kv_heads": meta.get("qwen4exp.attention.head_count_kv", 2),
            "experts": meta.get("qwen4exp.expert_count", 512),
            "top_k": meta.get("qwen4exp.expert_used_count", 10),
            "shared_experts": 1,
            "vocab_size": meta.get("qwen4exp.vocab_size", 248320),
            "context_length": meta.get("qwen4exp.context_length", 262144),
            "full_attention_interval": meta.get("qwen4exp.full_attention_interval", 4),
            "ssm": {
                "conv_kernel": meta.get("qwen4exp.ssm.conv_kernel", 4),
                "state_size": meta.get("qwen4exp.ssm.state_size", 128),
                "group_count": meta.get("qwen4exp.ssm.group_count", 16),
                "time_step_rank": meta.get("qwen4exp.ssm.time_step_rank", 48),
                "inner_size": meta.get("qwen4exp.ssm.inner_size", 6144),
            },
            "hyper_connection": {
                "count": meta.get("qwen4exp.hyper_connection.count", 4),
                "low_rank": meta.get("qwen4exp.hyper_connection.low_rank", 320),
            },
            "indexer": {
                "head_count": meta.get("qwen4exp.attention.indexer.head_count", 4),
                "key_length": meta.get("qwen4exp.attention.indexer.key_length", 128),
                "top_k": meta.get("qwen4exp.attention.indexer.top_k", 2048),
            },
            "ple": {
                "layers": [1],
                "ngram_size": meta.get("qwen4exp.ple.ngram_size", 3),
                "heads_per_ngram": meta.get("qwen4exp.ple.heads_per_ngram", 8),
                "conv_kernel": meta.get("qwen4exp.ple.conv_kernel", 4),
                "embedding_length_per_layer": meta.get("qwen4exp.embedding_length_per_layer_input", 160)
            },
            "total_tensors": len(all_tensors),
            "parts": parts_info,
            "total_physical_size_bytes": total_physical_size,
            "total_physical_size_gb": round(total_physical_size / (1024**3), 3)
        },
        "mtp": {
            "path": mtp_snap_path,
            "blob_path": mtp_blob_path,
            "size_bytes": mtp_size,
            "size_gb": round(mtp_size / (1024**3), 3),
            "architecture": mtp_meta.get("general.architecture", "qwen4exp"),
            "predict_layers": mtp_meta.get("qwen4exp.nextn_predict_layers", 1),
            "required_for_normal_inference": False,
            "tensor_count": len(mtp_tensors),
            "tensors": mtp_tensors
        }
    }

    # Now build the expert mapping
    # Each expert (layer, expert_id) has: gate, up, down chunks
    expert_map = {}
    n_layers = audit_result["model"]["layers"]
    n_experts = audit_result["model"]["experts"]

    for layer in range(n_layers):
        gate_name = f"blk.{layer}.ffn_gate_exps.weight"
        up_name = f"blk.{layer}.ffn_up_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"

        gate_t = file_tensor_map.get(gate_name)
        up_t = file_tensor_map.get(up_name)
        down_t = file_tensor_map.get(down_name)

        if not (gate_t and up_t and down_t):
            print(f"Warning: missing expert tensors for layer {layer}")
            continue

        gate_plane = gate_t["nbytes"] // n_experts
        up_plane = up_t["nbytes"] // n_experts
        down_plane = down_t["nbytes"] // n_experts
        expert_size = gate_plane + up_plane + down_plane

        for exp in range(n_experts):
            key = f"{layer}:{exp}"
            expert_map[key] = {
                "layer": layer,
                "expert_id": exp,
                "total_bytes": expert_size,
                "chunks": [
                    {
                        "kind": "gate",
                        "tensor": gate_name,
                        "file_index": gate_t["file_index"],
                        "blob_path": gate_t["blob_path"],
                        "file_offset": gate_t["offset"] + exp * gate_plane,
                        "size_bytes": gate_plane,
                        "type": gate_t["type"]
                    },
                    {
                        "kind": "up",
                        "tensor": up_name,
                        "file_index": up_t["file_index"],
                        "blob_path": up_t["blob_path"],
                        "file_offset": up_t["offset"] + exp * up_plane,
                        "size_bytes": up_plane,
                        "type": up_t["type"]
                    },
                    {
                        "kind": "down",
                        "tensor": down_name,
                        "file_index": down_t["file_index"],
                        "blob_path": down_t["blob_path"],
                        "file_offset": down_t["offset"] + exp * down_plane,
                        "size_bytes": down_plane,
                        "type": down_t["type"]
                    }
                ]
            }

    print(f"Built expert physical map for {len(expert_map)} experts across {n_layers} layers.")
    sample_key = "0:0"
    print(f"Sample expert {sample_key}: total_bytes={expert_map[sample_key]['total_bytes']} bytes (~{round(expert_map[sample_key]['total_bytes'] / 1048576, 2)} MB)")

    # Save to config/atlas_physical_map_qwen38.json
    out_map = {
        "format": "atlas-physical-map-v1",
        "model": audit_result["model"],
        "mtp": audit_result["mtp"],
        "expert_count_total": len(expert_map),
        "expert_size_bytes": expert_map["0:0"]["total_bytes"],
        "experts": expert_map
    }

    out_path = r"config\atlas_physical_map_qwen38.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_map, f, indent=2)
    print(f"Saved physical map to {out_path}")

    # Also save audit report
    report_path = r"config\qwen38_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)
    print(f"Saved audit report to {report_path}")

if __name__ == "__main__":
    inspect_all()
