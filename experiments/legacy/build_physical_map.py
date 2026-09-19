from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import defaultdict
from gguf_utils import parse_gguf, EXPERT_TENSOR_RE, exact_tensor_size


def pick_meta(meta, suffixes, default=None):
    for k, v in meta.items():
        if any(k.endswith(s) for s in suffixes):
            if isinstance(v, (int, float)):
                return int(v)
    return default


def bytes_for_tensor(t):
    return t.storage_bytes if t.storage_bytes is not None else t.storage_extent_bytes


def main():
    ap=argparse.ArgumentParser(description='Build Atlas physical GGUF map and join trace-derived logical map')
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--logical-map', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expert-count', type=int, default=None)
    args=ap.parse_args()

    parsed=parse_gguf(args.gguf)
    meta=parsed['metadata']
    expert_count=args.expert_count or pick_meta(meta, ('.expert_count', '.n_expert'), None)
    expert_used=pick_meta(meta, ('.expert_used_count', '.expert_used', '.n_expert_used'), None)
    block_count=pick_meta(meta, ('.block_count', '.n_layer'), None)
    arch=meta.get('general.architecture')

    logical={}
    if args.logical_map:
        logical=json.loads(Path(args.logical_map).read_text(encoding='utf-8'))

    tensor_index=[]
    experts=defaultdict(lambda: {'chunks':[], 'size_bytes':0, 'exact':True})
    matched=0; exact_slices=0; inexact_slices=0
    for t in parsed['tensors']:
        entry={
            'name':t.name,'dims':t.dims,'type_id':t.type_id,'offset':parsed['data_start']+t.offset,
            'offset_relative':t.offset,'storage_bytes':bytes_for_tensor(t),
            'storage_extent_bytes':t.storage_extent_bytes,'size_exact':t.exact_size,
        }
        tensor_index.append(entry)
        m=EXPERT_TENSOR_RE.match(t.name)
        if not m: continue
        matched += 1
        layer=int(m.group(1)); kind=m.group(2).lower(); sub=m.group(3) or 'weight'
        n_experts=expert_count
        axis=None
        if n_experts:
            for i,d in enumerate(t.dims):
                if d==n_experts: axis=i
        # Exact independent expert slices are possible only when expert is the outermost/last axis.
        if n_experts and axis == len(t.dims)-1 and bytes_for_tensor(t) is not None and bytes_for_tensor(t) % n_experts == 0:
            slice_bytes=bytes_for_tensor(t)//n_experts
            for e in range(n_experts):
                key=f'{layer}:{e}'
                off=parsed['data_start']+t.offset+e*slice_bytes
                experts[key]['chunks'].append({'tensor':t.name,'kind':kind,'subkind':sub,'offset':off,'size_bytes':slice_bytes,'contiguous_expert_slice':True})
                experts[key]['size_bytes'] += slice_bytes
            exact_slices += 1
        else:
            inexact_slices += 1
            # Keep tensor-level evidence but do not pretend it is an exact expert offset map.
            key_prefix=f'{layer}:-1'
            experts[key_prefix]['exact']=False
            experts[key_prefix]['chunks'].append({'tensor':t.name,'kind':kind,'subkind':sub,'offset':parsed['data_start']+t.offset,'size_bytes':bytes_for_tensor(t),'contiguous_expert_slice':False})

    payload={
      'format':'atlas-physical-map-v1',
      'source':Path(args.gguf).name,
      'file':{'path':str(Path(args.gguf).resolve()),'file_size':parsed['file_size'],'data_start':parsed['data_start'],'alignment':parsed['alignment']},
      'model':{'architecture':arch,'block_count':block_count,'expert_count':expert_count,'expert_used_count':expert_used},
      'notes':[
        'Physical offsets are read from the actual GGUF tensor index.',
        'Expert-level exact slices require an expert dimension on the outermost/last tensor axis and a supported/exact tensor size.',
        'Entries with key layer:-1 are tensor-level evidence only and must not be treated as exact per-expert offsets.',
      ],
      'logical_map_joined': bool(args.logical_map),
      'tensor_index':tensor_index,
      'experts':{k:v for k,v in sorted(experts.items(), key=lambda kv:(int(kv[0].split(':')[0]), int(kv[0].split(':')[1])))},
      'stats':{'expert_tensors_matched':matched,'exact_expert_slice_tensors':exact_slices,'inexact_expert_tensors':inexact_slices},
    }

    if logical:
        observed=logical.get('experts',{})
        missing=[]; exact_joined=0
        for key, info in observed.items():
            if key in payload['experts'] and payload['experts'][key].get('exact', True): exact_joined += 1
            else: missing.append(key)
        payload['join_stats']={'logical_experts':len(observed),'physical_exact_joined':exact_joined,'not_exact_or_missing':len(missing),'missing_or_inexact_examples':missing[:100]}

    Path(args.out).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[+] physical map: {args.out}')
    print(f'[+] architecture={arch!r} layers={block_count} experts={expert_count} used={expert_used}')
    print(f'[+] tensors={len(tensor_index)} expert-tensors={matched} exact-slice-tensors={exact_slices} inexact={inexact_slices}')
    if 'join_stats' in payload:
        j=payload['join_stats']; print(f'[+] logical join: {j["physical_exact_joined"]}/{j["logical_experts"]} exact')

if __name__=='__main__': main()
