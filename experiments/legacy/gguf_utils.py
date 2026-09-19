from __future__ import annotations
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
import re

GGUF_MAGIC = b'GGUF'

META_TYPES = {
    0: ('UINT8', '<B'), 1: ('INT8', '<b'), 2: ('UINT16', '<H'), 3: ('INT16', '<h'),
    4: ('UINT32', '<I'), 5: ('INT32', '<i'), 6: ('FLOAT32', '<f'), 7: ('BOOL', '<?'),
    8: ('STRING', None), 9: ('ARRAY', None), 10: ('UINT64', '<Q'), 11: ('INT64', '<q'),
    12: ('FLOAT64', '<d'),
}

# Exact storage size for common GGML types: (block elements, bytes per block).
# Unknown types are deliberately left unsupported; tensor extents can still be used
# as a conservative physical range.
TYPE_BLOCK = {
    0: (1, 4),   # F32
    1: (1, 2),   # F16
    2: (32, 18), # Q4_0
    3: (32, 20), # Q4_1
    6: (32, 22), # Q5_0
    7: (32, 24), # Q5_1
    8: (32, 34), # Q8_0
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292), # Q8_K
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 110),
    20: (256, 82),
    34: (256, 54),
    35: (256, 66),
}

EXPERT_TENSOR_RE = re.compile(
    r'^(?:blk|block)\.(\d+)\.ffn_(gate|up|down)_exps(?:\.(weight|bias))?$',
    re.IGNORECASE,
)

@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    type_id: int
    offset: int
    storage_bytes: int | None = None
    storage_extent_bytes: int | None = None
    exact_size: bool = False


def _read_u8(f): return struct.unpack('<B', f.read(1))[0]
def _read_u32(f): return struct.unpack('<I', f.read(4))[0]
def _read_u64(f): return struct.unpack('<Q', f.read(8))[0]

def read_gguf_string(f) -> str:
    n = _read_u64(f)
    raw = f.read(n)
    if len(raw) != n:
        raise EOFError('truncated GGUF string')
    return raw.decode('utf-8', errors='replace')


def skip_value(f, type_id: int):
    if type_id in META_TYPES and META_TYPES[type_id][1]:
        f.read(struct.calcsize(META_TYPES[type_id][1]))
        return
    if type_id == 8:
        n = _read_u64(f); f.seek(n, 1); return
    if type_id == 9:
        elem_type = _read_u32(f); n = _read_u64(f)
        for _ in range(n): skip_value(f, elem_type)
        return
    raise ValueError(f'unsupported GGUF metadata type {type_id}')


def read_value(f, type_id: int):
    if type_id in META_TYPES and META_TYPES[type_id][1]:
        return struct.unpack(META_TYPES[type_id][1], f.read(struct.calcsize(META_TYPES[type_id][1])))[0]
    if type_id == 8:
        return read_gguf_string(f)
    if type_id == 9:
        elem_type = _read_u32(f); n = _read_u64(f)
        return [read_value(f, elem_type) for _ in range(n)]
    raise ValueError(f'unsupported GGUF metadata type {type_id}')


def tensor_numel(dims: list[int]) -> int:
    n = 1
    for d in dims: n *= int(d)
    return n


def exact_tensor_size(dims: list[int], type_id: int) -> int | None:
    info = TYPE_BLOCK.get(type_id)
    if info is None:
        return None
    block_n, block_bytes = info
    n = tensor_numel(dims)
    if n % block_n:
        return None
    return (n // block_n) * block_bytes


def parse_gguf(path: str | Path):
    path = Path(path)
    file_size = path.stat().st_size
    with path.open('rb') as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f'{path} is not a GGUF file')
        version = _read_u32(f)
        if version not in (1, 2, 3):
            raise ValueError(f'unsupported GGUF version {version}')
        tensor_count = _read_u64(f)
        kv_count = _read_u64(f)
        meta = {}
        for _ in range(kv_count):
            key = read_gguf_string(f)
            value_type = _read_u32(f)
            meta[key] = read_value(f, value_type)
        tensor_infos = []
        for _ in range(tensor_count):
            name = read_gguf_string(f)
            n_dims = _read_u32(f)
            dims = [_read_u64(f) for _ in range(n_dims)]
            type_id = _read_u32(f)
            offset = _read_u64(f)
            tensor_infos.append(TensorInfo(name, [int(x) for x in dims], int(type_id), int(offset)))
        alignment = int(meta.get('general.alignment', 32) or 32)
        data_start = (f.tell() + alignment - 1) // alignment * alignment

    ordered = sorted(tensor_infos, key=lambda t: t.offset)
    for i, t in enumerate(ordered):
        next_off = ordered[i + 1].offset if i + 1 < len(ordered) else file_size - data_start
        extent = max(0, next_off - t.offset)
        exact = exact_tensor_size(t.dims, t.type_id)
        t.storage_bytes = exact
        t.storage_extent_bytes = extent
        t.exact_size = exact is not None
    # Restore header order for stable manifests.
    by_name = {t.name: t for t in ordered}
    tensor_infos = [by_name[t.name] for t in tensor_infos]
    return {
        'path': str(path),
        'file_size': file_size,
        'version': version,
        'tensor_count': tensor_count,
        'metadata_count': kv_count,
        'metadata': meta,
        'alignment': alignment,
        'data_start': data_start,
        'tensors': tensor_infos,
    }
