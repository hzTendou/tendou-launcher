import struct

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode('utf-8', errors='ignore')

def read_val(f, vtype):
    # GGUF types: 0: u8, 1: i8, 2: u16, 3: i16, 4: u32, 5: i32, 6: f32, 7: bool, 8: str, 9: arr, 10: u64, 11: i64, 12: f64
    if vtype == 0: return struct.unpack("<B", f.read(1))[0]
    elif vtype == 1: return struct.unpack("<b", f.read(1))[0]
    elif vtype == 2: return struct.unpack("<H", f.read(2))[0]
    elif vtype == 3: return struct.unpack("<h", f.read(2))[0]
    elif vtype == 4: return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5: return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6: return struct.unpack("<f", f.read(4))[0]
    elif vtype == 7: return struct.unpack("<?", f.read(1))[0]
    elif vtype == 8: return read_str(f)
    elif vtype == 9:
        elem_type = struct.unpack("<I", f.read(4))[0]
        elem_cnt = struct.unpack("<Q", f.read(8))[0]
        vals = [read_val(f, elem_type) for _ in range(elem_cnt)]
        return vals
    elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 11: return struct.unpack("<q", f.read(8))[0]
    elif vtype == 12: return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown type {vtype}")

with open("models/mtp/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf", "rb") as f:
    magic = f.read(4)
    assert magic == b"GGUF"
    version = struct.unpack("<I", f.read(4))[0]
    ti_count = struct.unpack("<Q", f.read(8))[0]
    kv_count = struct.unpack("<Q", f.read(8))[0]
    print(f"Version: {version}, Tensors: {ti_count}, KVs: {kv_count}")
    for i in range(kv_count):
        k = read_str(f)
        vt = struct.unpack("<I", f.read(4))[0]
        val = read_val(f, vt)
        if any(x in k for x in ["block", "layer", "nextn", "compress", "arch"]):
            if isinstance(val, list) and len(val) > 10:
                print(f"  {k} (len={len(val)}): {val[:5]}...{val[-2:]}")
            else:
                print(f"  {k}: {val}")
    print("Tensors:")
    for i in range(ti_count):
        name = read_str(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        print(f"  {name}: type={ttype}, dims={dims}")
