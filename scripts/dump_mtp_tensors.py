import struct

with open(r'models\mtp\Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf', 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    print(f'n_tensors: {n_tensors}, n_kv: {n_kv}')
    
    def skip_val(t):
        if t in (0, 1, 7): f.seek(1, 1)
        elif t in (2, 3): f.seek(2, 1)
        elif t in (4, 5, 12): f.seek(4, 1)
        elif t in (6, 10, 11): f.seek(8, 1)
        elif t == 8:
            slen = struct.unpack('<Q', f.read(8))[0]
            f.seek(slen, 1)
        elif t == 9:
            item_t = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if item_t in (0, 1, 7): f.seek(arr_len, 1)
            elif item_t in (2, 3): f.seek(arr_len * 2, 1)
            elif item_t in (4, 5, 12): f.seek(arr_len * 4, 1)
            elif item_t in (6, 10, 11): f.seek(arr_len * 8, 1)
            elif item_t == 8:
                for _ in range(arr_len):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.seek(slen, 1)
            else:
                for _ in range(arr_len): skip_val(item_t)

    for i in range(n_kv):
        k_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(k_len).decode('utf-8', errors='ignore')
        val_type = struct.unpack('<I', f.read(4))[0]
        skip_val(val_type)

    print('Tensors:')
    for i in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8', errors='ignore')
        n_dims = struct.unpack('<I', f.read(4))[0]
        f.seek(n_dims * 8 + 4 + 8, 1)
        print(f'{i}: {name}')
