import os
import shutil
import sys

def apply_version(version):
    src_dir = f"src_{version}"
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} does not exist!")
        sys.exit(1)
    
    file_mappings = [
        (os.path.join(src_dir, "llama.cpp", "ggml", "include", "ggml-cpu.h"), "llama.cpp/ggml/include/ggml-cpu.h"),
        (os.path.join(src_dir, "llama.cpp", "ggml", "src", "ggml-cpu", "quants.h"), "llama.cpp/ggml/src/ggml-cpu/quants.h"),
        (os.path.join(src_dir, "llama.cpp", "ggml", "src", "ggml-cpu", "ggml-cpu.c"), "llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c"),
        (os.path.join(src_dir, "llama.cpp", "ggml", "src", "ggml-cpu", "arch", "x86", "quants.c"), "llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c"),
        (os.path.join(src_dir, "llama.cpp", "src", "models", "qwen4exp.cpp"), "llama.cpp/src/models/qwen4exp.cpp"),
        (os.path.join(src_dir, "llama.cpp", "src", "models", "qwen3next.cpp"), "llama.cpp/src/models/qwen3next.cpp"),
        (os.path.join(src_dir, "llama.cpp", "ggml", "src", "ggml-backend.cpp"), "llama.cpp/ggml/src/ggml-backend.cpp"),
        (os.path.join(src_dir, "src", "atlas", "atlas-engine.cpp"), "src/atlas/atlas-engine.cpp"),
        (os.path.join(src_dir, "src", "atlas", "atlas-engine.h"), "src/atlas/atlas-engine.h"),
        (os.path.join(src_dir, "src", "atlas", "atlas-engine.cpp"), "llama.cpp/examples/atlas-engine/atlas-engine.cpp"),
        (os.path.join(src_dir, "src", "atlas", "atlas-engine.h"), "llama.cpp/examples/atlas-engine/atlas-engine.h"),
    ]

    for src_file, dst_file in file_mappings:
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Applied {src_file} -> {dst_file}")
        else:
            print(f"Warning: source file {src_file} not found!")

    print(f"\nSwitched to {version.upper()} successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python switch_version.py [phase8|phase9]")
        sys.exit(1)
    apply_version(sys.argv[1].lower())
