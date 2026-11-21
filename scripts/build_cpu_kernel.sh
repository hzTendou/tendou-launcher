#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cmake --build llama.cpp/build-linux --target ggml-cpu -j "$(nproc)"
mkdir -p llama.cpp/build/bin/Release releases/tendou-v0.16.3-stable/bin
cp -af llama.cpp/build-linux/bin/libggml-cpu.so* llama.cpp/build/bin/Release/
cp -af llama.cpp/build-linux/bin/libggml-cpu.so* releases/tendou-v0.16.3-stable/bin/
echo "[BUILD_CPU_KERNEL] Success: libggml-cpu.so rebuilt with 512-bit AVX-512 VNNI (-march=native)."
