#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cp -f src/atlas/atlas-engine.cpp llama.cpp/examples/atlas-engine/atlas-engine.cpp
cp -f src/atlas/atlas-engine.h   llama.cpp/examples/atlas-engine/atlas-engine.h
cp -f src/atlas/atlas-moe-cache.h llama.cpp/examples/atlas-engine/atlas-moe-cache.h
cp -f src/atlas/atlas-iouring-dma.h llama.cpp/examples/atlas-engine/atlas-iouring-dma.h
cp -f src/atlas/CMakeLists.txt   llama.cpp/examples/atlas-engine/CMakeLists.txt

mkdir -p \
    releases/tendou-v0.16.3-stable/source/src/atlas \
    releases/tendou-v0.16.3-stable/source/llama.cpp/src \
    releases/tendou-v0.16.3-stable/source/llama.cpp/ggml/include \
    releases/tendou-v0.16.3-stable/source/llama.cpp/ggml/src/ggml-cpu/arch/x86
cp -f src/atlas/atlas-engine.cpp releases/tendou-v0.16.3-stable/source/src/atlas/atlas-engine.cpp
cp -f src/atlas/atlas-engine.cpp releases/tendou-v0.16.3-stable/source/atlas-engine.cpp
cp -f src/atlas/atlas-engine.h   releases/tendou-v0.16.3-stable/source/src/atlas/atlas-engine.h
cp -f src/atlas/atlas-engine.h   releases/tendou-v0.16.3-stable/source/atlas-engine.h
cp -f src/atlas/atlas-moe-cache.h releases/tendou-v0.16.3-stable/source/src/atlas/atlas-moe-cache.h
cp -f src/atlas/atlas-moe-cache.h releases/tendou-v0.16.3-stable/source/atlas-moe-cache.h
cp -f src/atlas/atlas-iouring-dma.h releases/tendou-v0.16.3-stable/source/src/atlas/atlas-iouring-dma.h
cp -f src/atlas/atlas-iouring-dma.h releases/tendou-v0.16.3-stable/source/atlas-iouring-dma.h
cp -f src/atlas/CMakeLists.txt   releases/tendou-v0.16.3-stable/source/src/atlas/CMakeLists.txt
cp -f src/atlas/__init__.py      releases/tendou-v0.16.3-stable/source/src/atlas/__init__.py
cp -f src/atlas/native_server.py releases/tendou-v0.16.3-stable/source/src/atlas/native_server.py
cp -f src/atlas/autotune.py      releases/tendou-v0.16.3-stable/source/src/atlas/autotune.py
cp -f src/atlas/autotune.py      releases/tendou-v0.16.3-stable/source/autotune.py
cp -f src/atlas/server.py        releases/tendou-v0.16.3-stable/source/src/atlas/server.py
cp -f src/atlas/server.py        releases/tendou-v0.16.3-stable/source/server.py
cp -f src/atlas/ui_config.py     releases/tendou-v0.16.3-stable/source/src/atlas/ui_config.py
cp -f src/atlas/ui_telemetry.py  releases/tendou-v0.16.3-stable/source/src/atlas/ui_telemetry.py
cp -f src/atlas/ui_chats.py      releases/tendou-v0.16.3-stable/source/src/atlas/ui_chats.py
cp -f src/atlas/ui_api.py        releases/tendou-v0.16.3-stable/source/src/atlas/ui_api.py
cp -f llama.cpp/src/llama-model.cpp releases/tendou-v0.16.3-stable/source/llama.cpp/src/llama-model.cpp
cp -f llama.cpp/src/llama-model-loader.cpp releases/tendou-v0.16.3-stable/source/llama.cpp/src/llama-model-loader.cpp
cp -f llama.cpp/ggml/include/ggml-cpu.h releases/tendou-v0.16.3-stable/source/llama.cpp/ggml/include/ggml-cpu.h
cp -f llama.cpp/ggml/src/ggml-cpu/quants.h releases/tendou-v0.16.3-stable/source/llama.cpp/ggml/src/ggml-cpu/quants.h
cp -f llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c releases/tendou-v0.16.3-stable/source/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
cp -f llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c releases/tendou-v0.16.3-stable/source/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c

export PATH="/usr/local/cuda-13.2/bin:/usr/local/cuda/bin:$PATH"

if [ ! -f llama.cpp/build-linux/CMakeCache.txt ]; then
    cmake -S llama.cpp -B llama.cpp/build-linux -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc-14 \
        -DCMAKE_CXX_COMPILER=g++-14 \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="120a" \
        -DGGML_NATIVE=ON \
        -DGGML_OPENMP=ON \
        -DBUILD_SHARED_LIBS=ON
fi

cmake --build llama.cpp/build-linux --target llama-atlas-engine llama-server -j "$(nproc)"

mkdir -p llama.cpp/build/bin/Release releases/tendou-v0.16.3-stable/bin
cp -f llama.cpp/build-linux/bin/llama-atlas-engine llama.cpp/build-linux/bin/tendou-engine
cp -f llama.cpp/build-linux/bin/llama-atlas-engine llama.cpp/build/bin/Release/llama-atlas-engine
cp -f llama.cpp/build-linux/bin/llama-atlas-engine llama.cpp/build/bin/Release/tendou-engine
cp -f llama.cpp/build-linux/bin/llama-atlas-engine releases/tendou-v0.16.3-stable/bin/tendou-engine
cp -f llama.cpp/build-linux/bin/llama-atlas-engine releases/tendou-v0.16.3-stable/bin/llama-atlas-engine
if [ -f llama.cpp/build-linux/bin/llama-server ]; then
    cp -f llama.cpp/build-linux/bin/llama-server llama.cpp/build/bin/Release/llama-server
    cp -f llama.cpp/build-linux/bin/llama-server releases/tendou-v0.16.3-stable/bin/llama-server
fi
cp -af llama.cpp/build-linux/bin/lib*.so* llama.cpp/build/bin/Release/
cp -af llama.cpp/build-linux/bin/lib*.so* releases/tendou-v0.16.3-stable/bin/
echo "[BUILD_RUNTIME] Success: Linux tendou-engine v0.16.3-stable, llama-server, and shared libraries synced."
