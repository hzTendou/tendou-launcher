#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CXX="${CXX:-g++-14}"
if ! command -v "$CXX" >/dev/null 2>&1; then
    CXX="g++"
fi

BIN_DIR="$ROOT_DIR/llama.cpp/build-linux/bin"
REL_DIR="$ROOT_DIR/llama.cpp/build/bin/Release"
mkdir -p "$BIN_DIR" "$REL_DIR"

INCLUDES=(
    -I"$ROOT_DIR/src/atlas"
    -I"$ROOT_DIR/llama.cpp/examples/atlas-engine"
    -I"$ROOT_DIR/llama.cpp/common"
    -I"$ROOT_DIR/llama.cpp/vendor"
    -I"$ROOT_DIR/llama.cpp/include"
    -I"$ROOT_DIR/llama.cpp/src"
    -I"$ROOT_DIR/llama.cpp/ggml/include"
    -I"$ROOT_DIR/llama.cpp/ggml/src"
    -I"$ROOT_DIR/llama.cpp/ggml/src/ggml-cpu"
    -I"$ROOT_DIR/llama.cpp/ggml/src/ggml-cpu/arch/x86"
)

LIBS=(
    -L"$BIN_DIR"
    -Wl,-rpath,"\$ORIGIN"
    -lllama-common -lllama -lggml -lggml-cpu -lggml-cuda -lggml-base
    -lpthread -ldl
)

"$CXX" -std=c++17 -O2 -march=native "${INCLUDES[@]}" \
    "$ROOT_DIR/tests/native/test_async_prefetch.cpp" \
    "${LIBS[@]}" \
    -o "$BIN_DIR/atlas-native-tests"

"$CXX" -std=c++17 -O2 -march=native "${INCLUDES[@]}" \
    "$ROOT_DIR/tests/native/test_expert_partition.cpp" \
    "${LIBS[@]}" \
    -o "$BIN_DIR/atlas-partition-tests"

cp -f "$BIN_DIR/atlas-native-tests" "$REL_DIR/atlas-native-tests"
cp -f "$BIN_DIR/atlas-partition-tests" "$REL_DIR/atlas-partition-tests"
echo "[BUILD_NATIVE_TESTS] Success: atlas-native-tests and atlas-partition-tests built."
