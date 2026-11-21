#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export LD_LIBRARY_PATH="$ROOT_DIR/releases/tendou-v0.16.3-stable/bin:$ROOT_DIR/releases/tendou-v0.16.2-stable/bin:$ROOT_DIR/releases/tendou-v0.16.1-stable/bin:$ROOT_DIR/releases/tendou-v0.16.0-stable/bin:$ROOT_DIR/releases/tendou-v0.15.0-stable/bin:$ROOT_DIR/llama.cpp/build-linux/bin:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

PYTHON_BIN="$ROOT_DIR/.venv-audit/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/atlas_server.py" "$@"
