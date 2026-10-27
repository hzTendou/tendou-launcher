@echo off
title Tendou Launcher API Server (Port 8000 - 256k Context Max)
cd /d "%~dp0"

:: ---------------------------------------------------------------------------
:: Tendou Launcher - Peak Performance Configuration
:: Model: Qwen3.6-35B-A3B-UD-Q8_K_XL (38.5 GB, 40L, 256 Experts, Native 256k Ctx)
:: Hardware: NVIDIA RTX 5060 Laptop (8GB VRAM) + AMD Ryzen 7 260 (16 Threads)
:: ---------------------------------------------------------------------------

set "TENDOU_PHYSICAL_MAP=config\atlas_physical_map_35b.json"

set "MODEL_PATH=C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
set "PYTHON=.venv-audit\Scripts\python.exe"

echo ============================================================
echo   TENDOU LAUNCHER - LOCAL OPENAI-COMPATIBLE API SERVER
echo ============================================================
echo   Model       : Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
echo   Context Max : 262,144 tokens (256k Full Native Context)
echo   Threads     : 16 CPU Threads (Ryzen 7 260 Peak Throughput)
echo   GPU / VRAM  : 2 Dense Layers + 3000 MB Dynamic Expert Cache
echo   KV Cache    : Q4_0 Quantized (11.25 KB/tok, Ultra-Low VRAM)
echo   Engine Mode : Boost Mode (Adaptive K=5 Pruning, 32-40 TPS)
echo   Server URL  : http://127.0.0.1:8000/v1
echo ============================================================
echo.

%PYTHON% atlas_server.py ^
  --model-path "%MODEL_PATH%" ^
  --threads 16 ^
  --gpu-layers 2 ^
  --vram-cache 3000 ^
  --ctx-size 262144 ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --prompt-cache-mb 256 ^
  --boost ^
  --cuda-sched yield ^
  --odmoe-lead 2 ^
  --spice-conf-high 0.65 ^
  --spice-conf-mid 0.30 ^
  --prefetch-candidates 10 ^
  --tutti-queue-depth 64 ^
  --readback-interval 8 ^
  --port 8000 ^
  --force

pause
