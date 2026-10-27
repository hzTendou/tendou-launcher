@echo off
setlocal
set TENDOU_MODEL_PATH=C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf

echo Running Telemetry OFF 5 times...
set ATLAS_DEEP_TELEMETRY=0
for /l %%i in (1, 1, 5) do (
    echo Run %%i
    .venv-audit\Scripts\python.exe scripts/run_continuous_benchmark.py --context short
)

echo Running Telemetry ON 5 times...
set ATLAS_DEEP_TELEMETRY=1
for /l %%i in (1, 1, 5) do (
    echo Run %%i
    .venv-audit\Scripts\python.exe scripts/run_continuous_benchmark.py --context short
)

echo Done.
