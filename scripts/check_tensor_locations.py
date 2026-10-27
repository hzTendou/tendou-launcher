import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend

MODEL_PATH = os.environ.get(
    "TENDOU_MODEL_PATH",
    r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"
)
MTP_PATH = str(ROOT / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_0.gguf")

async def main():
    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=True,
        mtp="vram",
        mtp_draft_n=2,
        mtp_path=MTP_PATH,
        readback_interval=64,
        p3_async_transfer=False,
        prompt_cache_mb=0,
    )
    print("Starting backend...")
    await backend.start()
    print("Backend ready. Checking stderr lines:")
    for line in backend.stderr_lines:
        if "MTP" in line or "output" in line or "buft" in line or "free_vram" in line or "override" in line:
            print("  ", line)
    
    # Generate 5 tokens to see step latency
    async for ev in backend.generate(prompt="Hello", n_predict=5, temp=0.0):
        if ev.get("event") == "token":
            print("Token:", ev.get("token"))
        elif ev.get("event") == "done":
            print("Done:", ev)

    print("\n--- STDERR LINES AFTER GENERATION ---")
    for line in backend.stderr_lines:
        if "DIAG" in line or "MTP" in line or "decode_us" in line:
            print("  ", line)

    await backend.stop()

if __name__ == "__main__":
    asyncio.run(main())
