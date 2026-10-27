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
    b = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=True,
        mtp="vram",
        mtp_draft_n=2,
        mtp_path=MTP_PATH,
        prompt_cache_mb=0,
    )
    await b.start()
    try:
        print("\n--- Engine Stderr Filtered ---")
        for line in b.stderr_lines:
            if any(k in line for k in ["DEBUG-MTP", "ATLAS-MTP", "free_vram", "Aliased", "draft", "VRAM", "CUDA"]):
                print(line)
    finally:
        await b.stop()

if __name__ == "__main__":
    asyncio.run(main())
