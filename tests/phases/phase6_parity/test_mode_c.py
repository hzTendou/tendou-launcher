import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

MODEL_PATH = os.environ.get(
    "TENDOU_MODEL_PATH",
    r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"
)
MTP_PATH = os.environ.get(
    "TENDOU_MTP_PATH",
    str(ROOT / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf")
)

async def main():
    print(f"Testing Mode C initialization with MTP draft model: {MTP_PATH}")
    os.environ["ATLAS_TLEP"] = "1"
    os.environ["ATLAS_MTP"] = "ram"
    os.environ["ATLAS_MTP_PATH"] = MTP_PATH

    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=True,
        mtp="ram",
        mtp_draft_n=1,
        mtp_path=MTP_PATH,
        gpu_layers=2,
        readback_interval=64,
        p3_async_transfer=False,
    )

    print("Starting backend...")
    await backend.start()
    print("Backend started successfully!")
    try:
        prompt = format_chatml_prompt([ChatMessage(role="user", content="17 ile 19 carpimini adim adim yaz. 17 * 20 = 340, 340 - 17 = 323. Sonucu ve islemi acikla.")])
        tokens = []
        done_ev = {}
        t0 = time.perf_counter()
        async for ev in backend.generate(prompt=prompt, n_predict=64, temp=0.0):
            if ev.get("event") == "token":
                tokens.append(ev.get("token", ""))
            elif ev.get("event") == "done":
                done_ev = ev
        print(f"Output: '{''.join(tokens).strip()}'")
        print(f"Done event: {done_ev}")
        print(f"\n--- Engine STDERR Lines (Total: {len(backend.stderr_lines)}) ---")
        for line in backend.stderr_lines:
            print("STDERR:", line.strip())
    finally:
        await backend.stop()

if __name__ == "__main__":
    asyncio.run(main())
