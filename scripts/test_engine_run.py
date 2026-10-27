import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt, DEFAULT_MODEL_PATH

async def test_run():
    print(f"Testing engine run with model: {DEFAULT_MODEL_PATH}")
    b = AtlasSubprocessBackend(
        model_path=DEFAULT_MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        gpu_layers=2,
        mtp="off",
        prompt_cache_mb=0,
    )
    print("Starting backend...")
    await b.start()
    print("Backend started successfully!")
    p = format_chatml_prompt([ChatMessage(role="user", content="17 * 19 = ?")])
    toks = []
    t0 = time.perf_counter()
    async for ev in b.generate(prompt=p, n_predict=10, temp=0.0):
        if ev.get("event") == "token":
            toks.append(ev.get("token", ""))
            print(ev.get("token", ""), end="", flush=True)
        elif ev.get("event") == "done":
            print(f"\nDone event: {ev}")
    await b.stop()
    print("\nBackend stopped successfully!")

if __name__ == "__main__":
    asyncio.run(test_run())
