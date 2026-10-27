import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

MODEL_PATH = os.environ.get(
    "TENDOU_MODEL_PATH",
    r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"
)
MTP_PATH = str(ROOT / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf")

CANDIDATES = [
    # Candidate 1: The original from run_continuous_benchmark.py
    ("c1", "17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzunca acikla."),
    # Candidate 2: Clear operator anchor
    ("c2", "17 ile 19 carpimini adim adim yaz. Carpma sembolu olarak yalnizca '*' kullan. 17 * 20 = 340, 340 - 17 = 323. Islemi acikla."),
    # Candidate 3: Number-first explanation
    ("c3", "17 ile 19 carpimini acikla: 17 * 19 = 323. Adimlari sirala."),
]

async def test_backend(name, enable_tlep, mtp_mode):
    b = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=enable_tlep,
        mtp=mtp_mode,
        mtp_path=MTP_PATH if mtp_mode != "off" else None,
        gpu_layers=2,
        readback_interval=64,
        p3_async_transfer=False,
        prompt_cache_mb=0,
    )
    await b.start()
    res = {}
    try:
        for cid, query in CANDIDATES:
            p = format_chatml_prompt([ChatMessage(role="user", content=query)])
            toks = []
            async for ev in b.generate(prompt=p, n_predict=64, temp=0.0):
                if ev.get("event") == "token":
                    toks.append(ev.get("token", ""))
            out = "".join(toks).strip()
            res[cid] = out
            print(f"[{name}] {cid}: {repr(out[:50])}")
    finally:
        await b.stop()
    return res

async def main():
    a = await test_backend("Mode A", False, "off")
    c = await test_backend("Mode C", True, "ram")
    for cid, query in CANDIDATES:
        match = (a[cid] == c[cid])
        print(f"\n{cid} Match: {match}")
        if not match:
            print("  A:", repr(a[cid][:60]))
            print("  C:", repr(c[cid][:60]))
        else:
            print("  Exact Parity Match:", repr(a[cid][:60]))

if __name__ == "__main__":
    asyncio.run(main())
