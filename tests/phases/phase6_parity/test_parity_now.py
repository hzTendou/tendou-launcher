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

PROMPTS = [
    ("short-math-01", "17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.", 24),
    ("short-code-01", "Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])", 24),
    ("math-explain-01", "17 ile 19 carpimini adim adim yaz. 17 * 20 = 340, 340 - 17 = 323. Sonucu ve islemi acikla.", 64),
    ("repetitive-code-01", "Repeat the pattern for the next item without explanation. Write only Python code:\nitems = ['a']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['b']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['c']\nfor x in items:\n", 48),
]

async def run_mode(name, enable_tlep, mtp_mode):
    print(f"\n================ Running {name} ================")
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
    results = {}
    try:
        for tid, query, n_pred in PROMPTS:
            p = format_chatml_prompt([ChatMessage(role="user", content=query)])
            toks = []
            async for ev in b.generate(prompt=p, n_predict=n_pred, temp=0.0):
                if ev.get("event") == "token":
                    toks.append(ev.get("token", ""))
            out = "".join(toks).strip()
            print(f"[{name}] {tid}: {repr(out[:40])} ({len(toks)} toks)")
            results[tid] = out
    finally:
        await b.stop()
    return results

async def main():
    res_a = await run_mode("Mode A", False, "off")
    res_c = await run_mode("Mode C", True, "ram")

    print("\n================ PARITY CHECK ================")
    all_match = True
    for tid, _, _ in PROMPTS:
        match = (res_a[tid] == res_c[tid])
        print(f"Task {tid}: {'MATCH (%100 parite)' if match else 'MISMATCH'}")
        if not match:
            all_match = False
            print(f"  A: {repr(res_a[tid])}")
            print(f"  C: {repr(res_c[tid])}")
    print(f"\nOVERALL PARITY: {'PASS (%100)' if all_match else 'FAIL'}")

if __name__ == "__main__":
    asyncio.run(main())
