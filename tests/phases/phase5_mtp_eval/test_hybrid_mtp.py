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
MTP_PATH = str(ROOT / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf")

async def test_hybrid(draft_n):
    print(f"\n================ Running MTP in Hybrid Mode (draft_n={draft_n}) ================")
    b = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=True,
        mtp="hybrid",
        mtp_draft_n=draft_n,
        mtp_path=MTP_PATH,
        gpu_layers=2,
        prompt_cache_mb=0,
    )
    await b.start()
    try:
        tasks = [
            ("short-math-01", "17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.", 24),
            ("short-code-01", "Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])", 24),
            ("repetitive-code-01", "Repeat the pattern for the next item without explanation. Write only Python code:\nitems = ['a']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['b']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['c']\nfor x in items:\n", 48),
        ]
        for tid, query, n_pred in tasks:
            p = format_chatml_prompt([ChatMessage(role="user", content=query)])
            toks = []
            first_t = None
            last_t = None
            done_ev = {}
            t0 = time.perf_counter()
            async for ev in b.generate(prompt=p, n_predict=n_pred, temp=0.0):
                if ev.get("event") == "token":
                    now = time.perf_counter()
                    if first_t is None: first_t = now
                    last_t = now
                    toks.append(ev.get("token", ""))
                elif ev.get("event") == "done":
                    done_ev = ev
            dur = last_t - first_t if first_t and last_t else 0.0
            tps = (len(toks) - 1) / dur if dur > 0 else 0.0
            total_dur = time.perf_counter() - t0
            eff_tps = len(toks) / total_dur if total_dur > 0 else 0.0
            out = "".join(toks).strip()
            print(f"[{tid}] {repr(out[:40])} | Toks: {len(toks)} | Decode TPS: {tps:.2f} | Eff TPS: {eff_tps:.2f}")
            print(f"  Draft time: {done_ev.get('spec_draft_time_ms', 0):.1f}ms | Verify time: {done_ev.get('spec_verify_time_ms', 0):.1f}ms")
            print(f"  Drafted: {done_ev.get('spec_draft_tokens', 0)}, Accepted: {done_ev.get('spec_accepted_tokens', 0)}")
    finally:
        await b.stop()

async def main():
    await test_hybrid(2)

if __name__ == "__main__":
    asyncio.run(main())
