import asyncio
import time
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

MODEL_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"

async def test():
    p = format_chatml_prompt([ChatMessage(role="user", content="17 ile 19 carpimini adim adim yaz. 17 * 20 = 340, 340 - 17 = 323. Sonucu ve islemi acikla.")])
    b = AtlasSubprocessBackend(model_path=MODEL_PATH, threads=8, boost=True, boost_level=2, gpu_layers=1, prompt_cache_mb=0)
    await b.start()
    try:
        t0, first_t, last_t, toks = time.perf_counter(), None, None, []
        async for ev in b.generate(prompt=p, n_predict=64, temp=0.0):
            if ev.get("event") == "token":
                now = time.perf_counter()
                if first_t is None: first_t = now
                last_t = now
                toks.append(ev.get("token", ""))
        dur = last_t - first_t if first_t and last_t else 0
        tps = (len(toks) - 1) / dur if dur > 0 else 0
        out_text = "".join(toks).strip()
        print(f"gpu_layers=1: {len(toks)} toks, decode_tps={tps:.2f}, text={repr(out_text[:50])}")
        return out_text
    finally:
        await b.stop()

if __name__ == "__main__":
    asyncio.run(test())
