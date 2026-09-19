import asyncio
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

TASKS = [
    ("turkish-01", "'Ayse sabah kitabi masaya birakti ve aksam oradan aldi.' Kitabi kim aldi? Yalnizca adi yaz.", "^Ayse$"),
    ("logic-01", "Tum laleler canlidir. Tum canlilar enerji kullanir. Tum laleler enerji kullanir mi? Yalnizca evet veya hayir yaz.", "^evet$"),
    ("instruction-02", "Yalnizca BUYUK HARFLERLE 'tendou' yaz; aciklama ve noktalama ekleme.", "^TENDOU$"),
]

async def main():
    backend = AtlasSubprocessBackend(
        threads=8,
        ctx_size=2048,
        extra_args=[
            "--atlas-k", "5",
            "--atlas-dynamic-k", "0.55",
            "--atlas-dynamic-k-min", "1",
            "--atlas-dynamic-k-max", "2",
            "--atlas-dynamic-k-min-weight", "0.1",
            "--atlas-dynamic-k-end", "47",
            "--atlas-layer-adapt", "1",
        ],
    )
    await backend.start()
    try:
        for tid, p, exp in TASKS:
            prompt = format_chatml_prompt([ChatMessage(role="user", content=p)])
            tokens = []
            async for ev in backend.generate(prompt, n_predict=16, temp=0):
                if ev.get("event") == "token":
                    tokens.append(ev.get("token", ""))
            ans = "".join(tokens).strip()
            print(f"[{tid}] Got: '{ans}' | Expected: '{exp}'")
    finally:
        await backend.stop()

if __name__ == "__main__":
    asyncio.run(main())

