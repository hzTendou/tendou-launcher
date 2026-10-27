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

PROMPT = format_chatml_prompt([ChatMessage(role="user", content="17 ile 19 carpimini adim adim yaz. 17 * 20 = 340, 340 - 17 = 323. Sonucu ve islemi acikla.")])

async def test_mode(name, enable_tlep, mtp_mode):
    print(f"\n--- Testing {name} ---")
    b = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=enable_tlep,
        mtp=mtp_mode,
        mtp_path=MTP_PATH if mtp_mode != "off" else None,
        gpu_layers=0,
        readback_interval=64,
        p3_async_transfer=False,
    )
    await b.start()
    try:
        tokens = []
        async for ev in b.generate(prompt=PROMPT, n_predict=64, temp=0.0):
            if ev.get("event") == "token":
                tokens.append(ev.get("token", ""))
        out = "".join(tokens).strip()
        print(f"[{name}] Output ({len(tokens)} toks): {out[:80]}")
        return out
    finally:
        await b.stop()

async def main():
    out_a = await test_mode("Mode A", False, "off")
    out_c = await test_mode("Mode C", True, "vram")
    print("\nParity:", "MATCH" if out_a == out_c else "MISMATCH")
    if out_a != out_c:
        print("Diff:\n  A:", out_a, "\n  C:", out_c)

if __name__ == "__main__":
    asyncio.run(main())
