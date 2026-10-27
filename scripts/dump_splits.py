import asyncio
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

MODEL_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"

async def main():
    os.environ["TENDOU_DUMP_SPLITS"] = "1"
    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        ctx_size=2048,
        extra_args=["--atlas-k", "5"]
    )
    await backend.start()
    try:
        prompt = format_chatml_prompt([ChatMessage(role="user", content="Hi")])
        tokens = []
        async for ev in backend.generate(prompt, n_predict=2, temp=0):
            if ev.get("event") == "token":
                tokens.append(ev.get("token", ""))
        print("Generated:", "".join(tokens))
    finally:
        await backend.stop()
    
    print("\nCaptured stderr lines mentioning SPLIT:")
    for line in backend.stderr_lines:
        if "SPLIT" in line or "Split" in line or "input[" in line or "first:" in line:
            print(line)

if __name__ == "__main__":
    asyncio.run(main())
