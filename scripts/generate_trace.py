import os
import asyncio
import sys

# Ensure environment is set
os.environ["ATLAS_DEEP_TELEMETRY"] = "1"
os.environ["ATLAS_TRACE_PATH"] = "trace_explicit.jsonl"

from src.atlas.server import AtlasSubprocessBackend

async def test_trace():
    backend = AtlasSubprocessBackend(
        model_path=r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf",
        threads=8,
        boost_level=2
    )
    print("Starting backend...")
    await backend.start()
    print("Backend ready.")
    
    print("Generating short math measure...")
    prompt = "17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzunca acikla. Bunu yaparken mumkun oldugunca fazla detay ver ki token sayisi 64'u gecsin."
    async for item in backend.generate(prompt=prompt, n_predict=128):
        pass
    
    print("Done. Stopping backend...")
    await backend.stop()
    print("Stopped.")

if __name__ == "__main__":
    asyncio.run(test_trace())
