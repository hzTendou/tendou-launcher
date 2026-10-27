import os, sys, asyncio
sys.path.insert(0, '.')
os.environ['TENDOU_DUMP_SPLITS'] = '1'
from src.atlas.server import AtlasSubprocessBackend

async def main():
    b = AtlasSubprocessBackend(
        model_path=r'C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf',
        mtp='vram',
        mtp_path=r'C:\Users\Ali\WebProjects\tendou-launcher\models\mtp\Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf',
        tlep=True,
        threads=8,
        boost=True,
        boost_level=2,
        prompt_cache_mb=0
    )
    await b.start()
    async for _ in b.generate('test', 2, 0.0):
        pass
    dumps = [l for l in b.stderr_lines if 'SPLIT DUMP' in l or 'Split' in l]
    print(f"Total split lines: {len(dumps)}")
    header_indices = [i for i, l in enumerate(dumps) if 'SPLIT DUMP' in l]
    for idx in header_indices:
        print(f"\n--- Header at index {idx}: {dumps[idx]} ---")
        for l in dumps[idx:idx+15]:
            print("  ", l)
    await b.stop()

if __name__ == '__main__':
    asyncio.run(main())
