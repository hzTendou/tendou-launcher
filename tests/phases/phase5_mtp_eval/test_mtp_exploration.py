import asyncio
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

async def run():
    backend = AtlasSubprocessBackend(
        model_path=r'C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf',
        threads=8,
        boost=True,
        boost_level=2,
        tlep=True,
        mtp='hybrid',
        mtp_path=str(ROOT / 'models' / 'mtp' / 'Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf'),
        prompt_cache_mb=0,
    )
    await backend.start()
    
    prompts = [
        format_chatml_prompt([ChatMessage(role='user', content='17 ile 19 carpimi kactir? Yalnizca sayi.')]),
        format_chatml_prompt([ChatMessage(role='user', content='Repeat the pattern: foo bar foo bar foo bar foo bar')]),
        format_chatml_prompt([ChatMessage(role='user', content='Python kodu yaz: def add(a, b): return a + b')])
    ]
    for i, p in enumerate(prompts):
        tokens = []
        done_ev = {}
        async for ev in backend.generate(prompt=p, n_predict=24, temp=0.0):
            if ev.get('event') == 'done':
                done_ev = ev
            elif ev.get('event') == 'token':
                tokens.append(ev.get('token', ''))
        out = ''.join(tokens).strip()
        print(f"Prompt {i+1}: out={out!r}")
        d = done_ev.get('spec_draft_tokens', 0)
        a = done_ev.get('spec_accepted_tokens', 0)
        rate = (a / d * 100) if d else 0
        print(f"   MTP: drafted={d}, accepted={a}, rate={rate:.1f}%")
        print(f"   TLEP: tlep_draft={done_ev.get('tlep_draft_tokens')}, tlep_acc={done_ev.get('tlep_accepted_tokens')}, max_depth={done_ev.get('tlep_max_depth')}")
    await backend.stop()

if __name__ == '__main__':
    asyncio.run(run())
