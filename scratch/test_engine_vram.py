import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asyncio
import time
from src.atlas.server import (
    AtlasSubprocessBackend,
    DEFAULT_EXE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_ID,
)
from scratch.benchmark_server_tps import get_cuda_vram_info

async def main():
    f0, t0, u0 = get_cuda_vram_info()
    print(f"Initial: Used={u0:.1f} MiB, Free={f0:.1f} MiB")

    backend = AtlasSubprocessBackend(
        exe_path=DEFAULT_EXE_PATH,
        model_path=DEFAULT_MODEL_PATH,
        threads=8,
        mtp="pld",
        boost_level=2,
        prompt_cache_mb=0,
    )
    print("Starting backend...")
    t_start = time.perf_counter()
    await backend.start()
    t_init = time.perf_counter() - t_start
    print(f"Backend started in {t_init:.2f}s")

    await asyncio.sleep(1.0)
    f1, t1, u1 = get_cuda_vram_info()
    delta = u1 - u0
    print(f"After load: Used={u1:.1f} MiB, Free={f1:.1f} MiB, Delta={delta:.1f} MiB")

    # Test tool-call-edit-01 prompt
    from src.atlas.server import format_chatml_prompt, ChatMessage, ToolDefinition
    tools = [
        ToolDefinition.model_validate({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit or write file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        })
    ]
    prompt = format_chatml_prompt(
        messages=[ChatMessage(role="user", content="Edit main.py to add a healthcheck endpoint.")],
        tools=tools,
    )
    print(f"Prompt length chars: {len(prompt)}")
    toks = []
    t_gen_start = time.perf_counter()
    first_tok_time = None
    async for ev in backend.generate(prompt=prompt, n_predict=48, temp=0.0):
        if ev.get("event") == "token":
            now = time.perf_counter()
            if first_tok_time is None:
                first_tok_time = now
                print(f"First token received in: {(first_tok_time - t_gen_start)*1000.0:.1f}ms")
            toks.append(ev.get("token", ""))
        elif ev.get("event") == "done":
            t_done = time.perf_counter()
            print(f"Done event: {ev}, total gen time: {(t_done - t_gen_start):.2f}s")
    print("Generated:", repr("".join(toks)))

    f2, t2, u2 = get_cuda_vram_info()
    print(f"After gen: Used={u2:.1f} MiB, Free={f2:.1f} MiB, Delta={u2 - u0:.1f} MiB")

    await backend.stop()
    print("Backend stopped.")

if __name__ == "__main__":
    asyncio.run(main())
