"""Measure real IPC latency and retain tokens for full-router fidelity checks."""
import argparse
import asyncio
import json
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atlas.server import AtlasSubprocessBackend, DEFAULT_EXE_PATH


async def run(args):
    backend = AtlasSubprocessBackend(
        exe_path=args.exe, ctx_size=2048, boost=True, threads=args.threads,
        extra_args=["--atlas-k", "0"] + args.engine_args,
    )
    cases = [
        ("math", "What is 17 times 19? Answer with just the number."),
        ("code", "What does Python sorted([3, 1, 2]) return? Reply with only the list."),
        ("logic", "All ravens are birds. All birds are animals. Are all ravens animals? Answer yes or no."),
    ]
    if args.cases:
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    results = []
    metadata = dict(exe=str(Path(args.exe).resolve()),
                    exe_sha256=hashlib.sha256(Path(args.exe).read_bytes()).hexdigest(),
                    runtime_dll_sha256={n:hashlib.sha256((Path(args.exe).parent/n).read_bytes()).hexdigest()
                                        for n in ("llama.dll", "llama-common.dll", "ggml-cpu.dll")},
                    native_router=True, engine_args=args.engine_args, threads=args.threads,
                    context=2048, max_tokens=args.tokens, temperature=0)
    try:
        t0 = time.perf_counter()
        await backend.start()
        startup = time.perf_counter() - t0
        for name, question in cases:
            prompt = ("<|im_start|>user\n" + question +
                      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            for rep in range(args.repetitions):
                started = time.perf_counter()
                first = None
                last = None
                tokens, pieces, done = [], [], None
                async for event in backend.generate(prompt, n_predict=args.tokens, temp=0):
                    if event["event"] == "error":
                        raise RuntimeError(event)
                    if event["event"] == "token":
                        last = time.perf_counter()
                        first = first or last
                        tokens.append(event["token_id"])
                        pieces.append(event["token"])
                    elif event["event"] == "done":
                        done = event
                elapsed = time.perf_counter() - started
                if not tokens or not done:
                    raise RuntimeError("Incomplete generation")
                row = dict(case=name, prompt=prompt, repetition=rep, ttft_ms=(first-started)*1000,
                           elapsed_s=elapsed, effective_tps=len(tokens)/elapsed,
                           decode_tps=(len(tokens)-1)/(last-first) if len(tokens)>1 else None,
                           token_ids=tokens, text="".join(pieces), done=done)
                results.append(row)
                Path(args.output).write_text(json.dumps(dict(**metadata, startup_s=startup, results=results), indent=2), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=True), flush=True)
    finally:
        await backend.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default=DEFAULT_EXE_PATH)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--cases", help="JSON array of [name, question] pairs")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.engine_args[:1] == ["--"]:
        args.engine_args = args.engine_args[1:]
    asyncio.run(asyncio.wait_for(run(args), timeout=1200))
