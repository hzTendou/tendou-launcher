"""Paired real HTTP smoke tests of the native single-slot serving path."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atlas.native_server import make_launch, parser


def run(output, modes):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for mode in modes:
        args = parser().parse_args(["--mtp", mode, "--port", "18081"])
        command, env = make_launch(args)
        metadata = dict(command=command, transport="default HTTP client",
                        runtime_sha256={name: hashlib.sha256((Path(args.exe_path).parent/name).read_bytes()).hexdigest()
                                        for name in ("llama-server.exe", "llama-server-impl.dll", "llama.dll", "llama-common.dll")})
        (output/f"{mode}-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (output / f"{mode}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                with httpx.Client(base_url="http://127.0.0.1:18081", timeout=180) as client:
                    deadline = time.monotonic() + 180
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f"Native server exited with {proc.returncode}; see {log.name}")
                        try:
                            if client.get("/health").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError("Native server startup timed out")
                        time.sleep(0.5)
                    client.get("/v1/models").raise_for_status()
                    cases = [
                        ("math", "What is 17 times 19? Answer only with the number.", r"\b323\b"),
                        ("code", "What does Python sorted([3, 1, 2]) return? Reply only with the list.", r"\[1,\s*2,\s*3\]"),
                        ("logic", "All ravens are birds. All birds are animals. Are all ravens animals? Answer yes or no.", r"(?i)\byes\b"),
                    ]
                    for name, question, expected in cases:
                        for rep in range(2):
                            started = time.perf_counter()
                            first = None
                            pieces, finish, usage = [], None, None
                            payload = dict(model="qwen3.8-flash-next", messages=[dict(role="user", content=question)],
                                           temperature=0, max_tokens=32, seed=42, stream=True,
                                           stream_options={"include_usage": True},
                                           chat_template_kwargs={"enable_thinking": False})
                            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                                response.raise_for_status()
                                for line in response.iter_lines():
                                    if not line.startswith("data: ") or line == "data: [DONE]":
                                        continue
                                    event = json.loads(line[6:])
                                    usage = event.get("usage") or usage
                                    for choice in event.get("choices", []):
                                        piece = choice.get("delta", {}).get("content", "") or ""
                                        if piece:
                                            first = first or time.perf_counter()
                                            pieces.append(piece)
                                        finish = choice.get("finish_reason") or finish
                            elapsed = time.perf_counter() - started
                            text = "".join(pieces)
                            row = dict(mode=mode, case=name, repetition=rep, text=text, usage=usage,
                                       finish_reason=finish, elapsed_s=elapsed,
                                       ttft_ms=(first-started)*1000 if first else None,
                                       effective_tps=usage["completion_tokens"]/elapsed if usage else None,
                                       smoke_correct=bool(re.search(expected, text)),
                                       general_quality_status="NOT_EVALUATED")
                            rows.append(row)
                            (output/"results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                            print(json.dumps(row), flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    assert all(r["smoke_correct"] and r["finish_reason"] == "stop" for r in rows), "HTTP smoke quality failed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="experiments/runtime_audit/ram16/native_api")
    p.add_argument("--modes", nargs="+", choices=["off", "ram", "dense-gpu"], default=["off", "ram"])
    args = p.parse_args()
    run(Path(args.output), args.modes)
