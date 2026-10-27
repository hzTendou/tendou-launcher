import sys
import time
import json
import socket
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import uvicorn
from src.atlas.server import create_app, AtlasSubprocessBackend

MODEL_35B = Path(r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
ENGINE_V13 = ROOT / "releases" / "tendou-v0.13.0-stable" / "bin" / "tendou-engine.exe"
MAP_35B_3TIER = ROOT / "config" / "atlas_physical_map_35b_3tier.json"

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def test_speed():
    backend = AtlasSubprocessBackend(
        exe_path=str(ENGINE_V13),
        model_path=str(MODEL_35B),
        threads=8,
        gpu_layers=2,
        mtp="off",
        boost_level=2,
        prompt_cache_mb=0,
        extra_args=["--atlas-physical-map", str(MAP_35B_3TIER)],
    )

    app = create_app(backend=backend)
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    for _ in range(60):
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    prompt_rust = "Write a highly optimized lock-free concurrent ring buffer in Rust with memory ordering rationale."
    req_payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt_rust}],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": 40,
    }

    try:
        with httpx.Client(base_url=base_url, timeout=120.0) as client:
            client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "1+1="}], "max_tokens": 4})
            for rep in (1, 2):
                t0 = time.perf_counter()
                first_t = None
                last_t = None
                final_usage = None
                with client.stream("POST", "/v1/chat/completions", json=req_payload) as resp:
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            c = json.loads(raw)
                            if "usage" in c and c["usage"]:
                                final_usage = c["usage"]
                            delta = c["choices"][0]["delta"]
                            if delta.get("content"):
                                now = time.perf_counter()
                                if first_t is None:
                                    first_t = now
                                last_t = now
                        except Exception:
                            continue
                t1 = time.perf_counter()
                tokens = final_usage.get("completion_tokens", 0) if final_usage else 0
                dur = (last_t - first_t) if (first_t and last_t and last_t > first_t) else (t1 - t0)
                tps = (tokens - 1) / dur if dur > 0 and tokens > 1 else 0.0
                print(f"  Rep {rep} ({'Cold' if rep==1 else 'Warm'}): Tokens={tokens} | Decode Dur={dur:.3f}s | TPS={tps:.2f} tok/s")
    finally:
        try:
            import asyncio
            asyncio.run(backend.stop())
        except Exception:
            pass
        server.should_exit = True

if __name__ == "__main__":
    print("Testing 35B with Elastic 3-Tier Map (Zero Disk Paging)...")
    test_speed()
