"""
Ablation Test: Config-D TPS Düşüşü Kök Neden İzolasyonu
Test değişkenleri:
1. v0.11 vs v0.13 (mtp off) -> Motor binary farkı var mı?
2. v0.13 mtp off vs mtp pld (draft_n=3) -> PLD cezası var mı?
3. 35B: physical_map_35b (eski) vs physical_map_35b_3tier (yeni)
"""

import sys
import os
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
MODEL_38 = Path(r"C:\Users\Ali\WebProjects\tendou-launcher\models\qwen3.8-gsq-rco-iq3_xxs\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")

ENGINE_V11 = ROOT / "releases" / "tendou-v0.11.0-stable" / "bin" / "tendou-engine.exe"
ENGINE_V13 = ROOT / "releases" / "tendou-v0.13.0-stable" / "bin" / "tendou-engine.exe"

MAP_35B_OLD = ROOT / "config" / "atlas_physical_map_35b.json"
MAP_35B_3TIER = ROOT / "config" / "atlas_physical_map_35b_3tier.json"

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def test_ablation(desc, exe_path, model_path, mtp_mode, physical_map, prompt, max_tokens=40):
    print(f"\n>>> TESTING: {desc}")
    extra_args = []
    if physical_map and physical_map.is_file():
        extra_args.extend(["--atlas-physical-map", str(physical_map)])

    backend = AtlasSubprocessBackend(
        exe_path=str(exe_path),
        model_path=str(model_path),
        threads=8,
        gpu_layers=2,
        mtp=mtp_mode,
        mtp_draft_n=3 if mtp_mode == "pld" else 2,
        boost_level=2,
        prompt_cache_mb=0,
        extra_args=extra_args,
    )

    app = create_app(backend=backend)
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Preflight
    started = False
    for _ in range(60):
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.1)

    if not started:
        print("  [ERROR] Server failed to start")
        return

    req_payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    try:
        with httpx.Client(base_url=base_url, timeout=120.0) as client:
            # Warmup
            client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "1+1="}], "max_tokens": 4})
            
            # Rep 1 & 2
            for rep in (1, 2):
                t0 = time.perf_counter()
                first_t = None
                last_t = None
                final_usage = None
                text = ""

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
                                text += delta["content"]
                        except Exception:
                            continue
                t1 = time.perf_counter()
                tokens = final_usage.get("completion_tokens", 0) if final_usage else 0
                dur = (last_t - first_t) if (first_t and last_t and last_t > first_t) else (t1 - t0)
                tps = (tokens - 1) / dur if dur > 0 and tokens > 1 else 0.0
                print(f"  Rep {rep} ({'Cold' if rep==1 else 'Warm'}): Tokens={tokens} | Decode Dur={dur:.3f}s | TPS={tps:.2f} tok/s | Total={t1-t0:.2f}s")
    finally:
        try:
            import asyncio
            asyncio.run(backend.stop())
        except Exception:
            pass
        server.should_exit = True

if __name__ == "__main__":
    prompt_code = (
        "Repeat the pattern for the next item without explanation. Write only Python code:\n"
        "items = ['a']\nfor x in items:\n    print('Element')\n    print('Done')\n\n"
        "items = ['b']\nfor x in items:\n    print('Element')\n    print('Done')\n\n"
        "items = ['c']\nfor x in items:\n"
    )
    prompt_rust = "Write a highly optimized lock-free concurrent ring buffer in Rust with memory ordering rationale."

    print("=================================================================")
    print("ABLATION STUDY: ISOLATING PERFORMANCE DIFFERENCES")
    print("=================================================================")

    # Test A1: Qwen3.8 Flash Next - v0.11 Engine (mtp=off) - Rust Greenfield
    test_ablation("Qwen3.8 | v0.11 Engine | MTP=OFF | Rust Greenfield", ENGINE_V11, MODEL_38, "off", None, prompt_rust)

    # Test A2: Qwen3.8 Flash Next - v0.13 Engine (mtp=off) - Rust Greenfield
    test_ablation("Qwen3.8 | v0.13 Engine | MTP=OFF | Rust Greenfield", ENGINE_V13, MODEL_38, "off", None, prompt_rust)

    # Test A3: Qwen3.8 Flash Next - v0.13 Engine (mtp=pld) - Rust Greenfield
    test_ablation("Qwen3.8 | v0.13 Engine | MTP=PLD | Rust Greenfield", ENGINE_V13, MODEL_38, "pld", None, prompt_rust)

    # Test B1: Qwen3.6-35B - v0.13 Engine | MTP=OFF | Old Map (35b.json) - Rust Greenfield
    test_ablation("Qwen3.6-35B | v0.13 Engine | MTP=OFF | Old Map | Rust Greenfield", ENGINE_V13, MODEL_35B, "off", MAP_35B_OLD, prompt_rust)

    # Test B2: Qwen3.6-35B - v0.13 Engine | MTP=OFF | 3-Tier Map (35b_3tier.json) - Rust Greenfield
    test_ablation("Qwen3.6-35B | v0.13 Engine | MTP=OFF | 3-Tier Map | Rust Greenfield", ENGINE_V13, MODEL_35B, "off", MAP_35B_3TIER, prompt_rust)

    # Test B3: Qwen3.6-35B - v0.13 Engine | MTP=PLD | 3-Tier Map (35b_3tier.json) - Rust Greenfield
    test_ablation("Qwen3.6-35B | v0.13 Engine | MTP=PLD | 3-Tier Map | Rust Greenfield", ENGINE_V13, MODEL_35B, "pld", MAP_35B_3TIER, prompt_rust)
