"""
Teşhis Betiği: Qwen3.8 Flash Next TPS Düşüşü Kök Neden Analizi ve Doğrulaması.
Karşılaştırma:
1. Yanlış Harita ile (Cold & Warm)
2. Haritasız / Temiz v0.13.0-stable (Cold & Warm)
3. 23+ TPS alınan orijinal senaryo (creative-chat ve repetitive-code)
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

MODEL_PATH = Path(r"C:\Users\Ali\WebProjects\tendou-launcher\models\qwen3.8-gsq-rco-iq3_xxs\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
ENGINE_PATH = ROOT / "releases" / "tendou-v0.13.0-stable" / "bin" / "tendou-engine.exe"
WRONG_MAP = ROOT / "config" / "atlas_physical_map_qwen38.json"

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def run_test_server(use_wrong_map: bool, desc: str):
    print(f"\n=======================================================")
    print(f"TEST BAŞLATILIYOR: {desc}")
    print(f"Exe: {ENGINE_PATH}")
    print(f"Wrong Map: {use_wrong_map}")
    print(f"=======================================================")

    extra_args = []
    if use_wrong_map and WRONG_MAP.is_file():
        extra_args.extend(["--atlas-physical-map", str(WRONG_MAP)])

    backend = AtlasSubprocessBackend(
        exe_path=str(ENGINE_PATH),
        model_path=str(MODEL_PATH),
        threads=8,
        mtp="pld",
        mtp_draft_n=3,
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
        print("Sunucu başlatılamadı!")
        return

    test_prompt = (
        "Repeat the pattern for the next item without explanation. Write only Python code:\n"
        "items = ['a']\n"
        "for x in items:\n"
        "    print('Element')\n"
        "    print('Done')\n\n"
        "items = ['b']\n"
        "for x in items:\n"
        "    print('Element')\n"
        "    print('Done')\n\n"
        "items = ['c']\n"
        "for x in items:\n"
    )

    creative_prompt = "Gunes Sistemi'ndeki sekiz gezegeni gunese olan uzakliklarina gore sirala ve her bir gezegen hakkinda ayrintili bir paragraf yaz."

    prompts = [
        ("Repetitive Code (PLD Template)", test_prompt, 32),
        ("Creative Chat (Free Text)", creative_prompt, 32),
    ]

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        for p_name, p_text, max_tok in prompts:
            print(f"\n--- İş Yükü: {p_name} ---")
            for rep in range(1, 3):
                req = {
                    "model": "auto",
                    "messages": [{"role": "user", "content": p_text}],
                    "stream": True,
                    "temperature": 0.0,
                    "max_tokens": max_tok,
                }
                t0 = time.perf_counter()
                first_t = None
                last_t = None
                n_tokens = 0
                text = ""

                with client.stream("POST", "/v1/chat/completions", json=req) as resp:
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            c = json.loads(raw)
                            delta = c["choices"][0]["delta"]
                            if delta.get("content"):
                                now = time.perf_counter()
                                if first_t is None:
                                    first_t = now
                                last_t = now
                                n_tokens += 1
                                text += delta["content"]
                        except Exception:
                            continue
                t1 = time.perf_counter()
                ttft_ms = (first_t - t0) * 1000.0 if first_t else 0.0
                decode_dur = (last_t - first_t) if (first_t and last_t and last_t > first_t) else (t1 - t0)
                decode_tps = (n_tokens / decode_dur) if decode_dur > 0 else 0.0
                print(f"  Rep {rep} ({'SOĞUK (COLD)' if rep == 1 else 'SICAK (WARM)'}): Toks={n_tokens} | TTFT={ttft_ms:.1f}ms | Decode TPS={decode_tps:.2f} tok/s | Elapsed={t1-t0:.2f}s")

    # Stop server
    try:
        import asyncio
        asyncio.run(backend.stop())
    except Exception:
        pass
    server.should_exit = True

if __name__ == "__main__":
    # Test 1: Clean (Doğru, Haritasız, v0.13.0-stable)
    run_test_server(use_wrong_map=False, desc="TEMİZ v0.13.0-stable (Haritasız, Native IQ3_XXS)")

    # Test 2: Yanlış Harita ile (Q5_K_S haritası zorlanmış)
    run_test_server(use_wrong_map=True, desc="YANLIŞ HARİTALI v0.13.0-stable (atlas_physical_map_qwen38.json)")
