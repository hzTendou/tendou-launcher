#!/usr/bin/env python3
"""
Tendou Launcher — TLEP Phase 3 Full Audit & Verification Benchmark.
Compares:
  Mode A: Native Target (TLEP OFF, MTP OFF)
  Mode C: TLEP + 100% VRAM-Resident MTP Q4_0 (Zero-Copy aliasing, draft head in VRAM)
Measures:
  - TTFT, Decode TPS, Effective TPS
  - Speculative draft tokens, accepted tokens, acceptance rate
  - Speculative draft time, verify time, ms/committed token
  - Exact token parity on deterministic tasks
"""
import os
import sys
import time
import json
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

MODEL_PATH = os.environ.get(
    "TENDOU_MODEL_PATH",
    r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"
)

MTP_PATH = str(ROOT / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_0.gguf")

TASKS = [
    {
        "id": "short-math-01",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.")]),
        "expected": "323",
        "n_predict": 24,
    },
    {
        "id": "short-code-01",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])")]),
        "expected": "42",
        "n_predict": 24,
    },
    {
        "id": "math-explain-01",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="17 ile 19 carpimini adim adim yaz. 17 * 20 = 340, 340 - 17 = 323. Sonucu ve islemi acikla.")]),
        "expected": "323",
        "n_predict": 64,
    },
    {
        "id": "repetitive-code-01",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="Repeat the pattern for the next item without explanation. Write only Python code:\nitems = ['a']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['b']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['c']\nfor x in items:\n")]),
        "expected": "print",
        "n_predict": 48,
    },
    {
        "id": "creative-chat-01",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="Gunes sistemindeki gezegenleri gunese olan uzakliklarina gore sirala ve her biri hakkinda tek cumlelik ozet yaz.")]),
        "expected": "Merk",
        "n_predict": 64,
    }
]

async def run_single_mode(mode_name: str, enable_tlep: bool, mtp_mode: str, mtp_path: str = None, draft_n: int = 2):
    env_backup = dict(os.environ)
    if enable_tlep:
        os.environ["ATLAS_TLEP"] = "1"
    else:
        os.environ.pop("ATLAS_TLEP", None)

    if mtp_mode != "off":
        os.environ["ATLAS_MTP"] = mtp_mode
        if mtp_path:
            os.environ["ATLAS_MTP_PATH"] = mtp_path
    else:
        os.environ.pop("ATLAS_MTP", None)
        os.environ.pop("ATLAS_MTP_PATH", None)

    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=enable_tlep,
        mtp=mtp_mode,
        mtp_draft_n=draft_n,
        mtp_path=mtp_path if mtp_mode != "off" else None,
        readback_interval=64,
        p3_async_transfer=False,
        prompt_cache_mb=0,
    )
    print(f"\n=======================================================")
    print(f"Launching [{mode_name}] (TLEP={enable_tlep}, MTP={mtp_mode}, draft_n={draft_n})...")
    await backend.start()
    print("Backend started successfully.")

    task_results = []
    try:
        for t in TASKS:
            t0 = time.perf_counter()
            first_t = None
            last_t = None
            tokens = []
            done_ev = {}

            async for ev in backend.generate(prompt=t["prompt"], n_predict=t["n_predict"], temp=0.0):
                if ev.get("event") == "token":
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                    last_t = now
                    tokens.append(ev.get("token", ""))
                elif ev.get("event") == "done":
                    done_ev = ev

            t_end = time.perf_counter()
            output_text = "".join(tokens).strip()
            total_dur = t_end - t0
            ttft = (first_t - t0) * 1000.0 if first_t else 0.0
            decode_span = (last_t - first_t) if (first_t and last_t and last_t > first_t) else 0.0
            n_gen = len(tokens)
            decode_tps = (n_gen - 1) / decode_span if decode_span > 0 else 0.0
            eff_tps = n_gen / total_dur if total_dur > 0 else 0.0
            passed = t["expected"] in output_text

            task_results.append({
                "task": t["id"],
                "expected": t["expected"],
                "passed": passed,
                "tokens": n_gen,
                "ttft_ms": ttft,
                "decode_span_s": decode_span,
                "decode_tps": decode_tps,
                "effective_tps": eff_tps,
                "output": output_text,
                "done_event": done_ev,
            })
            draft_info = ""
            if done_ev.get("spec_draft_tokens"):
                d = done_ev.get("spec_draft_tokens", 0)
                a = done_ev.get("spec_accepted_tokens", 0)
                dt = done_ev.get("spec_draft_time_ms", 0)
                vt = done_ev.get("spec_verify_time_ms", 0)
                draft_info = f" | spec: {a}/{d} acc ({a/d*100:.1f}%), dft_t={dt:.1f}ms, ver_t={vt:.1f}ms"
            print(f"  {t['id']}: {decode_tps:.2f} dec TPS | {eff_tps:.2f} eff TPS | {n_gen} tok | pass={passed}{draft_info}")
    finally:
        await backend.stop()
        os.environ.clear()
        os.environ.update(env_backup)

    return task_results

async def main():
    print("================================================================================")
    print("   TENDOU LAUNCHER — TLEP PHASE 3 FINAL BENCHMARK (Mode A vs Mode C VRAM MTP)")
    print("================================================================================")

    res_a = await run_single_mode(
        mode_name="Mode A: Native Target (TLEP OFF)",
        enable_tlep=False,
        mtp_mode="off",
    )

    res_c = await run_single_mode(
        mode_name="Mode C: TLEP + Real Full-VRAM MTP Q4_0 (draft_n=2)",
        enable_tlep=True,
        mtp_mode="vram",
        mtp_path=MTP_PATH,
        draft_n=2,
    )

    print("\n================================================================================")
    print("   TLEP PHASE 3 EMPIRICAL COMPARISON REPORT")
    print("================================================================================")

    for a, c in zip(res_a, res_c):
        task_id = a["task"]
        parity_c = (a["output"] == c["output"])
        print(f"\nTask: {task_id}")
        print(f"  Accuracy: Mode A={a['passed']}, Mode C={c['passed']}")
        print(f"  Token Parity (A vs C): {'MATCH (%100)' if parity_c else 'MISMATCH'}")
        print(f"  Decode TPS:    Mode A={a['decode_tps']:.2f} | Mode C={c['decode_tps']:.2f}")
        print(f"  Effective TPS: Mode A={a['effective_tps']:.2f} | Mode C={c['effective_tps']:.2f}")
        
        dec_c = c["done_event"]
        if dec_c:
            d_c = dec_c.get("spec_draft_tokens", 0)
            a_c = dec_c.get("spec_accepted_tokens", 0)
            dt_c = dec_c.get("spec_draft_time_ms", 0)
            vt_c = dec_c.get("spec_verify_time_ms", 0)
            print(f"  Mode C Specs: {d_c} drafted, {a_c} accepted ({(a_c/d_c*100) if d_c else 0:.1f}%), draft_ms={dt_c:.1f}, verify_ms={vt_c:.1f}")
            if d_c > 0:
                print(f"  Single draft step latency: {dt_c / d_c:.2f} ms/token")

    out_file = ROOT / "scripts" / "phase3_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"mode_a": res_a, "mode_c": res_c}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved empirical results to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
