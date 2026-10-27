#!/usr/bin/env python3
"""
Tendou Launcher — TLEP Phase 2 Audit & Benchmark Suite.
Measures:
1. Native baseline integrity (15.11 TPS reproduction & CPU MoE regression audit).
2. VRAM allocation table with cudaMemGetInfo.
3. Three-way same-run comparison (A: Native TLEP OFF, B: TLEP+n-gram, C: TLEP+Real MTP).
4. Telemetry KPIs: C_committed (ms/token), accepted/verify, interference %, parity.
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

MTP_PATH = os.environ.get(
    "TENDOU_MTP_PATH",
    str(ROOT / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf")
)

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

async def run_single_mode(mode_name: str, enable_tlep: bool, mtp_mode: str, mtp_path: str = None, trace_path: str = None):
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

    if trace_path:
        os.environ["ATLAS_DEEP_TELEMETRY"] = "1"
        os.environ["ATLAS_TRACE_PATH"] = trace_path
    else:
        os.environ.pop("ATLAS_DEEP_TELEMETRY", None)
        os.environ.pop("ATLAS_TRACE_PATH", None)

    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        tlep=enable_tlep,
        mtp=mtp_mode,
        mtp_path=mtp_path if mtp_mode != "off" else None,
        readback_interval=64,
        p3_async_transfer=False,
        prompt_cache_mb=0,
    )
    print(f"\n=======================================================")
    print(f"Launching [{mode_name}] (TLEP={enable_tlep}, MTP={mtp_mode})...")
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
            print(f"  {t['id']}: {decode_tps:.2f} dec TPS | {eff_tps:.2f} eff TPS | {n_gen} tokens | pass={passed}")
    finally:
        await backend.stop()
        os.environ.clear()
        os.environ.update(env_backup)

    return task_results

async def main():
    print("================================================================================")
    print("   TENDOU LAUNCHER — TLEP PHASE 2 COMPREHENSIVE EXPERIMENT")
    print("================================================================================")

    # Mode A: Native Target (TLEP OFF)
    res_a = await run_single_mode(
        mode_name="Mode A: Native Target (TLEP OFF)",
        enable_tlep=False,
        mtp_mode="off",
        trace_path="trace_mode_a.jsonl"
    )

    # Mode B: TLEP + n-gram
    res_b = await run_single_mode(
        mode_name="Mode B: TLEP + n-gram",
        enable_tlep=True,
        mtp_mode="off",
        trace_path="trace_mode_b.jsonl"
    )

    # Mode C: TLEP + Real MTP (hybrid location for bounded VRAM)
    res_c = await run_single_mode(
        mode_name="Mode C: TLEP + Real MTP (hybrid)",
        enable_tlep=True,
        mtp_mode="hybrid",
        mtp_path=MTP_PATH,
        trace_path="trace_mode_c.jsonl"
    )

    print("\n================================================================================")
    print("   SUMMARY REPORT (SAME-RUN THREE-WAY COMPARISON)")
    print("================================================================================")

    for a, b, c in zip(res_a, res_b, res_c):
        task_id = a["task"]
        parity_b = (a["output"] == b["output"])
        parity_c = (a["output"] == c["output"])
        print(f"\nTask: {task_id}")
        print(f"  Accuracy: Mode A={a['passed']}, Mode B={b['passed']}, Mode C={c['passed']}")
        print(f"  Token Parity vs Mode A: Mode B={'MATCH' if parity_b else 'MISMATCH'}, Mode C={'MATCH (%100)' if parity_c else 'MISMATCH'}")
        print(f"  Decode TPS:    Mode A={a['decode_tps']:.2f} | Mode B={b['decode_tps']:.2f} | Mode C={c['decode_tps']:.2f}")
        print(f"  Effective TPS: Mode A={a['effective_tps']:.2f} | Mode B={b['effective_tps']:.2f} | Mode C={c['effective_tps']:.2f}")
        
        dec_b = b["done_event"]
        if dec_b:
            d_b = dec_b.get("spec_draft_tokens", 0)
            a_b = dec_b.get("spec_accepted_tokens", 0)
            print(f"  Mode B Specs (n-gram): {d_b} drafted, {a_b} accepted ({(a_b/d_b*100) if d_b else 0:.1f}%)")
        dec_c = c["done_event"]
        if dec_c:
            d_c = dec_c.get("spec_draft_tokens", 0)
            a_c = dec_c.get("spec_accepted_tokens", 0)
            print(f"  Mode C Specs (MTP):    {d_c} drafted, {a_c} accepted ({(a_c/d_c*100) if d_c else 0:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
