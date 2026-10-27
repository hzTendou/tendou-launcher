#!/usr/bin/env python3
"""
TLEP (Tendou Lookahead Execution Pipeline) Benchmark & Parity Verification.
Tests deterministic output parity, effective sustained TPS, and telemetry metrics.
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

RAW_TASKS = [
    {
        "id": "short-math-01",
        "user_query": "17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.",
        "expected": "323",
        "n_predict": 24,
    },
    {
        "id": "short-code-01",
        "user_query": "Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])",
        "expected": "42",
        "n_predict": 24,
    },
    {
        "id": "math-explain-01",
        "user_query": "17 ile 19 carpimini adim adim yaz. 17 * 20 = 340, 340 - 17 = 323. Sonucu ve islemi acikla.",
        "expected": "323",
        "n_predict": 64,
    },
    {
        "id": "repetitive-code-01",
        "user_query": "Repeat the pattern for the next item without explanation. Write only Python code:\nitems = ['a']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['b']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['c']\nfor x in items:\n",
        "expected": "print",
        "n_predict": 48,
    },
]

TASKS = [
    {
        **t,
        "prompt": format_chatml_prompt([ChatMessage(role="user", content=t["user_query"])]),
    }
    for t in RAW_TASKS
]

async def run_benchmark(
    mode_name: str,
    enable_tlep: bool,
    mtp_mode: str = "off",
    mtp_path: str = None,
    trace_path: str = None,
    tasks_override: list = None,
):
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
    print(f"Starting Backend [{mode_name}] (TLEP={enable_tlep}, MTP={mtp_mode})...")
    await backend.start()
    print("Backend ready.")

    results = []
    try:
        tasks_to_run = tasks_override or TASKS
        for t in tasks_to_run:
            print(f"\n--- Running Task: {t['id']} (expected '{t['expected']}') ---")
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

            res = {
                "task": t["id"],
                "passed": passed,
                "output": output_text,
                "tokens": n_gen,
                "ttft_ms": ttft,
                "decode_tps": decode_tps,
                "effective_tps": eff_tps,
                "done_event": done_ev,
            }
            results.append(res)
            print(f"Output: '{output_text[:60]}' | Pass: {passed}")
            print(f"Generated: {n_gen} toks | TTFT: {ttft:.1f} ms | Decode TPS: {decode_tps:.2f} | Effective TPS: {eff_tps:.2f}")
            if done_ev.get("tlep_enabled") or done_ev.get("spec_draft_tokens", 0) > 0:
                draft_toks = done_ev.get("spec_draft_tokens", 0)
                acc_toks = done_ev.get("spec_accepted_tokens", 0)
                acc_rate = (acc_toks / draft_toks * 100.0) if draft_toks > 0 else 0.0
        diag_lines = [line for line in backend.stderr_lines if "[DIAG" in line]
        if diag_lines:
            print("  --- Diagnostic Samples (last 10) ---")
            for dl in diag_lines[-10:]:
                print(f"    {dl}")
    finally:
        await backend.stop()
        os.environ.clear()
        os.environ.update(env_backup)

    return results

async def main():
    print("================================================================")
    print("=== TLEP FULL A/B/C BENCHMARK & DETERMINISTIC PARITY SUITE ===")
    print("================================================================")

    print("\n>>> MODE A: NATIVE BASELINE (TLEP OFF, MTP OFF) <<<")
    res_a = await run_benchmark(
        mode_name="Mode A: Baseline",
        enable_tlep=False,
        mtp_mode="off"
    )

    print("\n>>> MODE B: TLEP + N-GRAM SPECULATION <<<")
    res_b = await run_benchmark(
        mode_name="Mode B: TLEP+ngram",
        enable_tlep=True,
        mtp_mode="off"
    )

    mtp_mode = os.environ.get("BENCH_MTP_MODE", "ram")
    print(f"\n>>> MODE C: TLEP + REAL NATIVE MTP DRAFT HEAD ({mtp_mode}) <<<")
    res_c = await run_benchmark(
        mode_name=f"Mode C: TLEP+MTP ({mtp_mode})",
        enable_tlep=True,
        mtp_mode=mtp_mode,
        mtp_path=MTP_PATH,
        trace_path="trace_tlep.jsonl"
    )

    print("\n\n==================================================================================")
    print("=== FULL SAME-RUN THREE-WAY COMPARISON (MODE A vs MODE B vs MODE C) ===")
    print("==================================================================================")

    total_tokens_a, total_tokens_b, total_tokens_c = 0, 0, 0
    total_time_a, total_time_b, total_time_c = 0.0, 0.0, 0.0

    all_parity_match = True

    for a, b, c in zip(res_a, res_b, res_c):
        t_id = a["task"]
        speedup_b = b["decode_tps"] / a["decode_tps"] if a["decode_tps"] > 0 else 1.0
        speedup_c = c["decode_tps"] / a["decode_tps"] if a["decode_tps"] > 0 else 1.0
        eff_speedup_c = c["effective_tps"] / a["effective_tps"] if a["effective_tps"] > 0 else 1.0

        parity_b = (a["output"] == b["output"])
        parity_c = (a["output"] == c["output"])
        if not parity_c:
            all_parity_match = False

        total_tokens_a += a["tokens"]
        total_tokens_b += b["tokens"]
        total_tokens_c += c["tokens"]

        dur_a = (a["tokens"] / a["effective_tps"]) if a["effective_tps"] > 0 else 0.0
        dur_b = (b["tokens"] / b["effective_tps"]) if b["effective_tps"] > 0 else 0.0
        dur_c = (c["tokens"] / c["effective_tps"]) if c["effective_tps"] > 0 else 0.0
        total_time_a += dur_a
        total_time_b += dur_b
        total_time_c += dur_c

        print(f"\nTask: {t_id} (Expected: '{a.get('expected', '')}')")
        print(f"  Accuracy: Mode A={a['passed']}, Mode B={b['passed']}, Mode C={c['passed']}")
        print(f"  Parity vs Base: Mode B={'MATCH' if parity_b else 'MISMATCH'}, Mode C={'MATCH (%100 parite)' if parity_c else 'MISMATCH'}")
        print(f"    Mode A Out: '{a['output'][:50]}...' ({a['tokens']} toks)")
        print(f"    Mode B Out: '{b['output'][:50]}...' ({b['tokens']} toks)")
        print(f"    Mode C Out: '{c['output'][:50]}...' ({c['tokens']} toks)")
        print(f"  Decode TPS:    Mode A={a['decode_tps']:.2f} | Mode B={b['decode_tps']:.2f} ({speedup_b:.2f}x) | Mode C={c['decode_tps']:.2f} ({speedup_c:.2f}x)")
        print(f"  Effective TPS: Mode A={a['effective_tps']:.2f} | Mode B={b['effective_tps']:.2f} | Mode C={c['effective_tps']:.2f} ({eff_speedup_c:.2f}x)")
        
        de_c = c["done_event"]
        if de_c:
            d_toks = de_c.get("spec_draft_tokens", 0)
            a_toks = de_c.get("spec_accepted_tokens", 0)
            acc_rate = (a_toks / d_toks * 100.0) if d_toks > 0 else 0.0
            print(f"  Mode C Telemetry:")
            print(f"    Draft / Accepted: {d_toks} drafted, {a_toks} accepted ({acc_rate:.1f}% acceptance rate)")
            print(f"    Spec Efficiency: {de_c.get('tlep_spec_efficiency', 0)*100:.1f}%")
            print(f"    Prefetch Precision: {de_c.get('tlep_prefetch_precision', 0)*100:.1f}%")
            print(f"    Accepted / Verify: {de_c.get('tlep_accepted_per_verify', 0):.2f}")
            print(f"    Draft Wall Time: {de_c.get('spec_draft_time_ms', 0):.1f} ms, Verify Wall Time: {de_c.get('spec_verify_time_ms', 0):.1f} ms")

    agg_eff_a = total_tokens_a / total_time_a if total_time_a > 0 else 0.0
    agg_eff_b = total_tokens_b / total_time_b if total_time_b > 0 else 0.0
    agg_eff_c = total_tokens_c / total_time_c if total_time_c > 0 else 0.0

    print("\n==================================================================================")
    print("=== AGGREGATE SUMMARY ===")
    print("==================================================================================")
    print(f"Aggregate Effective TPS:")
    print(f"  Mode A (Baseline):  {agg_eff_a:.2f} TPS")
    print(f"  Mode B (TLEP+ngram): {agg_eff_b:.2f} TPS ({agg_eff_b / agg_eff_a:.2f}x)")
    print(f"  Mode C (TLEP+MTP):   {agg_eff_c:.2f} TPS ({agg_eff_c / agg_eff_a:.2f}x)")
    print(f"100% Deterministic Parity across all tasks: {'PASS' if all_parity_match else 'FAIL'}")
    print("==================================================================================")

if __name__ == "__main__":
    asyncio.run(main())
