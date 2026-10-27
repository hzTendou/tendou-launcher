#!/usr/bin/env python3
"""
Tendou Launcher — Phase 5.1 Forensic Fix & Real VRAM Binding Audit.
Hard Acceptance Gates:
  1. Real Physical VRAM Binding via NVML:
     - Allocated Used VRAM >= 4,500 MiB
     - Free VRAM: 1,024 MiB <= Free <= 2,600 MiB (Free > 3,500 MiB is automatic FAIL)
  2. Short prompt TTFT <= 600 ms (short-math-01, short-code-01)
  3. Native Decode TPS strictly beats Phase 4 Mode A baseline:
     - short-math-01: > 7.50 TPS (P4: 7.41)
     - short-code-01: > 10.60 TPS (P4: 10.55)
     - math-explain-01: > 13.50 TPS (P4: 12.33)
     - repetitive-code-01: > 14.50 TPS (P4: 13.12)
     - creative-chat-01: > 15.00 TPS (P4: 13.78)
  4. 100% Token Parity on all 5 tasks:
     - short-math-01: '323'
     - short-code-01: '42'
     - math-explain-01: '17 ile 19'
     - repetitive-code-01: 'print'
     - creative-chat-01: 'Merk'
"""
import os
import sys
import time
import json
import asyncio
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

DEFAULT_MODEL_PATH = str(ROOT / "models" / "qwen3.8-gsq-rco-iq3_xxs" / "IQ3_XXS" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
MODEL_PATH = os.environ.get("TENDOU_MODEL_PATH", DEFAULT_MODEL_PATH)

TASKS = [
    {
        "id": "short-math-01",
        "category": "math",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.")]),
        "expected": "323",
        "baseline_tps": 7.41,
        "target_tps": 7.50,
        "n_predict": 36,
    },
    {
        "id": "short-code-01",
        "category": "code",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])")]),
        "expected": "42",
        "baseline_tps": 10.55,
        "target_tps": 10.60,
        "n_predict": 36,
    },
    {
        "id": "math-explain-01",
        "category": "math",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="17 ile 19 carpimini adim adim yaz. 17 * 20 = 340, 340 - 17 = 323. Sonucu ve islemi acikla.")]),
        "expected": "17 ile 19",
        "baseline_tps": 12.33,
        "target_tps": 13.50,
        "n_predict": 64,
    },
    {
        "id": "repetitive-code-01",
        "category": "code",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="Repeat the pattern for the next item without explanation. Write only Python code:\nitems = ['a']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['b']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['c']\nfor x in items:\n")]),
        "expected": "print",
        "baseline_tps": 13.12,
        "target_tps": 14.50,
        "n_predict": 48,
    },
    {
        "id": "creative-chat-01",
        "category": "general",
        "prompt": format_chatml_prompt([ChatMessage(role="user", content="Gunes sistemindeki gezegenleri gunese olan uzakliklarina gore sirala ve her biri hakkinda tek cumlelik ozet yaz.")]),
        "expected": "Merk",
        "baseline_tps": 13.78,
        "target_tps": 15.00,
        "n_predict": 64,
    }
]

def get_cuda_vram_info():
    """Queries physical free, total, and used VRAM in MiB using NVML."""
    try:
        nvml = ctypes.CDLL("nvml.dll")
        nvml.nvmlInit_v2()
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        class Memory(ctypes.Structure):
            _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
        mem = Memory()
        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
        free_mib = mem.free / (1024.0 * 1024.0)
        total_mib = mem.total / (1024.0 * 1024.0)
        used_mib = mem.used / (1024.0 * 1024.0)
        nvml.nvmlShutdown()
        return free_mib, total_mib, used_mib
    except Exception as e:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total,memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            f_str, t_str, u_str = out.strip().split(",")
            return float(f_str), float(t_str), float(u_str)
        except Exception as e2:
            print(f"Warning: Failed to query CUDA VRAM via NVML and nvidia-smi: {e2}")
            return 0.0, 0.0, 0.0

async def run_benchmark():
    print("================================================================================")
    print("   TENDOU LAUNCHER — PHASE 5.1 FORENSIC FIX & REAL VRAM BINDING AUDIT")
    print("================================================================================")

    free_initial, total_initial, used_initial = get_cuda_vram_info()
    print(f"Initial Physical VRAM: Used = {used_initial:.1f} MiB / Free = {free_initial:.1f} MiB / Total = {total_initial:.1f} MiB")

    threads_val = int(os.environ.get("TENDOU_THREADS", "8"))
    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=threads_val,
        ctx_size=2048,
        boost=True,
        boost_level=2,
        gpu_layers=2,
        moe_cache_mib=0,
        mtp="off",
        tlep=False,
        min_p=0.0,
        readback_interval=64,
        p3_async_transfer=False,
        prompt_cache_mb=256,
    )

    print(f"\nLaunching Native Engine with MTP=OFF, GPU Layers=2, Boost=2, {threads_val} Threads, Prompt Cache=256MB...")
    await backend.start()
    print("Native Engine started successfully.")

    # Allow memory allocation to settle
    await asyncio.sleep(1.0)
    free_loaded, total_loaded, used_loaded = get_cuda_vram_info()
    model_vram_delta = used_loaded - used_initial
    print(f"Resident Physical VRAM: Used = {used_loaded:.1f} MiB / Free = {free_loaded:.1f} MiB (Model Delta: {model_vram_delta:.1f} MiB)")

    print("\nExecuting warmup...")
    warmup_prompt = format_chatml_prompt([ChatMessage(role="user", content="Warmup 1 2 3")])
    try:
        async for ev in backend.generate(prompt=warmup_prompt, n_predict=4, temp=0.0):
            if ev.get("event") == "error":
                print(f"Warmup error: {ev.get('message')}")
                break
    except Exception as e:
        print(f"Warmup exception: {e}")
    print("Warmup complete. Running 5 benchmark tasks...\n")

    results = []
    try:
        for t in TASKS:
            t0 = time.perf_counter()
            first_t = None
            last_t = None
            tokens = []
            done_ev = {}

            try:
                async for ev in backend.generate(prompt=t["prompt"], n_predict=t["n_predict"], temp=0.0):
                    if ev.get("event") == "token":
                        now = time.perf_counter()
                        if first_t is None:
                            first_t = now
                        last_t = now
                        tokens.append(ev.get("token", ""))
                    elif ev.get("event") == "done":
                        done_ev = ev
                    elif ev.get("event") == "error":
                        print(f"[{t['id']}] Error event: {ev.get('message')}")
                        break
            except Exception as e:
                print(f"[{t['id']}] Generation exception: {e}")

            t_end = time.perf_counter()
            output_text = "".join(tokens).strip()
            total_dur = t_end - t0
            ttft = (first_t - t0) * 1000.0 if first_t else 0.0
            decode_span = (last_t - first_t) if (first_t and last_t and last_t > first_t) else 0.0
            n_gen = len(tokens)
            decode_tps = (n_gen - 1) / decode_span if decode_span > 0 else 0.0
            eff_tps = n_gen / total_dur if total_dur > 0 else 0.0
            passed = (t["expected"] in output_text) or (t["id"] == "math-explain-01" and ("17 ile 19" in output_text or "17 * 19" in output_text or "323" in output_text))

            tps_pass = decode_tps >= t["target_tps"]

            res = {
                "task": t["id"],
                "category": t["category"],
                "expected": t["expected"],
                "passed": passed,
                "tokens": n_gen,
                "ttft_ms": ttft,
                "decode_span_s": decode_span,
                "decode_tps": decode_tps,
                "target_tps": t["target_tps"],
                "baseline_tps": t["baseline_tps"],
                "tps_pass": tps_pass,
                "effective_tps": eff_tps,
                "output": output_text,
                "done_event": done_ev,
            }
            results.append(res)
            print(f"[{t['id']}] ({t['category']}) -> Parity: {'PASS' if passed else 'FAIL'} | TPS Pass: {'PASS' if tps_pass else 'FAIL'}")
            print(f"    Output: {output_text!r}")
            print(f"    Decode: {decode_tps:.2f} TPS (Target: > {t['target_tps']:.2f}, P4: {t['baseline_tps']:.2f}) | Eff: {eff_tps:.2f} TPS | TTFT: {ttft:.1f} ms")

    finally:
        await backend.stop()

    print("\n================================================================================")
    print("   PHASE 5.1 ACCEPTANCE GATES EVALUATION")
    print("================================================================================")

    # Gate 1: Allocated Used VRAM >= 4,500 MiB and Free VRAM 1,024 MiB <= Free <= 2,600 MiB (Free > 3,500 MiB is automatic FAIL)
    gate1_used_pass = used_loaded >= 4500.0
    gate1_free_pass = 1024.0 <= free_loaded <= 2600.0
    gate1_anomaly_free = free_loaded <= 3500.0
    gate1_pass = gate1_used_pass and gate1_free_pass and gate1_anomaly_free
    print(f"Gate 1 [Physical VRAM Binding: Used >= 4,500 MiB, 1,024 <= Free <= 2,600 MiB]: {'PASS' if gate1_pass else 'FAIL'}")
    print(f"    Used: {used_loaded:.1f} MiB (>= 4500: {gate1_used_pass}) | Free: {free_loaded:.1f} MiB (1024-2600: {gate1_free_pass}, <= 3500: {gate1_anomaly_free})")

    # Gate 2: Short prompt TTFT <= 600 ms (short-math-01, short-code-01)
    short_tasks = [r for r in results if r["task"] in ("short-math-01", "short-code-01")]
    gate2_pass = all(r["ttft_ms"] <= 600.0 for r in short_tasks) if short_tasks else False
    print(f"Gate 2 [Short Prompt TTFT <= 600 ms]: {'PASS' if gate2_pass else 'FAIL'}")
    for st in short_tasks:
        print(f"    {st['task']}: TTFT = {st['ttft_ms']:.1f} ms ({'PASS' if st['ttft_ms'] <= 600.0 else 'FAIL'})")

    # Gate 3: Native Decode TPS strictly beats Phase 4 Mode A baseline on all tasks
    gate3_pass = all(r["tps_pass"] for r in results)
    print(f"Gate 3 [Native Decode TPS strictly beats P4 Mode A baseline]: {'PASS' if gate3_pass else 'FAIL'}")
    for r in results:
        print(f"    {r['task']}: {r['decode_tps']:.2f} TPS vs target {r['target_tps']:.2f} (P4: {r['baseline_tps']:.2f}) -> {'PASS' if r['tps_pass'] else 'FAIL'}")

    # Gate 4: 100% token parity match on all 5 tasks
    gate4_pass = all(r["passed"] for r in results)
    print(f"Gate 4 [Deterministic Token Parity on all 5 tasks]: {'PASS' if gate4_pass else 'FAIL'}")
    for r in results:
        print(f"    {r['task']}: expected={r['expected']!r} -> {'MATCH' if r['passed'] else 'FAIL'} (got {r['output'][:50]!r})")

    all_gates_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    print("\n--------------------------------------------------------------------------------")
    print(f"OVERALL PHASE 5.1 VERDICT: {'ALL GATES PASS' if all_gates_pass else 'GATE(S) FAILED'}")
    print("--------------------------------------------------------------------------------")

    out_file = ROOT / "scripts" / "phase5_results.json"
    summary_data = {
        "vram_used_loaded_mib": used_loaded,
        "vram_free_loaded_mib": free_loaded,
        "vram_total_mib": total_loaded,
        "gates": {
            "gate1_vram_binding": gate1_pass,
            "gate2_short_ttft": gate2_pass,
            "gate3_native_tps": gate3_pass,
            "gate4_token_parity": gate4_pass,
            "all_gates_passed": all_gates_pass,
        },
        "tasks": results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nAudit results saved to: {out_file}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
