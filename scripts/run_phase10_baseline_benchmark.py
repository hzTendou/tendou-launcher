import os
import sys
import time
import json
import ctypes
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt, DEFAULT_MODEL_PATH, DEFAULT_MTP_PATH

BACKUP_DIR = ROOT / "backups" / "phase10_stable_20261020_123910"

WORKLOADS = [
    {
        "id": "repetitive-code-01",
        "category": "code_template",
        "prompt": "Repeat the pattern for the next item without explanation. Write only Python code:\nitems = ['a']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['b']\nfor x in items:\n    print('Element')\n    print('Done')\n\nitems = ['c']\nfor x in items:\n",
        "n_predict": 32,
    },
    {
        "id": "creative-chat-01",
        "category": "free_text",
        "prompt": "Gunes Sistemi'ndeki sekiz gezegeni gunese olan uzakliklarina gore sirala ve her bir gezegen hakkinda ayrintili bir paragraf yaz.",
        "n_predict": 32,
    },
    {
        "id": "math-explain-az",
        "category": "az_math_physics",
        "prompt": "Nyutonun ikinci qanununu (F = m * a) etrafli izah et. Kutlesi 5 kq olan cismin 4 m/s^2 tecille hereket etmesi ucun lazim olan quvveni addim-addim hesabla ve izahini yaz.",
        "n_predict": 32,
    },
]

def get_cuda_vram_info():
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
    except Exception:
        return 0.0, 0.0, 0.0

async def benchmark_config(config_code: str, mtp_mode: str, mtp_draft_n: int = 2) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  BENCHMARK CONFIG {config_code}: MTP={mtp_mode.upper()} (draft_n={mtp_draft_n})")
    print(f"=======================================================")
    
    free_init, total_init, used_init = get_cuda_vram_info()
    print(f"Initial VRAM: Used = {used_init:.1f} MiB / Free = {free_init:.1f} MiB")

    backend = AtlasSubprocessBackend(
        model_path=DEFAULT_MODEL_PATH,
        threads=8,
        boost=True,
        boost_level=2,
        gpu_layers=2 if mtp_mode == "off" else 0, # In full VRAM MTP, target gpu_layers=0 frees VRAM for MTP head
        mtp=mtp_mode,
        mtp_draft_n=mtp_draft_n,
        mtp_path=DEFAULT_MTP_PATH if mtp_mode != "off" else None,
        prompt_cache_mb=0,
    )
    
    await backend.start()
    await asyncio.sleep(1.0)
    free_res, total_res, used_res = get_cuda_vram_info()
    vram_increase_mib = max(0.0, used_res - used_init)
    print(f"Resident VRAM: Used = {used_res:.1f} MiB / Free = {free_res:.1f} MiB (Delta = {vram_increase_mib:.1f} MiB)")

    workload_results = []

    for wl in WORKLOADS:
        wl_id = wl["id"]
        p_text = wl["prompt"]
        n_pred = wl["n_predict"]
        formatted_prompt = format_chatml_prompt([ChatMessage(role="user", content=p_text)])
        
        print(f"\n--- Workload: {wl_id} ({wl['category']}) [5 repetitions] ---")
        reps = []

        for r in range(5):
            t0 = time.perf_counter()
            first_t = None
            last_t = None
            toks = []
            done_ev = {}

            async for ev in backend.generate(prompt=formatted_prompt, n_predict=n_pred, temp=0.0):
                if ev.get("event") == "token":
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                    last_t = now
                    toks.append(ev.get("token", ""))
                elif ev.get("event") == "done":
                    done_ev = ev

            t_end = time.perf_counter()
            total_dur_s = t_end - t0
            n_gen = len(toks)
            ttft_ms = (first_t - t0) * 1000.0 if first_t else 0.0
            decode_span_s = (last_t - first_t) if (first_t and last_t and last_t > first_t) else 0.0
            decode_tps = (n_gen - 1) / decode_span_s if decode_span_s > 0 else 0.0
            eff_tps = n_gen / total_dur_s if total_dur_s > 0 else 0.0

            expert_page_bytes = done_ev.get("expert_page_bytes", 0)
            kb_per_tok = (expert_page_bytes / 1024.0) / n_gen if n_gen > 0 else 0.0

            spec_draft = done_ev.get("spec_draft_tokens", 0)
            spec_accept = done_ev.get("spec_accepted_tokens", 0)
            acc_rate = (spec_accept / spec_draft) if spec_draft > 0 else 0.0
            t_draft_ms = (done_ev.get("spec_draft_time_ms", 0.0) / spec_draft) if spec_draft > 0 else 0.0
            t_verify_ms = done_ev.get("spec_verify_time_ms", 0.0)

            rep_data = {
                "rep": r + 1,
                "n_tokens": n_gen,
                "ttft_ms": round(ttft_ms, 2),
                "decode_tps": round(decode_tps, 2),
                "effective_tps": round(eff_tps, 2),
                "total_dur_s": round(total_dur_s, 4),
                "kb_per_tok": round(kb_per_tok, 2),
                "spec_draft_tokens": spec_draft,
                "spec_accepted_tokens": spec_accept,
                "acceptance_rate": round(acc_rate, 4),
                "t_draft_step_ms": round(t_draft_ms, 3),
                "t_cpu_moe_ms": round(decode_span_s * 1000.0 / max(1, n_gen - 1), 2),
                "output_text": "".join(toks).strip(),
            }
            reps.append(rep_data)
            print(f"  Rep {r+1}: Toks={n_gen} | TTFT={ttft_ms:.1f}ms | DecTPS={decode_tps:.2f} | EffTPS={eff_tps:.2f} | Draft={spec_draft} Acc={spec_accept} ({acc_rate*100:.1f}%)")

        # Deterministic parity check across all 5 reps
        first_out = reps[0]["output_text"]
        matches = sum(1 for rep in reps if rep["output_text"] == first_out)
        det_parity = f"{matches}/5"

        eff_tps_list = [rp["effective_tps"] for rp in reps]
        dec_tps_list = [rp["decode_tps"] for rp in reps]
        ttft_list = [rp["ttft_ms"] for rp in reps]
        acc_list = [rp["acceptance_rate"] for rp in reps if rp["spec_draft_tokens"] > 0]
        draft_time_list = [rp["t_draft_step_ms"] for rp in reps if rp["spec_draft_tokens"] > 0]
        moe_time_list = [rp["t_cpu_moe_ms"] for rp in reps]
        kb_tok_list = [rp["kb_per_tok"] for rp in reps]

        eff_tps_sorted = sorted(eff_tps_list)
        median_eff_tps = eff_tps_sorted[len(eff_tps_sorted) // 2]
        
        workload_results.append({
            "workload_id": wl_id,
            "category": wl["category"],
            "deterministic_parity": det_parity,
            "parity_pass": (matches == 5),
            "cold_ttft_ms": reps[0]["ttft_ms"],
            "warm_ttft_median_ms": round(sorted(ttft_list[1:])[len(ttft_list[1:])//2], 2) if len(ttft_list) > 1 else reps[0]["ttft_ms"],
            "effective_tps_median": median_eff_tps,
            "effective_tps_min": min(eff_tps_list),
            "effective_tps_max": max(eff_tps_list),
            "decode_tps_median": sorted(dec_tps_list)[len(dec_tps_list)//2],
            "acceptance_rate_median": round(sorted(acc_list)[len(acc_list)//2], 4) if acc_list else 0.0,
            "t_draft_step_median_ms": round(sorted(draft_time_list)[len(draft_time_list)//2], 3) if draft_time_list else 0.0,
            "t_cpu_moe_median_ms": round(sorted(moe_time_list)[len(moe_time_list)//2], 2),
            "kb_per_tok_median": round(sorted(kb_tok_list)[len(kb_tok_list)//2], 2),
            "reps": reps,
        })

    await backend.stop()

    return {
        "config_code": config_code,
        "mtp_mode": mtp_mode,
        "mtp_draft_n": mtp_draft_n,
        "vram_used_mib": used_res,
        "vram_free_mib": free_res,
        "vram_delta_mib": vram_increase_mib,
        "workloads": workload_results,
    }

async def run_baseline_benchmark():
    print("Running Native Tests for Phase 10 Baseline...")
    res_native = subprocess.run([str(ROOT / "llama.cpp" / "build" / "bin" / "Release" / "atlas-native-tests.exe")],
                                capture_output=True, text=True)
    native_pass = (res_native.returncode == 0) and ("PASS" in res_native.stdout)
    print(f"Native Tests: {'PASS' if native_pass else 'FAIL'}")

    # Config A: Phase 10 MTP off
    config_a = await benchmark_config(config_code="A", mtp_mode="off")

    # Config B: Phase 10 MTP on (vram)
    config_b = await benchmark_config(config_code="B", mtp_mode="vram", mtp_draft_n=2)

    baseline_data = {
        "timestamp": "20261020_123910",
        "phase": "phase10",
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 5060 Laptop (8151 MiB)",
            "cpu": "AMD Ryzen 7 260 (8C/16T, DDR5-5600, AVX-512 VNNI/VBMI)",
            "ram": "16 GB (OS: 15.29 GiB)",
            "threads": 8,
            "model": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS",
            "context_length": 8192,
        },
        "native_tests_pass": native_pass,
        "config_a_phase10_mtp_off": config_a,
        "config_b_phase10_mtp_on": config_b,
    }

    out_file = BACKUP_DIR / "baseline_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"\nPhase 10 Baseline metrics saved to: {out_file}")

if __name__ == "__main__":
    asyncio.run(run_baseline_benchmark())
