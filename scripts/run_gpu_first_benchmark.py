import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BASELINE_EXE = ROOT_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-atlas-engine-baseline-10tps.exe"
EXPERIMENTAL_EXE = ROOT_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-atlas-engine.exe"
MODEL_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"

PROMPTS = [
    ("coding", "Write a Python function to find the longest palindromic substring:"),
    ("physics", "Newton's first law states that"),
    ("math", "Solve for x: 3x^2 - 12x + 9 = 0. Show the steps."),
]

N_PREDICT = 48

def parse_engine_output(output_text):
    data = {
        "decode_tps": 0.0,
        "decode_ms_per_tok": 0.0,
        "prefill_tps": 0.0,
        "tokens_generated": 0,
        "vram_used_mb": 0.0,
        "vram_cap_mb": 8151.0,
        "vram_pct": 0.0,
        "resident_experts": 0,
        "resident_moe_mb": 0.0,
        "gpu_expert_computes": 0,
        "gpu_compute_pct": 0.0,
        "cpu_expert_computes": 0,
        "cpu_compute_pct": 0.0,
        "total_gpu_time_ms": 0.0,
        "total_cpu_time_ms": 0.0,
        "h2d_bw_gbps": 0.0,
        "d2h_bw_gbps": 0.0,
        "transfer_stall_ms": 0.0,
        "sync_stall_ms": 0.0,
        "gpu_util_pct": 0.0,
        "gpu_idle_pct": 0.0,
        "cpu_util_pct": 0.0,
        "cpu_idle_pct": 0.0,
        "vram_cache_hit_rate": 0.0,
        "expert_reuse_distance": 0.0,
        "expert_reuse_rate": 0.0,
        "avg_residency_duration": 0.0,
        "expert_eviction_rate": 0.0,
        "prefetch_success_rate": 0.0,
        "predictor_accuracy": 0.0,
        "quality_score": None,
        "quality_status": "NOT_EVALUATED",
        "response_sample": ""
    }

    # Decode TPS - Try GPU-first report first, then standard performance block
    m = re.search(r"Decode Performance:\s+([\d\.]+)\s+TPS\s+\(([\d\.]+)\s+ms/token\)", output_text)
    if m:
        data["decode_tps"] = float(m.group(1))
        data["decode_ms_per_tok"] = float(m.group(2))
    else:
        m2 = re.search(r"Generate:\s+(\d+)\s+tokens\s+/\s+([\d\.]+)\s+s\s+=\s+([\d\.]+)\s+TPS", output_text)
        if m2:
            data["tokens_generated"] = int(m2.group(1))
            sec = float(m2.group(2))
            data["decode_tps"] = float(m2.group(3))
            if data["tokens_generated"] > 0:
                data["decode_ms_per_tok"] = (sec / data["tokens_generated"]) * 1000.0

    # Prefill TPS
    m = re.search(r"Prefill Performance:\s+([\d\.]+)\s+TPS", output_text)
    if m:
        data["prefill_tps"] = float(m.group(1))
    else:
        m2 = re.search(r"Prompt:\s+\d+\s+tokens\s+/\s+[\d\.]+\s+s\s+=\s+([\d\.]+)\s+TPS", output_text)
        if m2:
            data["prefill_tps"] = float(m2.group(1))

    # Tokens generated
    m = re.search(r"Generate:\s+(\d+)\s+tokens", output_text)
    if m:
        data["tokens_generated"] = int(m.group(1))

    # VRAM Memory Budget / Utilization
    m = re.search(r"VRAM (?:Memory Budget|Utilization):\s+([\d\.]+)\s+MB\s+/\s+([\d\.]+)\s+MB\s+\(([\d\.]+)%\)", output_text)
    if m:
        data["vram_used_mb"] = float(m.group(1))
        data["vram_cap_mb"] = float(m.group(2))
        data["vram_pct"] = float(m.group(3))
    else:
        m2 = re.search(r"VRAM Cache:\s+([\d\.]+)\s+/\s+([\d\.]+)\s+MB", output_text)
        if m2:
            data["vram_used_mb"] = float(m2.group(1))
            data["vram_cap_mb"] = float(m2.group(2))
            if data["vram_cap_mb"] > 0:
                data["vram_pct"] = (data["vram_used_mb"] / data["vram_cap_mb"]) * 100.0

    # Dynamic Cache / Resident Experts
    m = re.search(r"VRAM (?:Dynamic Expert Cache|Resident Experts):\s+(\d+)\s+experts\s+\(~([\d\.]+)\s+MB", output_text)
    if m:
        data["resident_experts"] = int(m.group(1))
        data["resident_moe_mb"] = float(m.group(2))

    # MoE Workload Dispatches
    m = re.search(r"GPU Expert Computes:\s+(\d+)\s+\(([\d\.]+)%\s+of total MoE workload\)", output_text)
    if m:
        data["gpu_expert_computes"] = int(m.group(1))
        data["gpu_compute_pct"] = float(m.group(2))

    m = re.search(r"CPU Fallback Computes:\s+(\d+)\s+\(([\d\.]+)%\s+of total MoE workload\)", output_text)
    if m:
        data["cpu_expert_computes"] = int(m.group(1))
        data["cpu_compute_pct"] = float(m.group(2))

    # Latencies
    m = re.search(r"Total GPU Expert Time:\s+([\d\.]+)\s+ms", output_text)
    if m:
        data["total_gpu_time_ms"] = float(m.group(1))

    m = re.search(r"Total CPU Expert Time:\s+([\d\.]+)\s+ms", output_text)
    if m:
        data["total_cpu_time_ms"] = float(m.group(1))

    # Bandwidth and stalls
    m = re.search(r"PCIe H2D Bandwidth:\s+([\d\.]+)\s+GB/s", output_text)
    if m:
        data["h2d_bw_gbps"] = float(m.group(1))

    m = re.search(r"PCIe D2H Bandwidth:\s+([\d\.]+)\s+GB/s", output_text)
    if m:
        data["d2h_bw_gbps"] = float(m.group(1))

    m = re.search(r"Transfer Stall Time:\s+([\d\.]+)\s+ms", output_text)
    if m:
        data["transfer_stall_ms"] = float(m.group(1))

    m = re.search(r"CUDA Sync Stall Time:\s+([\d\.]+)\s+ms", output_text)
    if m:
        data["sync_stall_ms"] = float(m.group(1))

    # Utilizations
    m = re.search(r"GPU Utilization:\s+([\d\.]+)%\s+\(GPU Idle:\s+([\d\.]+)%\)", output_text)
    if m:
        data["gpu_util_pct"] = float(m.group(1))
        data["gpu_idle_pct"] = float(m.group(2))

    m = re.search(r"CPU Utilization:\s+([\d\.]+)%\s+\(CPU Idle:\s+([\d\.]+)%\)", output_text)
    if m:
        data["cpu_util_pct"] = float(m.group(1))
        data["cpu_idle_pct"] = float(m.group(2))

    # Cache hit rate
    m = re.search(r"VRAM Cache Hit Rate:\s+([\d\.]+)%", output_text)
    if m:
        data["vram_cache_hit_rate"] = float(m.group(1))

    m = re.search(r"Expert Reuse Distance:\s+([\d\.]+)\s+tokens\s+\(Reuse Rate:\s+([\d\.]+)%\)", output_text)
    if m:
        data["expert_reuse_distance"] = float(m.group(1))
        data["expert_reuse_rate"] = float(m.group(2))

    m = re.search(r"Average Residency Duration:\s+([\d\.]+)\s+tokens", output_text)
    if m:
        data["avg_residency_duration"] = float(m.group(1))

    m = re.search(r"Expert Eviction Rate:\s+([\d\.]+)\s+evictions/token", output_text)
    if m:
        data["expert_eviction_rate"] = float(m.group(1))

    m = re.search(r"Prefetch Success Rate:\s+([\d\.]+)%", output_text)
    if m:
        data["prefetch_success_rate"] = float(m.group(1))

    m = re.search(r"Predictor Online Accuracy:\s+([\d\.]+)%", output_text)
    if m:
        data["predictor_accuracy"] = float(m.group(1))

    # Quality Gate Assessment
    m = re.search(r"Quality Gate Assessment:\s+([\d\.]+)%\s+\((PASSED|FAILED)\)", output_text)
    if m:
        data["token_diversity_heuristic"] = float(m.group(1))
    else:
        m2 = re.search(r"Estimated Quality:\s+([\d\.]+)%", output_text)
        if m2:
            data["token_diversity_heuristic"] = float(m2.group(1))

    # Response sample
    m = re.search(r"\[RESPONSE\]:\s*(.*?)(?=\n\n\[ATLAS\]|$)", output_text, re.DOTALL)
    if m:
        data["response_sample"] = m.group(1).strip()[:160]

    return data

def run_single(exe_path, prompt, args, out_path):
    cmd = [
        str(exe_path),
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(N_PREDICT),
        "--temp", "0",
        "--seed", "42",
        "-t", "14"
    ] + args

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)

    parsed = parse_engine_output(proc.stdout)
    parsed["elapsed_sec"] = round(elapsed, 2)
    return parsed

def main():
    results_dir = ROOT_DIR / "experiments" / "gpu_first_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      ATLAS ENGINE: FROZEN BASELINE VS EXPERIMENTAL GPU-FIRST     ")
    print("=================================================================")
    print(f"Baseline:     {BASELINE_EXE.name}")
    print(f"Experimental: {EXPERIMENTAL_EXE.name}")
    print(f"Tokens/run:   {N_PREDICT}")
    print("=================================================================\n")

    summary = {
        "baseline": {},
        "experimental_gpu_first": {}
    }

    # 1. Benchmark Frozen Baseline across all 3 prompts
    print("\n>>> [1/2] RUNNING FROZEN BASELINE (Stride 1 + Adaptive-K=2 @ ~10 TPS) <<<")
    for domain, prompt in PROMPTS:
        log_file = results_dir / f"baseline_{domain}.log"
        print(f"Running Baseline: {domain.upper()} ('{prompt[:35]}...')...")
        res = run_single(BASELINE_EXE, prompt, ["--boost"], log_file)
        summary["baseline"][domain] = res
        print(f"  -> Decode: {res['decode_tps']:.2f} TPS | Quality: {res['quality_status']}")

    # 2. Benchmark Experimental Dynamic Expert-Level GPU-First across all 3 prompts
    print("\n>>> [2/2] RUNNING EXPERIMENTAL GPU-FIRST DYNAMIC ARCHITECTURE <<<")
    for domain, prompt in PROMPTS:
        log_file = results_dir / f"gpu_first_{domain}.log"
        print(f"Running GPU-First: {domain.upper()} ('{prompt[:35]}...')...")
        res = run_single(EXPERIMENTAL_EXE, prompt, ["--boost"], log_file)
        summary["experimental_gpu_first"][domain] = res
        print(f"  -> Decode: {res['decode_tps']:.2f} TPS | Quality: {res['quality_status']}")

    summary_file = results_dir / "gpu_first_vs_baseline_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n\n=================================================================")
    print("                    COMPARISON SUMMARY MATRIX                    ")
    print("=================================================================")
    header = f"{'Domain':<10} | {'Engine':<12} | {'Decode TPS':<10} | {'GPU MoE %':<10} | {'VRAM MB':<10} | {'Quality':<10} | {'Status':<6}"
    print(header)
    print("-" * len(header))
    for domain, _ in PROMPTS:
        b = summary["baseline"][domain]
        e = summary["experimental_gpu_first"][domain]
        print(f"{domain:<10} | {'Baseline':<12} | {b['decode_tps']:<10.2f} | Quality: {b['quality_status']}")
        print(f"{'':<10} | {'GPU-First':<12} | {e['decode_tps']:<10.2f} | Quality: {e['quality_status']}")
        print("-" * len(header))

    print(f"\nDetailed logs & JSON saved to: {summary_file}\n")

if __name__ == "__main__":
    main()
