import os
import sys
import subprocess
import json
import time
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXE_PATH = ROOT_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-atlas-engine.exe"
MODEL_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"
MTP_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\38bb39ee97821de2c9009abb7e93950eec396e66\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"

PROMPT = "Newton's first law states that"
N_PREDICT = 128

def parse_benchmark_output(output_text):
    data = {
        "tps": 0.0,
        "tokens_generated": 0,
        "generation_time_ms": 0.0,
        "startup_time_ms": 0.0,
        "vram_hit_rate": 0.0,
        "ram_hit_rate": 0.0,
        "total_hit_rate": 0.0,
        "vram_hits": 0,
        "ram_hits": 0,
        "nvme_misses": 0,
        "acceptance_rate": 0.0,
        "spec_draft_tokens": 0,
        "spec_accepted_tokens": 0,
        "spec_draft_time_ms": 0.0,
        "spec_verify_time_ms": 0.0,
        "expert_reuse_factor": 1.0,
        "prediction_precision": 0.0,
        "prediction_recall": 0.0,
        "prefetch_hits": 0,
        "prefetch_stalls": 0,
        "expert_compute_share": 0.0,
        "output_preview": ""
    }

    # Extract TPS
    m = re.search(r"Generate:\s+(\d+)\s+tokens\s+/\s+[\d\.]+\s+s\s+=\s+([\d\.]+)\s+TPS", output_text)
    if m:
        data["tokens_generated"] = int(m.group(1))
        data["tps"] = float(m.group(2))

    # Extract Startup
    m = re.search(r"Startup:\s+([\d\.]+)\s+ms", output_text)
    if m:
        data["startup_time_ms"] = float(m.group(1))

    # Extract Hit Rates
    m = re.search(r"Hit Rate:\s+VRAM=([\d\.]+)%\s+RAM=([\d\.]+)%\s+Total=([\d\.]+)%", output_text)
    if m:
        data["vram_hit_rate"] = float(m.group(1))
        data["ram_hit_rate"] = float(m.group(2))
        data["total_hit_rate"] = float(m.group(3))

    m = re.search(r"VRAM Cache:.*?Hits:\s+(\d+)", output_text)
    if m:
        data["vram_hits"] = int(m.group(1))

    m = re.search(r"RAM\s+Cache:.*?Hits:\s+(\d+)", output_text)
    if m:
        data["ram_hits"] = int(m.group(1))

    m = re.search(r"NVMe Misses:\s+(\d+)", output_text)
    if m:
        data["nvme_misses"] = int(m.group(1))

    # Extract MTP stats
    m = re.search(r"Acceptance Rate:\s+(\d+)\s+/\s+(\d+)\s+\(([\d\.]+)%\)", output_text)
    if m:
        data["spec_accepted_tokens"] = int(m.group(1))
        data["spec_draft_tokens"] = int(m.group(2))
        data["acceptance_rate"] = float(m.group(3))

    m = re.search(r"Draft Latency:\s+([\d\.]+)\s+ms", output_text)
    if m:
        data["spec_draft_time_ms"] = float(m.group(1))

    m = re.search(r"Verify Latency:\s+([\d\.]+)\s+ms", output_text)
    if m:
        data["spec_verify_time_ms"] = float(m.group(1))

    # Extract Reuse Factor
    m = re.search(r"Expert Reuse Factor:\s+([\d\.]+)x", output_text)
    if m:
        data["expert_reuse_factor"] = float(m.group(1))

    # Extract Precision and Recall
    m = re.search(r"Prediction Precision:\s+([\d\.]+)%", output_text)
    if m:
        data["prediction_precision"] = float(m.group(1))

    m = re.search(r"Prediction Recall:\s+([\d\.]+)%", output_text)
    if m:
        data["prediction_recall"] = float(m.group(1))

    # Extract Prefetch Hits and Stalls
    m = re.search(r"Prefetch Completed Before Demand \(Hits\):\s+(\d+)", output_text)
    if m:
        data["prefetch_hits"] = int(m.group(1))

    m = re.search(r"Prefetch Late / Demanded In-Flight \(Stalls\):\s+(\d+)", output_text)
    if m:
        data["prefetch_stalls"] = int(m.group(1))

    # Extract Expert Compute Share
    m = re.search(r"expert compute / graph execution.*?([\d\.]+)%", output_text)
    if m:
        data["expert_compute_share"] = float(m.group(1))

    # Extract Response preview
    m = re.search(r"\[RESPONSE\]:\s*(.*?)(?=\n\n\[ATLAS\]|$)", output_text, re.DOTALL)
    if m:
        data["output_preview"] = m.group(1).strip()[:100]

    return data

def run_single_benchmark(config_name, cmd_args, out_dir, repetitions=3):
    import statistics
    print(f"\n=================================================================")
    print(f"RUNNING CONFIGURATION: {config_name} ({repetitions} repetitions)")
    print(f"Args: {' '.join(cmd_args)}")
    print(f"=================================================================")

    runs = []
    for rep in range(1, repetitions + 1):
        print(f"--- Repetition {rep}/{repetitions} ---")
        start_t = time.time()
        full_cmd = [str(EXE_PATH), "-m", MODEL_PATH, "-p", PROMPT, "-n", str(N_PREDICT), "--ignore-eos", "--seed", "42"] + cmd_args
        proc = subprocess.run(full_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.time() - start_t

        log_file = out_dir / f"{config_name}_rep{rep}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(proc.stderr)

        parsed = parse_benchmark_output(proc.stdout)
        parsed["config_name"] = config_name
        parsed["repetition"] = rep
        parsed["wall_clock_time_s"] = round(elapsed, 2)
        runs.append(parsed)
        print(f"  Rep {rep}: TPS={parsed['tps']} | Time={elapsed:.2f}s | Acc={parsed['acceptance_rate']}% | Reuse={parsed['expert_reuse_factor']}x")

    tps_vals = [r["tps"] for r in runs if r["tps"] > 0]
    if not tps_vals:
        tps_vals = [0.0]

    summary = dict(runs[0])
    summary["repetition_count"] = repetitions
    summary["all_tps"] = tps_vals
    summary["mean_tps"] = round(statistics.mean(tps_vals), 2)
    summary["median_tps"] = round(statistics.median(tps_vals), 2)
    summary["best_tps"] = round(max(tps_vals), 2)
    summary["worst_tps"] = round(min(tps_vals), 2)
    summary["std_tps"] = round(statistics.stdev(tps_vals) if len(tps_vals) > 1 else 0.0, 3)
    summary["tps"] = summary["mean_tps"]

    print(f">> STATS for {config_name}: Mean={summary['mean_tps']} | Median={summary['median_tps']} | Best={summary['best_tps']} | Worst={summary['worst_tps']} | Std={summary['std_tps']}")
    return summary

def main():
    out_dir = ROOT_DIR / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define benchmark matrix per Section 21
    matrix = [
        # 1. Baseline MTP OFF
        ("baseline_mtp_off", ["--atlas-mtp", "off", "-t", "16"]),

        # 2-4. MTP RAM: N=2, N=4, N=6
        ("mtp_ram_n2", ["--atlas-mtp", "ram", "--atlas-mtp-draft-n", "2", "--atlas-mtws", "0", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
        ("mtp_ram_n4", ["--atlas-mtp", "ram", "--atlas-mtp-draft-n", "4", "--atlas-mtws", "0", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
        ("mtp_ram_n6", ["--atlas-mtp", "ram", "--atlas-mtp-draft-n", "6", "--atlas-mtws", "0", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),

        # 5-7. MTP VRAM: N=2, N=4, N=6
        ("mtp_vram_n2", ["--atlas-mtp", "vram", "--atlas-mtp-draft-n", "2", "--atlas-mtws", "0", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
        ("mtp_vram_n4", ["--atlas-mtp", "vram", "--atlas-mtp-draft-n", "4", "--atlas-mtws", "0", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
        ("mtp_vram_n6", ["--atlas-mtp", "vram", "--atlas-mtp-draft-n", "6", "--atlas-mtws", "0", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),

        # 8-9. Atlas MTWS RAM: N=2, N=4
        ("atlas_mtws_ram_n2", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "ram", "--atlas-mtp-draft-n", "2", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
        ("atlas_mtws_ram_n4", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "ram", "--atlas-mtp-draft-n", "4", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),

        # 10-12. Atlas MTWS VRAM: N=2, N=4, N=6
        ("atlas_mtws_vram_n2", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "vram", "--atlas-mtp-draft-n", "2", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
        ("atlas_mtws_vram_n4", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "vram", "--atlas-mtp-draft-n", "4", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
        ("atlas_mtws_vram_n6", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "vram", "--atlas-mtp-draft-n", "6", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),

        # 13. Atlas AUTO (adaptive placement & draft horizon)
        ("atlas_mtws_auto_adaptive", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "auto", "--atlas-adaptive", "1", "--atlas-mtp-draft-n", "3", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),

        # 14. Co-Designed Break-Ceiling Configuration: Atlas MTWS VRAM N=2 Confidence-Gated
        ("atlas_mtws_vram_n2_gated", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "vram", "--atlas-mtp-draft-n", "2", "--atlas-mtp-p-min", "0.60", "--atlas-mtp-path", MTP_PATH, "-t", "16"]),
    ]
    
    # Warmup run to warm filesystem cache
    print("Executing 8-token warmup to prime OS page cache...")
    subprocess.run([str(EXE_PATH), "-m", MODEL_PATH, "-p", "Warmup", "-n", "8", "--atlas-mtp", "off"],
                   capture_output=True, text=True)
    
    results = []
    for name, args in matrix:
        res = run_single_benchmark(name, args, out_dir)
        results.append(res)
        
    summary_path = out_dir / "benchmark_matrix_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n=================================================================")
    print(f"BENCHMARK COMPLETE. Summary saved to {summary_path}")
    print("=================================================================")

if __name__ == "__main__":
    main()
