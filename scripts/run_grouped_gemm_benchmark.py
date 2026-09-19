import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXE_PATH = ROOT_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-atlas-engine.exe"
MODEL_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\snapshots\06756566a4b4a29d0dee62ccb405914a15fdf80d\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"
MTP_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\38bb39ee97821de2c9009abb7e93950eec396e66\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"

PROMPTS = [
    ("physics", "Newton's first law states that"),
    ("coding", "Write a Python function to find the longest palindromic substring:"),
    ("math", "Solve for x: 3x^2 - 12x + 9 = 0. Show the steps."),
]

N_PREDICT = 32
REPETITIONS = 3

def parse_run_output(output_text):
    data = {
        "tps": 0.0,
        "tokens_generated": 0,
        "generation_time_ms": 0.0,
        "quality_score": None,
        "quality_status": "NOT_EVALUATED",
        "repetition_score": 100.0,
        "entropy_score": 100.0,
        "vram_used_mb": 0.0,
        "ram_used_mb": 0.0,
        "acceptance_rate": 0.0,
        "expert_reuse": 1.0,
        "active_k": 10,
        "grouped_gemm": 1,
        "cuda_streams": 1,
        "async_h2d": "DISABLED",
        "output_text": ""
    }
    
    m = re.search(r"Generate:\s+\d+\s+tokens\s+/\s+[\d\.]+\s+s\s+=\s+([\d\.]+)\s+TPS", output_text)
    if m:
        data["tps"] = float(m.group(1))
        
    m = re.search(r"Generate:\s+(\d+)\s+tokens\s+/\s+([\d\.]+)\s+s", output_text)
    if m:
        data["tokens_generated"] = int(m.group(1))
        data["generation_time_ms"] = float(m.group(2)) * 1000.0
        
    m = re.search(r"Estimated Quality:\s+([\d\.]+)%", output_text)
    if m:
        data["token_diversity_heuristic"] = float(m.group(1))
        
    m = re.search(r"Repetition Score:\s+([\d\.]+)%", output_text)
    if m:
        data["repetition_score"] = float(m.group(1))
        
    m = re.search(r"Entropy Score:\s+([\d\.]+)%", output_text)
    if m:
        data["entropy_score"] = float(m.group(1))
        
    m = re.search(r"VRAM Cache:\s+([\d\.]+)\s+/\s+([\d\.]+)\s+MB", output_text)
    if m:
        data["vram_used_mb"] = float(m.group(1))
        
    m = re.search(r"RAM  Cache:\s+([\d\.]+)\s+/\s+([\d\.]+)\s+MB", output_text)
    if m:
        data["ram_used_mb"] = float(m.group(1))
        
    m = re.search(r"Acceptance Rate:\s+\d+\s+/\s+\d+\s+\(([\d\.]+)%\)", output_text)
    if m:
        data["acceptance_rate"] = float(m.group(1))
        
    m = re.search(r"Expert Reuse Factor:\s+([\d\.]+)x", output_text)
    if m:
        data["expert_reuse"] = float(m.group(1))
        
    m = re.search(r"Active K=(\d+)", output_text)
    if m:
        data["active_k"] = int(m.group(1))
        
    m = re.search(r"Grouped-GEMM Pipeline:\s+(\d+)\s+concurrent groups\s+\|\s+(\d+)\s+CUDA streams\s+\|\s+Async H2D=(\w+)", output_text)
    if m:
        data["grouped_gemm"] = int(m.group(1))
        data["cuda_streams"] = int(m.group(2))
        data["async_h2d"] = m.group(3)
        
    m = re.search(r"\[RESPONSE\]:\s*(.*?)(?=\n\n\[ATLAS\]|$)", output_text, re.DOTALL)
    if m:
        data["output_text"] = m.group(1).strip()
        
    return data

def run_config_benchmark(config_name, cmd_args, prompt, out_dir, reps=3):
    print(f"\n=================================================================")
    print(f"BENCHMARKING: {config_name}")
    print(f"Prompt: \"{prompt[:40]}...\" | Reps: {reps}")
    print(f"Args: {' '.join(cmd_args)}")
    print(f"=================================================================")
    
    runs = []
    for r in range(1, reps + 1):
        full_cmd = [
            str(EXE_PATH),
            "-m", MODEL_PATH,
            "-p", prompt,
            "-n", str(N_PREDICT),
            "--temp", "0",
            "--seed", "42",
            "-t", "16"
        ] + cmd_args
        
        t0 = time.time()
        proc = subprocess.run(full_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.time() - t0
        
        parsed = parse_run_output(proc.stdout)
        parsed["repetition"] = r
        parsed["wall_time_s"] = round(elapsed, 2)
        runs.append(parsed)
        
        log_file = out_dir / f"{config_name}_rep{r}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n--- STDERR ---\n" + proc.stderr)
                
        print(f"  Rep {r}: TPS={parsed['tps']} | Qual={parsed['quality_score']}% | Time={elapsed:.2f}s | K={parsed['active_k']}")
        
    tps_list = [run["tps"] for run in runs if run["tps"] > 0]
    if not tps_list:
        tps_list = [0.0]
        
    summary = dict(runs[0])
    summary["config_name"] = config_name
    summary["repetition_count"] = reps
    summary["all_tps"] = tps_list
    summary["mean_tps"] = round(statistics.mean(tps_list), 2)
    summary["median_tps"] = round(statistics.median(tps_list), 2)
    summary["best_tps"] = round(max(tps_list), 2)
    summary["worst_tps"] = round(min(tps_list), 2)
    summary["std_tps"] = round(statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0, 3)
    
    print(f">> STATS: Mean={summary['mean_tps']} | Median={summary['median_tps']} | Best={summary['best_tps']} | Worst={summary['worst_tps']} | Std={summary['std_tps']} | Qual={summary['quality_score']}%")
    return summary

def main():
    out_dir = ROOT_DIR / "experiments" / "grouped_gemm_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    matrix = [
        # Stream and Grouped-GEMM scaling on K=3
        ("1_grouped1_stream1_k3", ["--atlas-mtp", "off", "--atlas-k", "3", "--atlas-grouped-gemm", "1", "--atlas-cuda-streams", "1", "--atlas-async-h2d", "0"]),
        ("2_grouped2_streams2_k3", ["--atlas-mtp", "off", "--atlas-k", "3", "--atlas-grouped-gemm", "2", "--atlas-cuda-streams", "2", "--atlas-async-h2d", "0"]),
        ("3_grouped3_streams3_k3", ["--atlas-mtp", "off", "--atlas-k", "3", "--atlas-grouped-gemm", "3", "--atlas-cuda-streams", "3", "--atlas-async-h2d", "0"]),
        ("4_grouped3_streams3_async_h2d_k3", ["--atlas-mtp", "off", "--atlas-k", "3", "--atlas-grouped-gemm", "3", "--atlas-cuda-streams", "3", "--atlas-async-h2d", "1"]),
        
        # Multi-Token Speculative MTP with Grouped-GEMM & Streams
        ("5_grouped_mtp_n2_streams2_k3", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "vram", "--atlas-mtp-draft-n", "2", "--atlas-mtp-p-min", "0.60", "--atlas-k", "3", "--atlas-grouped-gemm", "2", "--atlas-cuda-streams", "2", "--atlas-async-h2d", "1", "--atlas-mtp-path", MTP_PATH]),
        ("6_grouped_mtp_n3_streams3_k3", ["--atlas-mtp", "atlas", "--atlas-mtp-loc", "vram", "--atlas-mtp-draft-n", "3", "--atlas-mtp-p-min", "0.60", "--atlas-k", "3", "--atlas-grouped-gemm", "3", "--atlas-cuda-streams", "3", "--atlas-async-h2d", "1", "--atlas-mtp-path", MTP_PATH]),
        
        # Ceiling K=2 with Grouped-GEMM & Streams
        ("7_grouped2_streams2_k2_ceiling", ["--atlas-mtp", "off", "--atlas-k", "2", "--atlas-grouped-gemm", "2", "--atlas-cuda-streams", "2", "--atlas-async-h2d", "1"]),
        ("8_grouped2_streams2_gpu1_k2", ["--atlas-mtp", "off", "--atlas-k", "2", "--atlas-gpu-layers", "1", "--atlas-grouped-gemm", "2", "--atlas-cuda-streams", "2", "--atlas-async-h2d", "1"]),
    ]
    
    print("Performing warmup run...")
    subprocess.run([str(EXE_PATH), "-m", MODEL_PATH, "-p", "Warmup", "-n", "4", "--atlas-mtp", "off", "-t", "16"],
                   capture_output=True, text=True)
                   
    all_results = []
    for name, args in matrix:
        res = run_config_benchmark(name, args, PROMPTS[0][1], out_dir, reps=REPETITIONS)
        all_results.append(res)
        
    summary_file = out_dir / "grouped_gemm_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n\n=================================================================")
    print("           GROUPED-GEMM & STREAM OVERLAP MATRIX COMPLETE         ")
    print("=================================================================")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
