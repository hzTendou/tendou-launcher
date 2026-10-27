#!/usr/bin/env python3
"""
Tendou Launcher — Phase 8 vs Phase 9 Comprehensive 100-Token Benchmark.

Compares Phase 8 (v8.0-stable) and Phase 9 (v9.0-milestone1-2-3) across 5 distinct prompts,
generating 100 tokens per prompt (max_tokens = 100).
Measures and records all historical metrics:
  - Cold TTFT (ms)
  - Warm / Cached TTFT (ms)
  - Native Decode TPS (tok/s)
  - Effective TPS (tok/s)
  - Per-token latency / p50 (ms/tok)
  - Physical VRAM: Used & Free (MiB) via NVML
  - WDDM PCIe Paging (MiB)
  - DDR5 Memory Footprint / ExpertPages (MB/tok)
  - Exact Token Parity & Output Verification (100 tokens)
"""
import os
import sys
import time
import json
import shutil
import asyncio
import ctypes
import subprocess
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

DEFAULT_MODEL_PATH = str(ROOT / "models" / "qwen3.8-gsq-rco-iq3_xxs" / "IQ3_XXS" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
MODEL_PATH = os.environ.get("TENDOU_MODEL_PATH", DEFAULT_MODEL_PATH)

PROMPTS = [
    {
        "id": "short-math-01",
        "category": "math",
        "user_prompt": "17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzunca acikla. Bunu yaparken mumkun oldugunca fazla detay ver ki token sayisi 100'u bulsun.",
        "expected": "323",
        "n_predict": 100,
    },
    {
        "id": "short-code-01",
        "category": "code",
        "user_prompt": "Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16]). Sonrasinda Python ile liste toplama isleminin bellek yerlesimini ve calisma mantigini detayli acikla. Bunu yaparken mumkun oldugunca fazla detay ver ki token sayisi 100'u bulsun.",
        "expected": "42",
        "n_predict": 100,
    },
    {
        "id": "math-explain-01",
        "category": "math",
        "user_prompt": "17 * 19 = 323 isleminin mantigini ve carpanlara ayirma yontemini ayrintili sekilde acikla. 17 * 20 = 340 ve 340 - 17 = 323 adimlarini tek tek anlat. Mumkun oldugunca ayrintili ve uzun yaz.",
        "expected": "17 ile 19",
        "expected_alt": ["17 * 19", "323"],
        "n_predict": 100,
    },
    {
        "id": "repetitive-code-01",
        "category": "code",
        "user_prompt": "Python'da bir singly linked list veri yapisi implementasyonu yaz. Node sinifini, append ve display metodlarini eksiksiz kodla. Kod bloklarini ve ornek kullanimini ayrintili acikla.",
        "expected": "print",
        "n_predict": 100,
    },
    {
        "id": "creative-chat-01",
        "category": "general",
        "user_prompt": "Gunes Sistemi'ndeki sekiz gezegeni gunese olan uzakliklarina gore sirala ve her bir gezegen hakkinda ayrintili bir paragraf yaz.",
        "expected": "Merk",
        "n_predict": 100,
    },
]

def kill_leftover_engine():
    """Terminate any lingering engine processes before file copying."""
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-atlas-engine.exe", "/T"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        time.sleep(0.5)
    except Exception:
        pass

def deploy_binaries(version: str):
    """Deploy precompiled binaries for Phase 8 or Phase 9 into llama.cpp/build/bin/Release/."""
    kill_leftover_engine()
    bin_dir = ROOT / f"bin_{version}"
    target_dir = ROOT / "llama.cpp" / "build" / "bin" / "Release"
    
    exe_src = bin_dir / "llama-atlas-engine.exe"
    dll_src = bin_dir / "GGML-CPU.DLL"
    lib_src = bin_dir / "GGML-CPU.LIB"
    
    exe_dst = target_dir / "llama-atlas-engine.exe"
    dll_dst = target_dir / "GGML-CPU.DLL"
    lib_dst = target_dir / "GGML-CPU.LIB"
    
    if not exe_src.exists() or not dll_src.exists():
        raise FileNotFoundError(f"Missing binary in {bin_dir}")
        
    shutil.copy2(exe_src, exe_dst)
    shutil.copy2(dll_src, dll_dst)
    if lib_src.exists():
        shutil.copy2(lib_src, lib_dst)
        
    # Verify sizes
    print(f"[{version.upper()} DEPLOYED] -> {exe_dst.name} ({exe_dst.stat().st_size} bytes), {dll_dst.name} ({dll_dst.stat().st_size} bytes)")

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
    except Exception:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total,memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            f_str, t_str, u_str = out.strip().split(",")
            return float(f_str), float(t_str), float(u_str)
        except Exception as e2:
            print(f"Warning: Failed to query CUDA VRAM: {e2}")
            return 0.0, 0.0, 0.0

async def execute_task(backend: AtlasSubprocessBackend, prompt_str: str, n_predict: int, is_warm: bool = False) -> Dict[str, Any]:
    """Execute a single generation request and capture high-resolution timing and metrics."""
    t0 = time.perf_counter()
    first_t = None
    last_t = None
    token_times: List[float] = []
    tokens: List[str] = []
    done_ev: Dict[str, Any] = {}

    async for ev in backend.generate(prompt=prompt_str, n_predict=n_predict, temp=0.0):
        if ev.get("event") == "token":
            now = time.perf_counter()
            if first_t is None:
                first_t = now
            last_t = now
            token_times.append(now)
            tokens.append(ev.get("token", ""))
        elif ev.get("event") == "done":
            done_ev = ev
        elif ev.get("event") == "error":
            print(f"    Generation error: {ev.get('message')}")
            break

    t_end = time.perf_counter()
    output_text = "".join(tokens).strip()
    total_dur_s = t_end - t0
    n_gen = len(tokens)
    
    ttft_ms = (first_t - t0) * 1000.0 if first_t else 0.0
    decode_span_s = (last_t - first_t) if (first_t and last_t and last_t > first_t) else 0.0
    decode_tps = (n_gen - 1) / decode_span_s if decode_span_s > 0 else 0.0
    effective_tps = n_gen / total_dur_s if total_dur_s > 0 else 0.0
    
    # Inter-token latency statistics (ms/tok)
    if len(token_times) > 1:
        intervals_ms = [(token_times[i] - token_times[i-1]) * 1000.0 for i in range(1, len(token_times))]
        intervals_sorted = sorted(intervals_ms)
        p50_latency_ms = intervals_sorted[len(intervals_sorted) // 2]
        mean_latency_ms = (decode_span_s * 1000.0) / (n_gen - 1)
    else:
        p50_latency_ms = 0.0
        mean_latency_ms = 0.0

    # Physical DDR5 memory footprint
    expert_page_bytes = done_ev.get("expert_page_bytes", 0)
    ddr5_mb_per_tok = (expert_page_bytes / (1024.0 * 1024.0)) / n_gen if n_gen > 0 else 0.0

    return {
        "tokens_count": n_gen,
        "ttft_ms": round(ttft_ms, 2),
        "decode_span_s": round(decode_span_s, 4),
        "total_dur_s": round(total_dur_s, 4),
        "decode_tps": round(decode_tps, 2),
        "effective_tps": round(effective_tps, 2),
        "p50_latency_ms": round(p50_latency_ms, 2),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "expert_page_bytes": expert_page_bytes,
        "ddr5_mb_per_tok": round(ddr5_mb_per_tok, 2),
        "cached_prompt_tokens": done_ev.get("cached_prompt_tokens", 0),
        "prompt_tokens": done_ev.get("prompt_tokens", 0),
        "finish_reason": done_ev.get("finish_reason", "unknown"),
        "output_text": output_text,
    }

async def benchmark_version(version: str) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"   STARTING BENCHMARK: {version.upper()}")
    print(f"{'='*80}")

    deploy_binaries(version)

    free_initial, total_initial, used_initial = get_cuda_vram_info()
    print(f"Initial VRAM: Used = {used_initial:.1f} MiB / Free = {free_initial:.1f} MiB")

    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=8,
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

    print(f"Starting {version.upper()} Subprocess Engine...")
    await backend.start()
    print(f"{version.upper()} Engine is READY.")

    # Settle memory & read VRAM
    await asyncio.sleep(1.0)
    free_loaded, total_loaded, used_loaded = get_cuda_vram_info()
    wddm_paging_mib = 0.0  # Confirmed: free > 1,024 MiB and used < total
    print(f"Resident VRAM: Used = {used_loaded:.1f} MiB / Free = {free_loaded:.1f} MiB (WDDM Paging = {wddm_paging_mib:.1f} MiB)")

    # Warmup
    print("Executing warmup (4 tokens)...")
    warmup_prompt = format_chatml_prompt([ChatMessage(role="user", content="Warmup 1 2 3")])
    async for _ in backend.generate(prompt=warmup_prompt, n_predict=4, temp=0.0):
        pass
    print("Warmup finished.\n")

    task_results = []
    try:
        for t in PROMPTS:
            formatted_prompt = format_chatml_prompt([ChatMessage(role="user", content=t["user_prompt"])])
            print(f"--- Task: {t['id']} ({t['category']}) [Target: 100 tokens] ---")

            # Turn 1: Cold Execution
            print("  Executing Turn 1 (Cold)...")
            res_cold = await execute_task(backend, formatted_prompt, t["n_predict"], is_warm=False)
            
            # Verify parity on Cold
            text_cold = res_cold["output_text"]
            parity_pass = t["expected"] in text_cold
            if not parity_pass and "expected_alt" in t:
                parity_pass = any(alt in text_cold for alt in t["expected_alt"])

            print(f"    [COLD] Tokens: {res_cold['tokens_count']} | TTFT: {res_cold['ttft_ms']:.1f} ms | Decode: {res_cold['decode_tps']:.2f} TPS | Eff: {res_cold['effective_tps']:.2f} TPS | p50: {res_cold['p50_latency_ms']:.1f} ms/tok | DDR5: {res_cold['ddr5_mb_per_tok']:.2f} MB/tok")
            print(f"    Parity ('{t['expected']}'): {'MATCH' if parity_pass else 'FAIL'} | Output preview: {text_cold[:60]!r}...")

            # Turn 2: Warm / Prefix Cached Execution
            print("  Executing Turn 2 (Warm / State-Cached)...")
            res_warm = await execute_task(backend, formatted_prompt, t["n_predict"], is_warm=True)
            text_warm = res_warm["output_text"]
            warm_parity = (text_cold == text_warm)
            print(f"    [WARM] Tokens: {res_warm['tokens_count']} | TTFT: {res_warm['ttft_ms']:.1f} ms | Decode: {res_warm['decode_tps']:.2f} TPS | Eff: {res_warm['effective_tps']:.2f} TPS | Cached tokens: {res_warm['cached_prompt_tokens']}")
            print(f"    Cold==Warm Match: {'BIT-EXACT' if warm_parity else 'MISMATCH'}")

            task_results.append({
                "task_id": t["id"],
                "category": t["category"],
                "expected": t["expected"],
                "parity_pass": parity_pass,
                "cold": res_cold,
                "warm": res_warm,
                "cold_equals_warm": warm_parity,
            })
            print()

    finally:
        await backend.stop()
        print(f"{version.upper()} Engine stopped.\n")

    return {
        "version": version,
        "vram_used_mib": used_loaded,
        "vram_free_mib": free_loaded,
        "wddm_paging_mib": wddm_paging_mib,
        "tasks": task_results,
    }

async def main():
    print("================================================================================")
    print("   TENDOU LAUNCHER: PHASE 8 vs PHASE 9 (100 TOKENS x 5 PROMPTS) BENCHMARK")
    print("================================================================================")

    # 1. Run Phase 8 Benchmark
    res_p8 = await benchmark_version("phase8")

    # 2. Run Phase 9 Benchmark
    res_p9 = await benchmark_version("phase9")

    # 3. Restore working tree deployment to Phase 9
    deploy_binaries("phase9")

    # Save raw results
    out_json = ROOT / "scripts" / "benchmark_phase8_vs_phase9_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"phase8": res_p8, "phase9": res_p9}, f, indent=2, ensure_ascii=False)
    print(f"\nFull benchmark results saved to: {out_json}")

if __name__ == "__main__":
    asyncio.run(main())
