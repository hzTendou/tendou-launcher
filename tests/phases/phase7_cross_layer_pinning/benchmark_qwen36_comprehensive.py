"""
tendou launcher — Comprehensive Long-Generation Benchmark for Qwen3.6-35B-A3B (UD-Q8_K_XL).

Uncapped/Liberal generation mode: lets the model think and reason naturally without
truncation, measuring sustained decode TPS over hundreds of tokens, prefill speed,
TTFT, memory usage, and reasoning quality on the high-quant 38.5 GB model.
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend

MODEL_PATH = r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
PHYSICAL_MAP_PATH = r"config/atlas_physical_map_35b.json"

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

def get_ram_usage_gb() -> float:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return round((stat.ullTotalPhys - stat.ullAvailPhys) / (1024**3), 3)

def get_vram_usage_mb() -> float:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0

async def run_single_inference(
    backend: AtlasSubprocessBackend,
    prompt: str,
    n_predict: int = 512,
    temp: float = 0.0,
    expected_regex: Optional[str] = None,
) -> Dict[str, Any]:
    ram_before = get_ram_usage_gb()
    vram_before = get_vram_usage_mb()
    
    t_start = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    token_pieces: List[str] = []
    token_ids: List[int] = []
    done_event: Dict[str, Any] = {}
    
    async for ev in backend.generate(prompt=prompt, n_predict=n_predict, temp=temp):
        ev_type = ev.get("event")
        now = time.perf_counter()
        if ev_type == "token":
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            token_pieces.append(ev.get("token", ""))
            if "token_id" in ev:
                token_ids.append(ev["token_id"])
        elif ev_type == "done":
            done_event = ev
        elif ev_type == "error":
            raise RuntimeError(f"Engine error: {ev.get('message')}")
            
    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000.0
    num_tokens = len(token_pieces)
    
    ttft_ms = ((first_token_time - t_start) * 1000.0) if first_token_time else done_event.get("ttft_ms", 0.0)
    decode_span_s = (last_token_time - first_token_time) if (last_token_time and first_token_time) else 0.0
    decode_tps = ((num_tokens - 1) / decode_span_s) if (num_tokens > 1 and decode_span_s > 0) else 0.0
    effective_tps = (num_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
    
    prompt_tokens = done_event.get("prompt_tokens", 0)
    cached_tokens = done_event.get("cached_prompt_tokens", 0)
    prefill_compute_ms = done_event.get("prefill_compute_ms", 0.0)
    new_prefill_tokens = prompt_tokens - cached_tokens
    prefill_tps = (new_prefill_tokens / (prefill_compute_ms / 1000.0)) if (prefill_compute_ms > 0 and new_prefill_tokens > 0) else 0.0

    raw_output = done_event.get("text") or "".join(token_pieces)
    text_clean = raw_output.strip()
    
    thinking_part = ""
    response_part = text_clean
    if "</think>" in text_clean:
        parts = text_clean.split("</think>", 1)
        thinking_part = parts[0].replace("<think>", "").strip()
        response_part = parts[1].strip()
    elif text_clean.startswith("<think>"):
        thinking_part = text_clean.replace("<think>", "").strip()
        response_part = ""
        
    is_correct = None
    if expected_regex:
        is_correct = bool(re.search(expected_regex, text_clean, re.IGNORECASE))
        
    ram_after = get_ram_usage_gb()
    vram_after = get_vram_usage_mb()
    
    return {
        "output_text": text_clean,
        "thinking_text": thinking_part,
        "response_text": response_part,
        "is_correct": is_correct,
        "tokens_generated": num_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
        "ttft_ms": round(ttft_ms, 2),
        "decode_tps": round(decode_tps, 2),
        "effective_tps": round(effective_tps, 2),
        "prefill_tps": round(prefill_tps, 2),
        "prefill_compute_ms": round(prefill_compute_ms, 2),
        "checkpoint_ms": round(done_event.get("checkpoint_ms", 0.0), 2),
        "total_elapsed_ms": round(elapsed_ms, 2),
        "ram_before_gb": ram_before,
        "ram_after_gb": ram_after,
        "vram_before_mb": vram_before,
        "vram_after_mb": vram_after,
        "finish_reason": done_event.get("finish_reason", "stop")
    }

async def benchmark_scenario(
    name: str,
    threads: int = 16,
    gpu_layers: int = 2,
    vram_cache: int = 3200,
    k_override: int = 0,
    ctx_size: int = 4096,
    prompt_cache_mb: int = 512,
    cuda_sched: str = "yield",
) -> List[Dict[str, Any]]:
    print(f"\n================================================================================")
    print(f"--> SENARYO: {name}")
    print(f"    Threads={threads}, GPU_Layers={gpu_layers}, VRAM_Cache={vram_cache}MB, K={k_override}, Ctx={ctx_size}")
    print(f"================================================================================")
    
    backend = AtlasSubprocessBackend(
        model_path=MODEL_PATH,
        threads=threads,
        ctx_size=ctx_size,
        gpu_layers=gpu_layers,
        vram_cache=vram_cache,
        prompt_cache_mb=prompt_cache_mb,
        cuda_sched=cuda_sched,
        boost=True,
        extra_args=[
            "--atlas-physical-map", PHYSICAL_MAP_PATH,
            "--atlas-k", str(k_override)
        ]
    )
    
    results = []
    t_start_backend = time.perf_counter()
    await backend.start()
    init_time_s = time.perf_counter() - t_start_backend
    print(f"    [INIT] Motor baslatildi: {init_time_s:.2f}s | RAM={get_ram_usage_gb():.2f}GB | VRAM={get_vram_usage_mb():.1f}MB")

    try:
        # Görev 1: Matematik ve Çok Adımlı Akıl Yürütme (Serbest Bırakılmış Doğal Üretim, 384 token)
        p1 = (
            "<|im_start|>user\n"
            "Bir ciftlikte 17 tavuk ve 19 koyun bulunmaktadir. "
            "Ciftlikteki toplam bacak sayisini ve kafa sayisini adim adim hesapla. "
            "Sonucu acik ve net bir sekilde yaz.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        print("    [1/4] Cok Adimli Matematik & Akil Yurutme (Max 384t) ...", end="", flush=True)
        r1 = await run_single_inference(backend, p1, n_predict=384, expected_regex=r"(110|36)")
        r1["task_id"] = "math_reasoning_uncapped"
        r1["scenario"] = name
        results.append(r1)
        resp_sample = r1['response_text'][:60] if r1['response_text'] else r1['output_text'][:60]
        print(f" Tamamlandi! [Tokens={r1['tokens_generated']} | Decode={r1['decode_tps']} TPS | TTFT={r1['ttft_ms']}ms | Sonuc='{resp_sample}...']")

        # Görev 2: Sıcak Önbellek Çok Turlu İstem (Cache Hit TTFT & Hız Doğrulaması)
        p2 = (
            f"{p1}{r1['output_text']}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Simdi ciftlige 5 tavuk daha eklenirse ve 3 koyun satilirsa yeni toplam bacak sayisi kac olur? Kisa acikla.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        print("    [2/4] Sicak Onbellek Cok Turlu Istem (Cache Hit) ...", end="", flush=True)
        r2 = await run_single_inference(backend, p2, n_predict=256, expected_regex=r"(108|bacak)")
        r2["task_id"] = "multiturn_warm_cache_hit"
        r2["scenario"] = name
        results.append(r2)
        resp_sample2 = r2['response_text'][:60] if r2['response_text'] else r2['output_text'][:60]
        print(f" Tamamlandi! [Tokens={r2['tokens_generated']} | Decode={r2['decode_tps']} TPS | CacheHit={r2['cache_hit']} ({r2['cached_prompt_tokens']}t) | TTFT={r2['ttft_ms']}ms | Sonuc='{resp_sample2}...']")

        # Görev 3: Python Fonksiyon Sentezi (Genişletilmiş Kod Üretimi, 400 token)
        p3 = (
            "<|im_start|>user\n"
            "Python ile bir metindeki en sik gecen ilk 5 kelimeyi bulan, noktalama isaretlerini temizleyen "
            "ve buyuk/kucuk harf duyarsiz calisan temiz, aciklamali bir fonksiyon yaz. Ornek kullanim ekle.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        print("    [3/4] Genisletilmis Kod Uretimi (Max 400t) ...", end="", flush=True)
        r3 = await run_single_inference(backend, p3, n_predict=400, expected_regex=r"def\s+")
        r3["task_id"] = "code_synthesis_extended"
        r3["scenario"] = name
        results.append(r3)
        print(f" Tamamlandi! [Tokens={r3['tokens_generated']} | Decode={r3['decode_tps']} TPS | Eff={r3['effective_tps']} TPS | TTFT={r3['ttft_ms']}ms]")

        # Görev 4: Derin Sistem Mimarisi Analizi (Uzun Teknik Çıktı, 512 token)
        p4 = (
            "<|im_start|>user\n"
            "Mixture-of-Experts (MoE) mimarilerinde 'expert thrashing' ve 'memory wall' darbogazi nedir? "
            "DDR5 bellek bant genisligi buyuk MoE modellerinin calismasini nasil kisitlar? Net ve teknik olarak acikla.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        print("    [4/4] Derin MoE Sistem Mimarisi Analizi (Max 512t) ...", end="", flush=True)
        r4 = await run_single_inference(backend, p4, n_predict=512)
        r4["task_id"] = "deep_moe_architecture_analysis"
        r4["scenario"] = name
        results.append(r4)
        print(f" Tamamlandi! [Tokens={r4['tokens_generated']} | Decode={r4['decode_tps']} TPS | Eff={r4['effective_tps']} TPS | TTFT={r4['ttft_ms']}ms]")

    finally:
        await backend.stop()
        print(f"    [STOP] Motor guvenli sekilde durduruldu.")
        
    return results

async def main():
    print(f"********************************************************************************")
    print(f"  TENDOU LAUNCHER: Qwen3.6-35B-A3B-UD-Q8_K_XL (38.5 GB) SERBEST BENCHMARK")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Fiziksel Harita: {PHYSICAL_MAP_PATH}")
    print(f"  RAM: {get_ram_usage_gb():.2f} GB | VRAM: {get_vram_usage_mb():.1f} MB")
    print(f"********************************************************************************")
    
    all_runs = []

    # 1. 16-Thread SMT, GPU Layers=2 (Dengeli Baseline)
    s1 = await benchmark_scenario(
        name="16T_GPU2_NativeK8_Q8XL",
        threads=16,
        gpu_layers=2,
        vram_cache=3200,
        k_override=0,
        ctx_size=4096,
    )
    all_runs.extend(s1)

    # 2. 16-Thread SMT, GPU Layers=4, VRAM Cache=3500MB (Agresif GPU Dengeleme)
    s2 = await benchmark_scenario(
        name="16T_GPU4_NativeK8_Q8XL",
        threads=16,
        gpu_layers=4,
        vram_cache=3500,
        k_override=0,
        ctx_size=4096,
    )
    all_runs.extend(s2)

    # 3. 8-Thread (Physical Zen 4 Cores)
    s3 = await benchmark_scenario(
        name="8T_GPU2_NativeK8_Q8XL",
        threads=8,
        gpu_layers=2,
        vram_cache=3200,
        k_override=0,
        ctx_size=4096,
    )
    all_runs.extend(s3)

    # 4. Tendou Boost Mode (K=5 Expert Pruning)
    s4 = await benchmark_scenario(
        name="16T_GPU4_BoostK5_Q8XL",
        threads=16,
        gpu_layers=4,
        vram_cache=3500,
        k_override=5,
        ctx_size=4096,
    )
    all_runs.extend(s4)

    # JSON ve Markdown Raporlarını Kaydet
    out_json = ROOT / "experiments" / "qwen36_q8_benchmark_report.json"
    out_md = ROOT / "experiments" / "qwen36_q8_benchmark_report.md"
    os.makedirs(out_json.parent, exist_ok=True)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Yapilandirilmis JSON kaydedildi: {out_json}")

    lines = [
        "# Tendou Launcher — Qwen3.6-35B-A3B-UD-Q8_K_XL (38.5 GB) Kapsamlı Benchmark Raporu\n",
        f"**Tarih:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model:** `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (38.45 GB, 40 Katman, 256 Uzman, K=8, Q8_0 Ağırlıkları)  ",
        f"**Fiziksel Harita:** `config/atlas_physical_map_35b.json` (10,240 uzman parçası)  ",
        f"**Donanım:** AMD Ryzen 7 260 (8C/16T Zen 4), NVIDIA RTX 5060 Laptop (8151 MiB GDDR6), 16 GB DDR5 RAM  \n",
        "## 1. Performans ve Hız Karşılaştırması\n",
        "| Senaryo | Görev | Prompt Tok | Üretilen Tok | TTFT (ms) | Decode TPS | Efektif TPS | RAM (GB) | VRAM (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_runs:
        hit = " (HIT)" if r["cache_hit"] else ""
        lines.append(
            f"| `{r['scenario']}` | `{r['task_id']}{hit}` | {r['prompt_tokens']} | **{r['tokens_generated']}** | {r['ttft_ms']} | **{r['decode_tps']}** | {r['effective_tps']} | {r['ram_after_gb']} | {r['vram_after_mb']} |"
        )
    
    lines.append("\n## 2. Modelin Ürettiği Yanıtlar, Akıl Yürütme ve Kalite Çıktıları\n")
    for r in all_runs:
        lines.append(f"### `{r['scenario']}` — `{r['task_id']}`")
        lines.append(f"- **Decode Hızı:** {r['decode_tps']} TPS | **TTFT:** {r['ttft_ms']} ms | **Toplam Token:** {r['tokens_generated']} | **Süre:** {r['total_elapsed_ms']} ms")
        if r.get("thinking_text"):
            lines.append(f"- **Düşünme Süreci (`<think>`):**\n```text\n{r['thinking_text'][:1500]}\n```")
        lines.append(f"- **Nihai Model Yanıtı:**\n```text\n{r['response_text'] if r['response_text'] else r['output_text']}\n```\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] Markdown Raporu kaydedildi: {out_md}")
    print(f"\n================================================================================")
    print(f"  QWEN3.6-35B-A3B Q8_K_XL BENCHMARK TAMAMLANDI!")
    print(f"================================================================================")

if __name__ == "__main__":
    asyncio.run(main())
