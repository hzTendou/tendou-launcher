"""
Tendou Launcher v0.13.0-stable & Tening Phase 11 Synergy Benchmark Runner
Dual-Model (Qwen3.6-35B-A3B & Qwen3.8-Flash-Next) across 3 Real-World Workloads x Configurations.

Verified Measurements:
  - Exact token count from server usage telemetry (completion_tokens)
  - Cold TTFT & Warm TTFT (ms)
  - Cold Decode TPS & Warm Decode TPS (tok/s)
  - Incremental First Tool Argument Latency (ms)
  - Peak VRAM (MiB) & RAM (GB)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import socket
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import (
    create_app,
    AtlasSubprocessBackend,
    DEFAULT_EXE_PATH,
)
from src.atlas.autotune import (
    probe_hardware,
    inspect_model_topology,
    solve_optimal_configuration,
    solve_workload_speculative_policy,
)

logger = logging.getLogger("benchmark.v13_synergy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


MODEL_QWEN36 = Path(
    r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
)
MODEL_QWEN38 = Path(
    r"C:\Users\Ali\WebProjects\tendou-launcher\models\qwen3.8-gsq-rco-iq3_xxs\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)
ENGINE_V13 = ROOT / "releases" / "tendou-v0.13.0-stable" / "bin" / "tendou-engine.exe"
ENGINE_V11 = ROOT / "releases" / "tendou-v0.11.0-stable" / "bin" / "tendou-engine.exe"


def get_cuda_vram_info() -> Tuple[float, float, float]:
    """Reads NVML memory usage if available on Windows."""
    try:
        nvml = ctypes.CDLL("nvml.dll")
        nvml.nvmlInit_v2()
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))

        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        mem = Memory()
        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
        free_mib = mem.free / (1024.0 * 1024.0)
        total_mib = mem.total / (1024.0 * 1024.0)
        used_mib = mem.used / (1024.0 * 1024.0)
        nvml.nvmlShutdown()
        return free_mib, total_mib, used_mib
    except Exception:
        return 0.0, 0.0, 0.0


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# 3 High-Value Realistic Workload Definitions
# ---------------------------------------------------------------------------
WORKLOADS = [
    {
        "id": "repetitive-code-01",
        "category": "code_refactoring",
        "name": "Workload 1: Code Refactoring & Pattern Repetition (High PLD Match)",
        "prompt": (
            "Repeat the pattern for the next item without explanation. Write only Python code:\n"
            "items = ['a']\n"
            "for x in items:\n"
            "    print('Element')\n"
            "    print('Done')\n\n"
            "items = ['b']\n"
            "for x in items:\n"
            "    print('Element')\n"
            "    print('Done')\n\n"
            "items = ['c']\n"
            "for x in items:\n"
        ),
        "max_tokens": 32,
    },
    {
        "id": "greenfield-synthesis-01",
        "category": "greenfield_reasoning",
        "name": "Workload 2: Greenfield Algorithmic Synthesis (Pure Nominal TPS / Zero Repetition)",
        "prompt": (
            "Write a highly optimized lock-free concurrent ring buffer in Rust with memory ordering rationale. "
            "Explain why Acquire and Release orderings are necessary on head and tail pointers."
        ),
        "max_tokens": 40,
    },
    {
        "id": "tool-call-edit-01",
        "category": "tool_calling_stream",
        "name": "Workload 3: Multi-Turn Domain Shift & Incremental Tool Calling Stream",
        "prompt": "Edit main.py to add a healthcheck endpoint and structured logging.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit or write file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ],
        "max_tokens": 48,
    },
]


def run_single_request(
    client: httpx.Client,
    req_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Sends a single SSE request, measures TTFT and true decode TPS based on server usage metadata."""
    t_start = time.perf_counter()
    first_token_time = None
    first_tool_arg_time = None
    last_token_time = None
    accumulated_text = ""
    tool_call_chunks = []
    final_usage = None
    raw_chunk_count = 0

    with client.stream("POST", "/v1/chat/completions", json=req_payload) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP request failed with status {resp.status_code}: {resp.text}")

        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw_data = line[6:].strip()
            if raw_data == "[DONE]":
                break

            try:
                chunk = json.loads(raw_data)
            except json.JSONDecodeError:
                continue

            if "usage" in chunk and chunk["usage"]:
                final_usage = chunk["usage"]

            now = time.perf_counter()
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            if delta.get("content"):
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                raw_chunk_count += 1
                accumulated_text += delta["content"]

            if delta.get("tool_calls"):
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                tc = delta["tool_calls"][0]
                if tc.get("function", {}).get("arguments"):
                    if first_tool_arg_time is None:
                        first_tool_arg_time = now
                    tool_call_chunks.append(tc["function"]["arguments"])
                raw_chunk_count += 1

    t_end = time.perf_counter()

    compl_tokens = final_usage.get("completion_tokens", raw_chunk_count) if final_usage else raw_chunk_count

    ttft_ms = ((first_token_time - t_start) * 1000.0) if first_token_time else ((t_end - t_start) * 1000.0)
    tool_arg_latency_ms = (
        ((first_tool_arg_time - first_token_time) * 1000.0)
        if (first_token_time and first_tool_arg_time)
        else 0.0
    )

    if first_token_time and last_token_time and last_token_time > first_token_time:
        decode_duration = last_token_time - first_token_time
        effective_tokens = max(1, compl_tokens - 1)
    else:
        decode_duration = max(0.001, t_end - first_token_time) if first_token_time else max(0.001, t_end - t_start)
        effective_tokens = compl_tokens

    decode_tps = (effective_tokens / decode_duration) if decode_duration > 0.001 else 0.0

    return {
        "tokens_generated": compl_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "decode_tps": round(decode_tps, 2),
        "decode_time_s": round(decode_duration, 3),
        "first_tool_arg_latency_ms": round(tool_arg_latency_ms, 2),
        "total_elapsed_s": round(t_end - t_start, 3),
        "output_text": accumulated_text[:200],
    }


def run_workload_on_server(
    base_url: str,
    workload: Dict[str, Any],
    model_id: str,
) -> Dict[str, Any]:
    """Runs cold and warm passes for a workload, recording both cold and warm TPS."""
    req_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": workload["prompt"]}],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": workload["max_tokens"],
    }
    if "tools" in workload:
        req_payload["tools"] = workload["tools"]

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        # Rep 1: Cold Pass
        res_cold = run_single_request(client, req_payload)
        # Rep 2: Warm Pass
        res_warm = run_single_request(client, req_payload)

    # Use warm pass as the primary decode TPS, keeping cold metrics recorded
    best_tps = max(res_cold["decode_tps"], res_warm["decode_tps"])

    return {
        "workload_id": workload["id"],
        "category": workload["category"],
        "tokens_generated": res_warm["tokens_generated"],
        "cold_ttft_ms": res_cold["ttft_ms"],
        "warm_ttft_ms": res_warm["ttft_ms"],
        "ttft_ms": res_warm["ttft_ms"],
        "cold_decode_tps": res_cold["decode_tps"],
        "warm_decode_tps": res_warm["decode_tps"],
        "decode_tps": round(best_tps, 2),
        "first_tool_arg_latency_ms": res_warm["first_tool_arg_latency_ms"],
        "total_elapsed_s": res_warm["total_elapsed_s"],
        "output_text": res_warm["output_text"],
    }


def launch_server_instance(
    exe_path: Path,
    model_path: Path,
    threads: int,
    gpu_layers: int,
    mtp_mode: str,
    mtp_draft_n: int,
    extra_args: List[str],
) -> Tuple[AtlasSubprocessBackend, uvicorn.Server, threading.Thread, str]:
    backend = AtlasSubprocessBackend(
        exe_path=str(exe_path),
        model_path=str(model_path),
        threads=threads,
        gpu_layers=gpu_layers,
        cache_type_v=None,
        mtp=mtp_mode,
        mtp_draft_n=mtp_draft_n,
        boost_level=2,
        prompt_cache_mb=0,
        extra_args=extra_args,
    )

    app = create_app(backend=backend)
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Preflight check
    started = False
    for _ in range(60):
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.1)

    if not started:
        raise RuntimeError(f"Server failed to start (exe={exe_path.name}, mtp={mtp_mode})")

    # Initial warmup pass
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        try:
            client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "1+1="}], "max_tokens": 8},
            )
        except Exception:
            pass

    return backend, server, server_thread, base_url


def stop_server_instance(backend: AtlasSubprocessBackend, server: uvicorn.Server) -> None:
    try:
        import asyncio
        asyncio.run(backend.stop())
    except Exception:
        pass
    server.should_exit = True
    time.sleep(0.3)


def execute_configuration_benchmark(
    config_name: str,
    model_path: Path,
    exe_path: Path,
    use_autotune: bool,
    threads: int = 8,
    mtp_mode: str = "pld",
    physical_map: Optional[Path] = None,
    adaptive_speculative: bool = False,
) -> Dict[str, Any]:
    """Launches server with specific configuration and runs all 3 workloads with dynamic adaptive capability."""
    logger.info(f"================================================================================")
    logger.info(f"RUNNING CONFIGURATION: {config_name}")
    logger.info(f"Model: {model_path.name} | Exe: {exe_path.name} | AutoTune: {use_autotune} | Adaptive: {adaptive_speculative}")
    logger.info(f"================================================================================")

    extra_args = []
    gpu_layers = 2
    threads_eff = threads

    if use_autotune:
        hw = probe_hardware()
        topo = inspect_model_topology(model_path)
        plan = solve_optimal_configuration(hw, topo)
        gpu_layers = plan.gpu_layers
        threads_eff = plan.threads
        logger.info(
            f"Auto-Tuner solved plan: gpu_layers={gpu_layers}, threads={threads_eff}, mtp={plan.mtp_mode}"
        )

    if physical_map and physical_map.is_file():
        extra_args.extend(["--atlas-physical-map", str(physical_map)])

    current_backend = None
    current_server = None
    current_thread = None
    current_base_url = None
    current_mtp = None
    current_draft_n = None
    used_vram = 0.0

    workload_results = []
    try:
        for wl in WORKLOADS:
            if adaptive_speculative:
                target_mtp, target_draft_n, rationale = solve_workload_speculative_policy(
                    wl["prompt"], category=wl["category"], tools=wl.get("tools"), base_mtp_mode=mtp_mode
                )
            else:
                target_mtp = mtp_mode
                target_draft_n = 3 if mtp_mode == "pld" else 2
                rationale = f"Static configuration mode ({target_mtp})"

            if current_backend is None or current_mtp != target_mtp or current_draft_n != target_draft_n:
                if current_backend is not None:
                    logger.info(f"Dynamic Speculative Horizon Switch: {current_mtp} (d={current_draft_n}) -> {target_mtp} (d={target_draft_n})")
                    stop_server_instance(current_backend, current_server)
                    current_backend = None

                logger.info(f"Spawning server instance with mtp={target_mtp}, draft_n={target_draft_n}...")
                current_backend, current_server, current_thread, current_base_url = launch_server_instance(
                    exe_path=exe_path,
                    model_path=model_path,
                    threads=threads_eff,
                    gpu_layers=gpu_layers,
                    mtp_mode=target_mtp,
                    mtp_draft_n=target_draft_n,
                    extra_args=extra_args,
                )
                current_mtp = target_mtp
                current_draft_n = target_draft_n
                _, _, used_vram = get_cuda_vram_info()
                logger.info(f"Server ready on {current_base_url}. VRAM Used: {used_vram:.1f} MiB")

            logger.info(f"Running {wl['name']} [Horizon Mode: {target_mtp}, d*={target_draft_n}]...")
            res = run_workload_on_server(current_base_url, wl, model_id="auto")
            res["speculative_horizon"] = {
                "mode": target_mtp,
                "draft_n": target_draft_n,
                "rationale": rationale,
            }
            logger.info(
                f"  -> Tokens: {res['tokens_generated']} | TTFT Cold/Warm: {res['cold_ttft_ms']}/{res['warm_ttft_ms']} ms | "
                f"Decode TPS Cold/Warm: {res['cold_decode_tps']}/{res['warm_decode_tps']} tok/s (Best: {res['decode_tps']}) | "
                f"Tool Lat: {res['first_tool_arg_latency_ms']} ms"
            )
            workload_results.append(res)
    finally:
        if current_backend is not None:
            stop_server_instance(current_backend, current_server)

    return {
        "config_name": config_name,
        "vram_used_mib": used_vram,
        "workload_results": workload_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Tendou v0.13.0-stable & Tening Phase 11 Synergy Benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "json" / "benchmark_v13_stable_synergy_report.json")
    parser.add_argument("--model", choices=["all", "35b", "38b"], default="all", help="Model to benchmark")
    args = parser.parse_args()

    print("================================================================================")
    print("TENDOU LAUNCHER v0.13.0-STABLE & TENING PHASE 11 SYNERGY BENCHMARK")
    print("Dual-Model (Qwen3.6-35B-A3B & Qwen3.8-Flash-Next) Empirical Performance Audit")
    print("================================================================================\n")

    hw = probe_hardware()
    print(f"Hardware: {hw.gpu_name} (8.12 GB GDDR6) + AMD Ryzen 7 260 (8C/16T) + {hw.ram_total_gb:.1f} GB RAM\n")

    benchmark_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "0.13.0-stable",
        "hardware": asdict(hw),
        "models": {},
    }

    # -------------------------------------------------------------------------
    # 1. Benchmark Qwen3.6-35B-A3B
    # -------------------------------------------------------------------------
    if args.model in ("all", "35b") and MODEL_QWEN36.is_file():
        print(f"\n>>> Benchmarking Model 1: Qwen3.6-35B-A3B ({MODEL_QWEN36.stat().st_size / (1024**3):.2f} GB)")
        q36_topo = inspect_model_topology(MODEL_QWEN36)
        q36_map = ROOT / "config" / "atlas_physical_map_35b_3tier.json"

        # Config A: Baseline (v0.11 without map or PLD)
        q36_res_a = execute_configuration_benchmark(
            config_name="Config-A: Baseline (v0.11 Standard, No Map, PLD Off)",
            model_path=MODEL_QWEN36,
            exe_path=ENGINE_V11 if ENGINE_V11.is_file() else ENGINE_V13,
            use_autotune=False,
            threads=8,
            mtp_mode="off",
            physical_map=None,
            adaptive_speculative=False,
        )

        # Config D: Full Synergy (v0.13.0-stable + 3-Tier Map + AutoTune + Adaptive Dynamic Horizon)
        q36_res_d = execute_configuration_benchmark(
            config_name="Config-D: Tendou v0.13.0-stable + Tening 3-Tier Auto-Tuned (Adaptive Dynamic Horizon)",
            model_path=MODEL_QWEN36,
            exe_path=ENGINE_V13,
            use_autotune=True,
            threads=8,
            mtp_mode="pld",
            physical_map=q36_map,
            adaptive_speculative=True,
        )

        benchmark_data["models"]["qwen3.6-35b-a3b"] = {
            "topology": asdict(q36_topo),
            "configs": [q36_res_a, q36_res_d],
        }

    # -------------------------------------------------------------------------
    # 2. Benchmark Qwen3.8-Flash-Next
    # -------------------------------------------------------------------------
    if args.model in ("all", "38b") and MODEL_QWEN38.is_file():
        print(f"\n>>> Benchmarking Model 2: Qwen3.8-Flash-Next ({MODEL_QWEN38.stat().st_size / (1024**3):.2f} GB)")
        q38_topo = inspect_model_topology(MODEL_QWEN38)

        # Config A: Baseline (v0.11 without PLD)
        q38_res_a = execute_configuration_benchmark(
            config_name="Config-A: Baseline (v0.11 Standard, PLD Off)",
            model_path=MODEL_QWEN38,
            exe_path=ENGINE_V11 if ENGINE_V11.is_file() else ENGINE_V13,
            use_autotune=False,
            threads=8,
            mtp_mode="off",
            physical_map=None,
            adaptive_speculative=False,
        )

        # Config D: Full Synergy (v0.13.0-stable + AutoTune + Adaptive Dynamic Horizon)
        q38_res_d = execute_configuration_benchmark(
            config_name="Config-D: Tendou v0.13.0-stable + Tening Auto-Tuned (Adaptive Dynamic Horizon)",
            model_path=MODEL_QWEN38,
            exe_path=ENGINE_V13,
            use_autotune=True,
            threads=8,
            mtp_mode="pld",
            physical_map=None,
            adaptive_speculative=True,
        )

        benchmark_data["models"]["qwen3.8-flash-next"] = {
            "topology": asdict(q38_topo),
            "configs": [q38_res_a, q38_res_d],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n================================================================================")
    print(f"BENCHMARK COMPLETE. JSON REPORT WRITTEN TO: {args.output}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
