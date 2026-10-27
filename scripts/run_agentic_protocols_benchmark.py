"""
Tendou Launcher v0.14.0-stable Dual-Protocol Agentic Benchmark
Comparing OpenCode (OpenAI /v1/chat/completions) & Claude Code (Anthropic /v1/messages)
across Qwen3.6-35B-A3B and Qwen3.8-Flash-Next on Real Subprocess Hardware.
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

logger = logging.getLogger("benchmark.agentic_protocols")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_QWEN36 = Path(
    r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
)
MODEL_QWEN38 = Path(
    r"C:\Users\Ali\WebProjects\tendou-launcher\models\qwen3.8-gsq-rco-iq3_xxs\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)
MODEL_QWEN122 = ROOT / "models" / "MXFP4_MOE" / "Qwen3.5-122B-A10B-MXFP4_MOE-00001-of-00003.gguf"
ENGINE_V14 = ROOT / "releases" / "tendou-v0.14.0-stable" / "bin" / "tendou-engine.exe"


def get_cuda_vram_info() -> Tuple[float, float, float]:
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
# 3 High-Value Realistic Agentic Workloads
# ---------------------------------------------------------------------------
WORKLOADS = [
    {
        "id": "workload-1-refactor",
        "category": "code_refactoring",
        "name": "Workload 1: Code Refactoring & Repetitive Pattern (PLD d*=3 Candidate)",
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
        "id": "workload-2-greenfield",
        "category": "greenfield_reasoning",
        "name": "Workload 2: Greenfield Synthesis & Reasoning (Pure Nominal / Zero Repetition Fallback)",
        "prompt": (
            "Write a highly optimized lock-free concurrent ring buffer in Rust with memory ordering rationale. "
            "Explain why Acquire and Release orderings are necessary on head and tail pointers."
        ),
        "max_tokens": 40,
    },
    {
        "id": "workload-3-tool-calling",
        "category": "tool_calling_stream",
        "name": "Workload 3: Domain Shift & Incremental Tool Calling Stream",
        "prompt": "Use the edit_file tool to write main.py with a FastAPI /health endpoint and structured logging.",
        "tools": [
            {
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
            }
        ],
        "max_tokens": 48,
    },
]


def run_opencode_stream_request(
    client: httpx.Client,
    workload: Dict[str, Any],
    model_id: str = "auto",
) -> Dict[str, Any]:
    """Runs a streaming request using OpenCode / OpenAI protocol (POST /v1/chat/completions)."""
    req_payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": workload["prompt"]}],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": workload["max_tokens"],
    }
    if "tools" in workload:
        req_payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in workload["tools"]
        ]

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
            raise RuntimeError(f"OpenCode HTTP error {resp.status_code}: {resp.text}")

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
    prompt_tokens = final_usage.get("prompt_tokens", 0) if final_usage else 0
    cached_tokens = (
        final_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if final_usage
        else 0
    )

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
        "protocol": "opencode",
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "tokens_generated": compl_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "decode_tps": round(decode_tps, 2),
        "decode_time_s": round(decode_duration, 3),
        "first_tool_arg_latency_ms": round(tool_arg_latency_ms, 2),
        "total_elapsed_s": round(t_end - t_start, 3),
        "output_preview": (accumulated_text or "".join(tool_call_chunks))[:160],
    }


def run_claude_code_stream_request(
    client: httpx.Client,
    workload: Dict[str, Any],
    model_id: str = "claude-3-5-sonnet-20241022",
) -> Dict[str, Any]:
    """Runs a streaming request using Claude Code / Anthropic protocol (POST /v1/messages)."""
    req_payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": workload["prompt"]}],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": workload["max_tokens"],
    }
    if "tools" in workload:
        req_payload["tools"] = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in workload["tools"]
        ]

    t_start = time.perf_counter()
    first_token_time = None
    first_tool_arg_time = None
    last_token_time = None
    accumulated_text = ""
    tool_call_chunks = []
    compl_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    raw_event_count = 0

    with client.stream("POST", "/v1/messages", json=req_payload) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Claude Code HTTP error {resp.status_code}: {resp.text}")

        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw_data = line[6:].strip()

            try:
                ev = json.loads(raw_data)
            except json.JSONDecodeError:
                continue

            ev_type = ev.get("type")
            now = time.perf_counter()

            if ev_type == "message_start":
                usage = ev.get("message", {}).get("usage", {})
                if "input_tokens" in usage:
                    prompt_tokens = usage["input_tokens"]
                if "cache_read_input_tokens" in usage:
                    cached_tokens = usage["cache_read_input_tokens"]

            elif ev_type == "content_block_start":
                if first_token_time is None:
                    first_token_time = now

            elif ev_type == "content_block_delta":
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                raw_event_count += 1

                delta = ev.get("delta", {})
                if delta.get("type") == "text_delta":
                    accumulated_text += delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    if first_tool_arg_time is None:
                        first_tool_arg_time = now
                    tool_call_chunks.append(delta.get("partial_json", ""))

            elif ev_type == "message_delta":
                usage = ev.get("usage", {})
                if "output_tokens" in usage and usage["output_tokens"] > 0:
                    compl_tokens = usage["output_tokens"]
                if "cache_read_input_tokens" in usage:
                    cached_tokens = usage["cache_read_input_tokens"]

    t_end = time.perf_counter()
    if compl_tokens == 0:
        compl_tokens = raw_event_count

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
        "protocol": "claude-code",
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "tokens_generated": compl_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "decode_tps": round(decode_tps, 2),
        "decode_time_s": round(decode_duration, 3),
        "first_tool_arg_latency_ms": round(tool_arg_latency_ms, 2),
        "total_elapsed_s": round(t_end - t_start, 3),
        "output_preview": (accumulated_text or "".join(tool_call_chunks))[:160],
    }


def run_cline_stream_request(
    client: httpx.Client,
    workload: Dict[str, Any],
    model_id: str = "auto",
) -> Dict[str, Any]:
    """Runs a streaming request using Cline (NVIDIA NIM / Agentic XML + multi-part content blocks)."""
    if "tools" in workload:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Cline, a highly skilled software engineer. "
                            "Respond ONLY with a complete <write_to_file> XML tool call containing the full Python implementation with logging, FastAPI app, and /health endpoint (do not output prose or <thinking> tags):\n"
                            "<write_to_file>\n<path>main.py</path>\n<content>import logging\nfrom fastapi import FastAPI\n...</content>\n</write_to_file>"
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<task>\n{workload['prompt']}\n</task>"},
                    {"type": "text", "text": "<environment_details>\n# VSCode Visible Files\nmain.py\n</environment_details>"},
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": workload["prompt"]}],
            }
        ]

    req_payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": workload["max_tokens"],
    }

    t_start = time.perf_counter()
    first_token_time = None
    first_tool_arg_time = None
    last_token_time = None
    accumulated_text = ""
    final_usage = None
    raw_chunk_count = 0

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=req_payload,
        headers={"X-Title": "Cline", "HTTP-Referer": "https://cline.bot"},
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Cline HTTP error {resp.status_code}: {resp.text}")

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
                if first_tool_arg_time is None and ("<path>" in accumulated_text or "<content>" in accumulated_text):
                    first_tool_arg_time = now

    t_end = time.perf_counter()
    compl_tokens = final_usage.get("completion_tokens", raw_chunk_count) if final_usage else raw_chunk_count
    prompt_tokens = final_usage.get("prompt_tokens", 0) if final_usage else 0
    cached_tokens = (
        final_usage.get("cache_read_input_tokens", 0)
        or final_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if final_usage
        else 0
    )

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
        "protocol": "cline",
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "tokens_generated": compl_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "decode_tps": round(decode_tps, 2),
        "decode_time_s": round(decode_duration, 3),
        "first_tool_arg_latency_ms": round(tool_arg_latency_ms, 2),
        "total_elapsed_s": round(t_end - t_start, 3),
        "output_preview": accumulated_text[:160],
    }


def launch_server_instance(
    exe_path: Path,
    model_path: Path,
    threads: int,
    gpu_layers: int,
    mtp_mode: str,
    mtp_draft_n: int,
    extra_args: List[str],
    prompt_cache_mb: int = 512,
    moe_cache_mib: int = 2048,
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
        prompt_cache_mb=prompt_cache_mb,
        moe_cache_mib=moe_cache_mib,
        extra_args=extra_args,
    )

    app = create_app(backend=backend)
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Preflight check (up to 60s for cold 75GB multi-shard model + 30MB physical map init)
    started = False
    for _ in range(300):
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.2)

    if not started:
        raise RuntimeError(f"Server failed to start (exe={exe_path.name}, mtp={mtp_mode})")

    # Initial warmup pass
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        try:
            client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "1+1="}], "max_tokens": 4},
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
    time.sleep(2.0)  # Allow Windows WDDM driver to reclaim GPU VRAM cleanly


def run_benchmark_for_model(
    model_name: str,
    model_path: Path,
    exe_path: Path,
    physical_map: Optional[Path] = None,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(f"STARTING DUAL-PROTOCOL BENCHMARK: {model_name.upper()}")
    logger.info(f"Model File: {model_path.name}")
    logger.info("=" * 80)

    hw = probe_hardware()
    topo = inspect_model_topology(model_path)
    plan = solve_optimal_configuration(hw, topo)

    extra_args = []
    map_to_use = physical_map or (Path(plan.physical_map_path) if plan.physical_map_path else None)
    if map_to_use and map_to_use.is_file():
        extra_args.extend(["--atlas-physical-map", str(map_to_use)])

    if "122b" in model_name.lower():
        os.environ["ATLAS_PREFILL_TOP_K"] = "2"
        extra_args.extend(["--atlas-k", "4"])
    else:
        # Use native layer-aware prefill sparsity (Top-2 on reasoning/output layers, Top-1 on outer layers)
        os.environ.pop("ATLAS_PREFILL_TOP_K", None)

    current_backend = None
    current_server = None
    current_thread = None
    current_base_url = None
    current_mtp = None
    current_draft_n = None
    vram_peak = 0.0

    model_workload_results = []

    try:
        for wl in WORKLOADS:
            target_mtp, target_draft_n, rationale = solve_workload_speculative_policy(
                wl["prompt"], category=wl["category"], tools=wl.get("tools"), base_mtp_mode=plan.mtp_mode
            )

            if current_backend is None or current_mtp != target_mtp or current_draft_n != target_draft_n:
                if current_backend is not None:
                    stop_server_instance(current_backend, current_server)
                    current_backend = None

                logger.info(f"Spawning server instance [Mode: {target_mtp}, d*={target_draft_n}]...")
                current_backend, current_server, current_thread, current_base_url = launch_server_instance(
                    exe_path=exe_path,
                    model_path=model_path,
                    threads=plan.threads,
                    gpu_layers=plan.gpu_layers,
                    mtp_mode=target_mtp,
                    mtp_draft_n=target_draft_n,
                    extra_args=extra_args,
                    prompt_cache_mb=getattr(plan, "prompt_cache_mb", 512),
                    moe_cache_mib=getattr(plan, "moe_cache_mib", 2048),
                )
                current_mtp = target_mtp
                current_draft_n = target_draft_n
                _, _, vram_peak = get_cuda_vram_info()
                logger.info(f"Server ready on {current_base_url}. VRAM Used: {vram_peak:.1f} MiB")

            logger.info(f"\n--- Testing Workload: {wl['name']} ---")

            with httpx.Client(base_url=current_base_url, timeout=300.0) as client:
                # -------------------------------------------------------------
                # PROTOCOL 1: OpenCode (/v1/chat/completions)
                # -------------------------------------------------------------
                logger.info(">> Running OpenCode (OpenAI /v1/chat/completions) Cold & Warm...")
                opencode_cold = run_opencode_stream_request(client, wl)
                opencode_warm = run_opencode_stream_request(client, wl)
                opencode_best_tps = max(opencode_cold["decode_tps"], opencode_warm["decode_tps"])
                opencode_warm_speedup = (
                    round(opencode_cold["ttft_ms"] / max(0.1, opencode_warm["ttft_ms"]), 2)
                    if opencode_warm["ttft_ms"] > 0
                    else 1.0
                )
                logger.info(
                    f"   [OpenCode] Cold TPS: {opencode_cold['decode_tps']} | Warm TPS: {opencode_warm['decode_tps']} "
                    f"(Best: {opencode_best_tps} tok/s) | TTFT Cold/Warm: {opencode_cold['ttft_ms']}/{opencode_warm['ttft_ms']} ms "
                    f"(Speedup: {opencode_warm_speedup}x, Cached: {opencode_warm.get('cached_prompt_tokens',0)} toks)"
                )
                # -------------------------------------------------------------
                # PROTOCOL 2: Claude Code (/v1/messages)
                # -------------------------------------------------------------
                logger.info(">> Running Claude Code (Anthropic /v1/messages) Cold & Warm...")
                claude_cold = run_claude_code_stream_request(client, wl)
                claude_warm = run_claude_code_stream_request(client, wl)
                claude_best_tps = max(claude_cold["decode_tps"], claude_warm["decode_tps"])
                claude_warm_speedup = (
                    round(claude_cold["ttft_ms"] / max(0.1, claude_warm["ttft_ms"]), 2)
                    if claude_warm["ttft_ms"] > 0
                    else 1.0
                )
                logger.info(
                    f"   [Claude Code] Cold TPS: {claude_cold['decode_tps']} | Warm TPS: {claude_warm['decode_tps']} "
                    f"(Best: {claude_best_tps} tok/s) | TTFT Cold/Warm: {claude_cold['ttft_ms']}/{claude_warm['ttft_ms']} ms "
                    f"(Speedup: {claude_warm_speedup}x, Cached: {claude_warm.get('cached_prompt_tokens',0)} toks)"
                )

                # -------------------------------------------------------------
                # PROTOCOL 3: Cline (NVIDIA NIM / Agentic XML Stream)
                # -------------------------------------------------------------
                logger.info(">> Running Cline (Agentic XML / Multi-Part Stream) Cold & Warm...")
                cline_cold = run_cline_stream_request(client, wl)
                cline_warm = run_cline_stream_request(client, wl)
                cline_best_tps = max(cline_cold["decode_tps"], cline_warm["decode_tps"])
                cline_warm_speedup = (
                    round(cline_cold["ttft_ms"] / max(0.1, cline_warm["ttft_ms"]), 2)
                    if cline_warm["ttft_ms"] > 0
                    else 1.0
                )
                logger.info(
                    f"   [Cline] Cold TPS: {cline_cold['decode_tps']} | Warm TPS: {cline_warm['decode_tps']} "
                    f"(Best: {cline_best_tps} tok/s) | TTFT Cold/Warm: {cline_cold['ttft_ms']}/{cline_warm['ttft_ms']} ms "
                    f"(Speedup: {cline_warm_speedup}x, Cached: {cline_warm.get('cached_prompt_tokens',0)} toks)"
                )

            model_workload_results.append({
                "workload_id": wl["id"],
                "category": wl["category"],
                "name": wl["name"],
                "speculative_horizon": {
                    "mode": target_mtp,
                    "draft_n": target_draft_n,
                    "rationale": rationale,
                },
                "opencode": {
                    "prompt_tokens": opencode_cold.get("prompt_tokens", 0),
                    "cached_prompt_tokens": opencode_warm.get("cached_prompt_tokens", 0),
                    "cold_ttft_ms": opencode_cold["ttft_ms"],
                    "warm_ttft_ms": opencode_warm["ttft_ms"],
                    "ttft_speedup_x": opencode_warm_speedup,
                    "cold_decode_tps": opencode_cold["decode_tps"],
                    "warm_decode_tps": opencode_warm["decode_tps"],
                    "best_decode_tps": round(opencode_best_tps, 2),
                    "cold_effective_tps": round(opencode_cold["tokens_generated"] / max(0.001, opencode_cold["total_elapsed_s"]), 2),
                    "warm_effective_tps": round(opencode_warm["tokens_generated"] / max(0.001, opencode_warm["total_elapsed_s"]), 2),
                    "tokens_generated": opencode_warm["tokens_generated"],
                    "first_tool_arg_latency_ms": opencode_warm["first_tool_arg_latency_ms"],
                    "cold_elapsed_s": opencode_cold["total_elapsed_s"],
                    "warm_elapsed_s": opencode_warm["total_elapsed_s"],
                    "output_preview": opencode_warm["output_preview"],
                },
                "claude_code": {
                    "prompt_tokens": claude_cold.get("prompt_tokens", 0),
                    "cached_prompt_tokens": claude_warm.get("cached_prompt_tokens", 0),
                    "cold_ttft_ms": claude_cold["ttft_ms"],
                    "warm_ttft_ms": claude_warm["ttft_ms"],
                    "ttft_speedup_x": claude_warm_speedup,
                    "cold_decode_tps": claude_cold["decode_tps"],
                    "warm_decode_tps": claude_warm["decode_tps"],
                    "best_decode_tps": round(claude_best_tps, 2),
                    "cold_effective_tps": round(claude_cold["tokens_generated"] / max(0.001, claude_cold["total_elapsed_s"]), 2),
                    "warm_effective_tps": round(claude_warm["tokens_generated"] / max(0.001, claude_warm["total_elapsed_s"]), 2),
                    "tokens_generated": claude_warm["tokens_generated"],
                    "first_tool_arg_latency_ms": claude_warm["first_tool_arg_latency_ms"],
                    "cold_elapsed_s": claude_cold["total_elapsed_s"],
                    "warm_elapsed_s": claude_warm["total_elapsed_s"],
                    "output_preview": claude_warm["output_preview"],
                },
                "cline": {
                    "prompt_tokens": cline_cold.get("prompt_tokens", 0),
                    "cached_prompt_tokens": cline_warm.get("cached_prompt_tokens", 0),
                    "cold_ttft_ms": cline_cold["ttft_ms"],
                    "warm_ttft_ms": cline_warm["ttft_ms"],
                    "ttft_speedup_x": cline_warm_speedup,
                    "cold_decode_tps": cline_cold["decode_tps"],
                    "warm_decode_tps": cline_warm["decode_tps"],
                    "best_decode_tps": round(cline_best_tps, 2),
                    "cold_effective_tps": round(cline_cold["tokens_generated"] / max(0.001, cline_cold["total_elapsed_s"]), 2),
                    "warm_effective_tps": round(cline_warm["tokens_generated"] / max(0.001, cline_warm["total_elapsed_s"]), 2),
                    "tokens_generated": cline_warm["tokens_generated"],
                    "first_tool_arg_latency_ms": cline_warm["first_tool_arg_latency_ms"],
                    "cold_elapsed_s": cline_cold["total_elapsed_s"],
                    "warm_elapsed_s": cline_warm["total_elapsed_s"],
                    "output_preview": cline_warm["output_preview"],
                },
            })
    finally:
        if current_backend is not None:
            stop_server_instance(current_backend, current_server)

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "vram_peak_mib": vram_peak,
        "topology": asdict(topo),
        "plan": asdict(plan),
        "workloads": model_workload_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Tendou Launcher v0.14.0-stable Dual-Protocol Agentic Benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "json" / "benchmark_v14_claude_code_and_opencode_report.json",
    )
    parser.add_argument("--model", choices=["all", "35b", "38b", "122b"], default="all")
    args = parser.parse_args()

    print("================================================================================")
    print("TENDOU LAUNCHER v0.14.0-STABLE DUAL-PROTOCOL AGENTIC BENCHMARK")
    print("OpenCode (/v1/chat/completions) & Claude Code (/v1/messages) Comparative Audit")
    print("================================================================================\n")

    hw = probe_hardware()
    print(f"Hardware Detected: {hw.gpu_name} (8.12 GB GDDR6) | AMD Ryzen 7 260 | {hw.ram_total_gb:.1f} GB RAM\n")

    benchmark_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "0.14.0-stable",
        "engine": str(ENGINE_V14.relative_to(ROOT)),
        "hardware": asdict(hw),
        "results": {},
    }

    if args.output.is_file():
        try:
            with args.output.open("r", encoding="utf-8") as f:
                existing_report = json.load(f)
                if "results" in existing_report:
                    benchmark_report["results"].update(existing_report["results"])
        except Exception:
            pass
    if args.model in ("all", "35b") and MODEL_QWEN36.is_file():
        q36_map = ROOT / "config" / "atlas_physical_map_35b_3tier.json"
        res_35b = run_benchmark_for_model(
            model_name="qwen3.6-35b-a3b",
            model_path=MODEL_QWEN36,
            exe_path=ENGINE_V14,
            physical_map=q36_map,
        )
        benchmark_report["results"]["qwen3.6-35b-a3b"] = res_35b

    # 2. Benchmark Qwen3.8-Flash-Next
    if args.model in ("all", "38b") and MODEL_QWEN38.is_file():
        res_38b = run_benchmark_for_model(
            model_name="qwen3.8-flash-next",
            model_path=MODEL_QWEN38,
            exe_path=ENGINE_V14,
            physical_map=ROOT / "config" / "atlas_physical_map_qwen38_iq3xxs.json",
        )
        benchmark_report["results"]["qwen3.8-flash-next"] = res_38b

    # 3. Benchmark Qwen3.5-122B-A10B MXFP4
    if args.model in ("all", "122b") and MODEL_QWEN122.is_file():
        res_122b = run_benchmark_for_model(
            model_name="qwen3.5-122b-a10b-mxfp4",
            model_path=MODEL_QWEN122,
            exe_path=ENGINE_V14,
            physical_map=ROOT / "config" / "atlas_physical_map_122b_mxfp4.json",
        )
        benchmark_report["results"]["qwen3.5-122b-a10b-mxfp4"] = res_122b

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. JSON REPORT SAVED TO: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
