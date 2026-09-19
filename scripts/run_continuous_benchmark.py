"""
tendou launcher — Continuous Benchmark Runner.

Executes benchmark suites across short, 32k, 64k, and 128k contexts with mandatory
prompt caching on large contexts, extracting deep C++ timing metrics and recording
the complete AI generated model outputs. Appends results continuously to docs/BENCHMARK_LOG.md
and experiments/benchmark_history.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_logger import (
    BenchmarkItemResult,
    BenchmarkLogger,
    BenchmarkRunSummary,
    detect_hardware,
)
from src.atlas.server import (
    DEFAULT_EXE_PATH,
    DEFAULT_MODEL_PATH,
    AtlasSubprocessBackend,
    ChatMessage,
    format_chatml_prompt,
)


CONTEXT_CONFIGS = {
    "short": {
        "ctx_size": 2048,
        "kv_quant": "q8_0",
        "description": "Short context baseline (2048 tokens)",
    },
    "2k": {
        "ctx_size": 2048,
        "kv_quant": "q8_0",
        "description": "Short context baseline (2048 tokens)",
    },
    "32k": {
        "ctx_size": 32768,
        "kv_quant": "q8_0",
        "description": "32k context with Q8_0 KV cache and active prompt checkpointing",
    },
    "64k": {
        "ctx_size": 65536,
        "kv_quant": "q4_0",
        "description": "64k context with Q4_0 KV cache (preserving 8GB VRAM / 16GB RAM)",
    },
    "128k": {
        "ctx_size": 131072,
        "kv_quant": "q4_0",
        "description": "128k context with Q4_0 KV cache (preserving 8GB VRAM / 16GB RAM)",
    },
}


def build_suite_tasks(context_name: str) -> List[Dict[str, Any]]:
    """Return task definitions for each context tier."""
    if context_name in ("short", "2k"):
        return [
            {
                "task_id": "short-math-01",
                "category": "math",
                "workload": "short_request",
                "prompt": "17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.",
                "expected": r"323",
                "max_tokens": 24,
            },
            {
                "task_id": "short-code-01",
                "category": "code",
                "workload": "short_request",
                "prompt": "Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])",
                "expected": r"42",
                "max_tokens": 24,
            },
        ]
    elif context_name == "32k":
        # 32k multi-turn caching test
        sys_prompt = "You are an expert AI software architect for high-performance low-latency MoE inference engines."
        context_body = (
            "Tendou launcher is an edge-native MoE inference system running on consumer laptops.\n"
            "Hardware architecture: AMD Ryzen 7 Zen 4 (8 physical cores), 16 GB DDR5 RAM, NVIDIA RTX 5060 Laptop (8 GB GDDR6).\n"
            "Key architectural modules:\n"
            "1. P2 Dynamic GPU Expert Binding with LRU cache.\n"
            "2. P3 Asynchronous H2D Transfer Pipeline with pinned staging buffers and CUDA event fences.\n"
            "3. Prompt Cache with state serialization for instant multi-turn prefill reuse.\n"
            "4. Dynamic K layer-adaptive expert pruning.\n"
        )
        return [
            {
                "task_id": "32k-turn1-cold",
                "category": "architecture",
                "workload": "multi_turn_cache",
                "system": sys_prompt,
                "context": context_body,
                "prompt": "Based on the provided specification, list the primary GPU model and its VRAM size in one short line.",
                "expected": r"(RTX\s*5060|8\s*GB)",
                "max_tokens": 48,
                "turn": 1,
            },
            {
                "task_id": "32k-turn2-warm",
                "category": "architecture",
                "workload": "multi_turn_cache",
                "system": sys_prompt,
                "context": context_body,
                "prompt": "Now name the module responsible for asynchronous H2D transfer and explain its buffer mechanism in one concise sentence.",
                "expected": r"(P3|Async|pinned|staging)",
                "max_tokens": 64,
                "turn": 2,
            },
        ]
    elif context_name == "64k":
        # 64k large context test
        sys_prompt = "You are an accurate technical documentation analyzer specializing in large language model serving."
        doc_segment = (
            "Section 64K-MoE: When operating at 64,000 token context length, memory management is critical.\n"
            "KV cache memory scaling: At 64k tokens with 48 layers and 2 KV heads, fp16 KV cache requires 3.4 GB.\n"
            "By deploying Q4_0 KV cache quantization, KV footprint is reduced to approximately 850 MB,\n"
            "allowing full model weights and active context to coexist safely within 8 GB VRAM.\n"
            "Prefix prompt caching guarantees that repetitive context prefix evaluation is bypassed completely.\n"
        )
        return [
            {
                "task_id": "64k-turn1-analysis",
                "category": "memory_scaling",
                "workload": "large_context_cache",
                "system": sys_prompt,
                "context": doc_segment,
                "prompt": "What is the memory size of fp16 KV cache versus Q4_0 KV cache at 64k context according to the text?",
                "expected": r"(3\.4\s*GB|850\s*MB)",
                "max_tokens": 64,
                "turn": 1,
            },
            {
                "task_id": "64k-turn2-cache-hit",
                "category": "memory_scaling",
                "workload": "large_context_cache",
                "system": sys_prompt,
                "context": doc_segment,
                "prompt": "According to the section, what does prefix prompt caching guarantee for repetitive context prefixes?",
                "expected": r"(bypassed|guarantee|prefix)",
                "max_tokens": 64,
                "turn": 2,
            },
        ]
    elif context_name == "128k":
        # 128k context test
        sys_prompt = "You are a senior systems engineer specializing in extreme context MoE runtime benchmarks."
        spec_segment = (
            "Protocol 128K-Serving: The 131,072 token context window operates under strict host-RAM bounds.\n"
            "Hardware constraint: 15.29 GiB physical RAM available. Host swap thrashing must be prevented.\n"
            "Atlas Subprocess backend allocates Q4_0 KV cache tensors on GPU and locks the prompt checkpoint budget at 1024 MB.\n"
            "Evaluation throughput metrics: decode TPS targets 10+ TPS; prefill reuse provides up to 100x TTFT reduction.\n"
        )
        return [
            {
                "task_id": "128k-turn1-eval",
                "category": "extreme_context",
                "workload": "ultra_context_cache",
                "system": sys_prompt,
                "context": spec_segment,
                "prompt": "What is the prompt checkpoint budget locked at in Protocol 128K-Serving?",
                "expected": r"(1024\s*MB|1\s*GB)",
                "max_tokens": 48,
                "turn": 1,
            },
            {
                "task_id": "128k-turn2-cache-hit",
                "category": "extreme_context",
                "workload": "ultra_context_cache",
                "system": sys_prompt,
                "context": spec_segment,
                "prompt": "What decode TPS does Protocol 128K-Serving target and what speedup does prefill reuse provide?",
                "expected": r"(10\+|100x)",
                "max_tokens": 64,
                "turn": 2,
            },
        ]
    else:
        raise ValueError(f"Unknown context tier: {context_name}")


async def execute_task(
    backend: AtlasSubprocessBackend,
    task: Dict[str, Any],
    context_size: int,
    kv_quant: str,
    prior_prompt: Optional[str] = None,
    prior_response: Optional[str] = None,
    repetition: int = 0,
) -> Tuple[BenchmarkItemResult, str, str]:
    """Execute a single task and return rich extracted metrics, AI output, and prompt context."""
    system_msg = task.get("system")
    context_body = task.get("context")
    user_query = task["prompt"]

    if prior_prompt and prior_response is not None:
        # Guarantee exact prefix matching for C++ prompt cache reuse
        resp_prefix = prior_response
        if not resp_prefix.rstrip().endswith("<|im_end|>"):
            resp_prefix = resp_prefix.rstrip() + "<|im_end|>"
        else:
            resp_prefix = resp_prefix.rstrip()
        formatted_prompt = (
            f"{prior_prompt}{resp_prefix}\n"
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    else:
        messages: List[ChatMessage] = []
        if system_msg:
            messages.append(ChatMessage(role="system", content=system_msg))
        if context_body:
            messages.append(ChatMessage(role="user", content=f"Context Document:\n{context_body}\n\nQuestion: {user_query}"))
        else:
            messages.append(ChatMessage(role="user", content=user_query))
        formatted_prompt = format_chatml_prompt(messages)

    t_start = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    token_pieces: List[str] = []
    token_ids: List[int] = []
    done_event: Dict[str, Any] = {}

    async for ev in backend.generate(
        prompt=formatted_prompt,
        n_predict=task.get("max_tokens", 48),
        temp=0.0,
    ):
        ev_type = ev.get("event")
        if ev_type == "token":
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            token_pieces.append(ev.get("token", ""))
            if "token_id" in ev:
                token_ids.append(ev["token_id"])
        elif ev_type == "done":
            done_event = ev
        elif ev_type == "error":
            raise RuntimeError(f"Engine error during generation: {ev.get('message')}")

    t_end = time.perf_counter()
    total_elapsed_ms = (t_end - t_start) * 1000.0

    raw_output = done_event.get("text") or "".join(token_pieces)
    generated_text = raw_output.strip()

    num_tokens = len(token_pieces)
    ttft_ms = ((first_token_time - t_start) * 1000.0) if first_token_time else done_event.get("ttft_ms", 0.0)
    decode_tps = (
        ((num_tokens - 1) / (last_token_time - first_token_time))
        if (num_tokens > 1 and last_token_time and first_token_time and last_token_time > first_token_time)
        else 0.0
    )
    effective_tps = (num_tokens / (total_elapsed_ms / 1000.0)) if total_elapsed_ms > 0 else 0.0

    cached_prompt_tokens = done_event.get("cached_prompt_tokens", 0)
    prompt_tokens = done_event.get("prompt_tokens", 0)
    completion_tokens = done_event.get("completion_tokens", num_tokens)

    # Check accuracy if expected pattern provided
    expected = task.get("expected")
    task_pass = None
    if expected:
        task_pass = bool(re.search(expected, generated_text, re.IGNORECASE))

    item = BenchmarkItemResult(
        task_id=task["task_id"],
        category=task.get("category", "general"),
        workload=task.get("workload", "benchmark"),
        context_size=context_size,
        prompt=formatted_prompt,
        generated_output=generated_text,
        expected=expected,
        task_pass=task_pass,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        prompt_cache_bytes=done_event.get("prompt_cache_bytes", 0),
        cache_hit=(cached_prompt_tokens > 0),
        ttft_ms=round(ttft_ms, 2),
        decode_tps=round(decode_tps, 2),
        effective_tps=round(effective_tps, 2),
        prefill_compute_ms=round(done_event.get("prefill_compute_ms", 0.0), 2),
        prefill_io_ms=round(done_event.get("prefill_io_ms", 0.0), 2),
        gpu_compute_ms=round(done_event.get("gpu_compute_ms", 0.0), 2),
        checkpoint_ms=round(done_event.get("checkpoint_ms", 0.0), 2),
        elapsed_ms=round(total_elapsed_ms, 2),
        finish_reason=done_event.get("finish_reason", "stop"),
        repetition=repetition,
        turn=task.get("turn", 0),
        kv_quant=kv_quant,
    )
    return item, formatted_prompt, raw_output


async def run_suite_for_context(
    context_name: str,
    cfg: Dict[str, Any],
    threads: int = 8,
    cache_mb: int = 1024,
    exe_path: str = DEFAULT_EXE_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
) -> List[BenchmarkItemResult]:
    """Launch engine with configured context size and run its task set."""
    ctx_size = cfg["ctx_size"]
    kv_quant = cfg["kv_quant"]
    tasks = build_suite_tasks(context_name)

    print(
        f"\n>>> Launching Engine for Context: {context_name.upper()} "
        f"(ctx_size={ctx_size}, kv_quant={kv_quant}, cache_mb={cache_mb}, threads={threads}) ...",
        flush=True,
    )

    backend = AtlasSubprocessBackend(
        exe_path=exe_path,
        model_path=model_path,
        threads=threads,
        ctx_size=ctx_size,
        boost=True,
        prompt_cache_mb=cache_mb,
        cache_type_k=kv_quant,
        cache_type_v=kv_quant,
    )

    results: List[BenchmarkItemResult] = []
    await backend.start()
    try:
        cur_prompt: Optional[str] = None
        cur_response: Optional[str] = None
        for task in tasks:
            turn = task.get("turn", 0)
            if turn <= 1:
                cur_prompt = None
                cur_response = None
            print(f"  -> Executing task: {task['task_id']} (turn {turn}) ...", end="", flush=True)
            res, cur_prompt, cur_response = await execute_task(
                backend,
                task,
                context_size=ctx_size,
                kv_quant=kv_quant,
                prior_prompt=cur_prompt,
                prior_response=cur_response,
            )
            results.append(res)
            cache_info = f"CACHE HIT ({res.cached_prompt_tokens} toks)" if res.cache_hit else "COLD"
            pass_info = f"PASS ({res.expected})" if res.task_pass else f"GOT '{res.generated_output[:30]}...'"
            print(
                f" Done! [TTFT: {res.ttft_ms:.1f}ms | Decode: {res.decode_tps:.2f} TPS | "
                f"{cache_info} | {pass_info}]",
                flush=True,
            )
    finally:
        await backend.stop()

    return results


async def run_continuous_benchmark(
    contexts: List[str],
    threads: int = 8,
    cache_mb: int = 1024,
    exe_path: str = DEFAULT_EXE_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    report_md: Optional[Path] = None,
    report_jsonl: Optional[Path] = None,
    notes: str = "",
) -> BenchmarkRunSummary:
    """Execute the benchmark across all selected contexts and append to reports."""
    run_id = f"bench-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    timestamp = datetime.datetime.now().isoformat()
    t_suite_start = time.perf_counter()

    print(f"================================================================================")
    print(f"  TENDOU LAUNCHER — CONTINUOUS BENCHMARK RUN: {run_id}")
    print(f"  Selected Contexts: {', '.join(contexts)}")
    print(f"  Threads: {threads} | Cache Budget: {cache_mb} MB")
    print(f"================================================================================")

    all_items: List[BenchmarkItemResult] = []
    for ctx_raw in contexts:
        ctx_name = ctx_raw.strip().lower()
        if ctx_name == "2k":
            ctx_name = "short"
        if ctx_name not in CONTEXT_CONFIGS:
            print(f"Warning: Unknown context '{ctx_raw}', skipping.", file=sys.stderr)
            continue
        cfg = CONTEXT_CONFIGS[ctx_name]
        items = await run_suite_for_context(
            context_name=ctx_name,
            cfg=cfg,
            threads=threads,
            cache_mb=cache_mb,
            exe_path=exe_path,
            model_path=model_path,
        )
        all_items.extend(items)

    total_duration_s = time.perf_counter() - t_suite_start
    hw_info = detect_hardware()
    model_name = Path(model_path).stem or "Qwen3.8-Flash-Next-Uncensored-Q5_K_S"

    summary = BenchmarkRunSummary(
        run_id=run_id,
        timestamp=timestamp,
        model_name=model_name,
        threads=threads,
        cache_mb=cache_mb,
        hardware_info=hw_info,
        items=all_items,
        total_duration_s=total_duration_s,
        notes=notes,
    )

    # Append to continuous log files
    logger = BenchmarkLogger(report_md_path=report_md, report_jsonl_path=report_jsonl)
    logger.append_run(summary)

    print(f"\n================================================================================")
    print(f"  BENCHMARK COMPLETE ({total_duration_s:.2f}s total)")
    print(f"  Appended to Markdown Report:  {logger.report_md_path}")
    print(f"  Appended to Structured JSONL: {logger.report_jsonl_path}")
    print(f"================================================================================")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous benchmark suite for tendou launcher.")
    parser.add_argument(
        "--contexts",
        type=str,
        default="short,32k,64k,128k",
        help="Comma-separated context tiers to benchmark (e.g. 'short,32k,64k,128k' or 'all')",
    )
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads (default: 8)")
    parser.add_argument("--cache-mb", type=int, default=1024, help="Prompt cache budget in MB (default: 1024)")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to GGUF model file")
    parser.add_argument("--exe-path", type=str, default=DEFAULT_EXE_PATH, help="Path to llama-atlas-engine.exe")
    parser.add_argument("--report-md", type=Path, default=None, help="Path to markdown report file")
    parser.add_argument("--report-jsonl", type=Path, default=None, help="Path to structured jsonl report file")
    parser.add_argument("--notes", type=str, default="", help="Optional notes to include in the report")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.contexts.strip().lower() == "all":
        ctx_list = ["short", "32k", "64k", "128k"]
    else:
        ctx_list = [c.strip().lower() for c in args.contexts.split(",") if c.strip()]

    asyncio.run(
        run_continuous_benchmark(
            contexts=ctx_list,
            threads=args.threads,
            cache_mb=args.cache_mb,
            exe_path=args.exe_path,
            model_path=args.model_path,
            report_md=args.report_md,
            report_jsonl=args.report_jsonl,
            notes=args.notes,
        )
    )


if __name__ == "__main__":
    main()
