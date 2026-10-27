import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import AtlasSubprocessBackend, DEFAULT_EXE_PATH, DEFAULT_MODEL_PATH, ChatMessage, format_chatml_prompt

def get_tasks():
    long_context = "Tendou yerel bir MoE calistiricisidir. Router expert secimini degistirmez. " * 120
    return [
        {"task_id": "short", "workload": "short_request", "prompt": "17 kere 19 kac eder?", "max_tokens": 16},
        {"task_id": "long", "workload": "long_prompt", "prompt": long_context + " Metindeki sistemin adini yaz.", "max_tokens": 24},
        {"task_id": "multi", "workload": "multi_turn", "turns": [
            {"prompt": "Benim tuttugum sayi 7."},
            {"prompt": "Sayinin iki katini yaz."},
            {"prompt": "Bir fazlasini yaz."}
        ], "max_tokens": 16},
    ]

async def run(args):
    tasks = get_tasks()
    task_filter = getattr(args, "task", None)
    if isinstance(task_filter, str) and task_filter:
        selected = set(s.strip() for s in task_filter.split(","))
        tasks = [t for t in tasks if t["task_id"] in selected or t["workload"] in selected]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backend = AtlasSubprocessBackend(
        exe_path=str(args.exe), model_path=str(args.model),
        threads=args.threads, ctx_size=4096,
        p2_gpu_binding=bool(getattr(args, "p2_gpu_binding", False) is True),
        p2_vram_budget_mib=int(getattr(args, "p2_vram_budget_mib", 0)) if isinstance(getattr(args, "p2_vram_budget_mib", 0), (int, float)) else 0,
        p3_async_transfer=bool(getattr(args, "p3_async_transfer", False) is True),
        extra_args=["--atlas-k", "0", "--atlas-prompt-cache-mb", "0"]
    )
    
    results = []
    try:
        await backend.start()
        
        for task in tasks:
            for rep in range(args.repetitions):
                turns = task.get("turns", [{"prompt": task.get("prompt")}])
                messages = []
                for turn_idx, turn in enumerate(turns):
                    messages.append(ChatMessage(role="user", content=turn["prompt"]))
                    prompt = format_chatml_prompt(messages)
                    
                    start_time = time.perf_counter()
                    first_token_time = None
                    done_event = None
                    tokens_generated = 0
                    full_response = ""
                    
                    async for event in backend.generate(prompt, n_predict=task["max_tokens"], temp=0):
                        if event["event"] == "token":
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            tokens_generated += 1
                            full_response += event.get("token", "")
                        elif event["event"] == "done":
                            done_event = event
                            if "text" in event and event["text"]:
                                full_response = event["text"]
                        elif event["event"] == "error":
                            raise RuntimeError(f"Error: {event}")
                            
                    end_time = time.perf_counter()
                    messages.append(ChatMessage(role="assistant", content=full_response))
                    
                    if done_event:
                        wall_ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else (end_time - start_time) * 1000
                        wall_total_ms = (end_time - start_time) * 1000
                        decode_tps = ((tokens_generated - 1) / ((wall_total_ms - wall_ttft_ms) / 1000)) if (tokens_generated > 1 and wall_total_ms > wall_ttft_ms) else 0.0
                        
                        def get_metric(name):
                            if name not in done_event or done_event[name] is None:
                                return "NOT_MEASURED"
                            return done_event[name]

                        results.append({
                            "task_id": task["task_id"],
                            "workload": task["workload"],
                            "repetition": rep,
                            "turn": turn_idx,
                            "wall_ttft_ms": wall_ttft_ms,
                            "wall_total_ms": wall_total_ms,
                            "decode_tps": decode_tps,
                            "ttft_ms": get_metric("ttft_ms"),
                            "prefill_io_ms": get_metric("prefill_io_ms"),
                            "prefill_compute_ms": get_metric("prefill_compute_ms"),
                            "checkpoint_ms": get_metric("checkpoint_ms"),
                            "gpu_compute_ms": get_metric("gpu_compute_ms"),
                        })
    finally:
        await backend.stop()
        
    # Calculate medians for each turn
    import statistics
    grouped = {}
    for r in results:
        key = (r["task_id"], r["turn"])
        if key not in grouped: grouped[key] = []
        grouped[key].append(r)
        
    medians = []
    for (task_id, turn), group in grouped.items():
        median_entry = {
            "task_id": task_id,
            "workload": group[0]["workload"],
            "turn": turn,
            "wall_ttft_ms": statistics.median([g["wall_ttft_ms"] for g in group]),
            "wall_total_ms": statistics.median([g["wall_total_ms"] for g in group]),
            "decode_tps": statistics.median([g["decode_tps"] for g in group])
        }
        for k in ["ttft_ms", "prefill_io_ms", "prefill_compute_ms", "checkpoint_ms", "gpu_compute_ms"]:
            vals = [g[k] for g in group if g[k] != "NOT_MEASURED"]
            median_entry[k] = statistics.median(vals) if vals else "NOT_MEASURED"
        medians.append(median_entry)
        
    out_file = output_dir / f"ttft-breakdown-{args.profile}.json"
    out_file.write_text(json.dumps(medians, indent=2), encoding="utf-8")
    
    if medians:
        for m in medians:
            print(f"Task {m['task_id']} Turn {m['turn']}: TTFT={m['wall_ttft_ms']:.2f}ms, TPS={m['decode_tps']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path(DEFAULT_EXE_PATH))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/p1"))
    parser.add_argument("--profile", type=str, default="baseline")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--task", type=str, default=None, help="Filter tasks by task_id or workload (comma-separated)")
    parser.add_argument("--p2-gpu-binding", action="store_true", default=False)
    parser.add_argument("--p2-vram-budget-mib", type=int, default=0)
    parser.add_argument("--p3-async-transfer", action="store_true", default=False)
    args = parser.parse_args()
    asyncio.run(run(args))
