import asyncio
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.p0_benchmark import build_quality_tasks
from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt

async def run_slice(
    k: int = 5,
    threads: int = 8,
    p2: bool = False,
    p3: bool = False,
    aggressive: bool = False,
    dyn_k: float = 0.55,
    dyn_min: int = 1,
    dyn_max: int = 2,
    min_w: float = 0.10,
    layer_adapt: bool = True,
    readback_interval: int = 64,
    dyn_end: int = 47,
):
    all_tasks = build_quality_tasks()
    cats = {}
    selected_tasks = []
    for t in all_tasks:
        c = t["category"]
        if cats.get(c, 0) < 2:
            cats[c] = cats.get(c, 0) + 1
            selected_tasks.append(t)

    print(f"\n================ Running Quality Slice (10 tasks) with K={k}, threads={threads}, P2={p2}, P3={p3}, Aggressive={aggressive}, DynK={dyn_k}, DynMin={dyn_min}, DynMax={dyn_max}, MinW={min_w}, LayerAdapt={layer_adapt}, RB={readback_interval}, DynEnd={dyn_end} ================", flush=True)
    extra = ["--atlas-k", str(k)] if k > 0 else ["--atlas-k", "0"]
    if dyn_k > 0.0:
        extra.extend(["--atlas-dynamic-k", str(dyn_k)])
    if dyn_min > 0:
        extra.extend(["--atlas-dynamic-k-min", str(dyn_min)])
    if dyn_max > 0:
        extra.extend(["--atlas-dynamic-k-max", str(dyn_max)])
    if min_w > 0.0:
        extra.extend(["--atlas-dynamic-k-min-weight", str(min_w)])
    if dyn_end > 0:
        extra.extend(["--atlas-dynamic-k-end", str(dyn_end)])
    if layer_adapt:
        extra.extend(["--atlas-layer-adapt", "1"])
    else:
        extra.extend(["--atlas-layer-adapt", "0"])

    backend = AtlasSubprocessBackend(
        threads=threads,
        ctx_size=2048,
        odmoe_lead=4 if aggressive else 2,
        spice_conf_high=0.50 if aggressive else 0.65,
        spice_conf_mid=0.15 if aggressive else 0.30,
        tutti_async_io=aggressive,
        readback_interval=readback_interval,
        prefetch_candidates=16 if aggressive else 10,
        tutti_queue_depth=128 if aggressive else 64,
        p2_gpu_binding=p2,
        p2_vram_budget_mib=1600 if p2 else 0,
        p3_async_transfer=p3,
        p3_staging_slots=4 if p3 else 0,
        p3_queue_depth=8 if p3 else 0,
        extra_args=extra,
    )
    await backend.start()
    try:
        passed = 0
        total_decode_tps = []
        for t in selected_tasks:
            prompt = format_chatml_prompt([ChatMessage(role="user", content=t["prompt"])])
            t0 = time.perf_counter()
            first = None
            last = None
            tokens = []
            async for ev in backend.generate(prompt, n_predict=t["max_tokens"], temp=0):
                if ev.get("event") == "token":
                    last = time.perf_counter()
                    first = first or last
                    tokens.append(ev.get("token", ""))
            t1 = time.perf_counter()
            ans = "".join(tokens).strip()
            m = re.search(t["expected"], ans, re.IGNORECASE)
            is_ok = bool(m)
            if is_ok:
                passed += 1
            tps = (len(tokens) - 1) / (last - first) if (last and first and last > first and len(tokens) > 1) else 0.0
            if tps > 0:
                total_decode_tps.append(tps)
            print(f"[{t['category']:11s}] Task: {t['task_id']} | Pass: {str(is_ok):5s} | Expected: {t['expected']:12s} | Got: '{ans}' | TPS: {tps:.2f}", flush=True)
        acc = passed / len(selected_tasks) * 100.0
        avg_tps = sum(total_decode_tps) / len(total_decode_tps) if total_decode_tps else 0.0
        print(f"\n---> Slice Accuracy: {acc:.1f}% ({passed}/{len(selected_tasks)}) | Average Decode TPS: {avg_tps:.2f}", flush=True)
        return acc, avg_tps
    finally:
        await backend.stop()

if __name__ == "__main__":
    k_val = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    th_val = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    p2_flag = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    p3_flag = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    agg_flag = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    dyn_val = float(sys.argv[6]) if len(sys.argv) > 6 else 0.55
    dyn_min_val = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    dyn_max_val = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    min_w_val = float(sys.argv[9]) if len(sys.argv) > 9 else 0.10
    layer_adapt_val = bool(int(sys.argv[10])) if len(sys.argv) > 10 else True
    rb_val = int(sys.argv[11]) if len(sys.argv) > 11 else 64
    dyn_end_val = int(sys.argv[12]) if len(sys.argv) > 12 else 47
    asyncio.run(run_slice(k_val, th_val, p2_flag, p3_flag, agg_flag, dyn_val, dyn_min_val, dyn_max_val, min_w_val, layer_adapt_val, rb_val, dyn_end_val))
