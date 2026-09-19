"""In-process fast sweep of physical simulator parameters."""
import json
from pathlib import Path
from atlas_physical_simulator import PhysicalSim, load_sessions

pm = json.loads(Path("atlas_physical_map.json").read_text(encoding="utf-8"))
sessions = load_sessions("trace_35b_a3b_q2k.jsonl")

vram_gb = 7.0
ram_gb = 14.0
compute_ms = 50.0  # 20 TPS

candidates = [1, 2, 4, 8, 12, 16, 24, 32]
confidence_floors = [0.05, 0.10, 0.15, 0.20, 0.25]
budget_scales = [1.0, 1.05, 1.10, 1.20]
source_top_ns = [8, 16, 32]

results = []

# First baseline
sim_none = PhysicalSim(pm, vram_gb, ram_gb, 7.0, 12.0, 40.0, 0.08, 0.02, compute_ms)
r_none = sim_none.run(sessions, policy="none", predictor_kwargs={"preload_experts_to_ram": True})
print(f"BASELINE (none): TPS={r_none['effective_tps']:.3f} exposed={r_none['exposed_io_ms_mean']:.3f}ms P95={r_none['p95_ms']:.3f}ms")

# Oracle
sim_ora = PhysicalSim(pm, vram_gb, ram_gb, 7.0, 12.0, 40.0, 0.08, 0.02, compute_ms)
r_ora = sim_ora.run(sessions, policy="oracle", predictor_kwargs={"preload_experts_to_ram": True})
print(f"ORACLE:          TPS={r_ora['effective_tps']:.3f} exposed={r_ora['exposed_io_ms_mean']:.3f}ms P95={r_ora['p95_ms']:.3f}ms")

print("\nRunning in-process parameter sweep...")
for c in candidates:
    for f in confidence_floors:
        for b in budget_scales:
            for s in [32]:
                sim = PhysicalSim(pm, vram_gb, ram_gb, 7.0, 12.0, 40.0, 0.08, 0.02, compute_ms)
                kwargs = {
                    "confidence_floor": f,
                    "budget_scale": b,
                    "history_window": 4,
                    "max_candidates_per_layer": c,
                    "source_top_n": s,
                    "min_count": 2,
                    "persistent_fallback": True,
                    "preload_experts_to_ram": True,
                }
                r = sim.run(sessions, policy="atlas_predictor", predictor_kwargs=kwargs)
                res = {
                    "tps": r["effective_tps"],
                    "exposed_ms": r["exposed_io_ms_mean"],
                    "p95": r["p95_ms"],
                    "p99": r["p99_ms"],
                    "precision": r["prefetch_precision"],
                    "predicted": r["predicted_experts"],
                    "correct": r["correct_prefetch"],
                    "cand": c,
                    "floor": f,
                    "budget": b,
                    "source_n": s,
                }
                results.append(res)

results.sort(key=lambda x: (-x["tps"], x["exposed_ms"]))

print("\n" + "=" * 80)
print(f"{'Rank':<5} {'Cand':>5} {'Floor':>6} {'Budget':>7} {'TPS':>8} {'Exposed ms':>12} {'P95 ms':>10} {'Prec%':>8} {'Predicted':>10} {'Hits':>8}")
print("-" * 88)
for i, r in enumerate(results[:20], 1):
    print(
        f"{i:<5} {r['cand']:>5} {r['floor']:>6.2f} {r['budget']:>7.2f} "
        f"{r['tps']:>8.3f} {r['exposed_ms']:>12.3f} {r['p95']:>10.3f} "
        f"{r['precision']*100:>7.1f}% {r['predicted']:>10} {r['correct']:>8}"
    )

# Save best config
Path("physical_sweep_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\n[+] Results saved to physical_sweep_results.json")
