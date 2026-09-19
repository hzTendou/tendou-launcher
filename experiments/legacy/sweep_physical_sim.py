import subprocess
import json

candidates = [1, 2, 4, 8, 12, 16, 24]
confidence_floors = [0.05, 0.10, 0.15, 0.20]
budget_scales = [1.0, 1.05, 1.10, 1.20]

results = []

for c in candidates:
    for f in [0.05, 0.10, 0.15, 0.20]:
        for b in [1.0, 1.05, 1.10]:
            cmd = f"python atlas_physical_simulator.py --trace trace_35b_a3b_q2k.jsonl --physical-map atlas_physical_map.json --target-tps 20 --predictor-persistent-fallback --preload-experts-to-ram --predictor-max-candidates {c} --predictor-confidence-floor {f} --predictor-budget-scale {b}"
            out = subprocess.check_output(cmd, shell=True, text=True)
            for line in out.strip().split("\n"):
                line = line.strip()
                if line.startswith("{") and '"policy": "atlas_predictor"' in line:
                    d = json.loads(line)
                    res = {
                        "tps": d["effective_tps"],
                        "exposed_ms": d["exposed_io_ms_mean"],
                        "p95": d["p95_ms"],
                        "precision": d["prefetch_precision"],
                        "predicted": d["predicted_experts"],
                        "correct": d["correct_prefetch"],
                        "cand": c,
                        "floor": f,
                        "budget": b,
                    }
                    results.append(res)
                    print(
                        f"cand={c:2d} floor={f:.2f} budget={b:.2f} -> "
                        f"TPS={d['effective_tps']:.3f} exposed={d['exposed_io_ms_mean']:.3f}ms "
                        f"P95={d['p95_ms']:.3f}ms prec={d['prefetch_precision']*100:.1f}% "
                        f"pred={d['predicted_experts']} hit={d['correct_prefetch']}"
                    )

results.sort(key=lambda x: (-x["tps"], x["exposed_ms"]))
print("\n--- TOP 10 ---")
for r in results[:10]:
    print(r)
