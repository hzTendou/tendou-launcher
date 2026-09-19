"""Comprehensive V7 Benchmark Runner.

Evaluates all key hypotheses:
1. Baseline (No preload, LRU)
2. Preload RAM (Eliminates NVMe cold acquisition)
3. Frequency-biased VRAM (Hot-tier retention vs pure LRU)
4. Atlas Predictor with candidate tuning (1, 4, 8)
5. Deadline-aware budget capping
6. Oracle upper bounds
"""
import json
import time
from pathlib import Path
from atlas_simulator import load_sessions, evaluate_policy, build_affinity_regions
from atlas_physical_simulator import PhysicalSim, load_sessions as load_phys_sessions


def run_logical_comparisons():
    print("=" * 96)
    print("LOGICAL SIMULATOR ABLATION (trace_35b_a3b_q2k.jsonl, 20 TPS, 7GB VRAM / 14GB RAM)")
    print("=" * 96)
    
    sessions = load_sessions("trace_35b_a3b_q2k.jsonl")
    vram_mb = 7000.0
    ram_gb = 14.0
    compute_ms = 50.0  # 20 TPS
    
    configs = [
        # label, mode, preload, vpolicy, daware, horizon, step2_frac, pkwargs
        ("1. Baseline (no preload, LRU)", "none", False, "lru", False, 1, 0.0, None),
        ("2. RAM Preload (none, LRU)", "none", True, "lru", False, 1, 0.0, None),
        # --- 1-step atlas_predictor baselines ---
        ("3. 1-step cand=1 (preload)", "atlas_predictor", True, "lru", False, 1, 0.0,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 1, "persistent_fallback": True}),
        ("4. 1-step cand=4 (preload)", "atlas_predictor", True, "lru", False, 1, 0.0,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 4, "persistent_fallback": True}),
        ("5. 1-step cand=8 (preload)", "atlas_predictor", True, "lru", False, 1, 0.0,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 8, "persistent_fallback": True}),
        # --- Experiment F: 2-step-ahead prefetch ---
        ("6. 2-step cand=1+1 frac=1.0", "atlas_predictor", True, "lru", False, 2, 1.0,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 1, "persistent_fallback": True}),
        ("7. 2-step cand=4+2 frac=0.5", "atlas_predictor", True, "lru", False, 2, 0.5,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 4, "persistent_fallback": True}),
        ("8. 2-step cand=4+4 frac=1.0", "atlas_predictor", True, "lru", False, 2, 1.0,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 4, "persistent_fallback": True}),
        ("9. 2-step cand=8+4 frac=0.5", "atlas_predictor", True, "lru", False, 2, 0.5,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 8, "persistent_fallback": True}),
        # --- Oracle upper bound ---
        ("10. Oracle (upper bound)", "oracle", True, "lru", False, 1, 0.0, None),
    ]
    
    hdr = f"{'Configuration':<42} {'Expose ms':>10} {'P95 ms':>8} {'P99 ms':>8} {'TPS':>7} {'VHit%':>7} {'S1 pred':>8} {'S2 pred':>8} {'S2 prec':>8}"
    print(hdr)
    print("-" * len(hdr))
    
    logical_results = []
    for label, mode, preload, vpolicy, daware, horizon, step2_frac, pkwargs in configs:
        t0 = time.time()
        r = evaluate_policy(
            sessions, mode, vram_mb, 1.2, ram_gb, 7.0, 12.0, 40.0,
            0.08, 0.02, compute_ms, horizon, 0.75,
            predictor_kwargs=pkwargs,
            preload_ram=preload,
            vram_policy=vpolicy,
            deadline_aware=daware,
            step2_budget_fraction=step2_frac,
        )
        vhits = r['vram_hits']
        vtot = r['tokens'] * 320
        vhit_pct = (vhits / vtot * 100.0) if vtot else 0.0
        
        # Calculate pure decode TPS (exclude preload time)
        decode_active_s = (r['tokens'] * compute_ms + r['total_exposed_io_ms']) / 1000.0
        tps = r['tokens'] / decode_active_s if decode_active_s > 0 else 0.0
        
        s2prec = r.get('step2_precision', 0.0)
        s2cnt = r.get('step2_prefetch_count', 0)
        s1cnt = r.get('step1_prefetch_count', 0)
        
        print(f"{label:<42} {r['exposed_io_ms_mean']:>10.3f} {r['exposed_io_ms_p95']:>8.3f} "
              f"{r['exposed_io_ms_p99']:>8.3f} {tps:>7.2f} {vhit_pct:>6.1f}% "
              f"{s1cnt:>8} {s2cnt:>8} {s2prec*100:>7.1f}%")
        logical_results.append({
            "label": label,
            "mode": mode,
            "preload": preload,
            "vram_policy": vpolicy,
            "deadline_aware": daware,
            "prefetch_horizon": horizon,
            "step2_budget_fraction": step2_frac,
            "exposed_mean": r['exposed_io_ms_mean'],
            "p95": r['exposed_io_ms_p95'],
            "p99": r['exposed_io_ms_p99'],
            "tps": tps,
            "vram_hit_pct": vhit_pct,
            "nvme_mb": r['nvme_mb'],
            "ram_to_vram_mb": r['ram_to_vram_mb'],
            "vram_evictions": r['vram_evictions'],
            "prefetch_waits": r['prefetch_waits'],
            "precision": r['prefetch_precision'],
            "recall": r['prefetch_recall'],
            "step1_prefetch_count": s1cnt,
            "step2_prefetch_count": s2cnt,
            "step2_useful": r.get('step2_useful_prefetch', 0),
            "step2_wasted": r.get('step2_wasted_prefetch', 0),
            "step2_precision": s2prec,
        })
    
    return logical_results


def run_physical_comparisons():
    print("\n" + "=" * 88)
    print("PHYSICAL SIMULATOR ABLATION (exact GGUF tensor slices, 7GB VRAM / 14GB RAM)")
    print("=" * 88)
    
    pm = json.loads(Path("atlas_physical_map.json").read_text(encoding="utf-8"))
    sessions = load_phys_sessions("trace_35b_a3b_q2k.jsonl")
    vram_gb = 7.0
    ram_gb = 14.0
    compute_ms = 50.0  # 20 TPS
    
    phys_configs = [
        ("1. Physical Baseline (no preload)", "none", False, None),
        ("2. Physical RAM Preload (none)", "none", True, None),
        ("3. 1-step cand=1 (preload)", "atlas_predictor", True,
         {"confidence_floor": 0.10, "budget_scale": 1.00, "history_window": 4, "max_candidates_per_layer": 1, "source_top_n": 32, "min_count": 2, "persistent_fallback": True, "preload_experts_to_ram": True, "prefetch_horizon": 1}),
        ("4. 1-step cand=4 (preload)", "atlas_predictor", True,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 4, "source_top_n": 32, "min_count": 2, "persistent_fallback": True, "preload_experts_to_ram": True, "prefetch_horizon": 1}),
        ("5. 1-step cand=8 (preload)", "atlas_predictor", True,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 8, "source_top_n": 32, "min_count": 2, "persistent_fallback": True, "preload_experts_to_ram": True, "prefetch_horizon": 1}),
        ("6. 2-step cand=1+1 frac=1.0", "atlas_predictor", True,
         {"confidence_floor": 0.10, "budget_scale": 1.00, "history_window": 4, "max_candidates_per_layer": 1, "source_top_n": 32, "min_count": 2, "persistent_fallback": True, "preload_experts_to_ram": True, "prefetch_horizon": 2, "step2_budget_fraction": 1.0}),
        ("7. 2-step cand=4+2 frac=0.5", "atlas_predictor", True,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 4, "source_top_n": 32, "min_count": 2, "persistent_fallback": True, "preload_experts_to_ram": True, "prefetch_horizon": 2, "step2_budget_fraction": 0.5}),
        ("8. 2-step cand=4+4 frac=1.0", "atlas_predictor", True,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 4, "source_top_n": 32, "min_count": 2, "persistent_fallback": True, "preload_experts_to_ram": True, "prefetch_horizon": 2, "step2_budget_fraction": 1.0}),
        ("9. 2-step cand=8+4 frac=0.5", "atlas_predictor", True,
         {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4, "max_candidates_per_layer": 8, "source_top_n": 32, "min_count": 2, "persistent_fallback": True, "preload_experts_to_ram": True, "prefetch_horizon": 2, "step2_budget_fraction": 0.5}),
        ("10. Physical Oracle (preload)", "oracle", True, {"preload_experts_to_ram": True}),
    ]
    
    print(f"{'Configuration':<46} {'Exposed ms':>11} {'P95 ms':>9} {'P99 ms':>9} {'TPS':>8} {'Prec%':>8} {'Hits':>8}")
    print("-" * 96)
    
    phys_results = []
    for label, policy, preload, pkwargs in phys_configs:
        sim = PhysicalSim(pm, vram_gb, ram_gb, 7.0, 12.0, 40.0, 0.08, 0.02, compute_ms)
        r = sim.run(sessions, policy=policy, predictor_kwargs=pkwargs or ({"preload_experts_to_ram": True} if preload else {}))
        
        exposed_mean = r['exposed_io_ms_mean']
        p95 = r['p95_ms']
        p99 = r['p99_ms']
        tps = r['effective_tps']
        prec = r['prefetch_precision'] * 100.0
        hits = r['correct_prefetch']
        
        print(f"{label:<46} {exposed_mean:>11.3f} {p95:>9.3f} {p99:>9.3f} {tps:>8.2f} {prec:>7.1f}% {hits:>8}")
        phys_results.append({
            "label": label,
            "policy": policy,
            "exposed_mean": exposed_mean,
            "p95": p95,
            "p99": p99,
            "tps": tps,
            "precision": r['prefetch_precision'],
            "predicted": r['predicted_experts'],
            "correct": r['correct_prefetch'],
            "nvme_read_mb": r['nvme_read_mb'],
            "pcie_transfer_mb": r['pcie_transfer_mb'],
            "prefetch_waits": r['prefetch_waits'],
        })
    
    return phys_results


if __name__ == "__main__":
    log_res = run_logical_comparisons()
    phys_res = run_physical_comparisons()
    
    full_report = {
        "logical": log_res,
        "physical": phys_res,
    }
    Path("v7_benchmark_report.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print("\n[+] Full report written to v7_benchmark_report.json")
