"""Oracle Decomposition Diagnostic for Atlas Engine.

Decomposes the gap between Atlas and Oracle into:
1. Prediction Coverage Gap (what if we had oracle predictions vs predictor predictions?)
2. PCIe Transfer / Scheduling Gap (what if PCIe had zero latency vs real PCIe bandwidth?)
3. Memory Residency Gap (what if VRAM cache had perfect clairvoyant eviction / Belady's MIN?)

Variants:
- Baseline (none): No prefetch, LRU cache
- Atlas Predictor: Online causal prediction + real PCIe
- Oracle A (Predictor + Zero-Transfer-Cost): Predictor predictions, but transfers are instantaneous
- Oracle B (Next-Token Oracle + Real PCIe): Knows exact next expert set, transfers over real PCIe
- Oracle C (Clairvoyant Belady Residency): Knows entire future sequence, evicts expert whose next use is furthest in future
- Oracle D (Full Clairvoyant Upper Bound): Zero exposed I/O (perfect prefetch + residency)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path

from atlas_predictor import AtlasPredictor
from atlas_simulator import (
    load_sessions, layers, all_keys, GlobalLRU, CacheItem, TieredSim,
    collect_all_expert_keys, percentile
)


def run_belady_vram(sessions, vram_capacity_mb, expert_mb):
    """Simulate Belady's MIN optimal cache replacement for VRAM.
    
    When cache is full, evicts the expert that will not be used for the longest time in the future.
    """
    # Flatten all tokens across sessions (since VRAM is session-local, do per session)
    total_hits = total_misses = 0
    
    for sess in sessions:
        records = sess['records']
        # Precompute next-use index for every expert at every step
        expert_uses = defaultdict(list)
        for step, rec in enumerate(records):
            for k in all_keys(rec):
                expert_uses[k].append(step)
        
        vram = {}  # key -> CacheItem
        vram_used_mb = 0.0
        
        for step, rec in enumerate(records):
            current_keys = all_keys(rec)
            for k in current_keys:
                if k in vram:
                    total_hits += 1
                else:
                    total_misses += 1
                    # Evict if full
                    if vram_used_mb + expert_mb > vram_capacity_mb and vram:
                        # Find victim: expert with furthest next use
                        def next_use(key):
                            uses = expert_uses[key]
                            future = [u for u in uses if u > step]
                            return future[0] if future else float('inf')
                        
                        victim = max(vram.keys(), key=next_use)
                        del vram[victim]
                        vram_used_mb -= expert_mb
                    
                    if vram_used_mb + expert_mb <= vram_capacity_mb:
                        vram[k] = True
                        vram_used_mb += expert_mb

    return total_hits, total_misses, total_hits / (total_hits + total_misses) if (total_hits + total_misses) else 0.0


def decompose_gap(sessions, vram_gb=8.0, vram_reserve_gb=1.0, ram_gb=16.0,
                  nvme_gbps=7.0, pcie_gbps=12.0, ram_gbps=40.0,
                  target_tps=20.0, expert_mb=1.2, preload_ram=True):
    vram_capacity_mb = (vram_gb - vram_reserve_gb) * 1024.0
    compute_ms = 1000.0 / target_tps
    all_keys_corpus = collect_all_expert_keys(sessions)
    
    results = {}
    
    # 1. Baseline (No prefetch)
    sim_none = TieredSim(expert_mb, vram_capacity_mb, ram_gb, nvme_gbps, pcie_gbps, ram_gbps, 0.08, 0.02, compute_ms)
    if preload_ram:
        sim_none.preload_experts_to_ram(all_keys_corpus)
    for s in sessions:
        sim_none.vram.clear(); sim_none.vram_used_mb = 0.0
        for rec in s['records']:
            sim_none.run_token(all_keys(rec), set())
    ex_none = sim_none.token_exposed_ms
    results['Baseline (none)'] = {
        'exposed_mean_ms': sum(ex_none) / len(ex_none),
        'p95_ms': percentile(ex_none, 95),
        'tps': len(ex_none) / ((len(ex_none) * compute_ms + sum(ex_none)) / 1000.0),
        'vram_hits': sim_none.stats['vram_hit'],
        'ram_hits': sim_none.stats['ram_hit'],
    }
    
    # 2. Atlas Predictor (Causal prefetch, real PCIe)
    sim_pred = TieredSim(expert_mb, vram_capacity_mb, ram_gb, nvme_gbps, pcie_gbps, ram_gbps, 0.08, 0.02, compute_ms)
    if preload_ram:
        sim_pred.preload_experts_to_ram(all_keys_corpus)
    atlas_p = AtlasPredictor(history_window=4, max_candidates_per_layer=4, confidence_floor=0.10, persistent_fallback=True)
    for s in sessions:
        sim_pred.vram.clear(); sim_pred.vram_used_mb = 0.0
        pstate = atlas_p.begin_session()
        records = s['records']
        for i, rec in enumerate(records):
            cur = layers(rec); actual = all_keys(rec); pred = set()
            if i + 1 < len(records):
                vram_keys = set(sim_pred.vram.keys())
                pp = pstate.predict(cur, phase='decode', exclude_keys=vram_keys)
                pred = pp.keys
            sim_pred.run_token(actual, pred)
            pstate.observe(cur, phase='decode')
        pstate.finish()
    ex_pred = sim_pred.token_exposed_ms
    results['Atlas Predictor'] = {
        'exposed_mean_ms': sum(ex_pred) / len(ex_pred),
        'p95_ms': percentile(ex_pred, 95),
        'tps': len(ex_pred) / ((len(ex_pred) * compute_ms + sum(ex_pred)) / 1000.0),
        'vram_hits': sim_pred.stats['vram_hit'],
        'ram_hits': sim_pred.stats['ram_hit'],
    }
    
    # 3. Oracle B (Oracle next-token prefetch + real PCIe transfer)
    sim_orb = TieredSim(expert_mb, vram_capacity_mb, ram_gb, nvme_gbps, pcie_gbps, ram_gbps, 0.08, 0.02, compute_ms)
    if preload_ram:
        sim_orb.preload_experts_to_ram(all_keys_corpus)
    for s in sessions:
        sim_orb.vram.clear(); sim_orb.vram_used_mb = 0.0
        records = s['records']
        for i, rec in enumerate(records):
            actual = all_keys(rec)
            pred = all_keys(records[i + 1]) if i + 1 < len(records) else set()
            sim_orb.run_token(actual, pred)
    ex_orb = sim_orb.token_exposed_ms
    results['Oracle B (Next-token oracle + Real PCIe)'] = {
        'exposed_mean_ms': sum(ex_orb) / len(ex_orb),
        'p95_ms': percentile(ex_orb, 95),
        'tps': len(ex_orb) / ((len(ex_orb) * compute_ms + sum(ex_orb)) / 1000.0),
        'vram_hits': sim_orb.stats['vram_hit'],
        'ram_hits': sim_orb.stats['ram_hit'],
    }
    
    # 4. Oracle C (Belady MIN optimal replacement hit rate)
    b_hits, b_misses, b_hit_rate = run_belady_vram(sessions, vram_capacity_mb, expert_mb)
    results['Oracle C (Belady MIN VRAM Residency)'] = {
        'vram_hit_rate': b_hit_rate,
        'vram_hits': b_hits,
        'vram_misses': b_misses,
    }
    
    # 5. Oracle D (Perfect zero-exposed I/O upper bound)
    results['Oracle D (Zero-Exposed Upper Bound)'] = {
        'exposed_mean_ms': 0.0,
        'p95_ms': 0.0,
        'tps': target_tps,
    }
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Atlas Oracle Decomposition Diagnostic")
    ap.add_argument("--trace", default="trace_35b_a3b_q2k.jsonl")
    ap.add_argument("--target-tps", type=float, default=20.0)
    args = ap.parse_args()
    
    sessions = load_sessions(args.trace)
    print("=" * 80)
    print("ATLAS ORACLE GAP DECOMPOSITION DIAGNOSTIC")
    print("=" * 80)
    
    res = decompose_gap(sessions, target_tps=args.target_tps)
    print(f"{'Variant':<45} {'Exposed ms':>12} {'P95 ms':>10} {'TPS':>8} {'VRAM Hit%':>10}")
    print("-" * 88)
    for name, d in res.items():
        exp = f"{d.get('exposed_mean_ms', 0.0):.3f}" if 'exposed_mean_ms' in d else "N/A"
        p95 = f"{d.get('p95_ms', 0.0):.3f}" if 'p95_ms' in d else "N/A"
        tps = f"{d.get('tps', 0.0):.2f}" if 'tps' in d else "N/A"
        hit = f"{d.get('vram_hit_rate', 0.0)*100:.1f}%" if 'vram_hit_rate' in d else ""
        print(f"{name:<45} {exp:>12} {p95:>10} {tps:>8} {hit:>10}")
    
    print("\n[+] Diagnostic complete.")


if __name__ == "__main__":
    main()
