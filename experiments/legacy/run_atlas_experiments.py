"""One-command Atlas policy sweep for the real target hardware budget."""
import argparse, csv, json
from pathlib import Path
from atlas_simulator import load_sessions, build_affinity_regions, evaluate_policy


def floats(s): return [float(x) for x in s.split(',') if x.strip()]
def ints(s): return [int(x) for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--expert-size-mb', type=float, required=True)
    ap.add_argument('--vram-gbs', default='8', help='Physical VRAM sizes to sweep')
    ap.add_argument('--vram-reserve-gbs', default='1', help='VRAM reserved for runtime/KV/workspace')
    ap.add_argument('--ram-gbs', default='16', help='Physical host RAM sizes to sweep')
    ap.add_argument('--ram-reserve-gbs', default='2', help='Host RAM reserved for OS/runtime')
    ap.add_argument('--nvme-gbps', type=float, default=7.0)
    ap.add_argument('--pcie-gbps', type=float, default=12.0)
    ap.add_argument('--ram-bandwidth-gbps', type=float, default=40.0)
    ap.add_argument('--nvme-latency-ms', type=float, default=0.08)
    ap.add_argument('--pcie-latency-ms', type=float, default=0.02)
    ap.add_argument('--target-tps', default='5,10,20')
    ap.add_argument('--region-sizes', default='8', help='Logical affinity-region sizes to test for causal_region; use 4,8,16 for hypothesis sweep')
    ap.add_argument('--min-affinity-count', type=int, default=2)
    ap.add_argument('--preload-experts-to-ram', action='store_true', help='Preload entire expert corpus into RAM')
    ap.add_argument('--vram-policy', choices=['lru', 'freq'], default='lru')
    ap.add_argument('--deadline-aware-budget', action='store_true')
    ap.add_argument('--out-prefix', default='atlas_sweep')
    args = ap.parse_args()

    sessions = load_sessions(args.trace)
    with open(args.trace, encoding='utf-8') as f:
        raw = [json.loads(x) for x in f if x.strip()]
    for i, s in enumerate(sessions):
        s['_all_records'] = raw[i].get('records', [])
    region_maps = {}
    for region_size in ints(args.region_sizes):
        region_maps[region_size] = build_affinity_regions(sessions, region_size, args.min_affinity_count)

    rows = []
    for vram in floats(args.vram_gbs):
        for vram_reserve in floats(args.vram_reserve_gbs):
            if vram_reserve >= vram:
                raise SystemExit('--vram-reserve-gbs must be smaller than --vram-gbs')
            for ram in floats(args.ram_gbs):
                for ram_reserve in floats(args.ram_reserve_gbs):
                    if ram_reserve >= ram:
                        raise SystemExit('--ram-reserve-gbs must be smaller than --ram-gbs')
                    for tps in floats(args.target_tps):
                        compute_ms = 1000.0 / tps
                        vram_cache_gb = vram - vram_reserve
                        ram_cache_gb = ram - ram_reserve
                        for mode in ('none', 'causal', 'prompt', 'atlas_predictor', 'oracle'):
                            r = evaluate_policy(
                                sessions, mode, vram_cache_gb * 1024.0,
                                args.expert_size_mb, ram_cache_gb,
                                args.nvme_gbps, args.pcie_gbps,
                                args.ram_bandwidth_gbps, args.nvme_latency_ms,
                                args.pcie_latency_ms, compute_ms, 1, 0.75,
                                None, None, False,
                                {"confidence_floor": 0.10, "budget_scale": 1.05, "history_window": 4,
                                 "max_candidates_per_layer": 4, "min_count": 2, "persistent_fallback": True} if mode == 'atlas_predictor' else None,
                                preload_ram=args.preload_experts_to_ram,
                                vram_policy=args.vram_policy,
                                deadline_aware=args.deadline_aware_budget,
                            )
                            rows.append({
                                'vram_gb': vram,
                                'vram_reserve_gb': vram_reserve,
                                'vram_cache_gb': vram_cache_gb,
                                'ram_gb': ram,
                                'ram_reserve_gb': ram_reserve,
                                'ram_cache_gb': ram_cache_gb,
                                'target_tps': tps,
                                **{k: v for k, v in r.items() if k != 'mode'},
                                'policy': mode,
                                'region_size': None,
                            })
                        for region_size, (region_map, region_members) in region_maps.items():
                            r = evaluate_policy(
                                sessions, 'causal_region', vram_cache_gb * 1024.0,
                                args.expert_size_mb, ram_cache_gb, args.nvme_gbps,
                                args.pcie_gbps, args.ram_bandwidth_gbps, args.nvme_latency_ms,
                                args.pcie_latency_ms, compute_ms, 1, 0.75,
                                region_map, region_members, True,
                                preload_ram=args.preload_experts_to_ram,
                                vram_policy=args.vram_policy,
                                deadline_aware=args.deadline_aware_budget,
                            )
                            rows.append({
                                'vram_gb': vram,
                                'vram_reserve_gb': vram_reserve,
                                'vram_cache_gb': vram_cache_gb,
                                'ram_gb': ram,
                                'ram_reserve_gb': ram_reserve,
                                'ram_cache_gb': ram_cache_gb,
                                'target_tps': tps,
                                **{k: v for k, v in r.items() if k != 'mode'},
                                'policy': 'causal_region',
                                'region_size': region_size,
                            })

    csv_path = Path(args.out_prefix + '.csv')
    json_path = Path(args.out_prefix + '.json')
    keys = list(rows[0].keys())
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps({'config': vars(args), 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[+] CSV: {csv_path}')
    print(f'[+] JSON: {json_path}')

    grouped = {}
    for row in rows:
        key = (row['vram_gb'], row['vram_reserve_gb'], row['ram_gb'], row['ram_reserve_gb'], row['target_tps'])
        grouped.setdefault(key, []).append(row)
    print('\nDeployable policy summary (oracle excluded):')
    for key, rs in grouped.items():
        deployable = [r for r in rs if r['policy'] != 'oracle']
        best_mean = min(deployable, key=lambda r: (r['exposed_io_ms_mean'], r['exposed_io_ms_p95'], r['nvme_mb']))
        best_p95 = min(deployable, key=lambda r: (r['exposed_io_ms_p95'], r['exposed_io_ms_mean'], r['nvme_mb']))
        print(
            f'  VRAM={key[0]:.0f}GB(-{key[1]:.0f}) RAM={key[2]:.0f}GB(-{key[3]:.0f}) TPS={key[4]:.0f}: '
            f'mean-best={best_mean["policy"]}[r={best_mean["region_size"]}] '
            f'{best_mean["exposed_io_ms_mean"]:.3f}ms, P95={best_mean["exposed_io_ms_p95"]:.3f}ms; '
            f'p95-best={best_p95["policy"]}[r={best_p95["region_size"]}] '
            f'{best_p95["exposed_io_ms_mean"]:.3f}ms, P95={best_p95["exposed_io_ms_p95"]:.3f}ms'
        )


if __name__ == '__main__':
    main()
