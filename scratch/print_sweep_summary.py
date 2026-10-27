import json

with open('docs/json/benchmark_phase10_vs_phase11_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

sweep = report['horizon_sweep_d1_to_d5']
print("=" * 90)
print(f"{'d':<4} | {'Workload':<20} | {'Accept %':<10} | {'Dec TPS':<10} | {'Eff TPS':<10} | {'T_draft (ms)':<14} | {'EV(d)':<10}")
print("=" * 90)

for d_key, cfg in sweep.items():
    d_val = cfg['mtp_draft_n']
    for w in cfg['workloads']:
        print(f"{d_val:<4} | {w['workload_id']:<20} | {w['acceptance_rate_median']*100:<9.1f}% | {w['decode_tps_median']:<10.2f} | {w['effective_tps_median']:<10.2f} | {w['t_draft_step_median_ms']:<14.4f} | {w['theoretical_ev']:<10.2f}")
    print("-" * 90)
