import json

with open('docs/json/benchmark_phase10_vs_phase11_report.json', 'r', encoding='utf-8') as f:
    r = json.load(f)

cfgs = {
    'A': r['config_a_phase10_mtp_off'],
    'B': r['config_b_phase10_mtp_on'],
    'C': r['config_c_phase11_mtp_off'],
    'D': r['config_d_phase11_pld_mtp_on'],
}

print("=" * 110)
print(f"{'Config':<8} | {'Workload':<20} | {'Parity':<8} | {'Cold TTFT':<10} | {'Warm TTFT':<10} | {'Dec TPS':<10} | {'Eff TPS (med/min/max)':<24} | {'T_draft':<10} | {'Acc %':<8}")
print("=" * 110)

for code, cfg in cfgs.items():
    for w in cfg['workloads']:
        eff_str = f"{w['effective_tps_median']:.2f} [{w['effective_tps_min']:.2f}-{w['effective_tps_max']:.2f}]"
        print(f"{code:<8} | {w['workload_id']:<20} | {w['deterministic_parity']:<8} | {w['cold_ttft_ms']:<10.1f} | {w['warm_ttft_median_ms']:<10.1f} | {w['decode_tps_median']:<10.2f} | {eff_str:<24} | {w['t_draft_step_median_ms']:<10.3f} | {w['acceptance_rate_median']*100:<7.1f}%")
    print("-" * 110)

print("\nVRAM Statistics:")
for code, cfg in cfgs.items():
    print(f"Config {code}: Used = {cfg['vram_used_mib']:.1f} MiB | Free = {cfg['vram_free_mib']:.1f} MiB | Delta = {cfg['vram_delta_mib']:.1f} MiB")
