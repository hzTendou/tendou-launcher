# Atlas Engine — Physical V6 Experiment Report

## Baseline

Trace: `trace_35b_a3b_q2k.jsonl`
Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`
Usable memory: 7 GB VRAM + 14 GB RAM
NVMe: 7 GB/s
PCIe: 12 GB/s
RAM bandwidth: 40 GB/s
Target: 20 TPS

### Baseline no preload
- NVMe misses: 9573
- Cold NVMe misses: 9573
- Eviction NVMe misses: 0
- Exposed I/O mean: 7.657 ms
- Effective decode TPS: 13.159

### RAM-preload
All 10,240 exact experts fit in RAM:
- Expert pool: 10,040 MB
- Usable RAM: 14,336 MB
- Preload time: 1655.39 ms

No-prefetch policy after preload:
- NVMe misses: 0
- Exposed I/O mean: 5.034 ms
- Effective decode TPS: 13.674
- TPS including startup preload: 13.127

Atlas predictor, 4 candidates/layer + preload:
- NVMe misses: 0
- Predictions: 1756
- Correct: 1518
- False: 238
- Precision: 86.45%
- Exposed I/O mean: 4.987 ms
- Effective decode TPS: 13.686
- TPS including startup preload: 13.138

## Interpretation

The 9573 baseline NVMe misses are all cold acquisitions in this trace; no eviction misses were observed. The complete expert pool fits in host RAM, so a RAM-resident expert tier is a viable architecture for this workload. After preload, Atlas is no longer primarily solving NVMe cold acquisition; its remaining opportunity is hiding RAM->VRAM transfers behind compute and improving PCIe scheduling.

The Oracle policy reaches substantially lower exposed I/O, showing that additional timely prefetch signal can still produce gains. The next engineering target is therefore PCIe/RAM prefetch scheduling and prediction coverage, not another confidence sweep.
