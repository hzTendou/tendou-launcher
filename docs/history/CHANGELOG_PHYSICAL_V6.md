# Physical Simulator V6

## Changes

- Added explicit `cold_nvme_misses` and `eviction_nvme_misses` accounting.
- Added prediction outcome metrics: `predictions_cold`, `predictions_already_hot`, `predictions_that_prevented_nvme_miss`, `predictions_arrived_too_late`.
- Added prediction lead-time percentiles.
- Added `--preload-experts-to-ram` to model an expert corpus that fits entirely in host RAM.
- Added separate startup preload time and decode throughput metrics.
- Added `--predictor-persistent-fallback` as an opt-in experiment.

## Qwen3.6-35B-A3B result

At 8 GB VRAM / 16 GB RAM (7 GB / 14 GB usable), all 10,240 exact experts require 10,040 MB and therefore fit in RAM.
Preloading reduced decode NVMe misses from 9,573 to 0 on `trace_35b_a3b_q2k.jsonl`.
At target TPS 20, exposed I/O fell from 7.657 ms to 5.034 ms for the no-prefetch baseline.
Atlas predictor with 4 candidates/layer and preload measured 4.987 ms exposed I/O and 13.686 TPS.
The result moves the primary bottleneck to RAM→VRAM / PCIe scheduling rather than NVMe cold acquisition.
