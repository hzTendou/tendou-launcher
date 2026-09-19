# Atlas Engine V7 — Experiment F Implementation Walkthrough

## Summary of Completed Work

Experiment F (**2-Step-Ahead Prefetch**) has been fully implemented across both the discrete-event logical simulator ([`atlas_simulator.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/atlas_simulator.py)) and the exact-tensor-slice physical simulator ([`atlas_physical_simulator.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/atlas_physical_simulator.py)). 14 new dedicated unit tests were added to [`test_experiment_f.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/test_experiment_f.py), bringing the total test suite to **37 tests (all passing)**.

A complete benchmark comparison across 10 configurations for both logical and physical simulators was executed via [`run_v7_benchmark.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/run_v7_benchmark.py).

---

## 1. Architecture & Design Implementation Details

1. **Current Architecture**: Models the memory hierarchy `NVMe -> RAM -> PCIe -> VRAM -> GPU Execution`. The simulation isolates request-local VRAM from a global RAM warm-cache.
2. **Predictor Location**: [`atlas_predictor.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/atlas_predictor.py) (`AtlasPredictor` and `_SessionState`). Predictions are phase-aware (prompt vs decode) with transient session state and post-session commits.
3. **Prefetch Scheduling Location**: 
   - Logical simulator: [`atlas_simulator.py:evaluate_policy()`](file:///c:/Users/Ali/WebProjects/atlas-engine/atlas_simulator.py) inside the token loop before `sim.run_token()`.
   - Physical simulator: [`atlas_physical_simulator.py:PhysicalSim.run()`](file:///c:/Users/Ali/WebProjects/atlas-engine/atlas_physical_simulator.py) before `ensure_batch()`.
4. **Candidate Limit**: Controlled per-layer by `max_candidates_per_layer`, with dynamic budget expansion based on prediction confidence concentration: `budget_k = ceil(k * (1.0 + (budget_scale - 1.0) * conf))`.
5. **PCIe Budget Enforcement**: Derived from physical compute window: `max_deadline_candidates = floor((compute_ms - pcie_lat) / 1000 * pcie_gbps * 1024 / expert_size_mb)`.
6. **VRAM Residency Tracking**: Global byte-budget cache with LRU eviction (`OrderedDict`) or frequency-biased eviction (`FrequencyBiasedVRAM`).
7. **Clean Experiment F Implementation**:
   - At step $t$, compute step-1 prediction $H_1 = \text{predict}(C_t)$.
   - Synthesize hypothetical next router state: $L_{\text{hypo}} = \text{keys\_to\_layers}(H_1)$.
   - Generate step-2 candidates $H_2 = \text{predict}(L_{\text{hypo}}, \text{exclude}=V_{\text{VRAM}} \cup H_1)$.
   - Allocate step-2 budget: $\text{Cap}_2 = \min(|H_2|, \lfloor(\text{MaxBudget} - |H_1|) \times \text{step2\_budget\_fraction}\rfloor)$.
   - Combined prefetch set: $P = H_1 \cup H_2^{\text{top\_ranked}}$.
8. **Modified Files**:
   - [`atlas_simulator.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/atlas_simulator.py)
   - [`atlas_physical_simulator.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/atlas_physical_simulator.py)
   - [`run_v7_benchmark.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/run_v7_benchmark.py)
   - [`test_experiment_f.py`](file:///c:/Users/Ali/WebProjects/atlas-engine/test_experiment_f.py) (New)
9. **Tests Added**: 14 tests verifying deduplication, priority ordering, PCIe budget cap, zero-budget equivalence, and mode isolation.

---

## 2. Benchmark Results

### Logical Simulator Ablation (trace_35b_a3b_q2k.jsonl, 20 TPS target, 7GB VRAM / 14GB RAM)

| Configuration | Exposed I/O (ms) | P95 (ms) | P99 (ms) | TPS | VRAM Hit % | S1 Pred | S2 Pred | S2 Prec % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. Baseline (no preload, LRU) | 7.778 | 23.255 | 37.650 | 17.31 | 81.8% | 0 | 0 | 0.0% |
| 2. RAM Preload (none, LRU) | 6.854 | 19.919 | 37.650 | 17.59 | 81.8% | 0 | 0 | 0.0% |
| 3. 1-step cand=1 (preload) | 6.839 | 19.919 | 37.650 | 17.59 | 81.8% | 271 | 0 | 0.0% |
| 4. 1-step cand=4 (preload) | 6.783 | 19.837 | 37.650 | 17.61 | 82.0% | 1,340 | 0 | 0.0% |
| 5. 1-step cand=8 (preload) | 6.669 | 19.254 | 37.650 | 17.65 | 82.3% | 3,328 | 0 | 0.0% |
| 6. 2-step cand=1+1 frac=1.0 | 6.834 | 19.919 | 37.650 | 17.60 | 81.8% | 266 | 78 | 5.1% |
| 7. 2-step cand=4+2 frac=0.5 | 6.742 | 19.531 | 37.650 | 17.62 | 82.1% | 1,267 | 790 | 5.2% |
| 8. 2-step cand=4+4 frac=1.0 | 6.742 | 19.531 | 37.650 | 17.62 | 82.1% | 1,267 | 790 | 5.2% |
| 9. **2-step cand=8+4 frac=0.5** | **6.568** | **18.584** | **37.650** | **17.68** | **82.6%** | **3,033** | **1,949** | **3.2%** |
| 10. **Oracle (upper bound)** | **0.588** | **0.000** | **37.650** | **19.77** | **98.4%** | **0** | **0** | **0.0%** |

---

### Physical Simulator Ablation (exact GGUF tensor slices, 7GB VRAM / 14GB RAM)

| Configuration | Exposed I/O (ms) | P95 (ms) | P99 (ms) | TPS | Precision % | Correct Hits |
|---|---:|---:|---:|---:|---:|---:|
| 1. Physical Baseline (no preload) | 7.657 | 25.885 | 41.340 | 17.93 | 0.0% | 0 |
| 2. Physical RAM Preload (none) | 5.034 | 14.292 | 31.933 | 18.71 | 0.0% | 0 |
| 3. 1-step cand=1 (preload) | 5.015 | 14.244 | 31.933 | 18.72 | 68.8% | 461 |
| 4. 1-step cand=4 (preload) | 4.951 | 13.988 | 31.933 | 18.75 | 67.6% | 1,866 |
| 5. 1-step cand=8 (preload) | 4.839 | 13.909 | 31.933 | 18.80 | 55.4% | 3,141 |
| 6. 2-step cand=1+1 frac=1.0 | 5.013 | 14.244 | 31.933 | 18.72 | 64.6% | 462 |
| 7. 2-step cand=4+2 frac=0.5 | 4.935 | 13.988 | 31.933 | 18.76 | 56.0% | 1,878 |
| 8. 2-step cand=4+4 frac=1.0 | 4.935 | 13.988 | 31.933 | 18.76 | 56.0% | 1,878 |
| 9. **2-step cand=8+4 frac=0.5** | **4.790** | **13.909** | **31.933** | **18.82** | **43.9%** | **3,164** |
| 10. **Physical Oracle (preload)** | **0.371** | **0.000** | **31.933** | **20.00** | **100.0%** | **326,720** |

---

## 3. Analysis of Experiment F Results (Outcome B: Small Improvement)

Experiment F advanced the physical simulator from **18.80 TPS** to **18.82 TPS** (and logical TPS from **17.65** to **17.68**), with exposed I/O dropping to **4.790 ms** (from 4.839 ms) and hit count increasing to **3,164**.

### Key Findings & Why Improvement is Incremental:
1. **Compounding Uncertainty on High-Entropy Transitions**:
   - Top-1 transition probability on this trace is only ~7.4% (normalized entropy ~0.997).
   - Single-step prediction has ~31% recall on unseen transitions. Predicting step $t+2$ through a hypothetical step $t+1$ compounds the uncertainty ($P(\text{correct}_{t+2}) \approx 0.31 \times 0.31 \approx 9.6\%$).
   - Measured Step-2 precision is **3.2% - 5.2%**, meaning many step-2 prefetch candidates do not materialize at token $t+2$.
2. **PCIe Bandwidth Budget Headroom**:
   - The budget cap comfortably prevented bandwidth congestion (exposed I/O improved monotonically from 1-step to 2-step).
   - Wasted transfers did not cause catastrophic VRAM evictions because high-confidence step-1 candidates were prioritized.
3. **The Bottleneck Is Predictor Representation**:
   - As established in Experiment E (Oracle B = 19.77 TPS), the bottleneck is not scheduling lead time, but the accuracy/recall of the prediction itself.

---

## 4. Test Suite Verification

```powershell
python -m pytest test_atlas_predictor.py test_atlas_simulator.py test_physical_simulator.py test_physical_map.py test_v7_improvements.py test_oracle_decompose.py test_experiment_f.py -v
```
**Result**: `37 passed in 0.10s` (100% pass rate).
