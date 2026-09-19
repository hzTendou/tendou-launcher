# OD-MoE / SPICE / Tutti audit (2026-09-10)

The updated project is in `C:/Users/Ali/WebProjects/tendou-launcher`; the old
`atlas-engine-v1` directory is absent. RAM was rechecked: 16,421,978,112 bytes
(15.29 GiB); GPU: RTX 5060 Laptop, 8,151 MiB.

## Findings

1. A real CPU kernel defect was found during thread-count measurement. The old
   single-token MUL_MAT_ID path assigned one expert ID per thread. With 10 active
   experts and 8 threads, expert slots 4 and 9 were never computed. Reused output
   memory concealed the missing writes and corrupted generation. Its duplicate-ID
   shortcut also copied one slot's result into another despite different inputs.
2. The reported 12.17x mixes three first requests with three prompt checkpoint
   hits. Its own artifact says three smoke checks, with general quality unevaluated.
3. The new feature arguments did not attach the router callback. The measured
   launch did not run predictive scheduling. Python modules were offline simulators.
4. VRAM promotion changes placement accounting; no consumer binds these records
   to CUDA expert computation. The staging pointer getter had no caller.
5. Every Runtime nevertheless allocated a 668 MiB staging pool, including when
   Tutti was disabled. Inference never consumed the copied expert weights.
6. Ring offsets could overlap or exceed capacity. Invalid reads could publish
   success. Queue limits excluded active work; cancellation could strand waiters.
7. The callback ended the previous token after observing the next token's first
   layer. Multi-token rows were mixed by a learner with only one token slot.
8. Multi-label activations were normalized as a categorical distribution. Ten
   always-active targets each received about 1/10 of their conditional activation
   frequency. Eviction confused equal expert IDs in different layers.
9. Native HTTP accepted OD-MoE/SPICE/Tutti arguments without using them. Those
   unsupported arguments now fail rather than silently doing nothing.

## Corrections

- Replaced the CPU expert-ID thread groups with a contiguous partition of all
  `(expert slot, output row)` work. Every row is computed exactly once, including
  when threads are fewer than active experts. Duplicate IDs retain separate input
  activations. At 14 threads this also removes the old uneven row workload.
- Zero default staging allocation; no redundant copying or inference-side waiting
  for speculative hints. No new quantization, router pruning or surrogate weights.
- Opt-in asynchronous OS page warming uses physical-map file paths and a worker.
  Windows calls PrefetchVirtualMemory; Linux uses aligned madvise, but Linux has
  not been executed on this Windows host. There is no dynamic H2D consumer.
- Queue capacity includes active chunks. Whole expert requests are admitted
  together. Empty, overflowing and malformed requests fail. File bounds are
  checked without overflowing addition. Failed or cancelled requests cannot
  publish success. A condition variable replaces busy waiting.
- Cancellation resolves old requests; shutdown joins the worker before releasing
  mappings. Handles are released even after a partial mapping failure. Completed
  hints are not permanent residency guarantees.
- Async enablement attaches router observation. Unconsumed prefill readbacks are
  skipped. The existing sampled decode observation schedule remains in use.
- Token boundaries precede new observations. Multi-token verification rows do not
  train this single-token predictor. Lookahead does not wrap into another token.
- Predictions use per-source observation counts, with conservative shrinkage
  `hits / (observations + 2)` and deterministic ties. These are empirical scores,
  not calibrated confidence guarantees. Cold starts do not invent expert IDs.
- Eviction uses layer-qualified identities and remains simulation-only. Confidence
  validation rejects nonfinite values and invalid ordering. SPICE-off is respected.
- Python staging simulation reserves bounded, non-overlapping ranges and completes
  each request once. Its metrics explicitly identify simulation.
- IPC reports router observations, queued/completed hint chunks, staging capacity
  and `dynamic_expert_h2d: false`. Hint completion means OS acceptance, not guaranteed
  residency or measured compute/I/O overlap.

## Measurements

Artifacts: `experiments/runtime_audit/2026-09-10/`. Four distinct requests cover
code, science, Turkish explanation and arithmetic reasoning. Each generates 32
tokens at temperature 0, native K=10, context 2048, with two GPU expert anchor
layers. No request repeats within a run; all report zero reused prompt tokens.
The OS page cache is not flushed. These are sequential warm-system measurements,
not cold-NVMe measurements. Short outputs test fidelity, not complete task accuracy.
Temperature and OS cache state introduce run-to-run variation.

| Case | Decode TPS before | Decode TPS after | Effective TPS before | Effective TPS after | TTFT before/after (s) |
|---|---:|---:|---:|---:|---:|
| code | 1.887 | 2.075 | 1.176 | 1.185 | 10.785 / 12.064 |
| science | 1.994 | 2.247 | 1.219 | 1.295 | 10.693 / 10.914 |
| turkish | 2.026 | 2.622 | 1.277 | 1.449 | 9.752 / 10.256 |
| reason | 1.614 | 2.133 | 1.052 | 1.207 | 11.215 / 11.972 |

Paired 14-thread comparison: median per-case decode speedup **1.211x**, effective
speedup **1.098x**, TTFT speedup **0.944x** (roughly 6% longer TTFT). All four
sequences match exactly, 128/128 reference tokens. Only the CPU DLL differs in
this pair: Atlas executable, llama.dll, llama-common.dll and request settings match.
This is a modest measured decode improvement, not a demonstrated multi-fold gain
or a TTFT improvement. The first corrected-kernel run also matched all tokens.

Eight-thread diagnosis: the original kernel produced different/broken output in
all four cases. With the corrected kernel, 8 threads again match all 128 reference
tokens. Eight threads were slower overall, so the default remains 14. The rejected
`threads8.json` is retained to show why speed without fidelity must not pass.

On the corrected CPU kernel, async page warming completed 21,876 chunk hints over
960 router observations. All 128 reference tokens still matched. Relative to the
corrected kernel with warming disabled, median paired decode speedup was
0.928x and effective speedup 0.952x.
This did not establish an overall win; async warming therefore remains opt-in.
There was no pinned allocation or dynamic H2D transfer in either run.


Reproduce from the project root (the benchmark injects the local CUDA DLL path):

```powershell
.venv-audit/Scripts/python.exe scripts/benchmark_resident.py --output result.json --cases experiments/runtime_audit/2026-09-10/cases.json --tokens 32 --repetitions 1
# Add these final arguments to test page warming:
# -- --atlas-tutti-async-io 1
```

The isolated original CPU library lives in `baseline-runtime/`, alongside the same
Atlas executable. `kernel_evaluation.json` contains the paired comparison.
General task accuracy remains **NOT_EVALUATED**; matching short reference outputs
is not certification of the global 85% task-quality threshold.


## Verification

`tests/native/test_expert_partition.cpp` reproduced missing output rows and wrong
duplicate-ID results against the original CPU DLL. It now passes for F32 and Q5_K,
10 active experts, duplicate IDs with distinct inputs, and 1/2/3/8/10/14 threads,
comparing every output to the single-thread reference. Outputs are filled with NaN
before execution so skipped writes cannot hide behind a previous result.

`scripts/build_cpu_kernel.cmd` rebuilds only the changed CPU object and relinks the
existing CPU library. Original CPU source, object and DLL are saved under `before/`.
The recorded response files preserve the original compiler flags and are relocated
to the current checkout; they must be regenerated if the checkout moves again.

The native harness includes the actual C++ implementation and uses a real temporary
mapped file. It covers multi-chunk hints, failed files, invalid ranges, overflow,
queue rejection, repeated cancellation, zero allocation, multi-label confidence
and the actual router callback's token boundary using host tensors.

Build with `scripts/build_runtime.cmd` and `scripts/build_native_tests.cmd`.
Native tests use the same local CUDA DLL search path as the server. Python tests
cover policy regressions, server lifecycle and API mocks: **93 passed, 1 optional
real-cache test skipped**. Both native harnesses passed; the partition harness
also covers broadcast inputs and single-expert inputs (48 configurations). Checkpoint serialization
was not changed; its optional long resident regression remains separate.

## Research and limits

- [OD-MoE](https://arxiv.org/html/2512.03927v1) combines lookahead with distributed
  loading and computation. Its ten-node results are not single-laptop estimates.
  Atlas's empirical learner is not the paper's emulative predictor.
- [SPICE](https://arxiv.org/html/2608.21240v2) combines confidence-aware prefetch
  with heterogeneous recovery and low-rank surrogates. Atlas uses scheduling ideas;
  exact native expert computation remains authoritative. Excluded approximation
  components mean its published speedup cannot be assumed here.
- [Tutti storage architecture](https://github.com/xPU-IO/Tutti/blob/main/doc/architecture/system-architecture.md)
  describes GPU-issued NVMe work, a custom driver and ext4 resolution. Relevant
  lessons include bounded queues, per-request completion and decoupled work. This
  Windows implementation is OS page warming, not that GPU-direct storage backend.
- The [Tutti KV-cache paper](https://arxiv.org/html/2605.03375v1) concerns restoring
  cached attention state from SSD. That is distinct from streaming MoE weights.
- MTP remains experimental. The preceding audit measured a short decode gain but
  also token divergence on a longer code continuation. It is not enabled in the
  default lossless path; this patch does not certify MTP as token-exact.

A real dynamic GPU expert cache still needs backend binding of selected quantized
expert planes, compact ID remapping, compute/transfer events and buffer lifetime
protection through GEMM completion. Pinned allocation and stream-shaped accounting
cannot supply that missing implementation. Static llama.cpp offload executes the
current inference workload.
