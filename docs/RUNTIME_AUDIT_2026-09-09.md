# Atlas runtime audit — 2026-09-09

## Findings and implementation

1. **Quality was not measured.** `QualityEstimator` blends repetition and token
   entropy. Fluent incorrect answers can score 100%. Earlier benchmark parsers
   even defaulted absent quality to 100%. Production and parsers now report
   `NOT_EVALUATED`; diversity remains a diagnostic only. `--atlas-quality-target`
   accepts 0.85–1.0 and no longer secretly chooses a smaller expert count.
2. **Default routing changed the model.** Both the config and GPU/boost branches
   forced K=2, although the supplied model uses K=10. A later branch overwrote
   the quality target with 0.80. Defaults now preserve native routing and 0.85.
   Explicit `--atlas-k` remains an experimental approximation, not a quality guarantee.
3. **The dynamic GPU cache was a simulator.** `execute_gpu_first_token` adds
   fixed cost constants, changes residency bookkeeping, and queues C++ task
   records after `llama_decode`. `GroupedGemmStreamPipeline` creates no CUDA
   streams or kernels. These costs and router observation are off the default
   generation path. Static dense/expert-layer offload remains in llama.cpp.
4. **Every API request repeated prefill.** Two bounded checkpoints now cover the
   full prompt and an existing prefill batch boundary before the final assistant
   suffix. Full repeats restore logits; extensions decode their suffix. The
   latter boundary handles the API formatter omitting thinking markers from
   assistant history. Short prompts do not get split into an extra MoE pass just
   to create this second checkpoint. Total checkpoint storage is bounded by
   `--atlas-prompt-cache-mb` (256 MiB by default); oversized snapshots are skipped.
5. **Recurrent state needs real restoration.** Generic KV truncation is unsafe
   for this hybrid model. The complete memory state is saved. This fork's
   `llama_context::state_write_data` does NOT serialize logits, despite its API
   header comment; logits are therefore copied separately. A real-model repeat
   test caught this mismatch during development; the failed attempt is retained
   in `experiments/runtime_audit/cache_missing_logits_failed.json`.
6. **Unnecessary work and failure handling.** The final requested token no longer
   triggers another forward pass. Context limits stop generation cleanly and
   decode failures are labeled errors. Per-request quality state resets. Python
   startup detects an exited child immediately, active queues receive an error
   on EOF, and reader tasks/processes are joined at shutdown.

## Hardware and environment

Measured machine: Ryzen 7 260 (8 cores / 16 threads), RTX 5060 Laptop GPU
(8151 MiB), driver 591.91. Windows reports **8,368,914,432 bytes physical RAM**,
about 7.79 GiB, not the 16 GiB README target. See `hardware.json` in the results.
The model remains the original three-shard Q5_K_S; no weights were requantized.

The old `.venv` points to a removed Python 3.12 installation. `.venv-audit` uses
the installed Python 3.14. The old CMake build references missing CUDA 13.3
MSBuild files and a removed CMake path. `scripts/build_runtime.cmd` compiles only
the Atlas executable with installed VS 2026 and links existing unchanged llama
and ggml Release libraries; it is not a clean rebuild of those libraries.
Missing cuBLAS runtime DLLs were installed into `.venv-audit` using NVIDIA's
`nvidia-cublas` wheel. The server adds its virtual environment's NVIDIA DLL
directory to the child PATH without changing the system PATH.

`src/atlas/atlas-engine.{cpp,h}` is the editable source; the local build script
copies it to `llama.cpp/examples/atlas-engine/` before compiling. The pre-audit
executable and source copies are retained for comparison. No commits or remote
repository changes were made.

## What was learned from Unsloth and AirLLM

- [Unsloth Llama inference source](https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py)
  reuses KV and temporary attention buffers and has a distinct single-token
  inference path. The transferable principle applied here is preserving useful
  inference state and avoiding redundant work. Atlas uses native llama.cpp
  state serialization, not copied PyTorch/Triton kernels.
- [Unsloth README](https://github.com/unslothai/unsloth#readme) places its 70% VRAM
  reduction claim under fine-tuning. That figure is not evidence that a 134 GB
  GGUF will decode quickly on an 8 GB laptop.
- [AirLLM execution source](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_base.py)
  implements actual module load/eviction hooks and next-module prefetch with a
  worker future. It also bounds pinned host memory. This informed the bounded
  resident-state design and the audit distinction between completed work and
  predicted placement. Whole-layer streaming is a capacity technique: replacing
  sparse GGUF expert access with full-layer loads would not establish a speedup.
  No AirLLM speed or memory percentage is attributed to Atlas.

## Measurement protocol and limits

`scripts/benchmark_resident.py` measures real JSON IPC requests after model
startup. It stores binary SHA-256, prompts, emitted token IDs, timings and terminal
events. Baseline uses the frozen executable with `--atlas-k 0`; candidate uses the
same native router, model, context (2048), temperature (0), and 14 threads. Engines
run sequentially. Two requests per question distinguish first access and prompt
reuse; OS page cache, I/O and laptop power remain uncontrolled.

- TTFT: request submission to first emitted token, excluding model startup.
- Effective TPS: visible output token count / entire request duration.
- Decode TPS: `(visible tokens - 1) / (last token time - first token time)`;
  undefined for a one-token answer. EOS is excluded from visible token counts.
- Quality smoke test: independently checked arithmetic, Python list result and
  syllogism. Exact emitted token sequences are compared with the full-router
  baseline. Three questions are **not** a general coding/reasoning certification.
- The initial coding prompt exhausted its 16-token budget before answering:
  `comparison_initial.json` correctly fails the task gate at 66.7%, despite 100%
  token agreement. Both engines were rerun with an explicit short-answer
  instruction. The initial and intermediate reports remain available.
- This change targets repeated/extended request latency. It does not establish
  a large raw decode TPS jump, a real dynamic CUDA expert cache, or universal
  quality >=85%. Unrelated prompts and states above the budget still prefill.

## Initial 8 GB short-answer measurements

Results are in `experiments/runtime_audit/comparison.json`, with the two full
`*_short_answers.json` runs. Table below compares the second request per question
(warm prompt checkpoint for Atlas, repeated prefill for the frozen baseline).

| Case | Baseline TTFT | Atlas TTFT | Baseline effective TPS | Atlas effective TPS | Effective speedup |
|---|---:|---:|---:|---:|---:|
| 17 x 19 | 23.314 s | 0.111 s | 0.111 | 1.268 | 11.43x |
| Python sorted list | 16.145 s | 0.230 s | 0.356 | 0.948 | 2.66x |
| Syllogism | 19.734 s | 0.107 s | 0.047 | 2.581 | 55.36x |

The median request speedup across all six first/repeated requests is **2.11x**.
First-access speedups in this run were 1.16x–1.55x; there are too few samples and
too much uncontrolled page-cache variation to certify these as reliable gains.
Warm math decode was 0.731 -> 1.337 TPS, while warm code decode was 0.970 -> 0.963
TPS: **a general raw decode improvement is not established**.

All three smoke answers are correct in both engines, and all six emitted token
sequences match exactly. This clears the 85% floor **on this small smoke suite**;
general model task accuracy remains unmeasured. The initial optimized short-prompt cache uses
about 114.6 MiB. Its configured total ceiling remains 256 MiB including the optional
chat-boundary checkpoint. The earlier unaligned two-checkpoint attempt caused extra
MoE passes on short prompts; it was replaced with checkpoints at existing batch
boundaries before the final run. Its measurements are retained separately.

## Reproduction

From the workspace in PowerShell:

```powershell
cmd /c scripts\build_runtime.cmd
.venv-audit\Scripts\python.exe atlas_server.py --port 8000 --threads 14 --boost
```

Run each model benchmark separately:

```powershell
.venv-audit\Scripts\python.exe scripts\benchmark_resident.py --exe llama.cpp/build/bin/Release/llama-atlas-engine-pre-audit.exe --output experiments/runtime_audit/baseline_short_answers.json
.venv-audit\Scripts\python.exe scripts\benchmark_resident.py --output experiments/runtime_audit/optimized_short_answers.json
.venv-audit\Scripts\python.exe scripts\evaluate_resident.py experiments/runtime_audit/baseline_short_answers.json experiments/runtime_audit/optimized_short_answers.json --output experiments/runtime_audit/comparison.json
.venv-audit\Scripts\python.exe -m pytest tests/integration/test_openai_server.py tests/integration/test_runtime_regressions.py -q
$env:ATLAS_TEST_REAL = '1'
.venv-audit\Scripts\python.exe -m pytest tests/integration/test_runtime_regressions.py -k recurrent -q
Remove-Item Env:ATLAS_TEST_REAL
```

The last test compares uncached and cached inference on exact repeats, a naturally
formatted added chat turn, and unrelated prompts, checking emitted tokens and the
memory budget. Its token evidence is saved in `cache_regression.json`.


## 16 GB RAM and MTP follow-up

Windows now reports 16,421,978,112 bytes physical memory (15.29 GiB). The earlier
8 GB results above are retained as historical measurements. Hardware and fresh
runs are stored separately in `experiments/runtime_audit/ram16/`.

Further defects found and corrected:

- `--boost` silently enabled n-gram speculation, but the IPC loop never used it.
  This reserved extra recurrent rollback slots and inflated prompt snapshots.
  Boost now keeps the normal sampler and allocates no speculative state unless
  the user explicitly enables speculation.
- The Qwen4Exp MTP tensor names and memory plumbing existed, but its tensor loader
  and graph implementation were missing from `src/models/qwen4exp.cpp`. Selecting
  MTP constructed the trunk graph against the draft's attention-only memory and
  crashed in `llama.dll`. Ported the graph, head loading, and wide hidden-state
  export from Daniel Hanchen's upstream PR #28243, adapting to this tree's scalar
  expert FFN parameters and existing sidecar merger. Existing tensor enums,
  layer fields and graph declarations already covered this support.
- `common/speculative.cpp` incorrectly treated every draft with `ctx_other` as
  sharing the target KV. Qwen4Exp shares weights, but needs its own attention
  history and increasing draft positions. Ported the PR's Gemma-only sharing test.
- CLI speculative verification now checks decode/process/rollback failures and
  probes recurrent rollback requirements before prefill. Baseline and speculative
  generation use the same target sampler. Real token IDs and request/decode/TTFT
  timings are emitted as `[ATLAS_RESULT]` JSON.

The API/IPC loop still uses single-token generation with prompt checkpoints.
Explicit speculation in IPC now fails at startup with an actionable message;
previously it consumed extra memory while silently running ordinary generation.
A separate `atlas_server.py --native-mtp` preset now exposes the patched native
server on localhost:8001, using one slot and native routing. The existing Atlas
IPC API remains the default. The native preset uses explicit HTTP connection
closure after each response: a default-client repeated streaming test exposed a
connection reuse failure, so this preset sets `ATLAS_HTTP_CLOSE=1`.

The MTP graph currently uses dense draft attention instead of QSA. Its source
notes equivalence below the 2048-token indexer budget. All current measurements
use context 2048. Target verification always retains the full model; long-context,
non-greedy and concurrent serving still require separate validation.

### Research and provenance

- [Unsloth MTP head documentation](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md)
  reports 1.34-1.67x on one B200 with different target quantizations, and a loss
  at concurrency 8. These are not measurements on this laptop.
- [Qwen4Exp MTP implementation, PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243)
  supplies the model-specific implementation used here. The fetched diff and
  original local sources are retained under `experiments/runtime_audit/mtp_sources/`.
  This is a private local adaptation, not an upstream submission.
- [Earlier PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836) helped trace
  the incomplete integration; it was not applied wholesale.
- [DeepSeek-V3 report](https://arxiv.org/html/2412.19437v2), section 5.4.3, reports
  85-90% next-token acceptance and 1.8x TPS in its deployment. Acceptance is a
  performance statistic, not answer accuracy.

Expected gain is roughly `(1 + accepted draft tokens) * single_token_cost /
(drafting + verification + state maintenance)`. Longer drafts can lose on a
memory-constrained MoE because verification touches more distinct experts and
adds rollback memory. MTP mainly targets decode throughput; it does not remove
prefill. Prompt reuse addresses repeat-request TTFT separately.


### Measured MTP results (16 GB, Q5_K_S target, Q4_K_M head)

`ram16/mtp_complete_q4/summary.json`: greedy, 48 output tokens, one short Newton
continuation, context 2048, 14 threads. Effective TPS includes prompt evaluation
but excludes model startup. Decode TPS is `(tokens-1)/(last-first token time)`.

| Mode | Decode TPS | Effective TPS | TTFT (s) | Exact token sequence |
|---|---:|---:|---:|---|
| Off | 1.776 | 1.630 | 2.983 | Reference |
| RAM, 1 draft | 2.416 | 2.112 | 3.273 | Yes |
| RAM, 2 drafts | 1.915 | 1.682 | 3.990 | Yes |
| RAM, 4 drafts | 1.650 | 1.466 | see JSON | Yes |
| GPU dense head, 2 drafts | 2.318 | 1.987 | see JSON | Yes |

One-token drafting improved decode 36.0% and effective throughput 29.6% in this
run. Four-token drafting lost performance. The default explicit MTP draft length
is therefore now 1. This does not enable MTP in the normal API.

A 64-token Fibonacci continuation did NOT match byte/token-for-token: both MTP
placements changed the wording of the generated input prompt and ended at 60
tokens. The first generated Fibonacci function passed all 11 inputs (0..10) in
all three modes (`mtp_functional_smoke.json`). This is one function, not a coding
benchmark, and its differing lengths make a direct speed-win claim inappropriate.

`--atlas-verify-batch` adds expensive diagnostic sequential replay of the same
model state. The corrected diagnostic preserves the original MTP output and
reports 2 greedy-choice differences in 68 checked positions. At position 45,
which predicts the first differing emitted token, the sequential winner's margin
was only 0.061176 logits. This demonstrates batch-versus-single-token numerical
sensitivity; it does not prove all long-context rollback behavior correct.
The initial diagnostic incorrectly restored only the current state before later
rollback; it was corrected to rebuild the original batch and its rollback planes.
Both logs are retained, and diagnostic runs are excluded from speed scoring.

All general quality fields remain `NOT_EVALUATED`. The 85% requirement is a
release gate for a representative task suite, not a claim inferred from acceptance,
entropy, the one matching continuation, or the one correct function.

### Build and use

These incremental scripts reuse this workspace's existing Release objects and
CUDA libraries; they are not a replacement for a clean portable CMake build.
The HTTP build regenerates CMake's export definitions after recompiling.

```powershell
cmd /c scripts\build_mtp_graph.cmd
cmd /c scripts\build_mtp_common.cmd
cmd /c scripts\build_native_http.cmd
cmd /c scripts\build_runtime.cmd

# Existing Atlas API, native routing and prompt checkpoints:
.venv-audit\Scripts\python.exe atlas_server.py --port 8000 --threads 14 --boost

# Experimental native MTP API for longer generation, single slot, context 2048:
.venv-audit\Scripts\python.exe atlas_server.py --native-mtp --port 8001 --mtp ram --draft-n 1

# Native API without speculation, for a controlled comparison:
.venv-audit\Scripts\python.exe atlas_server.py --native-mtp --port 8001 --mtp off

.venv-audit\Scripts\python.exe scripts/benchmark_mtp.py --tokens 48 --modes off ram1 ram2 ram4 vram2 --output-dir experiments/runtime_audit/ram16/recheck
.venv-audit\Scripts\python.exe scripts/benchmark_native_api.py --output experiments/runtime_audit/ram16/http_recheck
```

Use OpenAI base URL `http://127.0.0.1:8001/v1`, model `qwen3.8-flash-next`, and a
placeholder API key for the localhost native preset. It serves only one slot;
requests queue instead of running independent sequences concurrently. Native
HTTP usage includes EOS in completion counts, so its TPS must not be compared
numerically to the Atlas IPC script's visible-token TPS without normalization.
