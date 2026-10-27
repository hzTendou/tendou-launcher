# llama.cpp integration

ATLAS V1 is implemented as a native llama.cpp example target. The integration deliberately uses existing llama.cpp primitives rather than an offline trace collector:

- tensor buffer overrides keep routed expert matrices on CPU/file-backed storage;
- `LLAMA_LOAD_MODE_MMAP` prevents Atlas from attempting to materialize the full expert corpus in RAM;
- `ggml_backend_sched_eval_callback` observes the final `ffn_moe_topk-*` tensor during live inference;
- a causal transition predictor ranks same-layer next-token experts;
- the prefetcher asks the OS to page those expert planes in early.

The callback is telemetry and scheduling only. It never mutates router logits, expert IDs, weights, or sampling.

## qwen4exp status

The supplied llama.cpp source has `qwen3next` and `qwen35moe`, but no `qwen4exp` architecture enum/graph/loader. The supplied Qwen3.8 Flash Next metadata declares `general.architecture=qwen4exp`. Therefore the runtime layer is ready, but exact Qwen3.8 execution requires a model adapter implementing that architecture.
