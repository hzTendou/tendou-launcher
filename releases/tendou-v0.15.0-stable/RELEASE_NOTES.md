# Tendou Launcher v0.15.0-stable

**Release Date:** 2026-11-14
**Platform:** Kubuntu Linux x86_64 (`g++-14`, CUDA 13.2 `sm_120a`, AVX-512 VNNI)
**Status:** Production Stable Release (`VERSION.txt`: `0.15.0-stable`)

---

## Key Changes in v0.15.0-stable

### 1. Engine Stability, Tool-Calling Quality & Prefix Cache (`atlas-engine.cpp`, `atlas-engine.h`, `ggml-cpu.c`)
- **Long-context VRAM cache fix:** Resolved the infinite eviction loop in `ExpertMemoryManager::demote_vram_to_ram_unlocked` under `-c 65536 --atlas-moe-cache-mib 2048` and clamped `CrossLayerGPUPinning::active_slots_per_layer`.
- **Cooperative cancellation & IPC robustness:** Integrated `llama_set_abort_callback`, mutex-synchronized `ipc_emit`, and periodic `5s` heartbeat (`hb`) signals.
- **Prefix cache & token emitter compile/runtime parity (`atlas-engine.cpp`):** Hoisted `PrefixCacheSlot`, `PrefixCache`, and `IpcTokenEmitter` to file/namespace scope so multi-turn prompt prefixes reuse warm KV state cleanly across requests.
- **Quality-safe prefill & boost Top-K tuning (`atlas-engine.cpp`, `ggml-cpu.c`):** Upgraded `execute_layer_wise_single_pass_prefill` from degenerate `K=2` to quality-safe layer-wise `K=4` (layers `0-1`, `15-30`, `44+`) / `K=3` (remaining layers), added `ggml_cpu_get_dyn_k()`, eliminated `K=1` from Boost Levels 1 & 2 (keeping `K=1` only for extreme Level 3), and restricted reasoning downgrade to `g_atlas_dyn_k >= 3`.

### 2. API Server & Tool-Calling Resilience (`src/atlas/server.py`, `start_server.sh`)
- **Root-cause fix for `<response_to_instruction>` loops & failed tool calls:** Switched default inference configuration from `--boost` (level 2) to quality-first `--no-boost --atlas-k 5` (`expert_k=5`, `--atlas-readback-warmup 0` when `expert_k <= 5`), raised Qwen3.8 default `ATLAS_PREFILL_TOP_K` from `"2"` to `"4"`, updated omitted Anthropic sampling defaults to `(0.6, 0.95)`, and trimmed leading unsent `<think>` prose in `StreamingTagRouter.finish()` when a `<tool_call>`/`<function=` tag completes at EOF.
- **Module-scope `convert_anthropic_payload` & fast-lane stability:** Hoisted `convert_anthropic_payload` to module scope, preserved non-blocking `< 5ms` side-query fastlane responses (`generate_session_title_schema`) even under busy GPU inference, and verified clean OpenAI/Anthropic streaming tool-call extraction without stop-token leaks.
- **Versioned release resolution & streaming compatibility:** Prioritized `releases/tendou-v0.15.0-stable/bin/tendou-engine`, exposed `"version"` and `"exe_path"` on health/debug endpoints, added `HEAD`/`GET /api/hello`, queued keepalive pings behind busy GPU locks, and mapped `context_overflow` to HTTP 400.

### 3. Build & Packaging Scripts (`scripts/build_runtime.sh`, `scripts/build_cpu_kernel.sh`)
- Syncs updated engine binaries (`tendou-engine`, `llama-atlas-engine`, `llama-server`), shared libraries (`lib*.so*`), and tracked C++/Python sources into `releases/tendou-v0.15.0-stable/`.

---

## Packaged Artifacts (`releases/tendou-v0.15.0-stable/`)

### `bin/` (Linux x86_64)
| File | Size (Bytes) | Notes |
|---|---:|---|
| `tendou-engine` | `770,840` | Primary patched MoE engine (`sha256: adb78c0e33309a69b3c56c57f02ba69149d11158268085992d866eaa4401f25e`) |
| `llama-atlas-engine` | `770,840` | Alias binary matching `tendou-engine` (`sha256: adb78c0e33309a69b3c56c57f02ba69149d11158268085992d866eaa4401f25e`) |
| `llama-server` | `17,904` | Native server launcher (`sha256: 3dfdf595f345aaa802f4d471938a6233abf3e77300879295fb94d8d925b04599`) |
| `libggml-cuda.so.0.22.0` | `48,324,600` | CUDA backend (`libggml-cuda.so` -> `.0` -> `.0.22.0`) |
| `libllama-server-impl.so` | `7,126,016` | Native HTTP server implementation shared library |
| `libllama-common.so.0.3.0` | `6,072,520` | Common helper library (`libllama-common.so` -> `.0` -> `.0.3.0`) |
| `libllama.so.0.3.0` | `4,782,464` | Core LLaMA library (`libllama.so` -> `.0` -> `.0.3.0`) |
| `libmtmd.so.0.3.0` | `1,986,560` | Multimodal helper library (`libmtmd.so` -> `.0` -> `.0.3.0`) |
| `libggml-cpu.so.0.22.0` | `1,510,424` | AVX-512 VNNI CPU backend (`libggml-cpu.so` -> `.0` -> `.0.22.0`) |
| `libggml-base.so.0.22.0` | `941,728` | GGML core runtime (`libggml-base.so` -> `.0` -> `.0.22.0`) |
| `libggml.so.0.22.0` | `59,880` | GGML backend dispatcher (`libggml.so` -> `.0` -> `.0.22.0`) |

> Note: Windows `.exe`/`.dll` artifacts are not built or bundled in this Linux release directory; `_resolve_default_engine_path()` falls back to `releases/tendou-v0.14.0-stable/bin/tendou-engine.exe` on Windows until a Windows build is synced.

### `source/`
- `src/atlas/{atlas-engine.cpp, atlas-engine.h, atlas-iouring-dma.h, autotune.py, CMakeLists.txt, __init__.py, native_server.py, server.py}` (plus top-level flat copies of `atlas-engine.cpp`, `atlas-engine.h`, `atlas-iouring-dma.h`, `autotune.py`, `server.py`)
- `llama.cpp/ggml/{include/ggml-cpu.h, src/ggml-cpu/ggml-cpu.c, src/ggml-cpu/arch/x86/quants.c}`

---

## Verification Summary (2026-11-14)

- **Native Tests:** `./llama.cpp/build-linux/bin/atlas-partition-tests` & `./llama.cpp/build-linux/bin/atlas-native-tests` -> **PASS** (exit 0)
- **Python Unit & Integration Suite:** `./.venv-audit/bin/python -m pytest -q --tb=short tests/unit tests/integration/test_ipc_resilience.py tests/integration/test_openai_server.py tests/integration/test_anthropic_messages.py tests/integration/test_cline_integration.py tests/integration/test_qwen35_protocol_and_streaming.py` -> **135 passed, 4 skipped** (`17.62s`)
- **Real GPU Engine IPC & Regression Suite (`ATLAS_TEST_REAL=1`):** `tests/integration/test_engine_ipc_e2e.py` & `tests/integration/test_runtime_regressions.py` -> **15 passed** (`54.58s`)
- **Live Server Multi-Turn Verification (`./start_server.sh --port 18090`, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`):**
  - **Claude Code Turn 1 Replay (`11,465` prompt tokens):** `stop_reason="tool_use"` (`Write` -> `/tmp/tmpjbv1q8mn/number_guess.py`, `508` output tokens, `4.88 TPS` decode, zero leaked `<response_to_instruction>`/`<|im_end|>`/`<function=` tags).
  - **Side-Query Fast-Lane (`generate_session_title_schema`):** Sequential `4.41 ms`, Concurrent behind busy GPU `1.77 ms` (`200 OK`, valid JSON).
  - **Claude Code Turn 2 Continuation (`11,976` prompt tokens):** Warm `PrefixCache` hit (`11,264` cached tokens), TTFT dropped **15.2x** (`245,472.9 ms` -> `16,149.8 ms`), `stop_reason="tool_use"` (`Bash`, `131` output tokens, `4.18 TPS` decode).
  - **OpenAI `/v1/chat/completions` Streaming:** `finish_reason="tool_calls"` (`write_file`, valid JSON arguments, `4.17 TPS` decode, zero leaked stop/tag tokens), clean shutdown with `0` stray GPU processes.

---

## Quick Start

```bash
./start_server.sh --port 8000
```
