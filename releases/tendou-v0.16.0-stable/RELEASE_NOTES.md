# Tendou Launcher v0.16.0-stable

**Release Date:** 2026-11-19
**Platform:** Kubuntu Linux x86_64 (`g++-14`, CUDA 13.2 `sm_120a`, AVX-512 VNNI, kernel `ntfs3`)
**Status:** Production Stable Release (`VERSION.txt`: `0.16.0-stable`)

---

## Key Changes in v0.16.0-stable (Phase 17 + Phase 18)

### 1. Long-Context FP8 E4M3 KV Cache & FlashAttention VEC (`fattn.cu`, `ggml.c`, `llama-kv-cache.cpp`)
- Added native IEEE/OCP FP8 E4M3 (`GGML_TYPE_F8_E4M3`) KV cache (`-ctk f8_e4m3 -ctv f8_e4m3`) across CPU/CUDA conversion, `set_rows`, `get_rows`, `cpy`, `build_rope_shift`, and FlashAttention (`fattn-vec-instance-f8_e4m3-*`).
- Fixed `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:614-635`, `672-685`) so `GGML_TYPE_F8_E4M3` takes the low-precision non-MMA branch and runs `BEST_FATTN_KERNEL_VEC` directly for `Q->ne[1] <= 2` at long contexts (`8k–64k`, `gqa_ratio > 4`) without full-KV F16 dequantization (`nsys` verified at `-c 16384`: `flash_attn_ext_vec<256, 1, 43, 43>` = **`7.05 µs` avg**).

### 2. Layer-Concentrated Unified VRAM+RAM MoE Cache & Multi-Token GPU Verify (`atlas-moe-cache.{h,cu}`, `llama-graph.cpp`, `ggml-cpu.c`, `atlas-engine.cpp`)
- Replaced thin 48-layer VRAM spread (`~9.5 slots/layer`, 48 graph splits) with layer-concentrated VRAM pinning (`--atlas-vram-layer-cap auto` -> `7/48` hot boundary layers `0xf00000000007`, eliminating 41 cross-backend `gpu_moe` + `ggml_add` splits).
- Added adaptive `<= 16 GiB` OS RAM lock guard (`--atlas-ram-lock-policy auto|on|off`) so `K <= 5` cold/large-model sessions preserve Linux `mmap` page cache instead of losing 2 GiB to `mlock`, jumping `K=5 --atlas-boost` from `4.43 TPS` (Phase 17) to **`6.07 TPS` (`q8_0`) / `6.00 TPS` (`f8_e4m3`)** (surpassing pre-Phase 17 `5.90 TPS`).
- Extended Tier-0 CUDA MoE cache + CPU split execution from `n_tokens == 1` to **`1 <= n_tokens <= 8`** (`atlas-moe-cache.cu`, `ggml-cpu.c`) so speculative verification batches (`MTP Q4_0` / `PLD`, `BS=2..4`) compute cached experts on GPU cleanly.
- Added PR #31047/#31533/#32067/#32159/#32385/#32392 CUDA MoE kernel upgrades, `GGML_OP_CUSTOM` fused `qwen4exp` HC kernels, PR #32114 `n_ffn_chunk`, and `v1.0.0` `sched_reserve` reuse.

### 3. Race-Free 3-Tensor NVMe -> Tier-0 VRAM `DirectStorageIoUringDMA` (`atlas-iouring-dma.h`, `atlas-moe-cache.h`, `atlas-engine.cpp`)
- Implemented `UnifiedTieredExpertCache::{reserve,commit,abort}_async_vram_slot` and `DirectStorageIoUringDMA::{acquire_free_slot,enqueue_expert_3tensor_read,poll_job_complete,abort_job}` with `SLOT_SIZE - SECTOR_ALIGNMENT` chunking and non-blocking `CUevent` completion, fixing single-chunk/miss-drop bugs and proving bit-for-bit 3-tensor unaligned NVMe -> VRAM parity (Gate 41).

### 4. Opt-In `MTP Q4_0` Support, Multi-Agent Swarm Priority Scheduler & Hierarchical KV Tree Forking (`server.py`, `atlas-engine.{h,cpp}`)
- Fixed `server.py` `--atlas-mtp-path` forwarding for `"ram"`, `"atlas"`, and `"auto"` modes while keeping MTP **OFF by default** (`--mtp off` unless explicitly enabled). In opt-in `K=5 --no-boost + MTP Q4_0 draft_n=3`, `Qwen3.8-IQ3_XXS` reached **`6.02 TPS`** (**29/29 = 100% accept**, +39.7% over `4.31 TPS` no-boost baseline); `K=5 --atlas-boost + MTP Q4_0 draft_n=3` reached **`6.20 TPS`**.
- Replaced FIFO `self.lock` in `AtlasSubprocessBackend` with `_PriorityEngineGate` (`priority=0` fast-lane, `priority=1` warm session affinity, `priority=2` cold with `15s` anti-starvation aging) and exposed `queue_depth`, `active_session_id`, `last_session_id`, and `prefix_fork_counts` on `/v1/debug/state`.
- Upgraded `PrefixCache` (`kMaxSlots = 16`) with protected shared system/tool `is_root_anchor` checkpoints for zero-cost multi-agent KV Tree Forking: Agent B sharing Agent A's system block cut TTFT **`23,369.6 ms -> 2,713.5 ms` (`8.61x` faster)**.

---

## Packaged Artifacts (`releases/tendou-v0.16.0-stable/`)

### `bin/` (Linux x86_64)
| File | Size (Bytes) | Notes |
|---|---:|---|
| `tendou-engine` | `844,096` | Primary patched MoE engine (`sha256: a0e43428f502f33b733b6ca9ed9600a8a84913148be6b08f3a0bc369068f0805`) |
| `llama-atlas-engine` | `844,096` | Alias binary matching `tendou-engine` |
| `llama-server` | `17,904` | Native server launcher (`sha256: 3dfdf595f345aaa802f4d471938a6233abf3e77300879295fb94d8d925b04599`) |
| `libggml-cuda.so.0.22.0` | `55,167,560` | CUDA `sm_120a` backend (`FP8 KV`, `atlas-moe-cache`, `atlas-hc`, PR #31047/#31533/#32067/#32159) |
| `libllama-server-impl.so` | `7,126,016` | Native HTTP server implementation shared library |
| `libllama-common.so.0.3.0` | `6,072,520` | Common helper library |
| `libllama.so.0.3.0` | `4,796,096` | Core LLaMA library (`n_ffn_chunk`, `sched_reserve` reuse, layer-masked `gpu_moe`) |
| `libmtmd.so.0.3.0` | `1,986,560` | Multimodal helper library |
| `libggml-cpu.so.0.22.0` | `1,545,192` | AVX-512 VNNI CPU backend (multi-token `BS=1..8` split bypass + layer mask) |
| `libggml-base.so.0.22.0` | `942,032` | GGML core runtime (`GGML_TYPE_F8_E4M3`) |
| `libggml.so.0.22.0` | `59,880` | GGML backend dispatcher |

> Note: `releases/tendou-v0.15.0-stable/` is preserved untouched as a clean rollback checkpoint. Windows `.exe`/`.dll` artifacts are not bundled in this Linux release directory.

---

## Verification Summary (Kubuntu 26.04, RTX 5060 Laptop 8151 MiB, Ryzen 7 260)
- `atlas-partition-tests`: **PASS**
- `atlas-native-tests` (Gates 1–44, including Phase 18 Gates 41–44): **PASS**
- Unit & Protocol Integration Pytest (`tests/unit`, `test_openai_server.py`, `test_anthropic_messages.py`, `test_ipc_resilience.py`, `test_qwen35_protocol_and_streaming.py`): **`135 passed, 4 skipped`** (`18.30s`)
- Real GPU E2E & Regressions (`ATLAS_TEST_REAL=1`: `test_engine_ipc_e2e.py`, `test_runtime_regressions.py`): **`15 passed`** (`53.05s`)
- Live `./start_server.sh --port 18090` HTTP smoke test (`/health`, `/v1/messages`, `/v1/chat/completions`, `/v1/debug/state`): **PASS** (`version="0.16.0-stable"`, ready in `2.3s`)
- Serial Hardware Benchmark Report: `docs/json/benchmark_phase18_kubuntu_report.json`
