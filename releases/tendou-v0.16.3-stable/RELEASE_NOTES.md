# Tendou Launcher `v0.16.3-stable` Release Notes

**Release Date:** 2026-11-20  
**Platforms:** Kubuntu 26.04 Linux (`x86_64`, CUDA 13.2 `sm_120a`, GCC 14) & Windows 11 (`x86_64`, MSVC + CUDA 13.2 `sm_120;89;86`)  
**Default Speculative Policy:** Neural MTP **OFF** (`--mtp off` by default; opt-in only)

---

## Highlights & Key Engineering Achievements

1. **3-Regime Zero-Copy Hot-RAM Planner & GPU Shadow Page Reclamation (`llama-model-loader.cpp`, `llama-model.cpp`, `atlas-moe-cache.h`, `atlas-engine.cpp`)**
   - **Root Cause Solved:** Previously (`v0.16.2-stable`), GPU-offloaded weights (~5.7 GiB on 35B, ~5.8 GiB on 38B) stayed cached as dead file pages in the Linux OS Page Cache after `ggml_backend_tensor_set()` uploaded them to CUDA VRAM, starving Host RAM and causing slow cold turns. Conversely, naive whole-layer `mlock` on near-fitting models (e.g. `11.6 GiB` CPU MoE on a `14.9 GiB` laptop) trapped unused cold experts in unreclaimable RAM and caused swap/NVMe thrashing on unpinned layers.
   - **GPU Shadow Page Reclamation (`llama-model-loader.cpp`):** Immediately issues `posix_fadvise(POSIX_FADV_DONTNEED)` + `madvise(MADV_DONTNEED)` on strictly interior 4 KB pages of GPU-bound tensors after synchronous CUDA upload, reclaiming ~4.7–6.0 GiB of Host RAM before CPU MoE execution begins. Tightens whole-file `MAP_POPULATE` threshold from `20 GiB` to `10 GiB` (`llama-model.cpp`).
   - **3-Regime Zero-Copy Hot-RAM Planner (`compute_zero_copy_hot_ram_plan`, Gate 48):**
     - **Full-Fit (`total_cpu_moe <= avail_ram - guard`, e.g. `>=24 GiB` RAM):** `mode=zero-copy-mlock` — locks all prioritized CPU MoE layers directly in mmap with selective `MADV_HUGEPAGE`, 4-thread atomic `in_flight_bytes` live-floor reservation, Windows `SetProcessWorkingSetSizeEx` quota lift, and RAII `HotRamUnlockGuard` ensuring `munlock`/`VirtualUnlock` runs strictly before `llama_model_free()`.
     - **Near-Fit (`full_fit_budget < total_cpu_moe <= 1.5× avail_ram`, e.g. `Qwen3.6-35B-A3B` on 16 GiB RAM):** `auto` enters **`mode=zero-copy-prewarm`** — applies `MADV_HUGEPAGE` strictly to selected layers and pre-warms **`~9.4–10.6 GiB`** of prioritized reasoning/anchor/output layers into reclaimable Hot RAM (`MADV_WILLNEED` + 4-thread touch) without locking cold unused experts in `mlock` jail. Explicit `"on"` caps `mlock` to `<=60%` of available RAM. Eliminates warm swap/NVMe thrashing while slashing 35B Cold TTFT by **-52.9% to -67.9%** and boosting Cold Decode TPS by **+106% to +154%**.
     - **Heavy NVMe-Streaming (`total_cpu_moe > 1.5× avail_ram`, e.g. `Qwen3.8-Flash-Next` 75.8 GB):** Automatically enters **`mode=reclaimed-pagecache`** (zero whole-layer `mlock` starvation, zero 2 MiB THP read-amplification), letting Linux's dynamic page cache dedicate the reclaimed Host RAM pool to Zipfian-hot experts across all 46 CPU layers.
   - **Real Kernel Process Telemetry (`ui_telemetry.py`):** Live `/proc/<pid>/status` + `/proc/<pid>/stat` (and cached Windows `psutil.Process`) polling surfaces real `engine_rss_mb`, `engine_locked_mb`, `engine_threads`, `engine_cpu_pct`, `hot_ram_mb`, and `hot_ram_mode` (`zero-copy-prewarm` / `zero-copy-mlock` / `reclaimed-pagecache`) at 4 Hz in Tendou Studio.

2. **True 512-Bit AVX-512F+BW+VL + VNNI (`vpdpwssd`) 4-Row Fused Kernels (`quants.c` + Gate 47, 0-ULP Parity)**
   - Upgraded BOTH `ggml_vec_dot_iq4_xs_q8_K_x4` (`Qwen3.6-35B`) and `ggml_vec_dot_iq3_xxs_q8_K_x4` (`Qwen3.8-Flash-Next`) to native **512-bit ZMM + AVX-512 VNNI (`_mm512_dpwssd_epi32`)** 4-row fused kernels:
     - `IQ4_XS_x4`: Pre-permutes `q8_512` once per 4 rows (`_mm512_shuffle_i64x2(..., 0xD8)`, eliminating 8 `vperm2i128` per 64 weights) + VNNI `vpdpwssd` → **`59.89 GB/s` @ 8T (+18.3% vs v0.16.2 AVX2, +17.3% @ 16T)**.
     - `IQ3_XXS_x4`: Compile-time 128-byte `ksign_mask8_q2xs[128]` direct `__mmask64` sign table + VNNI `vpdpwssd` → **`+4.9% @ 8T / +26.0% @ 16T` vs v0.16.2 AVX2**.
   - Verified mathematically **0-ULP IEEE-754 bit-exact** against scalar/single-row reference across `256 / 768 / 2048` lengths with full signed int8 adversarial inputs including `-128 (0x80)` in **Gate 47**.

3. **ChatGPT-Style Persistent Chat History, Background-Resilient Streaming & Context Growth Curve (`ui_chats.py`, `ChatContext.tsx`, `PlaygroundPage.tsx`, `server.py`)**
   - New atomic JSON conversation store under `data/chats/` (unique per-write temp files, strict ID regex validation, multi-line title derivation & Markdown export) with REST endpoints (`GET/POST /api/ui/chats`, `GET/PUT/DELETE /api/ui/chats/{id}`, `POST /api/ui/chats/{id}/messages`, `GET /api/ui/chats/{id}/export`).
   - Root-mounted `ChatProvider` (`App.tsx`) never unmounts when switching between Studio tabs — streaming inference continues seamlessly in the background across tabs and multiple conversations, guards against double-send duplicate chats and delete/clear-while-streaming races, binds each chat to `X-Tendou-Session-Id: studio-${chatId}` for PrefixCache hits, checks client disconnect every 8 SSE token events for prompt Stop cancellation, and survives full browser reloads.
   - Real server-measured `timings` (`ttft_ms`, `decode_tps`, `prefill_tps`, `prefix_fork_kind`, `cached_tokens`) power the built-in **Context Growth Curve** & persistent Inspector across every turn, plus chat search, rename, pin, delete, regenerate, edit-and-resend, and `.md` file download export.

4. **Apple-Inspired Obsidian & Alabaster Design Language (`ui/src/`)**
   - Completely replaced the generic neon/emerald/Sparkles dark UI across all 6 Studio surfaces (`Chats`, `Overview`, `Models`, `Engine`, `Agents`, `Benchmarks`) with a restrained, high-craft dual-theme design system: **Obsidian Dark** (`#0A0A0C`) & **Alabaster Light** (`#F5F5F7`), native `color-scheme` dark/light scrollbars & form controls, subtle translucent chrome, hairline borders, champagne-bronze precision accent (`#D4A574`), tabular numerals, custom geometric monochrome [`TendouMark`](../../ui/src/components/TendouMark.tsx) (plus inline SVG favicon), accurate MTP Draft head display on Models, and zero-bloat [`MarkdownRenderer`](../../ui/src/components/MarkdownRenderer.tsx) with fenced code copy.

5. **Large-Model (`Qwen3.8-Flash-Next` 512E) Mid-Output Cut-Off Fix & Unlimited Output (`∞ No Limit`, `max_tokens = 0`)**
   - **Engine (`ggml.h`, `ggml.c`, `ggml-cpu.c`, `atlas-moe-cache.{cuh,cu}`, `ggml-cuda.cu`, `llama-graph.cpp`, `atlas-engine.cpp`):** Fixed early repetition loops and spurious `<|im_end|>` cut-offs (~456–528 tokens under `--boost`) on 512-expert `qwen4exp` models via a unified runtime `ggml_atlas_get_moe_keep_k` policy (`keep_k >= 2` floor across all 512E layers in decode & prefill, `K=3..4` on reasoning layers `15..30` and entry/output bands), 512E MoE routing weight renormalization (`ATLAS_CUDA_OP_MOE_RENORM` + `atlas_moe_renorm_weights_cpu`, preserving `v0.16.3` unrenormalized scaling on `n_expert < 512` `Qwen3.6-35B-A3B` and `s_reasoning_k=4` on 256E L2), unified CUDA/CPU `keep_k` & `bypass_gpu` accounting, and fast `top_k=40` + `repeat_penalty=1.08` (`repeat_last_n=64`) sampling on the `!spec` IPC path.
   - **AutoTune, Unlimited `max_tokens = 0` & Studio UX (`autotune.py`, `server.py`, `ui_chats.py`, `ui_api.py`, `ui/src/`):** AutoTune selects balanced `boost_level = 1` automatically for 512E / 122B models (`boost_level = 2` remains default for 35B); raised `MAX_OUTPUT_TOKENS` to `65536` and added **unlimited generation (`max_tokens = 0` / `∞ No Limit`)** across OpenAI/Anthropic endpoints (`MockAtlasBackend` + `atlas-engine.cpp` remaining context), persistent `ChatParams` (`default=4096`), `localStorage` default parameter persistence, and **Continue** one-click continuation when `finish_reason === "length"`.

---

## Empirically Verified Benchmarks (`v0.16.3-stable`, MTP OFF)

Hardware: NVIDIA RTX 5060 Laptop (8,151 MiB GDDR6) + AMD Ryzen 7 260 (8C/16T Zen 4 AVX-512 VNNI) + 14.9 GiB RAM, Kubuntu 26.04, CUDA 13.2 (`sm_120a`), GCC 14.3, Neural MTP OFF (`--mtp off`).

### A. Controlled Direct GPU IPC Head-to-Head (`Qwen3.6-35B-A3B`, `-ngl 99 --atlas-gpu-first --atlas-gpu-layers 8 -t 8 -c 8192`)

| Mode | Engine Version | Cold Load | T1 Cold TTFT / TPS | T2 Warm TTFT / TPS | T3 Cold-Prompt TTFT / TPS | T4 Warm TTFT / TPS | Hot RAM (`VmRSS`) |
|---|---|---:|---:|---:|---:|---:|---:|
| Boost L2 (`K=5`) | `v0.16.2-stable` | ~10.7 s | 1,750 ms / 26.48 TPS | 78.81 TPS | 700 ms / 59.88 TPS | 83.70 TPS | ~8,600 MiB |
| Boost L2 (`K=5`) | **`v0.16.3-stable`** | **6.4 s** | **680.1 ms / 67.15 TPS (+153.6%)** | **9.8 ms / 83.27 TPS (+5.7%)** | **327.2 ms / 75.79 TPS (+26.6%)** | **10.0 ms / 85.21 TPS (+1.8%)** | **10,608 MiB** (`zero-copy-prewarm`) |
| Full `K=8` (`--no-boost`) | **`v0.16.3-stable`** | **6.4 s** | — | **10.5 ms / 34.28 TPS** | — | **9.9 ms / 32.41 TPS** | **Zero swap thrashing (-20.2% NVMe I/O vs naive mlock)** |

### B. Official Dual-Protocol Agentic Benchmark (`docs/json/benchmark_v16_3_kubuntu_report.json`)

| Model | Workload | Protocol | `v0.16.2` Cold TTFT / Cold TPS / Warm TPS | **`v0.16.3-stable` Cold TTFT / Cold TPS / Warm TPS** |
|---|---|---|---|---|
| `Qwen3.6-35B-A3B` | W1 Refactor (`pld d*=3`) | OpenCode | 2,063.2 ms / 30.88 / 83.82 TPS | **663.1 ms (-67.9%) / 69.10 (+123.8%) / 81.59 TPS** |
| `Qwen3.6-35B-A3B` | W2 Greenfield (`mtp=off`) | OpenCode | 1,347.0 ms / 28.83 / 63.52 TPS | **634.3 ms (-52.9%) / 59.41 (+106.1%) / 78.11 (+23.0%) TPS** |
| `Qwen3.6-35B-A3B` | W3 Tool Stream (`mtp=off`) | OpenCode | 3,806.1 ms / 58.09 / 97.66 TPS | **2,547.6 ms (-33.1%) / 74.05 (+27.5%) / 94.94 TPS** |
| `Qwen3.6-35B-A3B` | W3 Tool Stream (`mtp=off`) | Claude Code | 97.8 ms / 102.03 / 94.14 TPS | **94.4 ms / 102.79 (NEW PEAK) / 97.08 TPS** |
| `Qwen3.6-35B-A3B` | W3 Tool Stream (`mtp=off`) | Cline | 1,157.2 ms / 63.08 / 84.20 TPS | **856.9 ms (-26.0%) / 70.64 (+12.0%) / 79.82 TPS** |
| `Qwen3.8-Flash-Next` (`75.8 GB`) | W2 Greenfield (`mtp=off`) | Claude / Cline | 20.7 / 16.4 ms · 33.39 / 36.10 TPS | **19.2 / 15.9 ms · 34.56 / 35.71 TPS** (`reclaimed-pagecache`) |
| `Qwen3.8-Flash-Next` (`75.8 GB`) | W3 Tool Stream (`mtp=off`) | OpenCode / Claude / Cline | 9,038.4 / 226.8 / 3,677.5 ms · 43.09 TPS | **8,953.2 / 243.9 / 3,562.4 ms · 42.46 TPS · Cline Cold +2.9%** |

*Peak VRAM: `Qwen3.6-35B-A3B` = `6,395.4 MiB`; `Qwen3.8-Flash-Next` = `6,989.25 MiB` (cold 75.8 GB NVMe-streaming turns remain NVMe I/O-bound on 14.9 GiB RAM while GPU shadow eviction frees ~4.7 GiB of Host RAM for OS page cache).*

---

## Verification Summary

- `atlas-native-tests`: **Gates 1–49 PASS** (including Gate 47 512-bit AVX-512 VNNI `IQ4_XS_x4` + `IQ3_XXS_x4` 0-ULP adversarial parity, Gate 48 3-regime Zero-Copy Hot-RAM planner across 7 cases, and Gate 49 512E Keep-K floor & renorm parity).
- `atlas-partition-tests`: **PASS**.
- Python regression (`pytest tests/unit tests/integration -q --tb=short`): **172 passed, 5 skipped** (`tests/phases` not re-run in latest post-patch cycle; prior full-repo `pytest` recorded `273 passed, 5 skipped`).
- **Live RTX 5060 GPU `Qwen3.8` & `Qwen3.6-35B` Unlimited-Output (`max_tokens=0`) Check (`ctx=16384`, `--boost`):** **PASS** — `38B` AutoTune L1 (`1294` + `944` tokens @ `9.24–9.89 TPS`, `finish=stop`, `loop=False`), `38B` Turbo L2 (`1378` tokens @ `10.41 TPS`, `finish=stop`, `loop=False`), and `35B` Turbo L2 (`2192` tokens @ `52.18 TPS`, `finish=stop`, clean RAII C++17, `loop=False`).
- **Interactive Headless Firefox E2E** against compiled `ui/dist/`: **PASS** (`∞ No Limit` `max_tokens=0` toggle, `localStorage` & reload persistence, outgoing `/v1/chat/completions` `"max_tokens": 0`, persisted chat params, 0 JS errors; prior 5-tab dark/light & chat race suite PASS).
- `Tendou Studio_0.16.3_amd64.deb` (`3.39 MiB`, `tsc --noEmit` + Vite build + Tauri v2 bundle): **PASS**.
- `patches/llama.cpp.patch`: regenerated (`47` tracked files) and verified clean against `llama.cpp` working tree (`git apply --check --reverse`).
