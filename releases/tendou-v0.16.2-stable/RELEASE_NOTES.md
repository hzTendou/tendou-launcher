# Tendou Launcher `v0.16.2-stable` Release Notes

**Release Date:** 2026-11-20  
**Platforms:** Kubuntu 26.04 Linux (`x86_64`, CUDA 13.2 `sm_120a`, GCC 14) & Windows 11 (`x86_64`, MSVC + CUDA 13.2 `sm_120;89;86`)  
**Default Speculative Policy:** Neural MTP **OFF** (`--mtp off` by default; opt-in only)

---

## Highlights & Key Engineering Achievements

1. **Boost-Aware Non-Contiguous GPU MoE Layer Placement (`atlas-engine.cpp` + `compute_auto_plan`)**
   - `compute_auto_plan` computes exact per-layer MoE VRAM size (`expert_layer_vram_mb`) from GGUF metadata and available GPU VRAM after dense layers, KV budget, and `1024 MiB` safety headroom.
   - For `Qwen3.6-35B-A3B` (`16.51 GiB`, 40L × 256E), AutoTune places `6–8` MoE layers on GPU VRAM (`vram_peak = 6,378.8 MiB`, well within 8,151 MiB RTX 5060).
   - **Key Empirical Discovery:** Naive contiguous GPU placement (`layers 0..7`) displaced Dynamic-K=1 CPU layers (`2..7`) onto full Top-8 GPU execution, reducing warm decode from `80.06` to `71.73 TPS`. By offloading **non-contiguous high-K reasoning layers (`0..1` + `15..20`)** to GPU while retaining Dynamic-K=1 layers (`2..14`) on AVX2/AVX-512 CPU, `v0.16.2-stable` simultaneously achieved **-24% to -39% faster Cold TTFT** and **+6% to +20% faster Warm Decode TPS** (up to **102.03 TPS** on Claude Code W3 and **97.66 TPS** on OpenCode W3!).
   - For `Qwen3.8-Flash-Next` (`75.8 GB`, 48L × 512E), AutoTune preserves the safe `<= 2` GPU layer cap (`vram_peak = 6,934.8 MiB`, zero OOM). Explicit `--gpu-layers` / `--vram-layer-cap` flags are never overridden.

2. **4-Row Fused AVX2 `ggml_vec_dot_iq3_s_q8_K_x4` Kernel + Gate 46 (0-ULP Parity)**
   - Cuts `Q8_K` activation loads by 4× for mixed `IQ3_S` MoE layers inside Unsloth `UD-IQ4_XS` quantizations across all 4 `MUL_MAT` / `MUL_MAT_ID` dispatch paths and startup CPU quant benchmarking.
   - 100% GCC & MSVC compatible (`<immintrin.h>`, `GGML_BACKEND_API` DLL export, clean non-AVX2 fallback).
   - Verified bit-exact (**0-ULP IEEE-754 identity**) across single-block (`256`), odd multi-block (`768`), and even multi-block (`1024`) vector lengths in **Gate 46**.

3. **Context-Adaptive KV Cache Quantization (`q8_0 <= 8K`, `f8_e4m3 >= 16K`)**
   - Automatically selects `q8_0` at `ctx_size <= 8192` and `f8_e4m3` (FP8 E4M3) at `ctx_size >= 16384` (~50% KV VRAM reduction without Q4_0 degradation) unless the user passes explicit `-ctk`/`-ctv`/`--fp8-kv` overrides.

4. **Complete Windows 11 OS Integration (`MSVC + CUDA 13.2`)**
   - Audited & fixed Windows header/MSVC compatibility (`NOMINMAX` before `<windows.h>` in `atlas-moe-cache.h`, `<malloc.h>` in `atlas-iouring-dma.h`, `GGML_BACKEND_API` DLL exports in `quants.h`/`ggml-cpu.h`, CUDA/Win32 libraries in `src/atlas/CMakeLists.txt`).
   - Added one-click `scripts\build_windows_all.cmd` (clean CMake + `vswhere` + Ninja/MSBuild + Gates 1–46 verification), modernized `scripts\build_runtime.cmd`, `scripts\build_native_tests.cmd`, and `start_server.cmd` (1:1 parity with `start_server.sh`), plus `docs/WINDOWS_USER_GUIDE.md`.
   - Added `CREATE_NEW_PROCESS_GROUP` + graceful `CTRL_BREAK_EVENT` shutdown and `os.add_dll_directory` / `PATH` resolution in `src/atlas/server.py`.

5. **Tendou Studio Enterprise Web App & Tauri v2 Desktop App**
   - New FastAPI Control Plane (`/api/ui/*`, 14 REST + 4 Hz SSE Live TPS & Log endpoints in `src/atlas/{ui_config.py,ui_telemetry.py,ui_api.py}`).
   - Built-in Enterprise React + TypeScript + Tailwind SPA (`ui/dist/` served directly at `/` with 0 MB overhead), featuring always-visible Live TPS Counter + Sparkline, Streaming Playground with per-message TPS, all 52 CLI flags with live Linux/Windows CLI serializer, Hot-Swap model switcher, Agent Hub (Claude Code / OpenCode / Cline), and Benchmarks & Console Logs.
   - Cross-platform Tauri v2 Desktop wrapper (`desktop/src-tauri/`) with Native System Tray, Smart Backend Attach/Spawn, and zero-zombie cleanup (Linux `SIGTERM` + Windows Win32 Job Object).

---

## Verified Head-to-Head Kubuntu Benchmarks (`docs/json/benchmark_v16_2_kubuntu_report.json`)

Hardware: NVIDIA RTX 5060 Laptop (8,151 MiB GDDR6) + AMD Ryzen 7 260 (8C/16T, AVX-512 VNNI) + 14.9 GiB RAM, CUDA 13.2 (`sm_120a`), Neural MTP OFF (`--mtp off`).

| Model | Workload | Protocol | `v0.14.0` TPS / Cold TTFT | `v0.16.1` TPS / Cold TTFT | **`v0.16.2-stable` TPS / Cold TTFT** |
|---|---|---|---|---|---|
| `Qwen3.6-35B-A3B` | W1 Refactor (`pld d*=3`) | OpenCode | — / 6,435 ms | 76.11 / 2,110 ms | **83.82 TPS (+10.1%) / 2,063 ms** |
| `Qwen3.6-35B-A3B` | W1 Refactor (`pld d*=3`) | Claude / Cline | 83.11 / 82.88 | 83.95 / 80.83 | **87.34 / 86.32 TPS** |
| `Qwen3.6-35B-A3B` | W2 Greenfield (`mtp=off`) | Claude / Cline | 81.47 / 80.69 | 82.60 / 81.07 (1,769 ms) | **87.49 / 88.17 TPS (1,347 ms, -23.8% TTFT!)** |
| `Qwen3.6-35B-A3B` | W3 Tool Stream (`mtp=off`) | OpenCode / Claude | — / — (11,984 ms) | 81.41 / 96.96 (5,253 ms) | **97.66 (+20.0%) / 102.03 TPS (3,806 ms, -27.5% TTFT!)** |
| `Qwen3.8-Flash-Next` (`75.8 GB`) | W1 Refactor (`pld d*=3`) | OpenCode / Claude / Cline | — / 34.22 / 34.02 (16,212 ms) | 23.55 / 33.64 / 34.21 (4,876 ms) | **25.23 / 35.07 / 35.16 TPS (4,621 ms)** |
| `Qwen3.8-Flash-Next` (`75.8 GB`) | W2 Greenfield (`mtp=off`) | Cline | 20.26 (9,325 ms) | 35.77 (3,431 ms) | **36.10 TPS (3,333 ms)** |
| `Qwen3.8-Flash-Next` (`75.8 GB`) | W3 Tool Stream (`mtp=off`) | Claude / Cline | 34.83 / — (49,805 ms) | 43.00 / 27.63 (9,114 ms) | **43.09 / 28.97 TPS (9,038 ms)** |

## Native & Python Verification

- `atlas-native-tests`: **Gates 1–46 PASS** (including Gate 45 `IQ4_XS_x4` and Gate 46 `IQ3_S_x4` 0-ULP parity).
- `atlas-partition-tests`: **PASS**.
- `pytest` (`tests/unit`, `tests/integration`): **PASS**.
