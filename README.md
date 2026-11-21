<div align="center">

<img src="desktop/src-tauri/icons/icon.png" alt="Tendou Launcher Icon" width="112" />

# Tendou Launcher

### Independent MoE Inference Platform & Launcher for Gaming Hardware

Run massive Mixture-of-Experts (MoE) models locally with high throughput, low latency, and zero cloud dependencies on standard consumer laptops.

[![License: MIT](https://img.shields.io/badge/License-MIT-7c3aed.svg)](LICENSE)
![Version](https://img.shields.io/badge/Release-v0.16.3--stable-D4A574)
![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Linux & Windows 11](https://img.shields.io/badge/Platform-Kubuntu%20%7C%20Windows%2011-0078D4)
![CUDA 13.2](https://img.shields.io/badge/CUDA-13.2%20sm__120a-76B900?logo=nvidia&logoColor=white)
![AVX-512](https://img.shields.io/badge/SIMD-512--bit%20AVX--512%20x4%200--ULP-orange)
![Tests](https://img.shields.io/badge/Tests-48%20Native%20Gates%20%7C%20Unit%2FIntegration%20PASS-brightgreen)

**[Quick Start](#quick-start)** · **[Tendou Studio UI](#tendou-studio-web--desktop-app)** · **[Architecture](#architecture-overview)** · **[Benchmarks](#empirically-verified-benchmarks)** · **[Releases](#releases)** · **[Docs & Guides](#research--documentation-center)** · **[Agent Integration](#agent-integrations)**

</div>

---

## What is Tendou Launcher?

**Tendou Launcher (`v0.16.3-stable`)** is an **independent Mixture-of-Experts (MoE) inference platform, control plane, and desktop studio**. It runs massive MoE models (`Qwen3.6-35B-A3B` 16.5 GiB and `Qwen3.8-Flash-Next` 75.8 GB) at high token speeds (**up to 102.8 tokens/sec** on 35B and **42.5 tokens/sec** on 75.8 GB 38B, MTP strictly OFF by default) with zero Out-of-Memory crashes on everyday **8 GiB VRAM / 16 GiB RAM** laptops across **Linux (Kubuntu 26.04)** and **Windows 11**.

> [!IMPORTANT]
> **Independent Platform Principle:** Tendou Launcher is **NOT** a fork or simple wrapper of `llama.cpp`. It delivers:
> - **3-Regime Zero-Copy Hot-RAM Planner & GPU Shadow Page Reclamation** (`v0.16.3`): Evicts ~4.7–6.0 GiB of dead GPU-uploaded shadow pages from OS page cache; uses `zero-copy-mlock` when CPU MoE fits RAM cleanly, switches automatically to reclaimable `zero-copy-prewarm` (~9.5–10.6 GiB Hot RAM, selective THP) on near-fitting models (`Qwen3.6-35B` on 16 GiB RAM — **-52.9% to -67.9%** Cold TTFT and **+106% to +154%** Cold Decode TPS without swap thrashing), and switches to `reclaimed-pagecache` for 70+ GiB streaming models (**Gate 48**),
> - **True 512-Bit AVX-512F+BW+VL + VNNI (`vpdpwssd`) 4-Row Fused Kernels** (`ggml_vec_dot_iq4_xs_q8_K_x4`, `ggml_vec_dot_iq3_xxs_q8_K_x4`, `ggml_vec_dot_iq3_s_q8_K_x4`) cutting L1 activation loads by 75% (`+18.3%` @ 8T `IQ4_XS_x4`, `+26.0%` @ 16T `IQ3_XXS_x4` vs AVX2) with **0-ULP IEEE-754 parity (Gates 45–47)**,
> - **Boost-Aware Non-Contiguous GPU MoE Placement & AutoTune v0.16.3 Adaptive KV Cache** (`layers 0–1` + reasoning band `15–20` on GPU; `q8_0` at `<=8K`, `FP8 E4M3` at `>=16K` with strict `1,024 MiB` VRAM headroom),
> - **3-Tier Hybrid Memory Hierarchy**: CUDA GDDR6 VRAM (Tier 0) ↔ Zero-Copy Hot-RAM MMAP/THP (Tier 1) ↔ Async NVMe DMA (`io_uring` on Linux / `DirectStorage` & Overlapped I/O on Windows 11, Tier 2),
> - **Tendou Studio (`v0.16.3` Obsidian & Alabaster Edition — Web UI + Tauri v2 Desktop App)** with ChatGPT-style **persistent chat history (`data/chats/`)**, **background-resilient streaming** that survives tab switches and browser reloads, live **Context Growth Curve** backed by real server timings & PrefixCache fork telemetry, **real kernel Hot-RAM & CPU telemetry**, **all 52 CLI flags**, Zero-Zombie Hot-Swap, and Agent Hub for **Claude Code**, **OpenCode**, and **Cline**.
>
> `llama.cpp` and `ggml` serve strictly as the low-level tensor arithmetic backend (`patches/llama.cpp.patch`). Frozen stable releases live under [`releases/`](releases/).

---

## Tendou Studio (Web & Desktop App)

Starting the server automatically serves the prebuilt **Tendou Studio** React 19 + TypeScript + Tailwind v4 SPA at **`http://127.0.0.1:8000/`** (zero Node.js needed at runtime), redesigned in a restrained, Apple-inspired **Obsidian Dark (`#0A0A0C`) & Alabaster Light (`#F5F5F7`)** aesthetic with frosted glass chrome, hairline borders, champagne-bronze accents (`#D4A574`), and tabular numerals. An optional native **Tauri v2** desktop shell (`desktop/src-tauri/`) provides system-tray lifecycle management with zero-zombie cleanup (Unix process group & Win32 Job Object):

- **ChatGPT-Style Persistent Chats & Background-Resilient Streaming:** Full multi-conversation history stored atomically under `data/chats/` with search, pin, rename, delete, edit-and-resend, regenerate, and `.md` download export. Root-mounted `ChatProvider` keeps streams alive in the background across tab switches, binds each thread to `X-Tendou-Session-Id` for PrefixCache hits, and plots a live **Context Growth Curve** (context tokens vs real server TTFT, decode TPS & PrefixCache fork hits per turn).
- **Always-Visible Precision Telemetry Bar:** Live 4 Hz SSE decode TPS, EMA, TTFT, VRAM, **real kernel Hot-RAM (`VmRSS` / `VmLck` & mode)**, engine CPU utilization, and Start/Stop/Restart controls.
- **Engine Settings (All 52 CLI Flags):** 6 organized categories, one-click hardware presets (`MTP = OFF` by default), live Linux/Windows CLI serializer, and atomic profile persistence.
- **Models, Agents & Benchmarks:** Automatic GGUF scanner + AutoTune v0.16.3 planner + zero-zombie hot-swap, copyable Claude Code / OpenCode / Cline configs, benchmark history, and live SSE engine logs.

---

## Architecture Overview

Tendou Launcher solves the **MoE Memory Wall** through a 3-tier hybrid hierarchy coordinated by FastAPI (`/v1/*` + `/api/ui/*`) and a persistent IPC C++/CUDA runtime (`tendou-engine`):

```text
  Tendou Studio Web UI / Tauri v2  |  Claude Code (/v1/messages)  |  OpenCode & Cline (/v1/chat/completions)
                                                  │
                                                  ▼
               ┌────────────────────────────────────────────────────────────────────┐
               │   FastAPI Control Plane, Chat Store & Telemetry Hub (src/atlas/)   │
               │   • Atomic Chat History (/api/ui/chats) + Context Growth Curve     │
               │   • 4 Hz SSE Live TPS, Kernel VmRSS/VmLck Hot-RAM & Engine CPU     │
               │   • PriorityEngineGate (Swarm P0/P1/P2) + 16-Slot PrefixCache      │
               │   • AutoTune v0.16.3 (Non-Contiguous GPU Layers + Adaptive FP8 KV) │
               └──────────────────────────────────┬─────────────────────────────────┘
                                                  │ Persistent Line-Delimited JSON IPC
                                                  ▼
               ┌────────────────────────────────────────────────────────────────────┐
               │   tendou-engine v0.16.3-stable (C++ / CUDA 13.2 / AVX-512 VNNI)    │
               ├──────────────────────┬───────────────────────┬─────────────────────┤
               │ Tier 0: CUDA VRAM    │ Tier 1: Hot-RAM THP   │ Tier 2: NVMe Stream │
               │ 8 GiB GDDR6 (sm_120a)│ GPU-Shadow Evicted    │ Linux: io_uring     │
               │ Lay 0–1 + 15–20 MoE  │ 3-Regime Prewarm/Lock │ Win11: DirectStorage│
               │ FP8 E4M3 FlashAttn   │ 512b AVX-512 VNNI x4  │ Reclaimed PageCache │
               │ 1,024 MiB OOM Guard  │ 0-ULP (Gates 45–48)   │ 3-Tensor 4KB DMA    │
               └──────────────────────┴───────────────────────┴─────────────────────┘
```

---

## Empirically Verified Benchmarks (`v0.16.3-stable`, MTP OFF)

*Hardware: AMD Ryzen 7 260 (8C/16T Zen 4 AVX-512 VNNI), NVIDIA GeForce RTX 5060 Laptop (8 GiB / 8,151 MiB VRAM), 14.9 GiB System RAM, Kubuntu 26.04 (CUDA 13.2 `sm_120a`, GCC 14.3). All runs strictly **`--mtp off`**, `--auto-plan --boost --fp8-kv`. Source JSON: [`docs/json/benchmark_v16_3_kubuntu_report.json`](docs/json/benchmark_v16_3_kubuntu_report.json).*

| Model | Size on Disk | AutoTune GPU Layers | Peak VRAM | Hot RAM Mode | W1 OpenCode Cold TTFT / Cold TPS | W2 Greenfield Cold TTFT & Warm TPS | W3 Tool-Call Cold TTFT & Peak TPS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3.6-35B-A3B** | **16.5 GiB** | **8 non-contiguous** (`0..1 + 15..20`) | **6,395.4 MiB** | **~9.5–10.6 GiB** (`zero-copy-prewarm`) | **663.1 ms (-67.9%) · 69.1 tok/s (+123.8%)** | **634.3 ms (-52.9%) · 78.1–85.1 tok/s** | **2,547.6 ms (-33.1%) · 102.8 tok/s** |
| **Qwen3.8-Flash-Next** | **75.8 GB** | **2 safe cap** (OOM Guard) | **6,989.3 MiB** | **Reclaimed OS Page Cache** (`reclaimed-pagecache`) | **4,662.1 ms · 6.4 tok/s** | **3,337.1 ms · 35.7 tok/s** | **8,953.2 ms · 42.5 tok/s** |

**Controlled Direct GPU IPC (`Qwen3.6-35B-A3B`, `v0.16.2` vs `v0.16.3`, `-t 8`, `-c 8192`, `--boost`):**
- **`Qwen3.6-35B-A3B` (`zero-copy-prewarm` + AVX-512 VNNI):** Cold load **`6.4s`**, Turn 1 Cold **`1,750 -> 680.1 ms`** / **`26.48 -> 67.15 TPS` (+153.6%)**, Turn 2 Warm **`78.81 -> 83.27 TPS` (+5.7%)**, Turn 3 Cold-Prompt **`700 -> 327.2 ms`** / **`59.88 -> 75.79 TPS` (+26.6%)**, Turn 4 Warm **`83.70 -> 85.21 TPS`**, Full `K=8` (`--no-boost`) **`34.28 / 32.41 TPS`** with zero swap thrashing.
- **AVX-512 VNNI SIMD Microbenchmark (0-ULP Gate 47):** `IQ4_XS_x4` **`59.89 GB/s` @ 8T (+18.3% vs AVX2, +17.3% @ 16T)**; `IQ3_XXS_x4` **`+4.9% @ 8T / +26.0% @ 16T` vs AVX2**.

---

## Releases

All production-grade, frozen stable releases are published under [`releases/`](releases/):

- 🌟 **[`releases/tendou-v0.16.3-stable/`](releases/tendou-v0.16.3-stable/)**: **Current Flagship Release**
  - Zero-Copy Boost-Aware Hot-RAM THP Pinning + GPU Shadow Page Reclamation (Gate 48), 512-bit AVX-512 `IQ4_XS_x4` & `IQ3_XXS_x4` kernels (Gate 47 0-ULP), Real Kernel Hot-RAM/CPU Telemetry, ChatGPT-style Persistent Chat History & Background-Resilient Streaming with Context Growth Curve, and Apple-inspired Obsidian & Alabaster Tendou Studio redesign.
  - Full Linux binaries (`bin/tendou-engine`, `bin/llama-server`, `libggml-*.so`), full source snapshot (`source/`), and [`RELEASE_NOTES.md`](releases/tendou-v0.16.3-stable/RELEASE_NOTES.md).
- **[`releases/tendou-v0.16.2-stable/`](releases/tendou-v0.16.2-stable/)**: Boost-Aware Non-Contiguous GPU MoE placement, `IQ3_S_x4` (Gate 46), Windows 11 integration, initial Tendou Studio Web/Tauri v2.
- **[`releases/tendou-v0.16.1-stable/`](releases/tendou-v0.16.1-stable/)**: 4-Row `IQ4_XS_x4` (Gate 45) & Synchronized Dynamic-K Turbo.
- **[`releases/tendou-v0.16.0-stable/`](releases/tendou-v0.16.0-stable/)**: FP8 E4M3 FlashAttention VEC (`7.05 µs`), 3-Tensor NVMe `io_uring` DMA, Multi-Agent Swarm IPC.
- **[`releases/tendou-v0.15.0-stable/`](releases/tendou-v0.15.0-stable/)** · **[`v0.14.0`](releases/tendou-v0.14.0-stable/)** · **[`v0.13.0`](releases/tendou-v0.13.0-stable/)** · **[`v0.11.0`](releases/tendou-v0.11.0-stable/)** · **[`v0.9.0`](releases/tendou-v0.9.0-stable/)** · **[`v0.8.0`](releases/tendou-v0.8.0-stable/)**.

---

## Research & Documentation Center

- 🐧 **Kubuntu Linux Guide:** [`docs/KUBUNTU_USER_GUIDE.md`](docs/KUBUNTU_USER_GUIDE.md)
- 🪟 **Windows 11 Build & Runtime Guide:** [`docs/WINDOWS_USER_GUIDE.md`](docs/WINDOWS_USER_GUIDE.md)
- 📚 **Technical Articles ([`docs/articles/`](docs/articles/)):** `RESEARCH_HISTORY_AND_METHODOLOGY.md`, `HYPOTHESIS_ANALYSIS_PROVEN_AND_REFUTED.md`, `ARCHITECTURE_AND_INFERENCE_ENGINE.md`, `claude_code_and_opencode_integration_guide.md`.
- 📊 **Empirical Validation Datasets ([`docs/json/`](docs/json/)):** Full hardware JSON benchmark archive (`v0.14.0` through `v0.16.3`).

---

## Quick Start

### Linux (Kubuntu 26.04 / CUDA 13.2)
```bash
git clone https://github.com/hzTendou/tendou-launcher.git
cd tendou-launcher
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Launch API Server + Tendou Studio Web UI (http://127.0.0.1:8000/)
./start_server.sh --port 8000

# Optional: Build & run native Tauri v2 Desktop App
./scripts/build_desktop_linux.sh
./desktop/src-tauri/target/release/tendou-studio
```

### Windows 11 (MSVC + CUDA 13.2)
```cmd
git clone https://github.com/hzTendou/tendou-launcher.git
cd tendou-launcher
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt

:: 1. Build native C++/CUDA/AVX2 engine + 48 native tests
scripts\build_windows_all.cmd

:: 2. Start API Server + Tendou Studio Web UI (http://127.0.0.1:8000/)
start_server.cmd --port 8000

:: Optional: Build Tauri v2 Windows Desktop App (.exe / NSIS / MSI)
scripts\build_desktop_windows.cmd
```

*(Quick smoke test without model weights on either OS: `python atlas_server.py --mock --port 8000` and open `http://127.0.0.1:8000/`.)*

---

## Agent Integrations

### 1. OpenCode
Tendou Launcher provides native compatibility with OpenCode:
```powershell
Copy-Item opencode.json.example opencode.json
.\opencode.cmd
```

### 2. Claude Code CLI
Connect Claude Code directly to Tendou Launcher's native Anthropic `/v1/messages` endpoint:
```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8000"
$env:ANTHROPIC_API_KEY = "tendou-local"
claude --model qwen3.5-122b-mxfp4
```

### 3. Cline (VS Code Extension)
In Cline settings:
- **API Provider:** `OpenAI Compatible` or `NVIDIA NIM`
- **Base URL:** `http://127.0.0.1:8000/v1`
- **API Key:** `tendou-local`
- **Model ID:** `qwen3.5-122b-mxfp4` (or `qwen3.8-flash-next`)

---

## Native Runtime Compilation

If you wish to compile the engine binary from source instead of using the pre-compiled [`releases/`](releases/) binaries:

```powershell
# Clone the pinned backend base
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
git -C llama.cpp checkout 67a17c17caa95742186f8b1ecadd1b5abd6d5ebb
git -C llama.cpp apply ..\patches\llama.cpp.patch

# Build runtime with CMake & MSVC
cmake -S llama.cpp -B llama.cpp\build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp\build --config Release --target llama-atlas-engine -j
```

---

## Verification & Testing

### Python Test Suite (270 Tests)
```bash
# Quick core unit & OpenAI server suite (124 tests)
python -m pytest tests/unit tests/integration/test_openai_server.py -q

# Full repository pytest suite (270 passed, 5 optional real-GPU/E2E skipped by default)
python -m pytest -q
```

### Native Regression & Gate Suites (46 Gates)
```bash
# Linux (Kubuntu 26.04)
./scripts/build_native_tests.sh
./llama.cpp/build-linux/bin/atlas-native-tests
./llama.cpp/build-linux/bin/atlas-partition-tests
```
```powershell
# Windows 11 (MSVC + CUDA 13.2)
cmd /c scripts\build_native_tests.cmd
cmd /c tests\native\build_partition_tests.cmd

.\llama.cpp\build\bin\Release\atlas-native-tests.exe
.\llama.cpp\build\bin\Release\atlas-partition-tests.exe
```

---

## Repository Map

```text
tendou-launcher/
├── releases/       # Standalone frozen stable releases (v0.8.0 – v0.16.2-stable)
├── ui/             # Tendou Studio React 19 + TypeScript + Tailwind v4 Web UI (built into ui/dist/)
├── desktop/        # Native Tauri v2 Desktop App (Linux .deb / Windows .exe)
├── docs/           # Documentation center: guides, articles, benchmark JSONs, audits, specs
│   ├── articles/   # Comprehensive research papers and integration guides
│   ├── json/       # Empirical benchmark and validation JSON datasets
│   ├── audits/     # Hardware and thread execution audits
│   ├── specs/      # Runtime specifications and Llama-4 optimization plan
│   └── history/    # Historical changelogs and experiment reports
├── tests/          # Complete test suite: unit, integration, native (49 gates), benchmarks, phases 0-11
│   ├── phases/     # Chronological development phase test suites (phase0 - phase11)
│   ├── native/     # C++ parity (Gates 1–49), memory benchmarking, and partition tests
│   └── integration/# OpenCode, Claude Code, Cline, and IPC resilience protocol tests
├── src/atlas/      # Multi-protocol API server, Tendou Studio control plane (/api/ui/*), autotune & C++ engine
├── scripts/        # Linux/Windows build commands and benchmark scripts
├── config/         # Runtime example profiles and hardware topologies
├── patches/        # Reproducible backend engine integration patches (patches/llama.cpp.patch)
├── scratch/        # Diagnostic, sweep, and ablation tools
└── data/           # Historical physical reports and telemetry traces
```

---

## License

Tendou Launcher is released under the [MIT License](LICENSE).

<div align="center">

Built with precision for extreme local MoE inference.

</div>
