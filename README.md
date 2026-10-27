<div align="center">

# Tendou Launcher

### Independent MoE Inference Platform & Launcher for Gaming Hardware

Run massive Mixture-of-Experts (MoE) models locally with high throughput, low latency, and zero cloud dependencies on standard consumer laptops.

[![License: MIT](https://img.shields.io/badge/License-MIT-7c3aed.svg)](LICENSE)
![Version](https://img.shields.io/badge/Release-v0.14.0--stable-blue)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Windows 11](https://img.shields.io/badge/Windows-11-0078D4?logo=windows11&logoColor=white)
![CUDA 13](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![AVX-512 VNNI](https://img.shields.io/badge/AVX--512-VNNI%20x4%2Fnrc2-orange)
![Tests](https://img.shields.io/badge/Tests-109%20PASS%20%7C%2028%20Gates%20PASS-brightgreen)

**[Quick Start](#quick-start)** · **[Architecture](#architecture)** · **[Benchmarks](#empirically-verified-benchmarks)** · **[Releases](#releases)** · **[Research & Docs](#research--documentation-center)** · **[Agent Integration](#agent-integrations)**

</div>

---

## What is Tendou Launcher?

**Tendou Launcher** is an **independent Mixture-of-Experts (MoE) inference platform and local launcher**. It is designed from scratch to run large MoE models (such as Qwen 3.5 122B, Qwen 3.8 Flash Next 197B, and Qwen 3.6 35B) at practical token generation speeds (**20–72 tokens/sec**) on everyday hardware—specifically targeting gaming laptops with **8 GiB VRAM and 16 GiB RAM**.

> [!IMPORTANT]
> **Independent Platform Principle:** Tendou Launcher is **NOT** a fork, mod, or wrapper of `llama.cpp`. It features its own:
> - **Hierarchical Working Set Manager (HWSM)** with Dynamic 3-Tier memory tiering (VRAM Tier-0 ↔ RAM Tier-1 ↔ DirectStorage Tier-2),
> - **True Zero-Syscall Slab Arenas** (`ExpertSlabPool` with `VirtualLock` pre-allocation),
> - **Hardware-optimized AVX-512 VNNI CPU kernels** (`ggml_vec_dot_mxfp4_q8_0_x4` & `_nrc2`) with 0-ULP bit-exact parity,
> - **Unified Zero-Prefill Prompt KV Caching** for instant multi-turn conversational responses,
> - **DirectStorage NVMe DMA contiguous range coalescing**,
> - **Multi-Protocol Agent Server** natively serving OpenCode, Claude Code CLI, and Cline.
>
> `llama.cpp` and `ggml` are utilized strictly as an underlying low-level tensor arithmetic and compute backend dependency. All frozen production binaries and sources are published directly under [`releases/`](releases/).

---

## Architecture Overview

Tendou Launcher solves the **MoE Memory Wall** by decoupling activated parameters from resident weight storage through a dynamic 3-tier memory hierarchy:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TENDOU LAUNCHER ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────────┘

       OpenCode Client          Claude Code CLI             Cline (NIM/XML)
   (/v1/chat/completions)     (/v1/messages, /messages)   (/v1/chat/completions)
             │                           │                          │
             └───────────────────────────┼──────────────────────────┘
                                         ▼
                 ┌───────────────────────────────────────────┐
                 │         Multi-Protocol API Server         │  (src/atlas/server.py)
                 │  • Zero-overhead incremental JSON stream  │
                 │  • Anthropic <-> ChatML bi-directional    │
                 │  • Cline XML parser & single-tool breaker │
                 │  • Dynamic Speculative Horizon (d* in 0..3│
                 └─────────────────────┬─────────────────────┘
                                       │ Line-delimited JSON IPC
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       tendou-engine.exe (C++ Native)      │  (releases/v0.14.0-stable)
                 │  • Unified Zero-Prefill Prompt KV Cache   │
                 │  • Hierarchical Working Set Manager (HWSM)│
                 │  • Zero-Syscall ExpertSlabPool Arenas     │
                 │  • DirectStorage NVMe DMA Coalescing      │
                 └─────────────────────┬─────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
 │  Tier-0 Hot  │              │  Tier-1 Warm │              │  Tier-2 Cold │
 │   GPU VRAM   │ ◄──────────► │   Host RAM   │ ◄──────────► │  DirectStorage│
 │  (8GB GDDR6) │   Promotion  │ (VirtualLock)│   Demotion   │ (NVMe 4KB DMA│
 │ CUDA Compute │              │ AVX-512 VNNI │              │  Coalesced)  │
 └──────────────┘              └──────────────┘              └──────────────┘
```

---

## Empirically Verified Benchmarks

*Hardware: AMD Ryzen 7 (8C/16T, DDR5-5600), NVIDIA GeForce RTX 5060 Laptop (8 GiB VRAM), 16 GiB System RAM, Windows 11.*

| Model | Quantization | Total / Active Params | Layer / Expert Count | Warm Decode TPS | Warm TTFT | VRAM Peak | Speedup vs Baseline |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3.5-122B-A10B** | **MXFP4_MOE** | 122B / ~10B | 48L / 256E (Top-8) | **24.40 – 33.74 tok/s** | **21.7 – 84.1 ms** | 7,240 MiB | **6.4x – 8.9x vs Q6_K_P** |
| **Qwen3.8-Flash-Next** | **IQ3_XXS** | 197B / ~14B | 48L / 512E (Top-10) | **24.30 – 34.48 tok/s** | **20.1 – 43.3 ms** | 7,190 MiB | **4.2x vs Cold Start** |
| **Qwen3.6-35B-A3B** | **UD-IQ4_XS** | 35B / ~3B | 40L / 256E (Top-8) | **52.49 – 72.12 tok/s** | **14.7 – 39.3 ms** | 5,950 MiB | **2.8x vs Plain CPU** |

*All metrics are verified with 100% deterministic bit-exact parity (0-ULP) and KL-divergence $\text{KL} = 0.000000$ against native FP32/FP16 models. Detailed data points are available in [`docs/json/`](docs/json/).*

---

## Releases

All production-grade, frozen stable releases are bundled directly under [`releases/`](releases/):

- 🌟 **[`releases/tendou-v0.14.0-stable/`](releases/tendou-v0.14.0-stable/)**: Current Flagship Release
  - Complete pre-compiled binary package: `bin/tendou-engine.exe`, `bin/ggml-cpu.dll` (AVX-512 VNNI/VBMI), `bin/ggml-cuda.dll` (CUDA 13), `bin/llama.dll`, `bin/llama-common.dll`.
  - Full source snapshot: `source/` (C++ engine source, AVX-512 quants, autotune, server).
  - Release manifesto & verification record: [`RELEASE_NOTES.md`](releases/tendou-v0.14.0-stable/RELEASE_NOTES.md).
- **[`releases/tendou-v0.13.0-stable/`](releases/tendou-v0.13.0-stable/)**: Universal Auto-Tuner & Dynamic MoE Topology Milestone.
- **[`releases/tendou-v0.11.0-stable/`](releases/tendou-v0.11.0-stable/)**: Multi-Token Prediction (MTP) & Hybrid State Cache Milestone.
- **[`releases/tendou-v0.9.0-stable/`](releases/tendou-v0.9.0-stable/)**: Cross-Layer GPU Pinning & Zero-Copy Subprocess IPC.
- **[`releases/tendou-v0.8.0-stable/`](releases/tendou-v0.8.0-stable/)**: Initial Stable Runtime Baseline.

---

## Research & Documentation Center

Tendou Launcher maintains a comprehensive R&D and engineering record under [`docs/`](docs/):

- 📚 **Technical Articles ([`docs/articles/`](docs/articles/))**:
  - [`RESEARCH_HISTORY_AND_METHODOLOGY.md`](docs/articles/RESEARCH_HISTORY_AND_METHODOLOGY.md): Chronological Phase 0 through Phase 14 Ar-Ge journey and evidence-based engineering.
  - [`HYPOTHESIS_ANALYSIS_PROVEN_AND_REFUTED.md`](docs/articles/HYPOTHESIS_ANALYSIS_PROVEN_AND_REFUTED.md): Refuted hypotheses (Memory Wall, in-loop naive prefetch, speculative MTP traps) vs proven architectures (zero-syscall slab arenas, AVX-512 VNNI, DirectStorage coalescing).
  - [`ARCHITECTURE_AND_INFERENCE_ENGINE.md`](docs/articles/ARCHITECTURE_AND_INFERENCE_ENGINE.md): Deep-dive into memory managers, slab arenas, and tensor layouts.
  - [`claude_code_and_opencode_integration_guide.md`](docs/articles/claude_code_and_opencode_integration_guide.md): Tri-protocol guide for Claude Code, OpenCode, and Cline.
- 📊 **Empirical Validation Datasets ([`docs/json/`](docs/json/))**:
  - 18 empirical JSON reports verifying all speed, latency, memory, and parity claims ([`docs/json/README.md`](docs/json/README.md)).
- 🏛️ **Architecture & Specifications ([`docs/architecture/`](docs/architecture/), [`docs/specs/`](docs/specs/))**:
  - [`V1_RUNTIME_SPEC.md`](docs/specs/V1_RUNTIME_SPEC.md), [`LLAMA4_ARCHITECTURE_AND_OPTIMIZATION_PLAN.md`](docs/specs/LLAMA4_ARCHITECTURE_AND_OPTIMIZATION_PLAN.md).
- 📜 **Historical Engineering Archives ([`docs/history/`](docs/history/))**:
  - Full changelogs, experiment logs, and walkthroughs from earlier phases.
- 🧪 **Development Phases ([`tests/phases/`](tests/phases/))**:
  - Regression and architecture tests covering Phase 0 through Phase 11.

---

## Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone https://github.com/hzTendou/tendou-launcher.git
cd tendou-launcher

# Create and activate Python environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Start the Server

#### A. Mock Mode (Fast test without model weights or GPU)
```powershell
python atlas_server.py --mock --host 127.0.0.1 --port 8000
```

#### B. Native Mode (Using the v0.14.0-stable Engine)
```powershell
python atlas_server.py `
  --exe-path "releases\tendou-v0.14.0-stable\bin\tendou-engine.exe" `
  --model-path "D:\models\Qwen3.5-122B-A10B-MXFP4_MOE-00001-of-00003.gguf" `
  --threads 16 `
  --host 127.0.0.1 `
  --port 8000
```

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

### Python Test Suite (109 Tests)
```powershell
python -m pytest tests/unit tests/integration/test_openai_server.py -q
```

### Native Regression & Gate Suites (28 Gates)
```powershell
cmd /c scripts\build_native_tests.cmd
cmd /c tests\native\build_partition_tests.cmd

.\llama.cpp\build\bin\Release\atlas-native-tests.exe
.\llama.cpp\build\bin\Release\atlas-partition-tests.exe
```

---

## Repository Map

```text
tendou-launcher/
├── releases/       # Standalone frozen stable releases (v0.8.0, v0.9.0, v0.11.0, v0.13.0, v0.14.0)
├── docs/           # Documentation center: articles, benchmark JSONs, audits, specs, history
│   ├── articles/   # Comprehensive research papers and integration guides
│   ├── json/       # 18 empirical benchmark and validation JSON datasets
│   ├── audits/     # Hardware and thread execution audits
│   ├── specs/      # Runtime specifications and Llama-4 optimization plan
│   └── history/    # Historical changelogs and experiment reports
├── tests/          # Complete test suite: unit, integration, native, benchmarks, phases 0-11
│   ├── phases/     # Chronological development phase test suites (phase0 - phase11)
│   ├── native/     # C++ parity, memory benchmarking, and partition tests
│   └── integration/# OpenCode, Claude Code, and Cline protocol tests
├── src/atlas/      # Multi-protocol API server, autotune engine, and C++ engine sources
├── scripts/        # Build commands, physical map generators, and benchmark scripts
├── config/         # Runtime example profiles and hardware topologies
├── patches/        # Reproducible backend engine integration patches
├── scratch/        # Diagnostic, sweep, and ablation tools
└── data/           # Historical physical reports and telemetry traces
```

---

## License

Tendou Launcher is released under the [MIT License](LICENSE).

<div align="center">

Built with precision for extreme local MoE inference.

</div>
