# Tendou Launcher v0.16.3-stable — Windows 11 Build & Studio User Guide

This guide covers building the native `tendou-engine.exe` v0.16.3-stable engine on **Windows 11 (`x86_64`)** with **MSVC + CUDA 13.2**, launching **Tendou Studio (FastAPI Web UI + OpenAI/Anthropic API)**, optionally building the **Tauri v2 Desktop App (`.msi`/`.exe`)**, and connecting **Claude Code**, **OpenCode**, and **Cline**.

---

## 1. Prerequisites (Windows 11)

1. **Visual Studio 2022 / 2026 Build Tools** with the **Desktop development with C++** workload (`cl.exe`, Windows 11 SDK, CMake).
2. **NVIDIA CUDA Toolkit 13.2+** (`nvcc.exe`) and latest NVIDIA Game Ready / Studio Driver (RTX 30/40/50 series; Blackwell `sm_120`, Ada `sm_89`, Ampere `sm_86`).
3. **Python 3.11+** (`pip install fastapi uvicorn pydantic httpx psutil`).
4. *(Optional — only for building the Tauri v2 desktop wrapper or rebuilding the frontend)*:
   - **Node.js 22+** (`npm`)
   - **Rust toolchain (`rustup` / `cargo`)** — WebView2 is pre-installed on Windows 11.

---

## 2. One-Click Native Engine Build (`v0.16.3-stable`)

Open **Command Prompt** or **PowerShell** in the repository root and run:

```cmd
scripts\build_windows_all.cmd
```

What this single script does automatically:
1. Detects `vcvars64.bat` via `vswhere.exe` and checks `cmake` + `nvcc.exe`.
2. Syncs `src\atlas\{atlas-engine.cpp,atlas-engine.h,atlas-moe-cache.h,atlas-iouring-dma.h,CMakeLists.txt}` into `llama.cpp\examples\atlas-engine\`.
3. Builds `ggml-cuda.dll`, `ggml-cpu.dll` (including **AVX2/AVX-512 VNNI 4-row `IQ4_XS_x4`, `IQ3_XXS_x4` & `IQ3_S_x4`** kernels), `llama.dll`, `tendou-engine.exe`, and `llama-server.exe` with `CMAKE_CUDA_ARCHITECTURES="120;89;86"`.
4. Compiles and runs `atlas-native-tests.exe` (**Gates 1–48**, including Gate 47 AVX-512 VNNI 0-ULP parity and Gate 48 3-regime Zero-Copy Hot-RAM planner) and `atlas-partition-tests.exe`.
5. Copies all compiled `.exe` and `.dll` files into `releases\tendou-v0.16.3-stable\bin\`.

> **Tip:** To rebuild only the C++ runtime after editing `src\atlas\atlas-engine.cpp`, run `scripts\build_runtime.cmd`.

---

## 3. Starting Tendou Studio Web UI & Local API Server

```cmd
:: Start with AutoTune optimal settings (6-8 GPU layers on 35B, 2 GPU layers on 38B, Adaptive KV, MTP OFF)
start_server.cmd --port 8000

:: Open Studio Web UI immediately without auto-loading a model first
start_server.cmd --port 8000 --no-auto-start-engine --open-ui

:: Start with explicit model preset or custom GGUF path
start_server.cmd --model 35b --port 8000
start_server.cmd --model-path "D:\Models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" --port 8000
```

Once running:
- **Tendou Studio Enterprise UI:** `http://127.0.0.1:8000` (includes live TPS counter, full 49+ CLI flag settings, Playground, Models hot-swap, Swarm monitor, and Benchmarks)
- **OpenAI API Base URL:** `http://127.0.0.1:8000/v1`
- **Anthropic Messages API:** `http://127.0.0.1:8000/v1/messages`

> Pre-built frontend assets live in `ui\dist\`, so the Studio Web UI works out-of-the-box on Windows **without needing Node.js installed**.

---

## 4. Optional: Building the Native Windows Tauri v2 Desktop App

```cmd
scripts\build_desktop_windows.cmd
```

This builds `ui\dist\` and packages the lightweight native WebView2 desktop app (~40–60 MB RAM, System Tray, Win32 Job Object zero-zombie engine cleanup) as `.msi` / `.exe` installers under `desktop\src-tauri\target\release\bundle\`.

---

## 5. Connecting Coding Agents on Windows

### Claude Code (PowerShell)
```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8000"
$env:ANTHROPIC_API_KEY = "tendou-local"
claude
```

### OpenCode (`opencode.json`)
Use the one-click copy snippet inside **Tendou Studio -> Agent Hub** (`http://127.0.0.1:8000`), pointing `baseURL` to `http://127.0.0.1:8000/v1`.

### Cline / Roo Code (VS Code)
- **API Provider:** OpenAI Compatible
- **Base URL:** `http://127.0.0.1:8000/v1`
- **API Key:** `tendou-local`
- **Model ID:** `qwen3.6-35b` or `qwen3.8-flash-next`

---

## 6. Windows Performance Tips

- **Hardware-Accelerated GPU Scheduling & Power Plan:** Use **High Performance** or **Best Performance** power mode in Windows 11 Settings.
- **Model Discovery:** Place `.gguf` files in `models\` inside this repo or in `%USERPROFILE%\.cache\huggingface\hub` — Tendou Studio scans both automatically.
- **8 GB VRAM Safety:** Keep `--mtp off` (default) on 16 GB RAM / 8 GB VRAM laptops; `compute_auto_plan` automatically assigns `6–8` GPU MoE layers for 35B models and caps 38B (`75.8 GB`) at `2` GPU layers with `f8_e4m3` KV cache at `>= 16K` context.
