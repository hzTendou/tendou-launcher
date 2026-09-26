<div align="center">

# Tendou Launcher

### Run large MoE models locally—without pretending your laptop is a datacenter.

Windows-first, OpenAI-compatible inference for large Mixture-of-Experts GGUF
models on memory-constrained gaming laptops.

[![License: MIT](https://img.shields.io/badge/License-MIT-7c3aed.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Windows 11](https://img.shields.io/badge/Windows-11-0078D4?logo=windows11&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-accelerated-76B900?logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-f59e0b)

**[Quick start](#quick-start)** · **[Architecture](#architecture)** ·
**[Native build](#native-runtime)** · **[Configuration](#configuration)** ·
**[Testing](#testing)**

</div>

---

Tendou Launcher combines a patched `llama.cpp` runtime with a lightweight
FastAPI server. It keeps the model router authoritative while experimenting
with bounded expert prefetch, prompt-state reuse, CPU/GPU placement,
asynchronous transfer, and optional MTP speculative decoding.

> [!IMPORTANT]
> Tendou Launcher is experimental systems software, not a general-purpose
> `llama.cpp` distribution. The native path currently targets a specific
> Windows, NVIDIA CUDA, and Qwen3.8 Flash Next workflow. Begin with mock mode.

## Why Tendou Launcher?

Large MoE models may activate only a fraction of their experts per token, but
their complete weights can still exceed available RAM and VRAM. Tendou Launcher
explores a practical hierarchy for that constraint:

```text
NVMe / mmap  ──►  host RAM  ──►  GPU VRAM  ──►  CUDA execution
     cold            warm             hot
```

Prediction is advisory. It can influence where and when weights are fetched,
but it must never replace the model router, skip a selected expert, or silently
reduce the model's native expert count.

## Highlights

| Capability | What it provides |
|---|---|
| OpenAI-compatible API | Chat completions, text completions, streaming, model discovery, and tool-call parsing |
| Resident native runtime | A persistent JSON IPC subprocess keeps model state alive between requests |
| Prompt reuse | Bounded prompt checkpoints and suffix reuse reduce repeated-prefix work |
| MoE scheduling research | Router observation, expert prediction, page warming, and CPU/GPU placement |
| Optional MTP | Experimental speculative decoding with verification and acceptance metrics |
| Mock backend | API and client development without model weights, CUDA, or a native executable |
| Regression coverage | Portable Python tests plus native C++ scheduling and partition tests |

## Architecture

```text
┌────────────────────┐
│ OpenAI client      │
└─────────┬──────────┘
          │ HTTP / SSE
          ▼
┌────────────────────┐
│ FastAPI server     │  src/atlas/server.py
└─────────┬──────────┘
          │ line-delimited JSON IPC
          ▼
┌─────────────────────────────────────┐
│ patched llama-atlas-engine          │
│                                     │
│  • prompt-state cache               │
│  • router observation               │
│  • bounded expert prefetch          │
│  • RAM / VRAM placement             │
│  • optional MTP verification        │
└─────────┬───────────────────────────┘
          │
          ▼
     GGUF model shards
```

The repository contains Tendou source code and a reproducible patch for the
compatible `llama.cpp` base. Model weights, physical maps, compiled binaries,
benchmark logs, local agent state, and the dependency checkout stay local.

## Requirements

<table>
<thead>
<tr><th>Mock/API development</th><th>Native runtime</th></tr>
</thead>
<tbody>
<tr>
<td valign="top">
<ul>
<li>Python 3.12</li>
<li>Windows, Linux, or macOS</li>
</ul>
</td>
<td valign="top">
<ul>
<li>Windows 11</li>
<li>Visual Studio C++ Build Tools</li>
<li>CMake</li>
<li>NVIDIA CUDA toolkit and compatible GPU</li>
<li>Compatible multipart GGUF model</li>
<li>Sufficient local storage</li>
</ul>
</td>
</tr>
</tbody>
</table>

The primary development target is a laptop with **8 GiB VRAM and 16 GiB RAM**.
Results depend on the hardware, model, quantization, context, and cache state;
published measurements should not be treated as universal performance claims.

## Quick start

Mock mode is the fastest way to verify the API and client integration.

### 1. Install

```powershell
git clone <your-fork-url> tendou-launcher
cd tendou-launcher

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Start the mock server

```powershell
python atlas_server.py --mock --host 127.0.0.1 --port 8000
```

### 3. Check the API

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

### 4. Send a completion

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-only",
)

response = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[
        {"role": "user", "content": "Explain MoE routing briefly."},
    ],
)

print(response.choices[0].message.content)
```

## Native runtime

The nested `llama.cpp` checkout is deliberately ignored. Recreate it from the
pinned upstream base and apply the project patch:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
git -C llama.cpp checkout 67a17c17caa95742186f8b1ecadd1b5abd6d5ebb
git -C llama.cpp apply ..\patches\llama.cpp.patch

cmake -S llama.cpp -B llama.cpp\build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp\build --config Release --target llama-atlas-engine -j
```

The patch includes the Tendou example target, Qwen4Exp integration, runtime
hooks, and CPU/CUDA changes used by the current implementation. Its base commit
and update procedure are documented in
[`patches/README.md`](patches/README.md).

## Configuration

Keep model locations out of version control. Provide them through environment
variables:

```powershell
$env:TENDOU_MODEL_PATH = "D:\models\model-00001-of-00003.gguf"
$env:TENDOU_ENGINE_PATH = "$PWD\llama.cpp\build\bin\Release\llama-atlas-engine.exe"

python atlas_server.py --host 127.0.0.1 --port 8000
```

Or use CLI flags:

```powershell
python atlas_server.py `
  --exe-path ".\llama.cpp\build\bin\Release\llama-atlas-engine.exe" `
  --model-path "D:\models\model-00001-of-00003.gguf" `
  --threads 16 `
  --ctx-size 8192
```

### Environment variables

| Variable | Purpose |
|---|---|
| `TENDOU_MODEL_PATH` | Primary multipart GGUF path |
| `TENDOU_ENGINE_PATH` | Compiled `llama-atlas-engine` executable |
| `TENDOU_MTP_PATH` | Optional MTP GGUF used with `--mtp` |
| `TENDOU_MODEL_BLOB_1..3` | Shard blobs for experimental asynchronous page warming |

[`config/runtime_default.example.json`](config/runtime_default.example.json)
documents a complete local research profile. Copy it to
`config/runtime_default.json` for local tooling, but note that the API server
does not load this file automatically; pass server settings through environment
variables or CLI flags. The active local file is ignored by Git.

## OpenCode

The checked-in example points OpenCode to the local OpenAI-compatible endpoint:

```powershell
Copy-Item opencode.json.example opencode.json
opencode.cmd models atlas
```

The configured API key is a client placeholder. Tendou Launcher does not provide
production-grade authentication, so do not expose the server to an untrusted
network.

## Testing

### Portable Python suite

```powershell
python -m pytest tests/unit tests/integration/test_openai_server.py -q
```

### Native regression suites

```powershell
cmd /c scripts\build_native_tests.cmd
cmd /c scripts\build_partition_tests.cmd

.\llama.cpp\build\bin\Release\atlas-native-tests.exe
.\llama.cpp\build\bin\Release\atlas-partition-tests.exe
```

### Real-model integration

Real-model tests are opt-in because they require local weights and can be slow:

```powershell
$env:ATLAS_TEST_REAL = "1"
python -m pytest tests/integration/test_engine_ipc_e2e.py -q
```

## Runtime modes

| Mode | Purpose | Model | Native build |
|---|---|:---:|:---:|
| `--mock` | API and client development, CI | No | No |
| default | Persistent patched runtime over JSON IPC | Yes | Yes |
| `--native-mtp` | Experimental native llama-server/MTP path | Yes | Yes |

Run `python atlas_server.py --help` for the complete flag reference.
Experimental features remain opt-in unless a local profile enables them.

## Repository map

```text
src/atlas/          Python server, policy modules, and native runtime source
tests/              Python, API integration, and native regression tests
scripts/            Build, benchmark, and verification entry points
config/             Shareable configuration examples
patches/            Reproducible llama.cpp integration delta
docs/               Runtime specification and integration notes
experiments/legacy/ Historical simulator source retained for compatibility
```

Several root-level Python modules are compatibility shims for historical
`experiments.legacy` import paths.

## Status and limitations

- The project is under active research and remains Windows-first.
- Native compatibility is tied to the pinned `llama.cpp` base and included patch.
- Physical maps are model- and machine-specific and must be generated locally.
- MTP, predictive placement, and asynchronous transfer are experimental.
- Speed results require token-parity or an explicit task-quality gate.
- The API is intended for trusted local use; it does not provide hardened
  authentication, TLS termination, quotas, or multi-tenant isolation.

## Contributing

Keep changes focused and reproducible. For performance work, record the model,
quantization, native expert count, thread count, context size, cache state,
binary hashes, and at least three paired serial runs. A faster result is not
accepted without correctness evidence.

When changing the native runtime:

1. Update `patches/llama.cpp.patch` against the documented base.
2. Run the portable Python suite and both native regression suites.
3. Verify exact-token behavior when runtime semantics change.
4. Document changes to the pinned `llama.cpp` revision.

## License

Tendou Launcher is released under the [MIT License](LICENSE).

---

<div align="center">

Built for local inference experiments where every GiB matters.

</div>
