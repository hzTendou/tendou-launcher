# tendou launcher

Projenin güncel adı **tendou launcher**. Ajanlar yeni görevden önce
[ortak çalışma belleğini](AGENT.md) okumalıdır. Eski Atlas dosya, komut ve API
kimlikleri uyumluluk için korunur.

Aktif teknik aşamalar, kabul ölçütleri ve kesin devam noktaları
[geliştirme yol haritasında](docs/NEXT_STEPS.md) tutulur.

## Correctness and prefetch audit (2026-09-10)

The CPU expert kernel now computes every active expert even when the thread count
is lower than native top-k. A separate duplicate-expert input bug is also fixed.
The unused 668 MiB staging allocation is removed. The experimental
`--tutti-async-io` option now enables real background OS page warming, with bounded
requests and explicit completion; it is disabled by default pending a workload win.
Dynamic GPU expert caching and H2D staging remain unimplemented. Python policy
simulators and their estimated overlap are not inference benchmarks.

An experimental `--aggressive-prefetch` server profile increases lookahead to four
layers, expands the candidate and page-warming queue, and lowers confidence gates.
The P0 paired benchmark did not show a speed win, so this profile remains opt-in.

See [the new audit](docs/RUNTIME_AUDIT_2026-09-10.md) for reproduced failures,
native tests, original-versus-patched measurements, and implementation limits.

## Runtime audit (2026-09-09)

The current runtime preserves the model's native expert count by default, including
`--boost`. Older K=2 benchmarks are approximate model variants; their entropy and
repetition scores do **not** establish answer quality. Missing quality evidence now
remains `NOT_EVALUATED` in the benchmark parsers.

Resident IPC requests reuse bounded prompt checkpoints, including recurrent state
and separately saved logits. The default combined host-memory budget is 256 MiB;
`--atlas-prompt-cache-mb 0` disables caching. Long chat prompts also retain an
existing prefill batch boundary before the assistant suffix, so an appended chat
turn can reuse history even when thinking markers change. Different token prefixes
fall back to prefill; this is state reuse, not cached answers.

Static llama.cpp GPU offload performs inference. The old dynamic expert placement,
grouped stream and cache counters are a cost simulation, not a working CUDA expert
cache. Enable those diagnostics explicitly with `--atlas-simulate-placement`.

The 16 GB follow-up also repairs the model-specific MTP graph and draft history.
Explicit MTP now defaults to one draft token. An experimental native OpenAI API
is available with `atlas_server.py --native-mtp --port 8001`; the regular Atlas API
remains the default. MTP has measured decode gains on a short continuation, but
longer code output can differ numerically; broad 80% task quality is not certified.

See [the runtime audit](docs/RUNTIME_AUDIT_2026-09-09.md) for measurements, limits,
source research, and reproducible commands. Sections below describe the older V1
interface and architecture; historical performance/quality claims are superseded
by this audit.

tendou launcher is the first runtime-oriented implementation of the Atlas memory hierarchy:

`NVMe -> mmap/page cache -> RAM -> VRAM -> GPU`

The production runtime is integrated into llama.cpp under `examples/atlas-engine/`. Atlas keeps MoE expert weights in host memory, observes the final router-selected experts, predicts near-future expert reuse online, and proactively asks the operating system to page those expert slices into memory before they are consumed.

## Hardware target

- VRAM: 8 GiB
- RAM: 16 GiB
- Model storage: NVMe SSD
- Designed for very large MoE GGUF models whose full weights cannot fit in RAM.

## V1 design

1. Dense/trunk tensors follow normal llama.cpp GPU offload.
2. Routed MoE expert tensors are pinned to CPU/mmap instead of VRAM.
3. `ffn_moe_topk-*` is observed through llama.cpp's scheduler evaluation callback.
4. Atlas learns same-layer next-token expert transitions online.
5. Predicted expert planes are prefetched with OS virtual-memory hints.
6. The model router remains authoritative; Atlas prediction never changes selected experts.


## Local model path

The V1 runtime is configured for the three-part Hugging Face cache layout used on the target Windows machine. The first shard is passed to llama.cpp; GGUF multi-part loading resolves the remaining shards.

`C:\\Users\\Ali\\.cache\\huggingface\\hub\\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\\snapshots\\06756566a4b4a29d0dee62ccb405914a15fdf80d\\Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf`

The runtime profile records the full shard pattern so Atlas can identify the model as a single multi-part corpus. Do not copy the 134 GB model into the Atlas repository.

## Important compatibility note

The supplied llama.cpp tree currently contains Qwen3-Next/Qwen3.5-MoE support, but no `qwen4exp` architecture implementation. The supplied Qwen3.8 Flash Next metadata declares `general.architecture=qwen4exp`. Therefore V1's memory/runtime layer is implemented now, while the `qwen4exp` model graph remains a separate adapter task and is not falsely claimed as supported by the supplied fork.

## Directory layout

- `src/atlas/` production-side reference algorithms and OpenAI server implementation
- `config/` model/runtime profiles
- `tests/integration/` integration tests for OpenAI server and E2E engine
- `tests/unit/` unit tests for simulators and predictor
- `tests/legacy_v7/` previous simulator-era tests
- `experiments/legacy/` trace collection and simulators
- `data/traces/` old trace artifacts
- `data/reports/` historical benchmark outputs
- `docs/history/` old implementation notes and reports

## OpenAI-Compatible API Server

tendou launcher provides an OpenAI-compatible HTTP inference server (`/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/health`) maintaining resident GPU-first execution and MoE prefetching.

### Installation

Install runtime and testing dependencies using pip:

```bash
pip install -r requirements.txt
```

### Starting the Server

```bash
# Start with real compiled engine and resident model shards:
python atlas_server.py --port 8000 --threads 14 --boost

# Start in high-fidelity mock mode (for lightweight CI or environments without weights):
python atlas_server.py --port 8000 --mock
```

### Python OpenAI SDK Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",
)

# Chat Completion (Streaming)
response = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "Hello Atlas!"}],
    stream=True,
    temperature=0.7,
)
for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Function / Tool Calling
response = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }],
)
print(response.choices[0].message.tool_calls)
```

### Testing with Real OpenCode

OpenCode integrates with tendou launcher via an OpenAI-compatible provider definition.

#### Step 1: Configure OpenCode Provider

The project includes a ready-to-use [`opencode.json`](file:///c:/Users/Ali/WebProjects/atlas-engine-v1/opencode.json) in the workspace root (with [`opencode.json.example`](file:///c:/Users/Ali/WebProjects/atlas-engine-v1/opencode.json.example) as reference), configuring Atlas as a local provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "atlas": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "tendou launcher Local",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "opencode-local-key"
      },
      "models": {
        "qwen3.8-flash-next": {
          "name": "Qwen 3.8 Flash Next (Atlas V1)"
        },
        "atlas-engine": {
          "name": "tendou launcher"
        }
      }
    }
  }
}
```

*Windows PowerShell Note:* If PowerShell script execution policy blocks unsigned scripts (`opencode.ps1`), use `opencode.cmd` instead of `opencode`, or temporarily bypass the execution policy in your session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### Step 2: Start tendou launcher Server

In your server terminal, launch tendou launcher using python (or your virtual environment `.\.venv\Scripts\python.exe`):

```bash
# High-fidelity mock mode (for lightweight CI, testing OpenCode chat & agentic tools):
python atlas_server.py --port 8000 --mock

# Production mode with compiled engine and resident model shards:
python atlas_server.py --port 8000 --threads 14 --boost
```

Verify the server is healthy:
```bash
curl http://127.0.0.1:8000/health
```

#### Step 3: Verify Model Discovery in OpenCode

Check that OpenCode discovers the Atlas models from `opencode.json`:
```bash
# Windows (PowerShell / CMD):
opencode.cmd models atlas

# Linux / macOS:
opencode models atlas
```
Expected output:
```text
atlas/atlas-engine
atlas/qwen3.8-flash-next
```

#### Step 4: Verify Chat Completions & Streaming

Run an informational query to verify direct conversational generation without tool calls:

```powershell
# PowerShell:
$null | opencode.cmd run "What is the Atlas memory hierarchy?" -m atlas/qwen3.8-flash-next

# CMD:
opencode.cmd run "What is the Atlas memory hierarchy?" -m atlas/qwen3.8-flash-next < NUL
```

Expected output:
```text
> build · qwen3.8-flash-next

The tendou launcher memory hierarchy consists of:

1. **NVMe SSD (Model Storage)**: Houses multi-part 134GB GGUF model shards with OS mmap.
2. **Host RAM (16 GiB)**: Keeps routed MoE expert tensors in pinned host memory.
3. **GPU VRAM (8 GiB)**: Resident dense and trunk tensors offloaded to GPU.
4. **Online Expert Prefetcher**: Observes router-selected experts, predicts transitions online, and proactively asks the operating system to page expert slices into RAM before consumption.
```

#### Step 5: Verify Agentic Tool Calling

Test OpenCode's agentic loop (tool selection, execution, and final answer):

```powershell
# PowerShell:
$null | opencode.cmd run "Inspect requirements.txt and list the installed web server packages." -m atlas/qwen3.8-flash-next

# CMD:
opencode.cmd run "Inspect requirements.txt and list the installed web server packages." -m atlas/qwen3.8-flash-next < NUL
```

Expected output:
```text
> build · qwen3.8-flash-next

$ type requirements.txt
... (requirements.txt content) ...

I have inspected `requirements.txt`. The web server and API packages installed are FastAPI (v0.141.1) and Uvicorn (v0.52.4), along with Pydantic (v2.13.5), HTTPX (v0.28.1), and OpenAI SDK (v3.8.0).
```

#### Step 6: Interactive TUI Session

Launch the full interactive terminal interface for pair programming:
```powershell
opencode.cmd -m atlas/qwen3.8-flash-next
```

