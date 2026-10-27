from pathlib import Path

readme_path = Path("README.md")
content = readme_path.read_text(encoding="utf-8")

old_snippet = """Tendou Launcher combines a patched `llama.cpp` runtime with a lightweight
FastAPI server. It keeps the model router authoritative while experimenting
with bounded expert prefetch, prompt-state reuse, CPU/GPU placement,
asynchronous transfer, and optional MTP speculative decoding.

> [!IMPORTANT]
> Tendou Launcher is experimental systems software, not a general-purpose
> `llama.cpp` distribution. The native path currently targets a specific
> Windows, NVIDIA CUDA, and Qwen3.8 Flash Next workflow. Begin with mock mode."""

new_snippet = """Tendou Launcher is an **independent MoE inference platform and launcher** with its own
asynchronous orchestrator, physical memory manager (`PhysicalExpert`, `CrossLayerGPUPinning`),
zero-copy hybrid state cache (`AtlasHybridStateCache`), deep hardware telemetry, and OpenAI-compatible
server. It utilizes `llama.cpp` and `ggml` strictly as an underlying compute engine and tensor backend dependency,
not as a fork or mod.

> [!IMPORTANT]
> **Independent Project Principle:** Tendou Launcher is an independent standalone platform.
> Engine versions, architecture design, and release milestones are native to Tendou Launcher.
> `llama.cpp` serves purely as an execution backend. All stable releases are bundled under [`releases/`](releases/)."""

content = content.replace(old_snippet.replace("\r\n", "\n"), new_snippet.replace("\r\n", "\n"))
content = content.replace(old_snippet, new_snippet)

old_repo_map = """## Repository map

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
`experiments.legacy` import paths."""

new_repo_map = """## Repository map

```text
releases/           Frozen standalone stable releases (tendou-v0.8.0-stable, tendou-v0.9.0-stable)
docs/               Comprehensive documentation, reports, audits, release notes, and validation JSONs
docs/json/          Research benchmark and validation JSON outputs confirming technical milestones
tests/              Unified test suite structured by categories and development phases (phases/phase0-9)
src/atlas/          Python OpenAI API server, process manager, and native runtime source
scripts/            Build and environment management scripts
config/             Hardware topology, physical expert maps, and runtime configuration
patches/            Reproducible backend engine integration patch
scratch/            Diagnostic tools and temporary scripts
data/               Physical reports, traces, and execution logs
```"""

content = content.replace(old_repo_map.replace("\r\n", "\n"), new_repo_map.replace("\r\n", "\n"))
content = content.replace(old_repo_map, new_repo_map)

readme_path.write_text(content, encoding="utf-8")
print("README.md updated successfully!")
