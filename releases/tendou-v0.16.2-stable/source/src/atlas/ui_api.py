"""
Tendou Studio — FastAPI Control Plane Router (/api/ui/*).

Provides complete engine lifecycle management (start, stop, zero-zombie hot-swap restart),
4 Hz SSE real-time TPS/VRAM/MoE/Swarm telemetry, full 50+ CLI flag configuration &
preset management, local GGUF discovery with AutoTune projections, coding agent
config generation, built-in micro-benchmarking, historical benchmark comparison,
and live console log streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform as _platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.atlas.autotune import (
    HardwareProfile,
    compute_auto_plan,
    inspect_model_topology,
    probe_hardware,
    scan_local_models,
)
from src.atlas.ui_config import (
    DEFAULT_PROFILE_PATH,
    EngineSettingsModel,
    ProfileStore,
    compute_settings_diff,
    get_presets,
    get_settings_schema,
    serialize_to_cli_args,
    settings_from_backend,
)
from src.atlas.ui_telemetry import TelemetryHub, TelemetryLogHandler

logger = logging.getLogger("atlas.ui_api")
REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelScanRequest(BaseModel):
    directory: str = Field(..., min_length=1, description="Directory path to scan for .gguf models")


class EngineActionRequest(BaseModel):
    settings: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    save_profile: bool = True


class QuickBenchmarkRequest(BaseModel):
    prompt: str = Field(
        default="Write a concise Python function to merge two sorted lists with O(n+m) time complexity.",
        min_length=4,
    )
    max_tokens: int = Field(default=64, ge=8, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class StudioRuntimeContext:
    """Shared mutable control-plane state bridging FastAPI app.state and /api/ui/* routes."""

    def __init__(
        self,
        app: Any,
        backend_factory: Callable[[EngineSettingsModel], Any],
        version: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        profile_path: Path = DEFAULT_PROFILE_PATH,
        telemetry: Optional[TelemetryHub] = None,
        auto_started: bool = True,
    ):
        self.app = app
        self.backend_factory = backend_factory
        self.version = version
        self.host = host
        self.port = port
        self.profile_store = ProfileStore(profile_path)
        self.telemetry = telemetry or TelemetryHub()
        self.lifecycle_lock = asyncio.Lock()

        self.hw: HardwareProfile = probe_hardware()
        self.extra_model_dirs: List[str] = []
        self.last_error: Optional[str] = None
        self.engine_status: str = "ready" if auto_started else "stopped"

        # Snapshot current backend settings as baseline
        current_backend = getattr(app.state, "backend", None)
        self.running_settings: EngineSettingsModel = (
            settings_from_backend(current_backend, host=host, port=port)
            if current_backend is not None
            else EngineSettingsModel(host=host, port=port)
        )
        self.saved_settings: EngineSettingsModel = self.running_settings.model_copy(deep=True)
        self.last_benchmark_result: Optional[Dict[str, Any]] = None
        self.benchmark_running: bool = False

        # Attach server log handler once
        root_atlas_logger = logging.getLogger("atlas.server")
        if not any(isinstance(h, TelemetryLogHandler) and h.hub is self.telemetry for h in root_atlas_logger.handlers):
            handler = TelemetryLogHandler(self.telemetry)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
            root_atlas_logger.addHandler(handler)

        # Hook active backend telemetry reference
        self._attach_backend_telemetry(current_backend)

    def _attach_backend_telemetry(self, backend: Any) -> None:
        if backend is not None:
            setattr(backend, "telemetry_hub", self.telemetry)

    def compute_current_status(self) -> str:
        if self.engine_status in ("starting", "restarting"):
            return self.engine_status
        backend = getattr(self.app.state, "backend", None)
        if backend is None:
            return "stopped"
        if self.engine_status == "stopped":
            return "stopped"
        if self.engine_status == "error":
            return "error"
        try:
            return "ready" if backend.is_healthy() else "stopped"
        except Exception:
            return "error"

    def active_model_id(self) -> str:
        backend = getattr(self.app.state, "backend", None)
        if backend is not None and hasattr(backend, "get_model_info"):
            try:
                info = backend.get_model_info()
                if info.get("path"):
                    return Path(str(info["path"])).stem
                if info.get("id"):
                    return str(info["id"])
            except Exception:
                pass
        if self.running_settings.model_path:
            return Path(self.running_settings.model_path).stem
        return "tendou-local"

    def active_autotune_plan(self) -> Optional[Dict[str, Any]]:
        mpath = self.running_settings.model_path
        if mpath and os.path.isfile(mpath):
            try:
                topo = inspect_model_topology(mpath)
                ck, cv = self.running_settings.normalized_kv()
                plan = compute_auto_plan(
                    self.hw,
                    topo,
                    ctx_size=self.running_settings.ctx_size,
                    explicit_gpu_layers=self.running_settings.gpu_layers,
                    explicit_vram_layer_cap=self.running_settings.vram_layer_cap,
                    explicit_cache_type_k=ck,
                    explicit_cache_type_v=cv,
                    explicit_fp8_kv=self.running_settings.fp8_kv,
                )
                return asdict(plan)
            except Exception:
                return None
        return None

    async def start_engine(self, req: Optional[EngineActionRequest] = None) -> Dict[str, Any]:
        async with self.lifecycle_lock:
            target_cfg = self._resolve_target_settings(req)
            if req is None or req.save_profile:
                self.saved_settings = target_cfg.model_copy(deep=True)
                try:
                    self.profile_store.save(self.saved_settings)
                except Exception as exc:
                    logger.warning("Failed to save studio profile: %s", exc)

            cur = getattr(self.app.state, "backend", None)
            if cur is not None and self.compute_current_status() == "ready" and not compute_settings_diff(self.running_settings, target_cfg):
                return {"status": "ready", "message": "Engine is already running with the requested configuration."}

            self.engine_status = "starting"
            self.last_error = None
            try:
                if cur is not None:
                    await cur.stop()
                new_backend = self.backend_factory(target_cfg)
                self._attach_backend_telemetry(new_backend)
                await new_backend.start()
                self.app.state.backend = new_backend
                self.running_settings = target_cfg.model_copy(deep=True)
                self.engine_status = "ready"
                self.telemetry.reset_gpu_peak()
                await asyncio.to_thread(self.telemetry.maybe_refresh_system_metrics, True)
                self.telemetry.push_log(
                    f"[STUDIO] Engine started: model={self.active_model_id()} ctx={target_cfg.ctx_size} gpu_layers={target_cfg.gpu_layers}",
                    source="server",
                    level="INFO",
                )
                return {"status": "ready", "model": self.active_model_id()}
            except Exception as exc:
                self.engine_status = "error"
                self.last_error = str(exc)
                self.telemetry.push_log(f"[STUDIO] Engine start failed: {exc}", source="server", level="ERROR")
                raise HTTPException(status_code=500, detail=f"Engine failed to start: {exc}")

    async def _drain_active_requests(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while getattr(self.app.state, "active_requests", 0) > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def stop_engine(self) -> Dict[str, Any]:
        async with self.lifecycle_lock:
            await self._drain_active_requests(timeout_s=5.0)
            cur = getattr(self.app.state, "backend", None)
            if cur is not None:
                try:
                    await cur.stop()
                except Exception as exc:
                    logger.warning("Error stopping engine: %s", exc)
            self.engine_status = "stopped"
            self.last_error = None
            await asyncio.to_thread(self.telemetry.maybe_refresh_system_metrics, True)
            self.telemetry.push_log("[STUDIO] Engine stopped cleanly; VRAM released.", source="server", level="INFO")
            return {"status": "stopped"}

    async def restart_engine(self, req: Optional[EngineActionRequest] = None) -> Dict[str, Any]:
        async with self.lifecycle_lock:
            target_cfg = self._resolve_target_settings(req)
            diffs = compute_settings_diff(self.running_settings, target_cfg)
            if req is None or req.save_profile:
                self.saved_settings = target_cfg.model_copy(deep=True)
                try:
                    self.profile_store.save(self.saved_settings)
                except Exception as exc:
                    logger.warning("Failed to save studio profile: %s", exc)

            self.engine_status = "restarting"
            self.last_error = None
            await self._drain_active_requests(timeout_s=5.0)
            old_backend = getattr(self.app.state, "backend", None)
            try:
                if old_backend is not None:
                    await old_backend.stop()
                new_backend = self.backend_factory(target_cfg)
                self._attach_backend_telemetry(new_backend)
                await new_backend.start()
                self.app.state.backend = new_backend
                self.running_settings = target_cfg.model_copy(deep=True)
                self.engine_status = "ready"
                self.telemetry.reset_gpu_peak()
                await asyncio.to_thread(self.telemetry.maybe_refresh_system_metrics, True)
                self.telemetry.push_log(
                    f"[STUDIO] Hot-swap restart complete ({len(diffs)} setting(s) updated, model={self.active_model_id()}).",
                    source="server",
                    level="INFO",
                )
                return {
                    "status": "ready",
                    "model": self.active_model_id(),
                    "applied_diff": diffs,
                }
            except Exception as exc:
                self.engine_status = "error"
                self.last_error = str(exc)
                self.telemetry.push_log(f"[STUDIO] Hot-swap restart failed: {exc}", source="server", level="ERROR")
                raise HTTPException(status_code=500, detail=f"Engine restart failed: {exc}")

    def _resolve_target_settings(self, req: Optional[EngineActionRequest]) -> EngineSettingsModel:
        base = self.saved_settings.model_dump()
        if req is not None:
            if req.settings:
                base.update(req.settings)
            if req.model_path:
                base["model_path"] = req.model_path
                base["model"] = None
        return EngineSettingsModel(**base)


def _load_benchmark_history() -> Dict[str, Any]:
    """Reads verified benchmark JSON reports from docs/json/ for version comparison charts."""
    json_dir = REPO_ROOT / "docs" / "json"
    version_files = [
        ("v0.14.0", json_dir / "benchmark_v14_kubuntu_report.json"),
        ("v0.16.0", json_dir / "benchmark_v16_kubuntu_report.json"),
        ("v0.16.1", json_dir / "benchmark_v16_1_kubuntu_report.json"),
        ("v0.16.2", json_dir / "benchmark_v16_2_kubuntu_report.json"),
    ]
    versions: List[Dict[str, Any]] = []
    for label, fpath in version_files:
        if not fpath.is_file():
            continue
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        results = raw.get("results", {})
        models_summary: Dict[str, Any] = {}
        for mkey, mval in results.items():
            workloads = []
            for wl in mval.get("workloads", []):
                wl_entry: Dict[str, Any] = {
                    "workload_id": wl.get("workload_id"),
                    "category": wl.get("category"),
                    "name": wl.get("name"),
                }
                for proto in ("opencode", "claude_code", "cline"):
                    p = wl.get(proto)
                    if isinstance(p, dict):
                        wl_entry[proto] = {
                            "cold_ttft_ms": p.get("cold_ttft_ms", 0.0),
                            "warm_ttft_ms": p.get("warm_ttft_ms", 0.0),
                            "warm_decode_tps": p.get("warm_decode_tps", 0.0),
                            "best_decode_tps": p.get("best_decode_tps", 0.0),
                            "cold_decode_tps": p.get("cold_decode_tps", 0.0),
                        }
                workloads.append(wl_entry)
            models_summary[mkey] = {
                "vram_peak_mib": mval.get("vram_peak_mib", 0.0),
                "gpu_layers": (mval.get("plan") or {}).get("gpu_layers"),
                "cache_type_k": (mval.get("plan") or {}).get("cache_type_k"),
                "workloads": workloads,
            }
        versions.append({
            "label": label,
            "version": raw.get("version", label),
            "timestamp": raw.get("timestamp"),
            "file": fpath.name,
            "models": models_summary,
        })
    return {"versions": versions, "count": len(versions)}


def create_ui_router(ctx: StudioRuntimeContext) -> APIRouter:
    """Creates the /api/ui/* FastAPI APIRouter bound to StudioRuntimeContext."""
    router = APIRouter(prefix="/api/ui", tags=["Tendou Studio Control Plane"])

    @router.get("/state")
    async def get_studio_state():
        backend = getattr(ctx.app.state, "backend", None)
        info = backend.get_model_info() if (backend and hasattr(backend, "get_model_info")) else {}
        status = ctx.compute_current_status()
        snap = ctx.telemetry.snapshot(
            backend=backend,
            engine_status=status,
            active_http_requests=getattr(ctx.app.state, "active_requests", 0),
            record_history=False,
        )
        return {
            "status": status,
            "last_error": ctx.last_error,
            "version": ctx.version,
            "os": "windows" if os.name == "nt" else "linux",
            "platform": _platform.platform(),
            "host": ctx.host,
            "port": ctx.port,
            "exe_path": getattr(backend, "exe_path", None) if backend else ctx.running_settings.exe_path,
            "active_model": {
                "id": ctx.active_model_id(),
                "path": info.get("path") or ctx.running_settings.model_path,
                "backend": info.get("backend", "llama-atlas-engine"),
                "uptime_sec": info.get("uptime_sec", 0),
            },
            "hardware": asdict(ctx.hw),
            "autotune_plan": ctx.active_autotune_plan(),
            "settings": ctx.saved_settings.model_dump(),
            "pending_restart": len(compute_settings_diff(ctx.running_settings, ctx.saved_settings)) > 0,
            "telemetry": snap,
        }

    @router.get("/telemetry/stream")
    async def stream_telemetry(request: Request):
        async def _gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    backend = getattr(ctx.app.state, "backend", None)
                    payload = ctx.telemetry.snapshot(
                        backend=backend,
                        engine_status=ctx.compute_current_status(),
                        active_http_requests=getattr(ctx.app.state, "active_requests", 0),
                        record_history=True,
                    )
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0.25)  # 4 Hz live TPS updates
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/models")
    async def list_models(ctx_size: int = Query(default=8192, ge=2048, le=262144)):
        models = await asyncio.to_thread(
            scan_local_models,
            base_dir=REPO_ROOT,
            extra_dirs=ctx.extra_model_dirs,
            hw=ctx.hw,
            ctx_size=ctx_size,
        )
        active_path = os.path.abspath(ctx.running_settings.model_path) if ctx.running_settings.model_path else ""
        for m in models:
            try:
                m["active"] = bool(active_path and os.path.abspath(m["path"]) == active_path)
            except OSError:
                m["active"] = False
        return {
            "models": models,
            "count": len(models),
            "active_model_path": ctx.running_settings.model_path,
            "extra_dirs": ctx.extra_model_dirs,
        }

    @router.post("/models/scan")
    async def scan_models_directory(req: ModelScanRequest):
        target_dir = Path(req.directory).expanduser()
        if not target_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory}")
        resolved = str(target_dir.resolve())
        if resolved not in ctx.extra_model_dirs:
            ctx.extra_model_dirs.append(resolved)
        models = await asyncio.to_thread(
            scan_local_models,
            base_dir=REPO_ROOT,
            extra_dirs=ctx.extra_model_dirs,
            hw=ctx.hw,
            ctx_size=ctx.saved_settings.ctx_size,
        )
        return {
            "scanned": resolved,
            "models": models,
            "count": len(models),
            "extra_dirs": ctx.extra_model_dirs,
        }

    @router.get("/config")
    async def get_config():
        return {
            "settings": ctx.saved_settings.model_dump(),
            "running_settings": ctx.running_settings.model_dump(),
            "diff_from_running": compute_settings_diff(ctx.running_settings, ctx.saved_settings),
            "cli": {
                "linux": serialize_to_cli_args(ctx.saved_settings, platform="linux")["command"],
                "windows": serialize_to_cli_args(ctx.saved_settings, platform="windows")["command"],
            },
            "profile_path": str(ctx.profile_store.profile_path),
        }

    @router.put("/config")
    async def update_config(payload: Dict[str, Any]):
        merged = ctx.saved_settings.model_dump()
        merged.update(payload)
        try:
            new_cfg = EngineSettingsModel(**merged)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid settings: {exc}")
        ctx.saved_settings = new_cfg
        try:
            ctx.profile_store.save(new_cfg)
        except Exception as exc:
            logger.warning("Failed to persist config profile: %s", exc)
        return {
            "settings": new_cfg.model_dump(),
            "running_settings": ctx.running_settings.model_dump(),
            "diff_from_running": compute_settings_diff(ctx.running_settings, new_cfg),
            "cli": {
                "linux": serialize_to_cli_args(new_cfg, platform="linux")["command"],
                "windows": serialize_to_cli_args(new_cfg, platform="windows")["command"],
            },
            "profile_path": str(ctx.profile_store.profile_path),
            "saved": True,
        }

    @router.get("/config/schema")
    async def get_config_schema():
        return get_settings_schema(
            ram_total_gb=ctx.hw.ram_total_gb,
            vram_total_mb=ctx.hw.vram_total_mb,
        )

    @router.get("/config/presets")
    async def get_config_presets():
        return {"presets": get_presets()}

    @router.post("/engine/start")
    async def engine_start(req: Optional[EngineActionRequest] = None):
        return await ctx.start_engine(req)

    @router.post("/engine/stop")
    async def engine_stop():
        return await ctx.stop_engine()

    @router.post("/engine/restart")
    async def engine_restart(req: Optional[EngineActionRequest] = None):
        return await ctx.restart_engine(req)

    @router.get("/agents/configs")
    async def get_agent_configs():
        base_host = "127.0.0.1" if ctx.host in ("0.0.0.0", "::", "") else ctx.host
        base_url = f"http://{base_host}:{ctx.port}"
        model_id = ctx.active_model_id()

        opencode_json = json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "provider": {
                    "tendou-local": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Tendou Launcher (Local)",
                        "options": {"baseURL": f"{base_url}/v1", "apiKey": "tendou-local"},
                        "models": {
                            model_id: {
                                "name": f"Tendou {model_id}",
                                "limit": {"context": ctx.running_settings.ctx_size, "output": 16384},
                            }
                        },
                    }
                },
                "model": f"tendou-local/{model_id}",
            },
            indent=2,
        )

        claude_linux = (
            f'export ANTHROPIC_BASE_URL="{base_url}"\n'
            f'export ANTHROPIC_API_KEY="tendou-local"\n'
            f'claude --model "{model_id}"'
        )
        claude_windows = (
            f'$env:ANTHROPIC_BASE_URL = "{base_url}"\n'
            f'$env:ANTHROPIC_API_KEY = "tendou-local"\n'
            f'claude --model "{model_id}"'
        )

        cline_config = {
            "apiProvider": "OpenAI Compatible",
            "baseUrl": f"{base_url}/v1",
            "apiKey": "tendou-local",
            "modelId": model_id,
            "contextWindow": ctx.running_settings.ctx_size,
        }

        curl_snippet = (
            f"curl {base_url}/v1/chat/completions \\\n"
            f'  -H "Content-Type: application/json" \\\n'
            f'  -d \'{{"model": "{model_id}", "messages": [{{"role": "user", "content": "Hello Tendou!"}}], "stream": true}}\''
        )

        python_snippet = (
            "from openai import OpenAI\n\n"
            f'client = OpenAI(base_url="{base_url}/v1", api_key="tendou-local")\n'
            "resp = client.chat.completions.create(\n"
            f'    model="{model_id}",\n'
            '    messages=[{"role": "user", "content": "Explain 3-tier MoE caching in 3 bullets."}],\n'
            "    stream=True,\n"
            ")\n"
            "for chunk in resp:\n"
            '    print(chunk.choices[0].delta.content or "", end="", flush=True)\n'
        )

        return {
            "base_url": base_url,
            "openai_url": f"{base_url}/v1",
            "anthropic_url": f"{base_url}/v1/messages",
            "model_id": model_id,
            "claude_code": {"linux": claude_linux, "windows": claude_windows},
            "opencode": {"filename": "opencode.json", "content": opencode_json},
            "cline": cline_config,
            "curl": curl_snippet,
            "python": python_snippet,
        }

    @router.post("/benchmark/run")
    async def run_quick_benchmark(req: QuickBenchmarkRequest):
        backend = getattr(ctx.app.state, "backend", None)
        if backend is None or ctx.compute_current_status() != "ready":
            raise HTTPException(status_code=503, detail="Engine is not ready. Start the engine before benchmarking.")
        if ctx.benchmark_running:
            raise HTTPException(status_code=409, detail="A benchmark is already in progress.")

        ctx.benchmark_running = True
        ctx.app.state.active_requests = getattr(ctx.app.state, "active_requests", 0) + 1
        backend_has_telemetry = getattr(backend, "telemetry_hub", None) is ctx.telemetry
        try:
            prompt = f"<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n"
            est_prompt_tokens = max(8, len(prompt) // 4)

            async def _run_pass(pass_label: str) -> Dict[str, Any]:
                rid = f"studio-bench-{pass_label}-{int(time.time() * 1000)}"
                t0 = time.perf_counter()
                first_tok_t: Optional[float] = None
                tokens = 0
                out_chunks: List[str] = []
                if not backend_has_telemetry:
                    ctx.telemetry.on_request_start(rid, prompt_tokens=est_prompt_tokens)
                async for ev in backend.generate(
                    prompt=prompt,
                    n_predict=req.max_tokens,
                    temp=req.temperature,
                    top_p=0.9,
                    req_id=rid,
                ):
                    etype = ev.get("event")
                    if etype == "token":
                        now = time.perf_counter()
                        n_tok = int(ev.get("n_tokens", 1))
                        if first_tok_t is None:
                            first_tok_t = now
                            ctx.telemetry.on_prefill_done(
                                ttft_ms=(now - t0) * 1000.0,
                                prompt_tokens=est_prompt_tokens,
                            )
                        tokens += n_tok
                        if not backend_has_telemetry:
                            ctx.telemetry.on_token(n_tok)
                        out_chunks.append(ev.get("token", ""))
                    elif etype == "error":
                        raise RuntimeError(ev.get("message", "Engine error"))
                t_end = time.perf_counter()
                ttft_ms = round(((first_tok_t or t_end) - t0) * 1000.0, 2)
                dec_s = max(0.001, t_end - (first_tok_t or t0))
                dec_tps = round(max(1, tokens - 1) / dec_s, 2) if tokens > 1 else round(tokens / dec_s, 2)
                if not backend_has_telemetry:
                    ctx.telemetry.on_request_done(
                        prompt_tokens=est_prompt_tokens,
                        completion_tokens=tokens,
                        ttft_ms=ttft_ms,
                        decode_tps=dec_tps,
                    )
                return {
                    "pass": pass_label,
                    "ttft_ms": ttft_ms,
                    "decode_tps": dec_tps,
                    "tokens": tokens,
                    "elapsed_s": round(t_end - t0, 3),
                    "preview": "".join(out_chunks)[:240],
                }

            cold = await _run_pass("cold")
            warm = await _run_pass("warm")
            await asyncio.to_thread(ctx.telemetry.maybe_refresh_system_metrics, True)
            snap = ctx.telemetry.snapshot(backend=backend, engine_status="ready", record_history=False)
            result = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version": ctx.version,
                "model_id": ctx.active_model_id(),
                "cold": cold,
                "warm": warm,
                "ttft_speedup_x": round(cold["ttft_ms"] / max(0.1, warm["ttft_ms"]), 2),
                "vram_used_mb": snap["gpu"]["used_mb"],
                "vram_peak_mb": snap["gpu"]["peak_mb"],
            }
            ctx.last_benchmark_result = result
            return result
        except Exception as exc:
            if not backend_has_telemetry:
                ctx.telemetry.on_request_done()
            raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}")
        finally:
            ctx.app.state.active_requests = max(0, getattr(ctx.app.state, "active_requests", 1) - 1)
            ctx.benchmark_running = False

    @router.get("/benchmark/history")
    async def get_benchmark_history():
        hist = await asyncio.to_thread(_load_benchmark_history)
        hist["last_live_run"] = ctx.last_benchmark_result
        return hist

    @router.get("/logs/stream")
    async def stream_logs(request: Request, replay: int = Query(default=100, ge=0, le=500)):
        sub = ctx.telemetry.subscribe_logs()
        _, q = sub

        async def _gen():
            try:
                if replay > 0:
                    for entry in ctx.telemetry.get_recent_logs(limit=replay):
                        yield f"data: {json.dumps(entry)}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        entry = await asyncio.wait_for(q.get(), timeout=1.0)
                        yield f"data: {json.dumps(entry)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                return
            finally:
                ctx.telemetry.unsubscribe_logs(sub)

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
