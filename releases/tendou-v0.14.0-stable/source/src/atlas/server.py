"""
tendou launcher — OpenAI-Compatible API Server.

Provides a production-ready OpenAI-compatible HTTP API (/v1/models, /v1/chat/completions,
/v1/completions, /health) backed by the tendou launcher runtime with GPU-first parallel inference,
caching, and prefetching.
"""

import argparse
import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
import socket
import sys
import sysconfig
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("atlas.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

try:
    from src.atlas.autotune import estimate_prompt_pld_suitability
except ImportError:
    try:
        from autotune import estimate_prompt_pld_suitability
    except ImportError:
        estimate_prompt_pld_suitability = None

DEFAULT_MODEL_ID = os.environ.get("TENDOU_MODEL_ID", "qwen3.8-flash-next")
_NEW_IQ3_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "qwen3.8-gsq-rco-iq3_xxs", "IQ3_XXS", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)
DEFAULT_MODEL_PATH = os.environ.get("TENDOU_MODEL_PATH", _NEW_IQ3_MODEL if os.path.exists(_NEW_IQ3_MODEL) else "")
def _resolve_default_engine_path() -> str:
    if os.environ.get("TENDOU_ENGINE_PATH"):
        return os.environ["TENDOU_ENGINE_PATH"]
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(base_dir, "releases", "tendou-v0.14.0-stable", "bin", "tendou-engine.exe"),
        os.path.join(base_dir, "releases", "tendou-v0.13.0-stable", "bin", "tendou-engine.exe"),
        os.path.join(base_dir, "llama.cpp", "build", "bin", "Release", "tendou-engine.exe"),
        os.path.join(base_dir, "llama.cpp", "build", "bin", "Release", "llama-atlas-engine.exe"),
        os.path.join(base_dir, "releases", "tendou-v0.11.0-stable", "bin", "tendou-engine.exe"),
        os.path.join(base_dir, "releases", "tendou-v0.9.0-stable", "bin", "tendou-engine.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(base_dir, "llama.cpp", "build", "bin", "Release", "tendou-engine.exe")

DEFAULT_EXE_PATH = _resolve_default_engine_path()
DEFAULT_MTP_PATH = os.environ.get("TENDOU_MTP_PATH") or os.environ.get("ATLAS_MTP_PATH") or (
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "mtp", "Qwen3.8-Flash-Next-MTP-Q4_0.gguf")
    if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "mtp", "Qwen3.8-Flash-Next-MTP-Q4_0.gguf"))
    else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "mtp", "Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf")
)


# ---------------------------------------------------------------------------
# Backend Engine Interface
# ---------------------------------------------------------------------------

class BaseAtlasBackend:
    """Abstract interface for tendou launcher inference."""

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def generate(
        self,
        prompt: str,
        n_predict: int = 512,
        temp: float = 0.7,
        top_p: float = 0.9,
        min_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        req_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yields event dicts: {'event': 'token', 'token': ...} and {'event': 'done', ...}"""
        raise NotImplementedError

    async def cancel(self, req_id: str) -> None:
        raise NotImplementedError

    def is_healthy(self) -> bool:
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        raise NotImplementedError


class AtlasSubprocessBackend(BaseAtlasBackend):
    """
    Subprocess backend managing llama-atlas-engine.exe running in persistent --ipc mode.
    Maintains resident model weights and dispatches requests over JSON IPC.
    """

    def __init__(
        self,
        exe_path: str = DEFAULT_EXE_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        threads: int = 8,
        boost: bool = True,
        gpu: bool = True,
        gpu_layers: int = 2,
        ngl: int = 99,
        vram_cache: int = 3200,
        ctx_size: int = 8192,
        odmoe_lead: int = 2,
        odmoe_quick_evict: bool = True,
        spice_conf_high: float = 0.65,
        spice_conf_mid: float = 0.30,
        tutti_async_io: bool = False,
        readback_interval: int = 64,
        prefetch_candidates: int = 10,
        tutti_queue_depth: int = 64,
        p2_gpu_binding: bool = False,
        p2_vram_budget_mib: int = 0,
        p3_async_transfer: bool = False,
        p3_staging_slots: int = 4,
        p3_queue_depth: int = 8,
        prompt_cache_mb: int = 512,
        cuda_sched: Optional[str] = "yield",
        cache_type_k: Optional[str] = None,
        cache_type_v: Optional[str] = None,
        mtp: str = "pld",
        mtp_draft_n: int = 2,
        mtp_path: Optional[str] = None,
        mtp_loc: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        boost_level: int = 2,
        tlep: bool = False,
        tlep_max_depth: int = 4,
        tlep_min_guard_ms: float = 1.5,
        tlep_pf_bw_budget_mb: int = 128,
        moe_cache_mib: int = 2048,
        min_p: float = 0.0,
        cascading_migration: bool = True,
    ):
        self.exe_path = exe_path
        self.model_path = model_path
        self.threads = threads
        self.boost = boost
        self.boost_level = boost_level
        self.cascading_migration = cascading_migration
        self.tlep = tlep or (os.environ.get("ATLAS_TLEP") in ("1", "true"))
        self.tlep_max_depth = tlep_max_depth
        self.tlep_min_guard_ms = tlep_min_guard_ms
        self.tlep_pf_bw_budget_mb = tlep_pf_bw_budget_mb
        self.moe_cache_mib = moe_cache_mib
        self.min_p = min_p
        self.gpu = gpu
        self.gpu_layers = gpu_layers
        self.ngl = ngl
        self.vram_cache = vram_cache
        self.ctx_size = ctx_size
        self.odmoe_lead = odmoe_lead
        self.odmoe_quick_evict = odmoe_quick_evict
        self.spice_conf_high = spice_conf_high
        self.spice_conf_mid = spice_conf_mid
        self.tutti_async_io = tutti_async_io
        self.readback_interval = readback_interval
        self.prefetch_candidates = prefetch_candidates
        self.tutti_queue_depth = tutti_queue_depth
        self.p2_gpu_binding = p2_gpu_binding
        self.p2_vram_budget_mib = p2_vram_budget_mib
        self.p3_async_transfer = p3_async_transfer
        self.p3_staging_slots = p3_staging_slots
        self.p3_queue_depth = p3_queue_depth
        self.prompt_cache_mb = prompt_cache_mb
        self.cuda_sched = cuda_sched
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.mtp = mtp
        self.mtp_draft_n = mtp_draft_n
        if mtp_path is None and os.path.exists(DEFAULT_MTP_PATH):
            mtp_path = DEFAULT_MTP_PATH
        self.mtp_path = mtp_path
        self.mtp_loc = mtp_loc or os.environ.get("ATLAS_MTP_LOC")
        self.extra_args = extra_args or []

        self.proc: Optional[asyncio.subprocess.Process] = None
        self.ready_event = asyncio.Event()
        self.active_queues: Dict[str, asyncio.Queue] = {}
        self.lock = asyncio.Lock()
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._is_ready = False
        self._start_time = 0.0
        self.stderr_lines: List[str] = []

    async def start(self) -> None:
        self.ready_event.clear()
        self._is_ready = False
        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(f"tendou launcher binary not found at: {self.exe_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model GGUF file not found at: {self.model_path}")

        cmd = [
            self.exe_path,
            "-m", self.model_path,
            "--ipc",
            "-t", str(self.threads),
            "-c", str(self.ctx_size),
        ]
        if self.gpu:
            has_gpu_layers = any(arg in ("-ngl", "--n-gpu-layers", "--atlas-gpu-layers") for arg in self.extra_args)
            eff_gpu_layers = self.gpu_layers
            if not has_gpu_layers and self.mtp and self.mtp != "off" and self.gpu_layers == 2:
                # In full VRAM MTP mode, 8GB VRAM must house target dense layers + 2.15GB MTP draft head.
                # Setting target gpu_expert_layers=0 frees 3.8GB VRAM, keeping MTP 100% resident in VRAM.
                # In hybrid mode (mtp == "hybrid" or mtp_loc == "hybrid"), draft MoE experts stay in host RAM
                # (~250-800MB VRAM used), allowing target model to retain gpu_layers=2 without WDDM PCIe paging!
                if self.mtp == "vram" and self.mtp_loc != "hybrid":
                    eff_gpu_layers = 0
            cmd.extend([
                "-ngl", str(self.ngl),
                "--atlas-gpu-first",
                "--atlas-gpu-layers", str(eff_gpu_layers),
                "--atlas-vram-cache", str(self.vram_cache),
                "--atlas-readback-interval", str(self.readback_interval),
                "--atlas-readback-warmup", "0" if self.boost else "2",
            ])
            # Phase 8 Task D.1: Adaptive / Asymmetric KV Cache Quantization
            has_ctk = any(arg in ("-ctk", "--cache-type-k") for arg in self.extra_args)
            has_ctv = any(arg in ("-ctv", "--cache-type-v") for arg in self.extra_args)
            if self.ctx_size <= 8192:
                default_k = "q8_0"
                default_v = "q8_0"
            elif self.ctx_size <= 32768:
                default_k = "q8_0"
                default_v = "q4_0"
            else:
                default_k = "q4_0"
                default_v = "q4_0"
            ctk = self.cache_type_k or default_k
            ctv = self.cache_type_v or (default_v if self.cache_type_k is None else self.cache_type_k)
            if not has_ctk and ctk:
                cmd.extend(["-ctk", ctk])
            if not has_ctv and ctv:
                cmd.extend(["-ctv", ctv])
        else:
            cmd.append("--cpu-only")

        if self.boost:
            cmd.append("--boost")
            cmd.extend(["--atlas-boost-level", str(self.boost_level)])
            if not any(arg == "--atlas-k" or arg == "--atlas-adaptive-k" for arg in self.extra_args):
                cmd.extend(["--atlas-k", "5"])
        cmd.extend([
            "--atlas-odmoe-lead", str(self.odmoe_lead),
            "--atlas-odmoe-quick-evict", "1" if self.odmoe_quick_evict else "0",
            "--atlas-spice-conf-high", str(self.spice_conf_high),
            "--atlas-spice-conf-mid", str(self.spice_conf_mid),
            "--atlas-tutti-async-io", "1" if self.tutti_async_io else "0",
            "--atlas-prefetch-candidates", str(self.prefetch_candidates),
            "--atlas-tutti-queue-depth", str(self.tutti_queue_depth),
        ])
        if self.p2_gpu_binding:
            cmd.append("--atlas-p2-gpu-binding")
            if self.p2_vram_budget_mib > 0:
                cmd.extend(["--atlas-p2-vram-budget-mib", str(self.p2_vram_budget_mib)])
        if self.p3_async_transfer:
            cmd.append("--atlas-p3-async-transfer")
            cmd.extend([
                "--atlas-p3-staging-slots", str(self.p3_staging_slots),
                "--atlas-p3-queue-depth", str(self.p3_queue_depth),
            ])
        if self.prompt_cache_mb is not None and not any(arg == "--atlas-prompt-cache-mb" for arg in self.extra_args):
            cmd.extend(["--atlas-prompt-cache-mb", str(self.prompt_cache_mb)])
        if self.cuda_sched and not any(arg == "--atlas-cuda-sched" for arg in self.extra_args):
            cmd.extend(["--atlas-cuda-sched", str(self.cuda_sched)])
        if self.moe_cache_mib > 0 and not any(arg in ("--atlas-moe-cache-mib", "--moe-cache-mib") for arg in self.extra_args):
            cmd.extend(["--atlas-moe-cache-mib", str(self.moe_cache_mib)])
        if not any(arg == "--min-p" for arg in self.extra_args):
            cmd.extend(["--min-p", str(self.min_p)])
        if self.mtp and self.mtp != "off":
            if not any(arg == "--atlas-mtp" for arg in self.extra_args):
                cmd.extend(["--atlas-mtp", str(self.mtp)])
            if not any(arg in ("--atlas-mtp-draft-n", "--atlas-mtp-n") for arg in self.extra_args):
                cmd.extend(["--atlas-mtp-draft-n", str(self.mtp_draft_n)])
            if self.mtp_path and not any(arg == "--atlas-mtp-path" for arg in self.extra_args):
                cmd.extend(["--atlas-mtp-path", str(self.mtp_path)])
            if self.mtp_loc and not any(arg == "--atlas-mtp-loc" for arg in self.extra_args):
                cmd.extend(["--atlas-mtp-loc", str(self.mtp_loc)])
        else:
            if not any(arg == "--atlas-mtp" for arg in self.extra_args):
                cmd.extend(["--atlas-mtp", "off"])
        if self.tlep and not any(arg == "--atlas-tlep" for arg in self.extra_args):
            cmd.append("--atlas-tlep")
            if self.tlep_max_depth and not any(arg == "--atlas-tlep-depth" for arg in self.extra_args):
                cmd.extend(["--atlas-tlep-depth", str(self.tlep_max_depth)])
            if self.tlep_min_guard_ms and not any(arg == "--atlas-tlep-guard" for arg in self.extra_args):
                cmd.extend(["--atlas-tlep-guard", str(self.tlep_min_guard_ms)])
            if self.tlep_pf_bw_budget_mb and not any(arg == "--atlas-tlep-bw" for arg in self.extra_args):
                cmd.extend(["--atlas-tlep-bw", str(self.tlep_pf_bw_budget_mb)])

        is_legacy_exe = any(k in self.exe_path for k in ("v0.11", "v0.9", "v0.8", "baseline"))
        if self.cascading_migration and not is_legacy_exe and not any(arg in ("--atlas-cascading-working-set", "--atlas-3tier-migration") for arg in self.extra_args):
            cmd.extend(["--atlas-cascading-working-set", "1"])

        # Resolve physical map: respect explicit extra_args, env var, or auto-detect 35B
        if not any(arg == "--atlas-physical-map" for arg in self.extra_args):
            phys_env = os.environ.get("TENDOU_PHYSICAL_MAP")
            if phys_env and os.path.exists(phys_env):
                cmd.extend(["--atlas-physical-map", phys_env])
            elif self.model_path and ("35B" in self.model_path or "35b" in self.model_path):
                cand = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "atlas_physical_map_35b.json"
                )
                if os.path.exists(cand):
                    cmd.extend(["--atlas-physical-map", cand])

        cmd.extend(self.extra_args)

        logger.info(f"Launching tendou launcher: {' '.join(cmd)}")
        self._start_time = time.time()

        env = os.environ.copy()
        env["GGML_CUDA_REGISTER_HOST"] = os.environ.get("GGML_CUDA_REGISTER_HOST", "1")
        if self.cuda_sched:
            env["ATLAS_CUDA_SCHED"] = os.environ.get("ATLAS_CUDA_SCHED", str(self.cuda_sched))
        if self.boost:
            env["ATLAS_CPU_DYNAMIC_K"] = str(self.boost_level)
        if self.moe_cache_mib > 0:
            env["ATLAS_MOE_CACHE_MIB"] = str(self.moe_cache_mib)
        if self.mtp != "off":
            env["ATLAS_MTP_UNION_CAP"] = os.environ.get("ATLAS_MTP_UNION_CAP", "6")
        if "ATLAS_PREFILL_TOP_K" in os.environ:
            env["ATLAS_PREFILL_TOP_K"] = os.environ["ATLAS_PREFILL_TOP_K"]
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cuda_bin_candidates = [
            os.path.join(sysconfig.get_path("purelib"), "nvidia", "cu13", "bin", "x86_64"),
            os.path.join(repo_root, ".venv-audit", "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64"),
            os.path.join(repo_root, ".venv", "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64"),
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
        ]
        exe_dir = os.path.dirname(self.exe_path) if self.exe_path else os.path.dirname(DEFAULT_EXE_PATH)
        if os.name == "nt":
            extra_dirs = [d for d in [exe_dir, *cuda_bin_candidates] if os.path.isdir(d)]
            if extra_dirs:
                env["PATH"] = os.pathsep.join(extra_dirs) + os.pathsep + env.get("PATH", "")

        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._reader_task = asyncio.create_task(self._stdout_reader())
        self._stderr_task = asyncio.create_task(self._stderr_reader())

        # Wait up to 300s for [ATLAS_READY] (large multi-shard GGUF cold mmap safety)
        try:
            await asyncio.wait_for(self.ready_event.wait(), timeout=300.0)
            if self.proc.returncode is not None:
                code = self.proc.returncode
                stderr_dump = "\n".join(self.stderr_lines[-30:])
                await self.stop()
                raise RuntimeError(f"tendou launcher exited during startup (code {code}):\n{stderr_dump}")
            self._is_ready = True
            logger.info("tendou launcher is READY for inference.")
        except asyncio.TimeoutError:
            self._is_ready = False
            logger.error("Timeout waiting for tendou launcher [ATLAS_READY].")
            await self.stop()
            raise RuntimeError("tendou launcher failed to initialize in time.")

    async def _stdout_reader(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line_bytes = await self.proc.stdout.readline()
            if not line_bytes:
                self._is_ready = False
                await self.proc.wait()
                self.ready_event.set()
                for rid, queue in list(self.active_queues.items()):
                    queue.put_nowait({"event": "error", "id": rid,
                                      "message": f"tendou launcher exited (code {self.proc.returncode})"})
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if "[ATLAS_READY]" in line:
                logger.info("Received [ATLAS_READY] from engine stdout.")
                self.ready_event.set()
                continue

            # Parse JSON event
            try:
                data = json.loads(line)
                req_id = data.get("id")
                if req_id and req_id in self.active_queues:
                    await self.active_queues[req_id].put(data)
            except json.JSONDecodeError:
                # Debug output from engine
                logger.debug(f"[Engine Stdout] {line}")

    async def _stderr_reader(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line_bytes = await self.proc.stderr.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line:
                self.stderr_lines.append(line)
                logger.debug(f"[Engine Stderr] {line}")

    async def stop(self) -> None:
        self._is_ready = False
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(b'{"cmd":"exit"}\n')
                    await self.proc.stdin.drain()
            except Exception:
                pass

            try:
                await asyncio.wait_for(self.proc.wait(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                try:
                    if os.name == "nt":
                        import subprocess
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.proc.kill()
                except Exception:
                    pass

            if self._reader_task:
                self._reader_task.cancel()
            if self._stderr_task:
                self._stderr_task.cancel()

            if hasattr(self.proc, "_transport") and self.proc._transport:
                try:
                    self.proc._transport.close()
                except Exception:
                    pass

            try:
                await asyncio.wait_for(self.proc.wait(), timeout=1.0)
            except Exception:
                pass

            self._reader_task = None
            self._stderr_task = None
        self.proc = None

    async def generate(
        self,
        prompt: str,
        n_predict: int = 512,
        temp: float = 0.7,
        top_p: float = 0.9,
        min_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        req_id: Optional[str] = None,
        mtp_draft_n: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not self._is_ready or not self.proc or not self.proc.stdin:
            raise RuntimeError("tendou launcher backend is not ready.")

        rid = req_id or f"req-{uuid.uuid4().hex[:8]}"
        q: asyncio.Queue = asyncio.Queue()
        self.active_queues[rid] = q

        cmd_payload = {
            "cmd": "generate",
            "id": rid,
            "prompt": prompt,
            "n_predict": n_predict,
            "temp": temp,
            "top_p": top_p,
            "min_p": min_p if min_p is not None else self.min_p,
            "stop": stop or [],
        }
        if mtp_draft_n is not None:
            cmd_payload["mtp_draft_n"] = int(mtp_draft_n)

        async with self.lock:
            try:
                line = json.dumps(cmd_payload) + "\n"
                self.proc.stdin.write(line.encode("utf-8"))
                await self.proc.stdin.drain()

                consecutive_pings = 0
                max_consecutive_pings = 60  # Max 120s of silence before aborting
                while True:
                    if not self._is_ready or (self.proc and self.proc.returncode is not None):
                        yield {"event": "error", "message": f"tendou launcher process exited (code {self.proc.returncode if self.proc else 'unknown'})"}
                        break
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=2.0)
                        consecutive_pings = 0
                    except asyncio.TimeoutError:
                        consecutive_pings += 1
                        if not self._is_ready or (self.proc and self.proc.returncode is not None):
                            yield {"event": "error", "message": f"tendou launcher process exited (code {self.proc.returncode if self.proc else 'unknown'})"}
                            break
                        if consecutive_pings >= max_consecutive_pings:
                            yield {"event": "error", "message": "Inference timed out after 120 seconds of silence"}
                            break
                        yield {"event": "ping"}
                        continue
                    # Micro-Batch Queue Drain: PLD çoklu token ürettiğinde kuyrukta bekleyen
                    # ardışık tokenları tek event-loop adımında birleştirerek IPC gecikmesini sıfırla
                    if isinstance(event, dict) and event.get("event") == "token":
                        merged_text = event.get("token", "")
                        merged_count = int(event.get("n_tokens", 1))
                        leftover = None
                        while not q.empty():
                            try:
                                nxt = q.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if nxt is None:
                                await q.put(None)
                                break
                            if isinstance(nxt, dict) and nxt.get("event") == "token":
                                merged_text += nxt.get("token", "")
                                merged_count += int(nxt.get("n_tokens", 1))
                            else:
                                leftover = nxt
                                break
                        event = dict(event, token=merged_text, n_tokens=merged_count)
                        yield event
                        if leftover is not None:
                            yield leftover
                            if leftover.get("event") in ("done", "error"):
                                break
                    else:
                        yield event
                        if event.get("event") in ("done", "error"):
                            break
            finally:
                self.active_queues.pop(rid, None)

    async def cancel(self, req_id: str) -> None:
        if self.proc and self.proc.stdin and not self.proc.stdin.is_closing():
            try:
                cmd = json.dumps({"cmd": "cancel", "id": req_id}) + "\n"
                self.proc.stdin.write(cmd.encode("utf-8"))
                await self.proc.stdin.drain()
            except Exception as e:
                logger.warning(f"Failed to send cancel to engine: {e}")

    def is_healthy(self) -> bool:
        return self._is_ready and (self.proc is not None) and (self.proc.returncode is None)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "id": DEFAULT_MODEL_ID,
            "path": self.model_path,
            "boost": self.boost,
            "boost_level": self.boost_level,
            "gpu": self.gpu,
            "gpu_layers": self.gpu_layers if self.gpu else 0,
            "vram_capacity_mb": None,  # No measured device telemetry is available over IPC.
            "vram_status": "Static GPU offload" if self.gpu else "CPU-only",
            "threads": self.threads,
            "backend": "llama-atlas-engine",
            "uptime_sec": round(time.time() - self._start_time, 1) if self._start_time else 0,
        }


def extract_last_user_message(prompt: str) -> str:
    """Extract the last user turn from a ChatML prompt."""
    matches = re.findall(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)", prompt, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def parse_tools_from_prompt(prompt: str) -> List[Dict[str, Any]]:
    """Extract tool definitions from <tools> JSON block in prompt."""
    matches = re.findall(r"<tools>\s*(.*?)\s*</tools>", prompt, re.DOTALL)
    for m in matches:
        s = m.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                tools = json.loads(s)
                if isinstance(tools, list):
                    return tools
            except Exception:
                pass
    return []


def generate_valid_arguments_for_tool(tool_def: Dict[str, Any], user_context: str) -> Dict[str, Any]:
    """Generates schema-compliant arguments for a tool definition to satisfy client validation."""
    func = tool_def.get("function", {})
    params = func.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", list(props.keys()))

    args: Dict[str, Any] = {}
    target_keys = required if required else list(props.keys())

    for key in target_keys:
        p_info = props.get(key, {})
        p_type = p_info.get("type", "string")
        key_lower = key.lower()

        if key_lower in ("command", "cmd"):
            if "requirements.txt" in user_context.lower():
                args[key] = "type requirements.txt"
            else:
                args[key] = "dir"
        elif key_lower in ("filepath", "path", "filename", "file"):
            if "requirements.txt" in user_context.lower():
                args[key] = "requirements.txt"
            elif "main.py" in user_context.lower():
                args[key] = "main.py"
            else:
                args[key] = "README.md"
        elif key_lower in ("location", "city"):
            args[key] = "Tokyo" if "tokyo" in user_context.lower() else "San Francisco"
        elif key_lower in ("query", "search", "q"):
            args[key] = "tendou launcher"
        elif key_lower in ("action", "operation"):
            args[key] = "first"
        elif key_lower in ("oldstring", "old_str"):
            args[key] = "fastapi"
        elif key_lower in ("newstring", "new_str"):
            args[key] = "fastapi"
        elif key_lower in ("content", "text"):
            args[key] = "tendou launcher"
        elif p_type == "string":
            args[key] = "test"
        elif p_type in ("integer", "number"):
            args[key] = 1
        elif p_type == "boolean":
            args[key] = True
        elif p_type == "array":
            args[key] = []
        elif p_type == "object":
            args[key] = {}

    if not args and props:
        for k in props.keys():
            args[k] = "test"
    return args


class MockAtlasBackend(BaseAtlasBackend):
    """
    High-fidelity mock backend for integration testing and environments
    without the 134GB GGUF model shards. Supports streaming, tool calling,
    JSON mode, cancellation, and stop sequences matching OpenAI behavior.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, mtp: str = "pld"):
        self.model_id = model_id
        self.mtp = mtp
        self._is_ready = True
        self._cancelled_requests = set()
        self._start_time = time.time()
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        self._is_ready = True

    async def stop(self) -> None:
        self._is_ready = False

    async def generate(
        self,
        prompt: str,
        n_predict: int = 512,
        temp: float = 0.7,
        top_p: float = 0.9,
        min_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        req_id: Optional[str] = None,
        mtp_draft_n: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        rid = req_id or f"req-{uuid.uuid4().hex[:8]}"

        async with self.lock:
            last_user_msg = extract_last_user_message(prompt)
            user_msg_clean = re.sub(r"<tool_response>.*?</tool_response>", "", last_user_msg, flags=re.DOTALL).strip()
            user_msg_lower = user_msg_clean.lower()
            tools_list = parse_tools_from_prompt(prompt)

            # 1. Check if this is a follow-up turn after tool execution
            if "<tool_response>" in prompt and prompt.rfind("<tool_response>") > prompt.rfind("<tools>"):
                if "requirements.txt" in prompt.lower() or "package" in prompt.lower() or "web server" in prompt.lower():
                    response_text = (
                        "I have inspected `requirements.txt`. The web server and API packages installed are "
                        "FastAPI (v0.141.1) and Uvicorn (v0.52.4), along with Pydantic (v2.13.5), HTTPX (v0.28.1), "
                        "and OpenAI SDK (v3.8.0)."
                    )
                else:
                    response_text = "I have inspected the tool results. The operation succeeded and the requested details are verified."

            # 2. Check if JSON mode requested
            elif "Respond with a valid JSON object only" in prompt or "valid JSON object" in prompt:
                response_text = '{"status": "success", "data": {"result": 42, "items": ["alpha", "beta"]}}'

            # 3. Check if Tool Calling is explicitly or implicitly requested
            elif tools_list and (
                "You must call at least one function" in prompt
                or "You must call the function" in prompt
                or any(kw in user_msg_lower for kw in [
                    "weather", "search", "inspect", "check", "run", "bash", "execute",
                    "read", "edit", "call both", "call tool", "use tool", "multiple", "main.py"
                ])
            ):
                multiple_requested = (
                    "call both" in user_msg_lower
                    or "multiple" in user_msg_lower
                    or ("both" in user_msg_lower and "tool" in user_msg_lower)
                    or ("two" in user_msg_lower and "tool" in user_msg_lower)
                )

                if multiple_requested and len(tools_list) >= 2:
                    t1 = tools_list[0]
                    t2 = tools_list[1]
                    name1 = t1.get("function", {}).get("name", "tool1")
                    name2 = t2.get("function", {}).get("name", "tool2")
                    args1 = generate_valid_arguments_for_tool(t1, user_msg_clean)
                    args2 = generate_valid_arguments_for_tool(t2, user_msg_clean)
                    if "action" in args1:
                        args1["action"] = "first"
                    if "action" in args2:
                        args2["action"] = "second"
                    response_text = (
                        f"<tool_call>\n"
                        f'{{"name": "{name1}", "arguments": {json.dumps(args1)}}}\n'
                        f"</tool_call>\n"
                        f"<tool_call>\n"
                        f'{{"name": "{name2}", "arguments": {json.dumps(args2)}}}\n'
                        f"</tool_call>"
                    )
                else:
                    target_tool = tools_list[0]
                    for t in tools_list:
                        fname = t.get("function", {}).get("name", "").lower()
                        if fname and fname in user_msg_lower:
                            target_tool = t
                            break
                        if "weather" in user_msg_lower and "weather" in fname:
                            target_tool = t
                            break
                        if ("read" in user_msg_lower or "inspect" in user_msg_lower or "check" in user_msg_lower) and ("read" in fname or "bash" in fname):
                            target_tool = t
                            break
                        if "search" in user_msg_lower and "search" in fname:
                            target_tool = t
                            break

                    fname = target_tool.get("function", {}).get("name", "tool")
                    args = generate_valid_arguments_for_tool(target_tool, user_msg_clean)
                    response_text = (
                        f"<tool_call>\n"
                        f'{{"name": "{fname}", "arguments": {json.dumps(args)}}}\n'
                        f"</tool_call>"
                    )

            # 3b. Check if Cline Native XML Tool Calling mode is active
            elif is_cline_xml_prompt(prompt):
                if "[replace_in_file for" in last_user_msg or "attempt_completion" in user_msg_lower or "complete the task" in user_msg_lower:
                    response_text = (
                        "<thinking>\n"
                        "The file modifications and tests have succeeded. I can now present the final result using attempt_completion.\n"
                        "</thinking>\n"
                        "<attempt_completion>\n"
                        "<result>All requested changes have been applied and verified on tendou launcher.</result>\n"
                        "</attempt_completion>"
                    )
                elif "[read_file for" in last_user_msg or "<file_content" in last_user_msg or "replace_in_file" in user_msg_lower or "fix" in user_msg_lower or "refactor" in user_msg_lower:
                    target_path = "src/atlas/server.py"
                    path_m = re.search(r"([a-zA-Z0-9_/\.\-]+\.(?:py|ts|js|cpp|c|h|md|json))", user_msg_clean)
                    if path_m:
                        target_path = path_m.group(1)
                    response_text = (
                        "<thinking>\n"
                        f"I have analyzed `{target_path}` and will apply the targeted modification using replace_in_file.\n"
                        "</thinking>\n"
                        "<replace_in_file>\n"
                        f"<path>{target_path}</path>\n"
                        "<diff>\n"
                        "<<<<<<< SEARCH\n"
                        "def legacy_handler():\n"
                        "    return False\n"
                        "=======\n"
                        "def legacy_handler():\n"
                        "    return True\n"
                        ">>>>>>> REPLACE\n"
                        "</diff>\n"
                        "</replace_in_file>"
                    )
                elif "execute_command" in user_msg_lower or "pytest" in user_msg_lower or "run test" in user_msg_lower:
                    response_text = (
                        "<thinking>\n"
                        "I will execute the test suite to verify system behavior.\n"
                        "</thinking>\n"
                        "<execute_command>\n"
                        "<command>pytest -q</command>\n"
                        "<requires_approval>false</requires_approval>\n"
                        "</execute_command>"
                    )
                else:
                    target_path = "src/atlas/server.py"
                    path_m = re.search(r"([a-zA-Z0-9_/\.\-]+\.(?:py|ts|js|cpp|c|h|md|json))", user_msg_clean)
                    if path_m:
                        target_path = path_m.group(1)
                    response_text = (
                        "<thinking>\n"
                        f"I need to inspect `{target_path}` first using read_file.\n"
                        "</thinking>\n"
                        "<read_file>\n"
                        f"<path>{target_path}</path>\n"
                        "</read_file>"
                    )

            # 4. Standard conversational responses
            elif "memory hierarchy" in user_msg_lower or "atlas" in user_msg_lower:
                response_text = (
                    "The tendou launcher memory hierarchy consists of:\n\n"
                    "1. **NVMe SSD (Model Storage)**: Houses multi-part 134GB GGUF model shards with OS mmap.\n"
                    "2. **Host RAM (16 GiB)**: Keeps routed MoE expert tensors in pinned host memory.\n"
                    "3. **GPU VRAM (8 GiB)**: Resident dense and trunk tensors offloaded to GPU.\n"
                    "4. **Online Expert Prefetcher**: Observes router-selected experts, predicts transitions online, "
                    "and proactively asks the operating system to page expert slices into RAM before consumption."
                )
            elif any(w in user_msg_lower for w in ["hello", "hi", "hey"]):
                response_text = (
                    "Hello! I am tendou launcher, an OpenAI-compatible local inference engine with GPU-first parallel MoE dispatch. How can I assist you today?"
                )
            elif "items = ['a']" in user_msg_lower or "repetitive" in user_msg_lower:
                response_text = "    print('Element')\n    print('Done')\n"
            elif "gezegen" in user_msg_lower or "gunes sistemi" in user_msg_lower:
                response_text = (
                    "Güneş Sistemi'ndeki sekiz gezegen Güneş'e olan mesafelerine göre şunlardır:\n"
                    "1. Merkür: Güneş'e en yakın gezegen.\n"
                    "2. Venüs: Kalın atmosferiyle en sıcak gezegen.\n"
                    "3. Dünya: Yaşam barındıran mavi gezegen.\n"
                    "4. Mars: Kızıl gezegen.\n"
                    "5. Jüpiter: En büyük gaz devi.\n"
                    "6. Satürn: Görkemli halkaları olan gaz devi.\n"
                    "7. Uranüs: Buz devi.\n"
                    "8. Neptün: En dıştaki buz devi."
                )
            else:
                response_text = (
                    "tendou launcher is a high-performance local inference server with GPU-first parallel MoE dispatch."
                )

            # Split into simulated token chunks
            tokens = re.findall(r"\S+|\s+", response_text)
            prompt_tokens = max(1, len(re.findall(r"\S+|\s+", prompt)))
            generated_tokens = 0
            full_text = ""

            for tok in tokens:
                if rid in self._cancelled_requests:
                    self._cancelled_requests.remove(rid)
                    yield {
                        "event": "done",
                        "id": rid,
                        "finish_reason": "cancelled",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": generated_tokens,
                        "text": full_text,
                    }
                    return

                await asyncio.sleep(0.01)  # Simulate decode latency (~100 TPS mock)
                full_text += tok
                generated_tokens += 1

                # Check stop sequences
                if stop:
                    matched = False
                    for s in stop:
                        if s in full_text:
                            matched = True
                            break
                    if matched:
                        yield {
                            "event": "done",
                            "id": rid,
                            "finish_reason": "stop",
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": generated_tokens,
                            "text": full_text,
                        }
                        return

                if generated_tokens >= n_predict:
                    yield {
                        "event": "done",
                        "id": rid,
                        "finish_reason": "length",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": generated_tokens,
                        "text": full_text,
                    }
                    return

                yield {
                    "event": "token",
                    "id": rid,
                    "token": tok,
                    "token_id": 100 + generated_tokens,
                }

            yield {
                "event": "done",
                "id": rid,
                "finish_reason": "stop",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
                "text": full_text,
            }

    async def cancel(self, req_id: str) -> None:
        self._cancelled_requests.add(req_id)

    def is_healthy(self) -> bool:
        return self._is_ready

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "backend": "mock-atlas-engine",
            "uptime_sec": round(time.time() - self._start_time, 1),
        }


# ---------------------------------------------------------------------------
# OpenAI API Request & Response Models (Pydantic)
# ---------------------------------------------------------------------------

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None


class ResponseFormat(BaseModel):
    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: Optional[str] = DEFAULT_MODEL_ID
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = Field(None, alias="max_tokens")
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None
    seed: Optional[int] = None
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    user: Optional[str] = None
    min_p: Optional[float] = 0.0
    category: Optional[str] = None
    mtp_draft_n: Optional[int] = None


class CompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: Optional[str] = DEFAULT_MODEL_ID
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    category: Optional[str] = None
    mtp_draft_n: Optional[int] = None


# ---------------------------------------------------------------------------
# Cline / Agentic XML Protocol Helpers & Prompt Formatting
# ---------------------------------------------------------------------------

CLINE_XML_TOOLS: tuple[str, ...] = (
    "execute_command",
    "read_file",
    "write_to_file",
    "replace_in_file",
    "search_files",
    "list_files",
    "list_code_definition_names",
    "ask_followup_question",
    "attempt_completion",
    "use_mcp_tool",
    "access_mcp_resource",
    "browser_action",
    "new_task",
    "plan_mode_respond",
)

CLINE_XML_CLOSING_TAGS: tuple[str, ...] = tuple(f"</{t}>" for t in CLINE_XML_TOOLS)


def extract_message_content_text(content: Any) -> str:
    """
    Extracts clean text from OpenAI/Anthropic/Cline message content.
    Supports plain strings as well as multi-part content arrays
    ([{"type": "text", "text": "..."}]) without dumping raw JSON brackets,
    preserving exact token prefixes for 100% KV cache hits in Cline.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                btype = item.get("type", "")
                if btype == "text" or "text" in item:
                    txt = item.get("text")
                    if txt is not None:
                        parts.append(str(txt))
                elif btype in ("image_url", "image"):
                    parts.append("[Image]")
                elif "content" in item:
                    parts.append(extract_message_content_text(item.get("content")))
        return "\n\n".join(p for p in parts if p)
    return json.dumps(content)


def is_cline_xml_prompt(prompt: str, headers: Optional[Any] = None) -> bool:
    """
    Detects whether a request originates from Cline (or Nvidia NIM Cline agent)
    using native XML tool tags (<read_file>, <replace_in_file>, <write_to_file>, etc.).
    """
    if headers:
        try:
            title = (headers.get("x-title") or "").lower()
            referer = (headers.get("http-referer") or "").lower()
            if "cline" in title or "cline" in referer:
                return True
        except Exception:
            pass
    if "You are Cline" in prompt:
        return True
    if "<replace_in_file>" in prompt and "<read_file>" in prompt:
        return True
    if "<environment_details>" in prompt and (
        "<attempt_completion>" in prompt or "<execute_command>" in prompt or "<write_to_file>" in prompt
    ):
        return True
    return False


def convert_qwen_tool_call_to_cline_xml(text: str) -> str:
    """
    Bridges Qwen native <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    into Cline XML tool syntax (<tool_name><param>val</param></tool_name>) when the
    client is operating in Cline XML mode (req.tools is None).
    """
    if "<tool_call>" not in text:
        return text

    def _replace_tc(match: re.Match) -> str:
        body = match.group(1).strip()
        single = parse_single_tool_body(body, allow_unclosed=True)
        if not single:
            return match.group(0)
        fn = single.get("function", {})
        fn_name = fn.get("name", "")
        if fn_name not in CLINE_XML_TOOLS:
            return match.group(0)
        raw_args = fn.get("arguments", "{}")
        try:
            args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args_obj = {}
        if not isinstance(args_obj, dict):
            return match.group(0)
        xml_lines = [f"<{fn_name}>"]
        for k, v in args_obj.items():
            val_str = v if isinstance(v, str) else json.dumps(v)
            xml_lines.append(f"<{k}>{val_str}</{k}>")
        xml_lines.append(f"</{fn_name}>")
        return "\n".join(xml_lines)

    converted = re.sub(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", _replace_tc, text, flags=re.DOTALL)
    return converted


def truncate_after_first_cline_xml_tool(text: str) -> tuple[str, bool]:
    """
    Enforces Cline's single-tool-per-turn protocol invariant.
    If any Cline XML closing tool tag (e.g. </read_file>, </replace_in_file>,
    </execute_command>, </attempt_completion>) is completed in `text`, returns
    `(text_up_to_closing_tag, True)` so generation stops immediately without
    hallucinating a second tool call or fake <environment_details>.
    """
    earliest_end = -1
    for close_tag in CLINE_XML_CLOSING_TAGS:
        idx = text.find(close_tag)
        if idx != -1:
            open_tag = "<" + close_tag[2:]
            # Verify matching open tag exists prior to close tag (prevent false match inside system/thinking)
            open_idx = text.rfind(open_tag, 0, idx)
            if open_idx != -1:
                end_pos = idx + len(close_tag)
                if earliest_end == -1 or end_pos < earliest_end:
                    earliest_end = end_pos
    if earliest_end != -1:
        return text[:earliest_end], True
    return text, False


def format_chatml_prompt(
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Formats messages into Qwen ChatML prompt with tools, Cline multi-part content, and JSON mode instructions."""
    prompt_parts = []
    system_content_parts = []

    # 1. System Prompt Construction (supports both 'system' and 'developer' roles)
    user_system_found = False
    for msg in messages:
        if msg.role in ("system", "developer") and msg.content:
            text = extract_message_content_text(msg.content)
            if text:
                system_content_parts.append(text)
                user_system_found = True

    if not user_system_found:
        system_content_parts.append("You are a helpful assistant.")

    # Tools injection
    if tools and tool_choice != "none":
        tools_dict = [t.model_dump() for t in tools]
        tools_instruction = (
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{json.dumps(tools_dict, indent=2)}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "IMPORTANT: Inside <tool_call></tool_call>, write strictly valid JSON. Properly close and escape all quotes and string values before closing braces."
        )
        system_content_parts.append(tools_instruction)

        if isinstance(tool_choice, dict) and "function" in tool_choice:
            fname = tool_choice["function"].get("name", "")
            system_content_parts.append(f'\nYou must call the function "{fname}".')
        elif tool_choice == "required":
            system_content_parts.append("\nYou must call at least one function.")

    # JSON mode injection
    rf_type = None
    rf_schema = None
    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
        rf_schema = response_format.get("json_schema", {}).get("schema")
    elif isinstance(response_format, ResponseFormat):
        rf_type = response_format.type
        if response_format.json_schema:
            rf_schema = response_format.json_schema.get("schema")

    if rf_type in ("json_object", "json_schema"):
        instruction = "\n\nRespond with a valid JSON object only. Do not wrap output in markdown codeblocks."
        if rf_schema:
            instruction += f"\nThe JSON output must strictly adhere to this schema:\n{json.dumps(rf_schema, indent=2)}"
        system_content_parts.append(instruction)

    full_system = "\n\n".join(system_content_parts) if len(system_content_parts) > 1 and not tools else "".join(system_content_parts)
    prompt_parts.append(f"<|im_start|>system\n{full_system}<|im_end|>\n")

    # 2. Conversation Turns
    for msg in messages:
        if msg.role in ("system", "developer"):
            continue

        role = msg.role
        content = extract_message_content_text(msg.content)

        if role == "assistant":
            prompt_parts.append("<|im_start|>assistant\n")
            if content:
                prompt_parts.append(f"{content}\n")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    fn_name = func.get("name", "")
                    fn_args = func.get("arguments", "{}")
                    if isinstance(fn_args, str):
                        try:
                            fn_args_obj = json.loads(fn_args)
                        except Exception:
                            fn_args_obj = {}
                    else:
                        fn_args_obj = fn_args
                    prompt_parts.append(
                        f"<tool_call>\n{{\"name\": \"{fn_name}\", \"arguments\": {json.dumps(fn_args_obj)}}}\n</tool_call>\n"
                    )
            prompt_parts.append("<|im_end|>\n")
        elif role == "tool":
            prompt_parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
        else:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    prompt_parts.append("<|im_start|>assistant\n")
    if (tools and tool_choice != "none") or enable_thinking is False or enable_thinking is None:
        prompt_parts.append("<think>\n\n</think>\n\n")
    else:
        prompt_parts.append("<think>\n")
    return "".join(prompt_parts)


def parse_single_tool_body(body: str, allow_unclosed: bool = True) -> Optional[Dict[str, Any]]:
    """Helper to parse a single tool call from JSON or Qwen native XML format."""
    body = body.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
        body = body.strip()

    # 1. JSON format: {"name": "...", "arguments": ...}
    if body.startswith("{"):
        r_idx = body.rfind("}")
        candidate = body[:r_idx + 1] if r_idx != -1 else body

        parsed_obj = None
        # Try direct parse with strict=False (allows raw newlines in string literals)
        try:
            parsed_obj = json.loads(candidate, strict=False)
        except Exception:
            pass

        # Try fixing invalid JSON escapes (e.g. \), \(, etc.)
        if parsed_obj is None:
            fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', candidate)
            try:
                parsed_obj = json.loads(fixed, strict=False)
            except Exception:
                pass

        # Try fixing unclosed string quotes before trailing braces (e.g. "content": "print(...)} -> "content": "print(...)")
        if parsed_obj is None:
            for rep in [
                re.sub(r'([^"\\])\s*(\}+)\s*$', r'\1"\2', candidate.strip()),
                re.sub(r'([^"\\])\s*(\}+)', r'\1"\2', candidate.strip()),
            ]:
                try:
                    parsed_obj = json.loads(rep, strict=False)
                    if parsed_obj:
                        break
                except Exception:
                    fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', rep)
                    try:
                        parsed_obj = json.loads(fixed, strict=False)
                        if parsed_obj:
                            break
                    except Exception:
                        pass

        # Try auto-closing unbalanced braces/quotes for cut-off completions (only when allow_unclosed=True)
        if parsed_obj is None and allow_unclosed:
            open_b = candidate.count("{")
            close_b = candidate.count("}")
            if open_b > close_b:
                for suffix in ['"' + ('}' * (open_b - close_b)), '}' * (open_b - close_b)]:
                    cand = candidate + suffix
                    try:
                        parsed_obj = json.loads(cand, strict=False)
                        break
                    except Exception:
                        fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', cand)
                        try:
                            parsed_obj = json.loads(fixed, strict=False)
                            break
                        except Exception:
                            pass

        if isinstance(parsed_obj, dict):
            if "function" in parsed_obj and isinstance(parsed_obj["function"], dict):
                parsed_obj = parsed_obj["function"]
            if parsed_obj.get("name") and "arguments" in parsed_obj:
                args = parsed_obj.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                return {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": str(parsed_obj["name"]),
                        "arguments": args_str,
                    },
                }

        # Fallback regex extraction of function name & arguments if JSON parsing failed
        if parsed_obj is None:
            name_m = re.search(r'"name"\s*:\s*"([a-zA-Z0-9_\.\-]+)"', candidate)
            if name_m:
                fn_name = name_m.group(1)
                args_m = re.search(r'"arguments"\s*:\s*(\{.*\}|\[.*\]|"(?:\\.|[^"\\])*")', candidate, re.DOTALL)
                if args_m:
                    return {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_m.group(1),
                        },
                    }
                args_trail = re.search(r'"arguments"\s*:\s*(\{.*)$', candidate, re.DOTALL)
                if args_trail:
                    raw_args = args_trail.group(1).rstrip()
                    diff = raw_args.count("{") - raw_args.count("}")
                    if diff > 0:
                        raw_args += '"' + ('}' * diff)
                    return {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": raw_args,
                        },
                    }

    # 2. Native Qwen XML format: <function=name><parameter=key>val</parameter></function>
    fn_match = re.search(r"<function=([a-zA-Z0-9_\.\-]+)>(.*?)(?:</function>|$)", body, re.DOTALL)
    if fn_match:
        fn_name = fn_match.group(1).strip()
        params_content = fn_match.group(2)
        params = {}
        for p_match in re.finditer(r"<parameter=([a-zA-Z0-9_\.\-]+)>\s*(.*?)\s*(?:</parameter>|$)", params_content, re.DOTALL):
            p_name = p_match.group(1).strip()
            p_val = p_match.group(2).strip()
            try:
                params[p_name] = json.loads(p_val, strict=False)
            except Exception:
                params[p_name] = p_val
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": json.dumps(params),
            },
        }

    return None


class IncrementalJsonArgumentTokenizer:
    """Zero-allocation incremental JSON argument tokenizer for streaming tool calls."""
    def __init__(self):
        self.arg_brace_depth = 0
        self.arg_in_string = False
        self.arg_escape = False
        self.arg_started = False

    def process(self, raw_str: str) -> tuple[str, Optional[str]]:
        emit_chars = []
        trailer_str = None
        for i, c in enumerate(raw_str):
            if not self.arg_started:
                if c in " \t\r\n":
                    emit_chars.append(c)
                    continue
                elif c == "{":
                    self.arg_started = True
                    self.arg_brace_depth = 1
                    emit_chars.append(c)
                    continue
                else:
                    self.arg_started = True
                    emit_chars.append(c)
                    continue
            if self.arg_escape:
                self.arg_escape = False
                emit_chars.append(c)
            elif c == "\\" and self.arg_in_string:
                self.arg_escape = True
                emit_chars.append(c)
            elif c == '"':
                self.arg_in_string = not self.arg_in_string
                emit_chars.append(c)
            elif not self.arg_in_string:
                if c == "{":
                    self.arg_brace_depth += 1
                    emit_chars.append(c)
                elif c == "}":
                    self.arg_brace_depth -= 1
                    emit_chars.append(c)
                    if self.arg_brace_depth == 0:
                        trailer_str = raw_str[i + 1:]
                        break
                else:
                    emit_chars.append(c)
            else:
                emit_chars.append(c)
        return "".join(emit_chars), trailer_str


def parse_tool_calls_from_text(text: str, allow_unclosed: bool = True) -> tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Parses <tool_call> XML tags and thinking blocks from model output.
    Returns (cleaned_content, list_of_tool_calls, reasoning_content).
    """
    reasoning_content = None
    # Extract <think>...</think>
    think_match = re.search(r"<think>\s*(.*?)\s*(?:</think>|$)", text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()

    tool_calls = []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = list(pattern.finditer(text))

    if matches:
        pre_text = text[:matches[0].start()].strip()
        content = pre_text if pre_text else None
        for m in matches:
            body = m.group(1).strip()
            tc = parse_single_tool_body(body, allow_unclosed=allow_unclosed)
            if tc:
                tool_calls.append(tc)
            else:
                logger.warning(f"Failed to parse tool_call body: {body[:100]}")

        # Check if there is also an unclosed trailing tool_call after the last match
        if allow_unclosed:
            last_end = matches[-1].end()
            unclosed = re.search(r"<tool_call>\s*(.*)$", text[last_end:], re.DOTALL)
            if unclosed:
                single = parse_single_tool_body(unclosed.group(1), allow_unclosed=True)
                if single:
                    tool_calls.append(single)

        if not tool_calls and not content:
            content = text.strip()

        return content, (tool_calls if tool_calls else None), reasoning_content

    if allow_unclosed:
        # Check unclosed <tool_call> tag (e.g. max_tokens cutoff or model omitted closing tag)
        unclosed = re.search(r"<tool_call>\s*(.*)$", text, re.DOTALL)
        if unclosed:
            single = parse_single_tool_body(unclosed.group(1), allow_unclosed=True)
            if single:
                pre_text = text[:unclosed.start()].strip()
                content = pre_text if pre_text else None
                return content, [single], reasoning_content

        # Check if entire trimmed text is a JSON object with name & arguments
        trimmed = text.strip()
        single = parse_single_tool_body(trimmed, allow_unclosed=True)
        if single:
            return None, [single], reasoning_content

    cleaned = text.strip() if text.strip() else None
    return cleaned, None, reasoning_content


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_app(
    backend: Optional[BaseAtlasBackend] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting tendou launcher backend...")
        if hasattr(app.state, "backend") and app.state.backend:
            await app.state.backend.start()
        logger.info(f"Server ready at http://{app.state.host}:{app.state.port}/v1")
        yield
        logger.info("Shutting down tendou launcher backend...")
        if hasattr(app.state, "backend") and app.state.backend:
            await app.state.backend.stop()

    app = FastAPI(
        title="tendou launcher OpenAI-Compatible API Server",
        version="1.0.0",
        description="High-performance local inference server for tendou launcher.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach backend
    app.state.backend = backend or MockAtlasBackend()
    app.state.active_requests = 0
    app.state.host = host
    app.state.port = port

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"Incoming HTTP request: {request.method} {request.url.path}")
        response = await call_next(request)
        if response.status_code >= 400:
            logger.warning(f"HTTP {response.status_code} on {request.method} {request.url.path}")
        return response

    # -----------------------------------------------------------------------
    # Health & Models Endpoints
    # -----------------------------------------------------------------------

    @app.get("/health")
    @app.get("/v1/health")
    async def health_check():
        is_healthy = app.state.backend.is_healthy()
        info = app.state.backend.get_model_info()
        return {
            "status": "ok" if is_healthy else "unhealthy",
            "model": info.get("id", DEFAULT_MODEL_ID),
            "backend": info.get("backend", "atlas-engine"),
            "uptime_sec": info.get("uptime_sec", 0),
            "active_requests": app.state.active_requests,
        }

    @app.get("/models")
    @app.get("/v1/models")
    async def list_models():
        info = app.state.backend.get_model_info()
        mid = info.get("id", DEFAULT_MODEL_ID)
        models = [
            {"id": "qwen3.6-35b", "object": "model", "created": int(time.time()), "owned_by": "atlas"},
            {"id": "atlas-engine", "object": "model", "created": int(time.time()), "owned_by": "atlas"},
            {"id": "qwen3.8-flash-next", "object": "model", "created": int(time.time()), "owned_by": "atlas"},
        ]
        if mid not in [m["id"] for m in models]:
            models.insert(0, {"id": mid, "object": "model", "created": int(time.time()), "owned_by": "atlas"})
        return {
            "object": "list",
            "data": models,
        }

    @app.get("/models/{model_id:path}")
    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str):
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "atlas",
        }

    # -----------------------------------------------------------------------
    # Exception Handlers
    # -----------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error" if exc.status_code < 500 else "api_error",
                    "param": None,
                    "code": exc.status_code,
                }
            },
        )

    # -----------------------------------------------------------------------
    # Chat Completions (/v1/chat/completions)
    # -----------------------------------------------------------------------

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw_request: Request):
        app.state.active_requests += 1
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model_name = req.model or DEFAULT_MODEL_ID

        # Format prompt
        prompt = format_chatml_prompt(
            req.messages,
            tools=req.tools,
            tool_choice=req.tool_choice,
            response_format=req.response_format,
            enable_thinking=req.enable_thinking,
        )

        n_predict = req.max_tokens or req.max_completion_tokens or 2048
        temp = req.temperature if req.temperature is not None else 0.7
        top_p = req.top_p if req.top_p is not None else 0.9
        min_p = req.min_p

        # Stop sequences
        stops = ["<|im_end|>", "<|endoftext|>"]
        if req.stop:
            if isinstance(req.stop, str):
                stops.append(req.stop)
            elif isinstance(req.stop, list):
                stops.extend(req.stop)

        # Detect Cline Native XML Tool Mode (when req.tools is None and Cline XML signature is present)
        cline_xml_mode = (not req.tools) and is_cline_xml_prompt(prompt, raw_request.headers)

        # Phase 14: Dynamic speculative horizon selection (d* in [0, d_max])
        d_star = req.mtp_draft_n
        if d_star is None and estimate_prompt_pld_suitability is not None:
            raw_tools = [t.model_dump() if hasattr(t, "model_dump") else (t.dict() if hasattr(t, "dict") else t) for t in req.tools] if req.tools else None
            _, d_star, pld_rat = estimate_prompt_pld_suitability(prompt, category=req.category, tools=raw_tools)
            logger.debug(f"Prompt PLD suitability for {rid}: d*={d_star} ({pld_rat})")

        # 1. Non-Streaming Response
        if not req.stream:
            try:
                full_text = ""
                finish_reason = "stop"
                prompt_tokens = 0
                completion_tokens = 0
                cached_tokens = 0
                cline_tool_closed = False

                async for event in app.state.backend.generate(
                    prompt=prompt,
                    n_predict=n_predict,
                    temp=temp,
                    top_p=top_p,
                    min_p=min_p,
                    stop=stops,
                    req_id=rid,
                    mtp_draft_n=d_star,
                ):
                    if await raw_request.is_disconnected():
                        logger.info(f"Client disconnected; cancelling {rid}")
                        await app.state.backend.cancel(rid)
                        raise HTTPException(status_code=499, detail="Client closed connection")

                    if event.get("event") == "error":
                        raise HTTPException(
                            status_code=500,
                            detail=event.get("message", "Inference error from tendou launcher"),
                        )
                    elif event.get("event") == "token":
                        if cline_tool_closed:
                            continue
                        full_text += event.get("token", "")
                        completion_tokens += int(event.get("n_tokens", 1))
                        if cline_xml_mode:
                            truncated_text, has_closed_tool = truncate_after_first_cline_xml_tool(full_text)
                            if has_closed_tool:
                                full_text = truncated_text
                                finish_reason = "stop"
                                cline_tool_closed = True
                                await app.state.backend.cancel(rid)
                    elif event.get("event") == "done":
                        finish_reason = "stop" if cline_tool_closed else event.get("finish_reason", "stop")
                        if "prompt_tokens" in event and event["prompt_tokens"] is not None:
                            prompt_tokens = event["prompt_tokens"]
                        elif not prompt_tokens:
                            prompt_tokens = len(prompt.split())

                        if "completion_tokens" in event and event["completion_tokens"] is not None:
                            completion_tokens = event["completion_tokens"]
                        elif not completion_tokens:
                            completion_tokens = len(full_text.split())

                        cached_tokens = event.get("cached_prompt_tokens", 0)
                        # Log MTP speculative metrics when available
                        spec_drafted = event.get("spec_draft_tokens", 0)
                        if spec_drafted:
                            acc = event.get("spec_acceptance_rate", 0.0)
                            eff_tps = event.get("spec_effective_tps", 0.0)
                            import sys
                            print(
                                f"[MTP] draft={spec_drafted} accepted={event.get('spec_accepted_tokens',0)}"
                                f" acc={acc:.2%} eff_tps={eff_tps:.1f}",
                                file=sys.stderr, flush=True
                            )

                if cline_xml_mode:
                    # In Cline XML mode, bridge any accidental <tool_call> JSON to Cline XML and enforce single-tool truncation
                    full_text = convert_qwen_tool_call_to_cline_xml(full_text)
                    full_text, _ = truncate_after_first_cline_xml_tool(full_text)
                    # Strip empty <think></think> while preserving Cline's <thinking>...</thinking>
                    cleaned_content = re.sub(r"<think>\s*</think>\s*", "", full_text).strip()
                    if "<think>" in cleaned_content and "<thinking>" not in cleaned_content:
                        cleaned_content = cleaned_content.replace("<think>", "<thinking>").replace("</think>", "</thinking>")
                    tool_calls = None
                    reasoning_content = None
                else:
                    # Parse tool calls and reasoning content
                    cleaned_content, tool_calls, reasoning_content = parse_tool_calls_from_text(full_text)
                    if tool_calls:
                        finish_reason = "tool_calls"

                # Check JSON mode format if requested
                rf_type = None
                if isinstance(req.response_format, dict):
                    rf_type = req.response_format.get("type")
                elif isinstance(req.response_format, ResponseFormat):
                    rf_type = req.response_format.type

                if rf_type in ("json_object", "json_schema") and cleaned_content:
                    # Strip markdown codeblocks if model wrapped in ```json
                    cleaned = cleaned_content.strip()
                    if cleaned.startswith("```json") and cleaned.endswith("```"):
                        cleaned = cleaned[7:-3].strip()
                    elif cleaned.startswith("```") and cleaned.endswith("```"):
                        cleaned = cleaned[3:-3].strip()
                    else:
                        json_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", cleaned, re.DOTALL)
                        if json_match:
                            cleaned = json_match.group(1).strip()
                    try:
                        json.loads(cleaned)
                        cleaned_content = cleaned
                    except Exception:
                        pass

                msg_obj = {
                    "role": "assistant",
                    "content": cleaned_content,
                    "tool_calls": tool_calls,
                }
                if reasoning_content:
                    msg_obj["reasoning_content"] = reasoning_content

                if not prompt_tokens:
                    prompt_tokens = len(prompt.split())

                return {
                    "id": rid,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": msg_obj,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "cache_read_input_tokens": cached_tokens,
                        "cache_creation_input_tokens": max(0, prompt_tokens - cached_tokens) if cached_tokens > 0 else 0,
                        "prompt_tokens_details": {
                            "cached_tokens": cached_tokens,
                        },
                    },
                }
            finally:
                app.state.active_requests -= 1

        # 2. Streaming Response (SSE)
        async def event_generator():
            try:
                accumulated_text = ""
                emitted_tool_call_count = 0
                emitted_role = False
                last_finish_reason = "stop"
                prompt_toks = 0
                compl_toks = 0
                cached_toks = 0
                tag_buffer = ""
                streamed_any_content = False
                first_token_time = None
                gen_start_time = time.perf_counter()

                # Incremental Tool-Call Streaming State (Phase 13)
                in_tool_call_stream = False
                in_tool_header = False
                current_tool_index = 0
                current_tool_id = None
                current_tool_name = None
                tool_header_buffer = ""
                tool_arg_buffer = ""
                cline_stream_text = ""
                cline_emitted_len = 0
                arg_tokenizer = IncrementalJsonArgumentTokenizer()

                def process_arg_stream_piece(raw_str: str):
                    return arg_tokenizer.process(raw_str)

                def make_chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
                    return {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }

                # Emit initial assistant role chunk immediately so client receives first event with zero delay
                initial_chunk = make_chunk({"role": "assistant"})
                yield f"data: {json.dumps(initial_chunk)}\n\n"
                emitted_role = True

                async for event in app.state.backend.generate(
                    prompt=prompt,
                    n_predict=n_predict,
                    temp=temp,
                    top_p=top_p,
                    min_p=min_p,
                    stop=stops,
                    req_id=rid,
                    mtp_draft_n=d_star,
                ):
                    if event.get("event") == "ping":
                        if await raw_request.is_disconnected():
                            logger.info(f"Client disconnected; cancelling request {rid}")
                            await app.state.backend.cancel(rid)
                            return
                        yield ": keepalive\n\n"
                        continue

                    if event.get("event") == "error":
                        err_chunk = {
                            "error": {
                                "message": event.get("message", "Inference error"),
                                "type": "server_error",
                                "code": 500,
                            }
                        }
                        yield f"data: {json.dumps(err_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    if event.get("event") == "token":
                        if getattr(raw_request.state, "cline_tool_closed", False):
                            continue
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        piece = event.get("token", "")
                        compl_toks += int(event.get("n_tokens", 1))
                        accumulated_text += piece

                        if req.tools:
                            # 1. Active tool-call argument streaming
                            if in_tool_call_stream:
                                close_idx = piece.find("</tool_call>")
                                if close_idx != -1:
                                    pre_close = piece[:close_idx]
                                    post_close = piece[close_idx + len("</tool_call>"):]
                                    arg_chunk, _ = process_arg_stream_piece(pre_close)
                                    clean_tail = arg_chunk.rstrip()
                                    if clean_tail.endswith("}") and not clean_tail.startswith("{"):
                                        clean_tail = clean_tail[:-1].rstrip()
                                    if clean_tail:
                                        tc_delta = {
                                            "tool_calls": [{
                                                "index": current_tool_index,
                                                "function": {"arguments": clean_tail},
                                            }]
                                        }
                                        yield f"data: {json.dumps(make_chunk(tc_delta))}\n\n"
                                    in_tool_call_stream = False
                                    emitted_tool_call_count += 1
                                    current_tool_index += 1
                                    current_tool_id = None
                                    current_tool_name = None
                                    arg_tokenizer = IncrementalJsonArgumentTokenizer()
                                    tag_buffer = post_close
                                    continue

                                arg_chunk, trailer = process_arg_stream_piece(piece)
                                if arg_chunk:
                                    tc_delta = {
                                        "tool_calls": [{
                                            "index": current_tool_index,
                                            "function": {"arguments": arg_chunk},
                                        }]
                                    }
                                    yield f"data: {json.dumps(make_chunk(tc_delta))}\n\n"

                                if trailer is not None:
                                    in_tool_call_stream = False
                                    emitted_tool_call_count += 1
                                    current_tool_index += 1
                                    current_tool_id = None
                                    current_tool_name = None
                                    arg_tokenizer = IncrementalJsonArgumentTokenizer()
                                    close_idx = trailer.find("</tool_call>")
                                    if close_idx != -1:
                                        tag_buffer = trailer[close_idx + len("</tool_call>"):]
                                    else:
                                        tag_buffer = trailer
                                    continue
                                continue

                            # 2. Header parsing state: inside <tool_call>, awaiting name & arguments
                            if in_tool_header:
                                tool_header_buffer += piece
                                close_idx = tool_header_buffer.find("</tool_call>")
                                if close_idx != -1:
                                    # Closed before standard header recognized; fallback to single body parser
                                    raw_body = tool_header_buffer[:close_idx]
                                    in_tool_header = False
                                    tag_buffer = tool_header_buffer[close_idx + len("</tool_call>"):]
                                    tool_header_buffer = ""
                                    single = parse_single_tool_body(raw_body, allow_unclosed=True)
                                    if single:
                                        tc_delta = {
                                            "tool_calls": [{
                                                "index": current_tool_index,
                                                "id": single["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": single["function"]["name"],
                                                    "arguments": single["function"]["arguments"],
                                                },
                                            }]
                                        }
                                        if not emitted_role:
                                            tc_delta["role"] = "assistant"
                                            emitted_role = True
                                        yield f"data: {json.dumps(make_chunk(tc_delta))}\n\n"
                                        emitted_tool_call_count += 1
                                        current_tool_index += 1
                                    continue

                                m_name = re.search(r'["\']name["\']\s*:\s*["\']([a-zA-Z0-9_\.\-]+)["\']', tool_header_buffer)
                                m_args = re.search(r'["\']arguments["\']\s*:\s*', tool_header_buffer)
                                if m_name and m_args and m_args.start() > m_name.start():
                                    current_tool_name = m_name.group(1)
                                    current_tool_id = f"call_{uuid.uuid4().hex[:8]}"
                                    args_start_pos = m_args.end()

                                    # Emit initial OpenAI tool call descriptor
                                    tc_delta = {
                                        "tool_calls": [{
                                            "index": current_tool_index,
                                            "id": current_tool_id,
                                            "type": "function",
                                            "function": {
                                                "name": current_tool_name,
                                                "arguments": "",
                                            },
                                        }]
                                    }
                                    if not emitted_role:
                                        tc_delta["role"] = "assistant"
                                        emitted_role = True
                                    yield f"data: {json.dumps(make_chunk(tc_delta))}\n\n"

                                    in_tool_header = False
                                    in_tool_call_stream = True
                                    arg_tokenizer = IncrementalJsonArgumentTokenizer()
                                    rem_args = tool_header_buffer[args_start_pos:]
                                    tool_header_buffer = ""

                                    if rem_args:
                                        close_idx = rem_args.find("</tool_call>")
                                        if close_idx != -1:
                                            pre_close = rem_args[:close_idx]
                                            post_close = rem_args[close_idx + len("</tool_call>"):]
                                            arg_chunk, _ = process_arg_stream_piece(pre_close)
                                            clean_tail = arg_chunk.rstrip()
                                            if clean_tail.endswith("}") and not clean_tail.startswith("{"):
                                                clean_tail = clean_tail[:-1].rstrip()
                                            if clean_tail:
                                                tc_delta = {
                                                    "tool_calls": [{
                                                        "index": current_tool_index,
                                                        "function": {"arguments": clean_tail},
                                                    }]
                                                }
                                                yield f"data: {json.dumps(make_chunk(tc_delta))}\n\n"
                                            in_tool_call_stream = False
                                            emitted_tool_call_count += 1
                                            current_tool_index += 1
                                            current_tool_id = None
                                            current_tool_name = None
                                            tag_buffer = post_close
                                            continue

                                        arg_chunk, trailer = process_arg_stream_piece(rem_args)
                                        if arg_chunk:
                                            tc_delta = {
                                                "tool_calls": [{
                                                    "index": current_tool_index,
                                                    "function": {"arguments": arg_chunk},
                                                }]
                                            }
                                            yield f"data: {json.dumps(make_chunk(tc_delta))}\n\n"

                                        if trailer is not None:
                                            in_tool_call_stream = False
                                            emitted_tool_call_count += 1
                                            current_tool_index += 1
                                            current_tool_id = None
                                            current_tool_name = None
                                            close_idx = trailer.find("</tool_call>")
                                            if close_idx != -1:
                                                tag_buffer = trailer[close_idx + len("</tool_call>"):]
                                            else:
                                                tag_buffer = trailer
                                            continue
                                    continue
                                else:
                                    # Still buffering header
                                    continue

                            # 3. IDLE mode: checking for <tool_call> tag start or normal text
                            tag_buffer += piece
                            call_pos = tag_buffer.find("<tool_call>")
                            if call_pos != -1:
                                pre_text = tag_buffer[:call_pos]
                                if pre_text:
                                    delta = {"content": pre_text}
                                    if not emitted_role:
                                        delta["role"] = "assistant"
                                        emitted_role = True
                                    yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                                    streamed_any_content = True
                                in_tool_header = True
                                tool_header_buffer = tag_buffer[call_pos + len("<tool_call>"):]
                                tag_buffer = ""
                                continue
                            elif "<tool_call>".startswith(tag_buffer):
                                continue
                            else:
                                to_send = tag_buffer
                                tag_buffer = ""
                                delta = {"content": to_send}
                                if not emitted_role:
                                    delta["role"] = "assistant"
                                    emitted_role = True
                                yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                                streamed_any_content = True
                        elif cline_xml_mode:
                            if getattr(raw_request.state, "cline_tool_closed", False):
                                continue
                            # Cline Native XML Streaming Mode:
                            # 1) Bridge accidental <tool_call> JSON into Cline XML if emitted
                            # 2) Enforce single-tool-per-turn truncation immediately upon closing XML tag
                            cline_stream_text += piece
                            unemitted = cline_stream_text[cline_emitted_len:]
                            if "<tool_call>" in unemitted:
                                if "</tool_call>" in unemitted:
                                    converted = convert_qwen_tool_call_to_cline_xml(unemitted)
                                    cline_stream_text = cline_stream_text[:cline_emitted_len] + converted
                                else:
                                    # Wait until </tool_call> arrives to convert JSON to Cline XML
                                    continue
                            elif "<tool" in unemitted and "<tool_call>".startswith(unemitted[unemitted.rfind("<tool"):]):
                                continue

                            # Check if a Cline XML tool tag just closed
                            truncated_all, has_closed_tool = truncate_after_first_cline_xml_tool(cline_stream_text)
                            if has_closed_tool:
                                to_emit = truncated_all[cline_emitted_len:] if len(truncated_all) > cline_emitted_len else ""
                                cline_emitted_len = len(truncated_all)
                                tag_buffer = ""
                                if to_emit:
                                    delta = {"content": to_emit}
                                    if not emitted_role:
                                        delta["role"] = "assistant"
                                        emitted_role = True
                                    yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                                    streamed_any_content = True
                                last_finish_reason = "stop"
                                raw_request.state.cline_tool_closed = True
                                await app.state.backend.cancel(rid)
                                continue

                            # Hold back trailing '<' only if it could be the start of '<tool_call>'
                            hold_back = 0
                            tail_check = cline_stream_text[cline_emitted_len:]
                            last_lt = tail_check.rfind("<")
                            if last_lt != -1:
                                suffix = tail_check[last_lt:]
                                if "<tool_call>".startswith(suffix):
                                    hold_back = len(suffix)

                            safe_end = len(cline_stream_text) - hold_back
                            if safe_end > cline_emitted_len:
                                to_send = cline_stream_text[cline_emitted_len:safe_end]
                                cline_emitted_len = safe_end
                                delta = {"content": to_send}
                                if not emitted_role:
                                    delta["role"] = "assistant"
                                    emitted_role = True
                                yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                                streamed_any_content = True
                        else:
                            delta = {"content": piece}
                            if not emitted_role:
                                delta["role"] = "assistant"
                                emitted_role = True
                            chunk = make_chunk(delta)
                            yield f"data: {json.dumps(chunk)}\n\n"
                            streamed_any_content = True


                    elif event.get("event") == "done":
                        last_finish_reason = "stop" if getattr(raw_request.state, "cline_tool_closed", False) else event.get("finish_reason", "stop")
                        if "prompt_tokens" in event and event["prompt_tokens"] is not None:
                            prompt_toks = event["prompt_tokens"]
                        elif not prompt_toks:
                            prompt_toks = len(prompt.split())

                        if not getattr(raw_request.state, "cline_tool_closed", False):
                            if "completion_tokens" in event and event["completion_tokens"] is not None:
                                compl_toks = event["completion_tokens"]
                            elif not compl_toks:
                                compl_toks = len(accumulated_text.split())

                        cached_toks = event.get("cached_prompt_tokens", 0)

                # Flush active streaming tool-call arguments if generator finished before </tool_call>
                if in_tool_call_stream:
                    if tool_arg_buffer:
                        raw_tail = tool_arg_buffer.rstrip()
                        if raw_tail.endswith("}"):
                            raw_tail = raw_tail[:-1].rstrip()
                        if raw_tail:
                            tc_delta = {
                                "tool_calls": [{
                                    "index": current_tool_index,
                                    "function": {"arguments": raw_tail},
                                }]
                            }
                            yield f"data: {json.dumps(make_chunk(tc_delta))}\n\n"
                    emitted_tool_call_count += 1
                    current_tool_index += 1
                    in_tool_call_stream = False
                    tool_arg_buffer = ""

                # Flush any un-emitted tool calls if present
                if req.tools and "<tool_call>" in accumulated_text:
                    _, final_tool_calls, _ = parse_tool_calls_from_text(accumulated_text, allow_unclosed=True)
                    if final_tool_calls and len(final_tool_calls) > emitted_tool_call_count:
                        for idx in range(emitted_tool_call_count, len(final_tool_calls)):
                            tc = final_tool_calls[idx]
                            delta = {
                                "tool_calls": [
                                    {
                                        "index": idx,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"],
                                        },
                                    }
                                ]
                            }
                            if not emitted_role:
                                delta["role"] = "assistant"
                                emitted_role = True
                            chunk = make_chunk(delta)
                            yield f"data: {json.dumps(chunk)}\n\n"
                        emitted_tool_call_count = len(final_tool_calls)

                # Akıllı Sonlandırma: Çift gönderimi önle VE tag_buffer'da kalan '<' kodlarını kurtar
                if emitted_tool_call_count == 0:
                    if cline_xml_mode and not getattr(raw_request.state, "cline_tool_closed", False):
                        if cline_emitted_len < len(cline_stream_text):
                            rem_cline = convert_qwen_tool_call_to_cline_xml(cline_stream_text[cline_emitted_len:])
                            rem_cline, _ = truncate_after_first_cline_xml_tool(rem_cline)
                            if rem_cline:
                                delta = {"content": rem_cline}
                                if not emitted_role:
                                    delta["role"] = "assistant"
                                    emitted_role = True
                                yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                                streamed_any_content = True
                            cline_emitted_len = len(cline_stream_text)
                    if not streamed_any_content and accumulated_text.strip():
                        out_text = convert_qwen_tool_call_to_cline_xml(accumulated_text) if cline_xml_mode else accumulated_text
                        if cline_xml_mode:
                            out_text, _ = truncate_after_first_cline_xml_tool(out_text)
                        delta = {"content": out_text}
                        if not emitted_role:
                            delta["role"] = "assistant"
                            emitted_role = True
                        yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                        streamed_any_content = True
                    elif streamed_any_content and tag_buffer:
                        delta = {"content": tag_buffer}
                        tag_buffer = ""
                        yield f"data: {json.dumps(make_chunk(delta))}\n\n"

                if not prompt_toks:
                    prompt_toks = len(prompt.split())

                final_finish = "tool_calls" if emitted_tool_call_count > 0 else last_finish_reason
                final_chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": final_finish,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": compl_toks,
                        "total_tokens": prompt_toks + compl_toks,
                        "cache_read_input_tokens": cached_toks,
                        "cache_creation_input_tokens": max(0, prompt_toks - cached_toks) if cached_toks > 0 else 0,
                        "prompt_tokens_details": {
                            "cached_tokens": cached_toks,
                        },
                    },
                }

                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                now = time.perf_counter()
                total_time = now - gen_start_time
                ttft_ms = (first_token_time - gen_start_time) * 1000.0 if first_token_time else 0.0
                decode_time = (now - first_token_time) if first_token_time else total_time
                effective_toks = max(1, compl_toks - 1) if first_token_time and compl_toks > 1 else compl_toks
                tps = (effective_toks / decode_time) if decode_time > 0.001 else 0.0
                backend_mtp = getattr(app.state.backend, "mtp", "off")
                logger.info(
                    f"[HTTP-STREAM] id={rid} prompt_tokens={prompt_toks} compl_tokens={compl_toks} "
                    f"ttft={ttft_ms:.1f}ms decode_time={decode_time:.2f}s -> {tps:.2f} TPS (mtp={backend_mtp})"
                )

            except asyncio.CancelledError:
                logger.info(f"Streaming task cancelled for request {rid}")
                await app.state.backend.cancel(rid)
            finally:
                app.state.active_requests -= 1

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


    # -----------------------------------------------------------------------
    # Text Completions (/v1/completions)
    # -----------------------------------------------------------------------

    @app.post("/completions")
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw_request: Request):
        app.state.active_requests += 1
        rid = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model_name = req.model or DEFAULT_MODEL_ID

        prompt_str = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
        n_predict = req.max_tokens or 256
        temp = req.temperature if req.temperature is not None else 0.7
        top_p = req.top_p if req.top_p is not None else 0.9
        min_p = req.min_p

        stops = []
        if req.stop:
            if isinstance(req.stop, str):
                stops.append(req.stop)
            elif isinstance(req.stop, list):
                stops.extend(req.stop)

        # Phase 14: Dynamic speculative horizon selection (d* in [0, d_max])
        d_star = req.mtp_draft_n
        if d_star is None and estimate_prompt_pld_suitability is not None:
            _, d_star, pld_rat = estimate_prompt_pld_suitability(prompt_str, category=req.category, tools=None)
            logger.debug(f"Prompt PLD suitability for {rid}: d*={d_star} ({pld_rat})")

        if not req.stream:
            try:
                full_text = ""
                finish_reason = "stop"
                prompt_tokens = 0
                completion_tokens = 0

                async for event in app.state.backend.generate(
                    prompt=prompt_str,
                    n_predict=n_predict,
                    temp=temp,
                    top_p=top_p,
                    min_p=min_p,
                    stop=stops,
                    req_id=rid,
                    mtp_draft_n=d_star,
                ):
                    if await raw_request.is_disconnected():
                        logger.info(f"Client disconnected; cancelling {rid}")
                        await app.state.backend.cancel(rid)
                        raise HTTPException(status_code=499, detail="Client closed connection")

                    if event.get("event") == "error":
                        raise HTTPException(
                            status_code=500,
                            detail=event.get("message", "Inference error from tendou launcher"),
                        )
                    elif event.get("event") == "token":
                        full_text += event.get("token", "")
                        completion_tokens += int(event.get("n_tokens", 1))
                    elif event.get("event") == "done":
                        finish_reason = event.get("finish_reason", "stop")
                        if "prompt_tokens" in event and event["prompt_tokens"] is not None:
                            prompt_tokens = event["prompt_tokens"]
                        elif not prompt_tokens:
                            prompt_tokens = len(prompt_str.split())

                        if "completion_tokens" in event and event["completion_tokens"] is not None:
                            completion_tokens = event["completion_tokens"]
                        elif not completion_tokens:
                            completion_tokens = len(full_text.split())

                        cached_tokens = event.get("cached_prompt_tokens", 0)

                return {
                    "id": rid,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": full_text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": cached_tokens,
                        },
                    },
                }
            finally:
                app.state.active_requests -= 1

        async def stream_generator():
            try:
                accumulated = ""
                last_reason = "stop"
                p_toks = 0
                c_toks = 0

                async for event in app.state.backend.generate(
                    prompt=prompt_str,
                    n_predict=n_predict,
                    temp=temp,
                    top_p=top_p,
                    min_p=min_p,
                    stop=stops,
                    req_id=rid,
                    mtp_draft_n=d_star,
                ):
                    if await raw_request.is_disconnected():
                        logger.info(f"Completions client disconnected; cancelling {rid}")
                        await app.state.backend.cancel(rid)
                        return

                    if event.get("event") == "ping":
                        yield ": keepalive\n\n"
                        continue

                    if event.get("event") == "error":
                        err_chunk = {
                            "error": {
                                "message": event.get("message", "Inference error"),
                                "type": "server_error",
                                "code": 500,
                            }
                        }
                        yield f"data: {json.dumps(err_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    if event.get("event") == "token":
                        chunk_text = event.get("token", "")
                        accumulated += chunk_text
                        c_toks += int(event.get("n_tokens", 1))
                        chunk = {
                            "id": rid,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": chunk_text,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif event.get("event") == "done":
                        last_reason = event.get("finish_reason", "stop")
                        if "prompt_tokens" in event and event["prompt_tokens"] is not None:
                            p_toks = event["prompt_tokens"]
                        elif not p_toks:
                            p_toks = len(prompt_str.split())

                        if "completion_tokens" in event and event["completion_tokens"] is not None:
                            c_toks = event["completion_tokens"]
                        elif not c_toks:
                            c_toks = len(accumulated.split())

                final_chunk = {
                    "id": rid,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": last_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": p_toks,
                        "completion_tokens": c_toks,
                        "total_tokens": p_toks + c_toks,
                    },
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                await app.state.backend.cancel(rid)
            finally:
                app.state.active_requests -= 1

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -----------------------------------------------------------------------
    # OpenAI Responses API Adapter (/v1/responses & /responses)
    # -----------------------------------------------------------------------

    @app.post("/responses")
    @app.post("/v1/responses")
    async def responses_endpoint(raw_request: Request):
        try:
            body = await raw_request.json()
        except Exception:
            body = {}
        model_name = body.get("model", DEFAULT_MODEL_ID)
        raw_input = body.get("input", "")
        instructions = body.get("instructions", "")
        messages = []
        if instructions:
            messages.append(ChatMessage(role="system", content=instructions))
        if isinstance(raw_input, str) and raw_input:
            messages.append(ChatMessage(role="user", content=raw_input))
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    if isinstance(content, list):
                        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                        content = "".join(text_parts) if text_parts else str(content)
                    messages.append(ChatMessage(role=role, content=content))
                elif isinstance(item, str):
                    messages.append(ChatMessage(role="user", content=item))
        if not messages:
            raw_msgs = body.get("messages", [])
            for m in raw_msgs:
                if isinstance(m, dict):
                    messages.append(ChatMessage(**m))
        if not messages:
            messages.append(ChatMessage(role="user", content="Hello"))

        chat_req = ChatCompletionRequest(
            model=model_name,
            messages=messages,
            stream=body.get("stream", False),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            max_tokens=body.get("max_output_tokens") or body.get("max_tokens") or 2048,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
        )
        return await chat_completions(chat_req, raw_request)

    # -----------------------------------------------------------------------
    # Anthropic Messages API Adapter (/v1/messages & /messages) for Claude Code
    # -----------------------------------------------------------------------

    def convert_anthropic_payload(body: Dict[str, Any]) -> tuple[List[ChatMessage], Optional[List[ToolDefinition]]]:
        """Translates Anthropic Messages API payload to ChatML messages and tools."""
        system_val = body.get("system", "")
        sys_prompt = ""
        if isinstance(system_val, str):
            sys_prompt = system_val
        elif isinstance(system_val, list):
            parts = [p.get("text", "") for p in system_val if isinstance(p, dict) and p.get("type") == "text"]
            sys_prompt = "\n".join(parts)

        raw_msgs = body.get("messages", [])
        chatml_messages: List[ChatMessage] = []
        if sys_prompt:
            chatml_messages.append(ChatMessage(role="system", content=sys_prompt))

        for m in raw_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                chatml_messages.append(ChatMessage(role=role, content=content))
            elif isinstance(content, list):
                text_acc: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                tool_results: List[ChatMessage] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_acc.append(block.get("text", ""))
                    elif btype == "tool_use":
                        inp = block.get("input", {})
                        inp_str = json.dumps(inp) if isinstance(inp, dict) else str(inp)
                        tool_calls.append({
                            "id": block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": inp_str,
                            },
                        })
                    elif btype == "tool_result":
                        t_res_content = block.get("content", "")
                        if isinstance(t_res_content, list):
                            sub_texts = [sp.get("text", "") for sp in t_res_content if isinstance(sp, dict) and sp.get("type") == "text"]
                            t_res_content = "\n".join(sub_texts)
                        tool_results.append(ChatMessage(
                            role="tool",
                            tool_call_id=block.get("tool_use_id", ""),
                            content=str(t_res_content),
                        ))
                if text_acc:
                    chatml_messages.append(ChatMessage(
                        role=role,
                        content="\n".join(text_acc),
                        tool_calls=tool_calls if tool_calls else None,
                    ))
                elif tool_calls:
                    chatml_messages.append(ChatMessage(
                        role=role,
                        content=None,
                        tool_calls=tool_calls,
                    ))
                for tr in tool_results:
                    chatml_messages.append(tr)

        raw_tools = body.get("tools", [])
        openai_tools: Optional[List[ToolDefinition]] = None
        if raw_tools:
            openai_tools = []
            for t in raw_tools:
                openai_tools.append(ToolDefinition(
                    type="function",
                    function={
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                ))
        return chatml_messages, openai_tools

    @app.post("/messages")
    @app.post("/v1/messages")
    async def anthropic_messages_endpoint(raw_request: Request):
        """Native Anthropic Messages API endpoint (/v1/messages) for Claude Code CLI and Anthropic SDK."""
        app.state.active_requests += 1
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        try:
            body = await raw_request.json()
        except Exception:
            body = {}

        model_name = body.get("model", DEFAULT_MODEL_ID)
        is_stream = bool(body.get("stream", False))
        max_tokens = body.get("max_tokens") or 4096
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)

        messages, tools = convert_anthropic_payload(body)
        if not messages:
            messages = [ChatMessage(role="user", content="Hello")]

        prompt = format_chatml_prompt(
            messages,
            tools=tools,
            enable_thinking=False,
        )

        stops = ["<|im_end|>", "<|endoftext|>"]
        extra_stops = body.get("stop_sequences", [])
        if extra_stops:
            if isinstance(extra_stops, list):
                stops.extend(extra_stops)
            elif isinstance(extra_stops, str):
                stops.append(extra_stops)

        # Detect Cline Native XML mode over Anthropic Messages API
        cline_xml_mode = (not tools) and is_cline_xml_prompt(prompt, raw_request.headers)

        # Dynamic speculative horizon selection
        d_star = None
        if estimate_prompt_pld_suitability is not None:
            raw_tools = [t.model_dump() if hasattr(t, "model_dump") else (t.dict() if hasattr(t, "dict") else t) for t in tools] if tools else None
            _, d_star, pld_rat = estimate_prompt_pld_suitability(prompt, category=body.get("category"), tools=raw_tools)
            logger.debug(f"Prompt PLD suitability for Anthropic {msg_id}: d*={d_star} ({pld_rat})")

        # 1. Non-Streaming Response
        if not is_stream:
            try:
                full_text = ""
                finish_reason = "end_turn"
                prompt_tokens = 0
                completion_tokens = 0
                cached_tokens = 0

                async for event in app.state.backend.generate(
                    prompt=prompt,
                    n_predict=max_tokens,
                    temp=temperature,
                    top_p=top_p,
                    stop=stops,
                    req_id=msg_id,
                    mtp_draft_n=d_star,
                ):
                    if event.get("event") == "token":
                        full_text += event.get("token", "")
                        if cline_xml_mode:
                            truncated_text, has_closed_tool = truncate_after_first_cline_xml_tool(full_text)
                            if has_closed_tool:
                                full_text = truncated_text
                                await app.state.backend.cancel(msg_id)
                                break
                    elif event.get("event") == "done":
                        finish_reason = event.get("finish_reason", "end_turn")
                        prompt_tokens = event.get("prompt_tokens", 0)
                        completion_tokens = event.get("completion_tokens", 0)
                        cached_tokens = event.get("cached_prompt_tokens", 0)
                        break

                if cline_xml_mode:
                    full_text = convert_qwen_tool_call_to_cline_xml(full_text)
                    full_text, _ = truncate_after_first_cline_xml_tool(full_text)
                    cleaned_content = re.sub(r"<think>\s*</think>\s*", "", full_text).strip()
                    tool_calls = None
                else:
                    cleaned_content, tool_calls, _ = parse_tool_calls_from_text(full_text, allow_unclosed=False)
                content_blocks = []
                stop_reason = "end_turn"
                if cleaned_content:
                    content_blocks.append({"type": "text", "text": cleaned_content})
                if tool_calls:
                    stop_reason = "tool_use"
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        fn_args = fn.get("arguments", "{}")
                        try:
                            input_obj = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        except Exception:
                            input_obj = {"raw": fn_args}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": fn.get("name", ""),
                            "input": input_obj,
                        })
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": full_text.strip()})

                if not completion_tokens:
                    completion_tokens = len(full_text.split())

                return {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model_name,
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "cache_creation_input_tokens": max(0, prompt_tokens - cached_tokens) if cached_tokens > 0 else 0,
                        "cache_read_input_tokens": cached_tokens,
                    },
                }
            finally:
                app.state.active_requests -= 1

        # 2. Streaming Response (SSE for Claude Code)
        async def anthropic_sse_generator():
            try:
                gen_start_time = time.perf_counter()
                first_token_time = None
                prompt_toks = 0
                compl_toks = 0
                cached_toks = 0

                # Initial Anthropic event: message_start
                msg_start_ev = {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model_name,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(msg_start_ev)}\n\n"

                # Anthropic streaming state
                block_index = 0
                emitted_text_start = False
                accumulated_text = ""
                tag_buffer = ""
                in_tool_header = False
                in_tool_stream = False
                tool_header_buffer = ""
                current_tool_id = None
                current_tool_name = None
                anth_tokenizer = IncrementalJsonArgumentTokenizer()
                current_stop_reason = "end_turn"
                emitted_tool_count = 0

                async for event in app.state.backend.generate(
                    prompt=prompt,
                    n_predict=max_tokens,
                    temp=temperature,
                    top_p=top_p,
                    stop=stops,
                    req_id=msg_id,
                    mtp_draft_n=d_star,
                ):
                    if event.get("event") == "ping":
                        if await raw_request.is_disconnected():
                            logger.info(f"Claude Code client disconnected; cancelling {msg_id}")
                            await app.state.backend.cancel(msg_id)
                            return
                        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
                        continue

                    if event.get("event") == "error":
                        err_ev = {
                            "type": "error",
                            "error": {"type": "api_error", "message": event.get("message", "Inference error")},
                        }
                        yield f"event: error\ndata: {json.dumps(err_ev)}\n\n"
                        return

                    if event.get("event") == "token":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token = event.get("token", "")
                        n_tokens = event.get("n_tokens", 1)
                        compl_toks += n_tokens
                        accumulated_text += token

                        if tools:
                            # 1. Inside active tool argument streaming
                            if in_tool_stream:
                                close_idx = token.find("</tool_call>")
                                if close_idx != -1:
                                    pre_close = token[:close_idx]
                                    post_close = token[close_idx + len("</tool_call>"):]
                                    arg_chunk, _ = anth_tokenizer.process(pre_close)
                                    clean_tail = arg_chunk.rstrip()
                                    if clean_tail.endswith("}") and not (clean_tail.startswith("{") and clean_tail.count("{") == clean_tail.count("}")):
                                        clean_tail = clean_tail[:-1].rstrip()
                                    if clean_tail:
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': clean_tail}})}\n\n"
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    in_tool_stream = False
                                    anth_tokenizer = IncrementalJsonArgumentTokenizer()
                                    emitted_tool_count += 1
                                    tag_buffer = post_close
                                    continue

                                arg_chunk, trailer = anth_tokenizer.process(token)
                                if arg_chunk:
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': arg_chunk}})}\n\n"
                                if trailer is not None:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    in_tool_stream = False
                                    anth_tokenizer = IncrementalJsonArgumentTokenizer()
                                    emitted_tool_count += 1
                                    close_idx = trailer.find("</tool_call>")
                                    if close_idx != -1:
                                        tag_buffer = trailer[close_idx + len("</tool_call>"):]
                                    else:
                                        tag_buffer = trailer
                                    continue
                                continue

                            # 2. Header parsing inside <tool_call>
                            if in_tool_header:
                                tool_header_buffer += token
                                close_idx = tool_header_buffer.find("</tool_call>")
                                if close_idx != -1:
                                    raw_body = tool_header_buffer[:close_idx]
                                    in_tool_header = False
                                    tag_buffer = tool_header_buffer[close_idx + len("</tool_call>"):]
                                    tool_header_buffer = ""
                                    single = parse_single_tool_body(raw_body, allow_unclosed=True)
                                    if single:
                                        fn = single.get("function", {})
                                        fn_name = fn.get("name", "")
                                        fn_args = fn.get("arguments", "{}")
                                        tc_id = single.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                                        if emitted_text_start:
                                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                            emitted_text_start = False
                                            block_index += 1
                                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': fn_name, 'input': {}}})}\n\n"
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': fn_args}})}\n\n"
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                        emitted_tool_count += 1
                                        current_stop_reason = "tool_use"
                                    continue

                                m_name = re.search(r'["\']name["\']\s*:\s*["\']([a-zA-Z0-9_\.\-]+)["\']', tool_header_buffer)
                                m_arg = re.search(r'["\']arguments["\']\s*:\s*(\{?)', tool_header_buffer)
                                if m_name and m_arg:
                                    current_tool_name = m_name.group(1)
                                    current_tool_id = f"toolu_{uuid.uuid4().hex[:12]}"
                                    if emitted_text_start:
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                        emitted_text_start = False
                                        block_index += 1
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': current_tool_id, 'name': current_tool_name, 'input': {}}})}\n\n"
                                    in_tool_header = False
                                    in_tool_stream = True
                                    current_stop_reason = "tool_use"
                                    raw_arg_prefix = tool_header_buffer[m_arg.start(1):]
                                    tool_header_buffer = ""
                                    if raw_arg_prefix:
                                        arg_chunk, trailer = anth_tokenizer.process(raw_arg_prefix)
                                        if arg_chunk:
                                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': arg_chunk}})}\n\n"
                                        if trailer is not None:
                                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                            in_tool_stream = False
                                            emitted_tool_count += 1
                                            anth_tokenizer = IncrementalJsonArgumentTokenizer()
                                    continue
                                continue

                            # 3. Plain text / check for <tool_call> tag
                            incoming = tag_buffer + token
                            tag_buffer = ""
                            tc_idx = incoming.find("<tool_call>")
                            if tc_idx != -1:
                                pre_tc = incoming[:tc_idx]
                                post_tc = incoming[tc_idx + len("<tool_call>"):]
                                if pre_tc:
                                    if not emitted_text_start:
                                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        emitted_text_start = True
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': pre_tc}})}\n\n"
                                if emitted_text_start:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    emitted_text_start = False
                                    block_index += 1
                                in_tool_header = True
                                tool_header_buffer = post_tc
                                continue
                            else:
                                if not emitted_text_start:
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                    emitted_text_start = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': incoming}})}\n\n"
                                continue

                        # If no tools defined, pure text / Cline XML streaming
                        if cline_xml_mode:
                            truncated_all, has_closed_tool = truncate_after_first_cline_xml_tool(accumulated_text)
                            if has_closed_tool:
                                prev_len = len(accumulated_text) - len(token)
                                to_emit = truncated_all[prev_len:] if len(truncated_all) > prev_len else ""
                                if to_emit:
                                    if not emitted_text_start:
                                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        emitted_text_start = True
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': to_emit}})}\n\n"
                                await app.state.backend.cancel(msg_id)
                                break

                        if not emitted_text_start:
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            emitted_text_start = True
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"

                    elif event.get("event") == "done":
                        prompt_toks = event.get("prompt_tokens", 0)
                        compl_toks = event.get("completion_tokens", compl_toks)
                        cached_toks = event.get("cached_prompt_tokens", 0)
                        break

                # Cleanup remaining open blocks
                if in_tool_stream:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    emitted_tool_count += 1
                elif emitted_text_start:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                elif emitted_tool_count == 0:
                    _, tool_calls, _ = parse_tool_calls_from_text(accumulated_text, allow_unclosed=True)
                    if tool_calls:
                        current_stop_reason = "tool_use"
                        for tc in tool_calls:
                            block_index += 1
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "")
                            fn_args = fn.get("arguments", "{}")
                            tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': fn_name, 'input': {}}})}\n\n"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': fn_args}})}\n\n"
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    else:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                # message_delta with stop reason and usage
                delta_usage = {"output_tokens": compl_toks}
                if cached_toks > 0:
                    delta_usage["cache_read_input_tokens"] = cached_toks
                    delta_usage["cache_creation_input_tokens"] = max(0, prompt_toks - cached_toks)
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': current_stop_reason, 'stop_sequence': None}, 'usage': delta_usage})}\n\n"
                # message_stop
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                now = time.perf_counter()
                total_time = now - gen_start_time
                ttft_ms = (first_token_time - gen_start_time) * 1000.0 if first_token_time else 0.0
                decode_time = (now - first_token_time) if first_token_time else total_time
                effective_toks = max(1, compl_toks - 1) if first_token_time and compl_toks > 1 else compl_toks
                tps = (effective_toks / decode_time) if decode_time > 0.001 else 0.0
                logger.info(
                    f"[ANTHROPIC-STREAM] id={msg_id} prompt_tokens={prompt_toks} compl_tokens={compl_toks} "
                    f"ttft={ttft_ms:.1f}ms decode_time={decode_time:.2f}s -> {tps:.2f} TPS (stop_reason={current_stop_reason})"
                )
            except asyncio.CancelledError:
                logger.info(f"Claude Code streaming cancelled for {msg_id}")
                await app.state.backend.cancel(msg_id)
            finally:
                app.state.active_requests -= 1

        return StreamingResponse(
            anthropic_sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


    return app


# ---------------------------------------------------------------------------
# Port Pre-Flight & Network Management
# ---------------------------------------------------------------------------

def is_port_in_use(host: str, port: int) -> bool:
    """Check whether a given host/port combination is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def get_process_using_port(port: int) -> Optional[Dict[str, Any]]:
    """Return process details occupying the specified port if psutil is available."""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                if conn.pid:
                    try:
                        p = psutil.Process(conn.pid)
                        return {
                            "pid": conn.pid,
                            "name": p.name(),
                            "cmdline": " ".join(p.cmdline()[:4]) if p.cmdline() else "",
                        }
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return {"pid": conn.pid, "name": "unknown", "cmdline": ""}
    except Exception:
        pass
    return None


def kill_process_on_port(port: int) -> bool:
    """Force terminate any process occupying the given port."""
    info = get_process_using_port(port)
    if not info:
        return False
    try:
        import psutil
        p = psutil.Process(info["pid"])
        p.kill()
        p.wait(timeout=3)
        logger.info(f"Terminated process {info['name']} (PID: {info['pid']}) occupying port {port}.")
        return True
    except Exception as e:
        logger.warning(f"Failed to kill process on port {port}: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI Entry Point & Runner
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tendou launcher OpenAI-Compatible Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--force", action="store_true", help="Force terminate any existing process occupying the target port")
    parser.add_argument("--exe-path", default=DEFAULT_EXE_PATH, help="Path to llama-atlas-engine.exe")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Path to model GGUF (or set TENDOU_MODEL_PATH)")
    parser.add_argument("--threads", "-t", type=int, default=8, help="Number of CPU threads (default: 8)")
    parser.add_argument("--prompt-cache-mb", type=int, default=512, help="Prompt cache memory budget in MB (default: 512)")
    parser.add_argument("--boost", action="store_true", default=True, help="Enable Atlas boost mode (default: True)")
    parser.add_argument("--boost-level", type=int, default=2, choices=[1, 2, 3],
                        help="Atlas boost level: 1 (Balanced K=2/K=1), 2 (Turbo K=2 core only), 3 (Ultra K=1 max TPS)")
    parser.add_argument("--gpu", dest="gpu", action="store_true", default=True, help="Enable GPU-first acceleration with full VRAM capacity (default: True)")
    parser.add_argument("--cpu-only", dest="gpu", action="store_false", help="Run CPU-only fallback engine (for systems without GPUs)")
    parser.add_argument("--gpu-layers", type=int, default=2, help="Number of full MoE layers in GPU VRAM (default: 2)")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers for attention/shared experts (default: 99)")
    parser.add_argument("--vram-cache", type=int, default=3200, help="VRAM dynamic cache budget in MB (default: 3200)")
    parser.add_argument("--ctx-size", "-c", type=int, default=8192,
                        help="Context window size in tokens (default: 8192). "
                             "n_ctx=4096 fits in VRAM only; n_ctx=8192 spills ~400 MiB KV cache to system RAM. "
                             "Maximum supported by engine in GPU-first mode: 8192.")
    parser.add_argument("--mock", action="store_true", help="Run with mock backend for testing without model weights")
    parser.add_argument("--odmoe-lead", type=int, default=2, help="OD-MoE advance prefetch lead in layers")
    parser.add_argument("--odmoe-quick-evict", action="store_true", default=True, help="OD-MoE rapid post-layer eviction")
    parser.add_argument("--spice-conf-high", type=float, default=0.65, help="SPICE high confidence threshold for VRAM prefetch")
    parser.add_argument("--spice-conf-mid", type=float, default=0.30, help="SPICE mid confidence threshold for RAM staging")
    parser.add_argument("--tutti-async-io", action=argparse.BooleanOptionalAction, default=False,
                        help="Asynchronous expert page warming (default: disabled)")
    parser.add_argument("--readback-interval", type=int, default=8, help="Router readback interval in decode tokens")
    parser.add_argument("--prefetch-candidates", type=int, default=10, help="Maximum predicted experts per layer")
    parser.add_argument("--tutti-queue-depth", type=int, default=64, help="Maximum queued expert chunks")
    parser.add_argument("--aggressive-prefetch", action="store_true",
                        help="Use the experimental lead=4, 16-candidate, async page-warming profile")
    parser.add_argument("--p2-gpu-binding", action="store_true", default=False,
                        help="Enable P2 dynamic GPU expert buffer binding")
    parser.add_argument("--p2-vram-budget-mib", type=int, default=0,
                        help="P2 VRAM cache budget in MiB (0 for default)")
    parser.add_argument("--p3-async-transfer", action="store_true", default=False,
                        help="Enable P3 async H2D transfer pipeline")
    parser.add_argument("--p3-staging-slots", type=int, default=4,
                        help="P3 pinned staging buffer slots (default: 4)")
    parser.add_argument("--p3-queue-depth", type=int, default=8,
                        help="P3 H2D queue depth (default: 8)")
    parser.add_argument("--cuda-sched", default="yield", choices=["yield", "blocking", "spin", "auto"],
                        help="CUDA primary context scheduling mode (default: yield)")
    parser.add_argument("--mtp", choices=["off", "ram", "vram", "atlas", "auto", "hybrid", "pld"], default="pld",
                        help="Enable MTP speculative decoding mode (off, ram, vram, atlas, auto, hybrid, pld) (default: pld)")
    parser.add_argument("--mtp-mode", choices=["off", "ram", "vram", "atlas", "auto", "hybrid", "pld"], default=None,
                        help="MTP speculative decoding mode alias (default: None, falls back to --mtp)")
    parser.add_argument("--moe-cache-mib", type=int, default=2048,
                        help="Split-MoE GPU Cache budget in MiB (default: 2048)")
    parser.add_argument("--min-p", type=float, default=0.0,
                        help="Minimum probability sampling threshold (default: 0.0)")
    parser.add_argument("--mtp-loc", choices=["ram", "vram", "hybrid", "auto"], default=None,
                        help="MTP tensor placement location (ram, vram, hybrid, auto)")
    parser.add_argument("--mtp-draft-n", type=int, default=2,
                        help="MTP speculative draft token count (default: 2)")
    parser.add_argument("--mtp-path", type=str, default=None,
                        help="Optional separate MTP draft model GGUF path")
    parser.add_argument("--cache-type-k", type=str, default=None,
                        help="KV cache K quantization type (e.g. q4_0, q8_0, f16)")
    parser.add_argument("--cache-type-v", type=str, default=None,
                        help="KV cache V quantization type (e.g. q4_0, q8_0, f16)")
    parser.add_argument("--tlep", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable TLEP (Tendou Lookahead Execution Pipeline)")
    parser.add_argument("--tlep-depth", type=int, default=4,
                        help="TLEP maximum lookahead horizon depth (default: 4)")
    parser.add_argument("--tlep-guard", type=float, default=1.5,
                        help="TLEP minimum slack guard margin in ms (default: 1.5)")
    parser.add_argument("--tlep-bw", type=int, default=128,
                        help="TLEP predictive prefetch bandwidth budget in MB (default: 128)")
    parser.add_argument("--cascading-migration", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Phase 14 Dynamic Cascading 3-Tier MoE Working Set Migration (default: True)")
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.aggressive_prefetch:
        args.odmoe_lead = 4
        args.spice_conf_high = 0.50
        args.spice_conf_mid = 0.15
        args.tutti_async_io = True
        args.readback_interval = 4
        args.prefetch_candidates = 16
        args.tutti_queue_depth = 128

    # Pre-flight check: ensure target port is free before allocating GPU VRAM / launching engine
    if is_port_in_use(args.host, args.port):
        occupant = get_process_using_port(args.port)
        if args.force:
            logger.warning(f"Port {args.port} is in use. --force supplied; terminating occupying process...")
            kill_process_on_port(args.port)
            time.sleep(0.5)
            if is_port_in_use(args.host, args.port):
                logger.error(f"Failed to free port {args.port} even after terminating occupying process. Aborting.")
                sys.exit(1)
        else:
            proc_desc = f" (PID: {occupant['pid']}, Name: {occupant['name']})" if occupant else ""
            logger.error(
                f"Port {args.port} is already in use by another process{proc_desc}! "
                f"Aborting startup before initializing backend/allocating VRAM. "
                f"To terminate the occupying process automatically, re-run with --force. "
                f"Or specify an available port with --port <PORT>."
            )
            sys.exit(1)

    if args.mock or not os.path.exists(args.exe_path) or not os.path.exists(args.model_path):
        if not args.mock:
            logger.warning("Atlas binary or model not found; falling back to high-fidelity mock backend.")
        backend = MockAtlasBackend(model_id=DEFAULT_MODEL_ID)
    else:
        chosen_exe = args.exe_path
        if not args.gpu:
            baseline_exe = os.path.join(
                os.path.dirname(args.exe_path), "llama-atlas-engine-baseline-10tps.exe"
            )
            if os.path.exists(baseline_exe):
                chosen_exe = baseline_exe
                logger.info(f"Using frozen baseline CPU fallback binary: {chosen_exe}")

        backend = AtlasSubprocessBackend(
            exe_path=chosen_exe,
            model_path=args.model_path,
            threads=args.threads,
            boost=args.boost,
            boost_level=args.boost_level,
            gpu=args.gpu,
            gpu_layers=args.gpu_layers,
            ngl=args.ngl,
            vram_cache=args.vram_cache,
            ctx_size=args.ctx_size,
            odmoe_lead=args.odmoe_lead,
            odmoe_quick_evict=args.odmoe_quick_evict,
            spice_conf_high=args.spice_conf_high,
            spice_conf_mid=args.spice_conf_mid,
            tutti_async_io=args.tutti_async_io,
            readback_interval=args.readback_interval,
            prefetch_candidates=args.prefetch_candidates,
            tutti_queue_depth=args.tutti_queue_depth,
            p2_gpu_binding=args.p2_gpu_binding,
            p2_vram_budget_mib=args.p2_vram_budget_mib,
            p3_async_transfer=args.p3_async_transfer,
            p3_staging_slots=args.p3_staging_slots,
            p3_queue_depth=args.p3_queue_depth,
            prompt_cache_mb=args.prompt_cache_mb,
            cuda_sched=args.cuda_sched,
            mtp=args.mtp_mode if args.mtp_mode is not None else (args.mtp or "pld"),
            mtp_draft_n=args.mtp_draft_n,
            mtp_path=args.mtp_path,
            mtp_loc=args.mtp_loc,
            moe_cache_mib=args.moe_cache_mib,
            min_p=args.min_p,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
            tlep=args.tlep,
            tlep_max_depth=args.tlep_depth,
            tlep_min_guard_ms=args.tlep_guard,
            tlep_pf_bw_budget_mb=args.tlep_bw,
            cascading_migration=getattr(args, "cascading_migration", True),
        )

    app = create_app(backend=backend, host=args.host, port=args.port)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    try:
        sock = config.bind_socket()
    except OSError as e:
        logger.error(f"Failed to bind socket on {args.host}:{args.port}: {e}")
        sys.exit(1)

    server = uvicorn.Server(config)
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()