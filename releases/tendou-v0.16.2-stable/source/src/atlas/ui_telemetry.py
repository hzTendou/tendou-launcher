"""
Tendou Studio — Real-Time Live TPS Telemetry Hub & System Metrics Collector.

Provides thread-safe, zero-latency token throughput tracking (500 ms sliding window
live_decode_tps, EMA smoothing, session peak, 30 s @ 4 Hz sparkline ring buffer),
1 Hz cached GPU/RAM/CPU telemetry, MoE 3-tier cache statistics, PrefixCache & Swarm
queue snapshots, and a 500-line streaming log ring buffer.
"""

from __future__ import annotations

import asyncio
from collections import deque
import ctypes
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple


class TelemetryHub:
    """Thread-safe real-time telemetry hub powering Tendou Studio's Live TPS counter."""

    def __init__(self, history_len: int = 120, log_capacity: int = 500):
        self._lock = threading.RLock()

        # Live TPS & Generation Counters
        self._is_generating: bool = False
        self._active_request_id: Optional[str] = None
        self._req_start_perf: float = 0.0
        self._first_token_perf: float = 0.0
        self._last_token_perf: float = 0.0
        self._token_window: Deque[Tuple[float, int]] = deque()
        self._tokens_this_turn: int = 0
        self._tokens_session_total: int = 0
        self._requests_total: int = 0

        self._live_decode_tps: float = 0.0
        self._ema_decode_tps: float = 0.0
        self._peak_decode_tps: float = 0.0
        self._last_prefill_tps: float = 0.0
        self._last_ttft_ms: float = 0.0
        self._last_turn_decode_tps: float = 0.0
        self._last_turn_tokens: int = 0
        self._last_prefix_fork_kind: str = "cold"
        self._prefill_recorded_this_turn: bool = False

        # 30s @ 4 Hz Sparkline Ring Buffer (120 points)
        self._tps_history: Deque[Dict[str, Any]] = deque(maxlen=history_len)
        self._last_history_push: float = 0.0

        # System Metrics (1 Hz cached — never blocks inference hot path)
        self._sys_cache_ts: float = 0.0
        self._gpu_used_mb: float = 0.0
        self._gpu_total_mb: float = 0.0
        self._gpu_peak_mb: float = 0.0
        self._gpu_util_pct: float = 0.0
        self._gpu_temp_c: float = 0.0
        self._gpu_name: str = "Detecting..."
        self._ram_used_gb: float = 0.0
        self._ram_avail_gb: float = 0.0
        self._ram_total_gb: float = 0.0
        self._cpu_util_pct: float = 0.0
        self._sys_refreshing: bool = False

        # MoE Cache & Dynamic-K Engine Telemetry (parsed from engine logs/IPC)
        self._moe_tier0_vram_hits: int = 0
        self._moe_tier1_ram_hits: int = 0
        self._moe_tier2_nvme_misses: int = 0
        self._avg_dynamic_k: float = 0.0
        self._prefix_cache_slots_used: int = 0
        self._prefix_cache_max_slots: int = 16

        # Log Ring Buffer (last 500 lines)
        self._logs: Deque[Dict[str, Any]] = deque(maxlen=log_capacity)
        self._log_seq: int = 0
        self._log_subscribers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []

        # Regex patterns for engine telemetry parsing
        self._re_prefill = re.compile(r"Prefill:\s+(\d+)\s+tokens\s+/\s+([0-9.]+)\s+s\s+=\s+([0-9.]+)\s+TPS", re.I)
        self._re_moe_hits = re.compile(r"vram_hit[s]?[=:]\s*(\d+).*ram_hit[s]?[=:]\s*(\d+).*miss(?:es)?[=:]\s*(\d+)", re.I)
        self._re_dyn_k = re.compile(r"(?:avg_k|dynamic_k|keep_k)[=:]\s*([0-9.]+)", re.I)

    # -----------------------------------------------------------------------
    # Zero-Overhead Inference Hot-Path Hooks
    # -----------------------------------------------------------------------

    def on_request_start(self, request_id: str, prompt_tokens: int = 0) -> None:
        now = time.perf_counter()
        with self._lock:
            self._is_generating = True
            self._active_request_id = request_id
            self._req_start_perf = now
            self._first_token_perf = 0.0
            self._last_token_perf = now
            self._prefill_recorded_this_turn = False
            self._token_window.clear()
            self._tokens_this_turn = 0
            self._live_decode_tps = 0.0
            self._requests_total += 1

    def on_prefill_done(
        self,
        ttft_ms: Optional[float] = None,
        prefill_tps: Optional[float] = None,
        prompt_tokens: int = 0,
        prefix_fork_kind: Optional[str] = None,
    ) -> None:
        now = time.perf_counter()
        with self._lock:
            if ttft_ms is not None and ttft_ms > 0:
                self._last_ttft_ms = round(float(ttft_ms), 2)
                self._prefill_recorded_this_turn = True
            elif self._req_start_perf > 0:
                self._last_ttft_ms = round((now - self._req_start_perf) * 1000.0, 2)
                self._prefill_recorded_this_turn = True
            if prefill_tps is not None and prefill_tps > 0:
                self._last_prefill_tps = round(float(prefill_tps), 2)
            elif prompt_tokens > 0 and self._last_ttft_ms > 0.1:
                self._last_prefill_tps = round(prompt_tokens / (self._last_ttft_ms / 1000.0), 2)
            if prefix_fork_kind:
                self._last_prefix_fork_kind = prefix_fork_kind

    def on_token(self, n_tokens: int = 1) -> None:
        """Called on every emitted token chunk; updates 500ms sliding-window live TPS."""
        n = max(1, int(n_tokens))
        now = time.perf_counter()
        with self._lock:
            if not self._is_generating:
                self._is_generating = True
                self._req_start_perf = now
                self._first_token_perf = 0.0
                self._prefill_recorded_this_turn = False
                self._token_window.clear()
                self._tokens_this_turn = 0

            if self._first_token_perf == 0.0:
                self._first_token_perf = now
                if not self._prefill_recorded_this_turn and self._req_start_perf > 0 and now >= self._req_start_perf:
                    self._last_ttft_ms = round((now - self._req_start_perf) * 1000.0, 2)

            self._tokens_this_turn += n
            self._tokens_session_total += n
            self._last_token_perf = now
            self._token_window.append((now, n))

            # Evict samples older than 500 ms sliding window
            cutoff = now - 0.5
            while len(self._token_window) > 1 and self._token_window[0][0] < cutoff:
                self._token_window.popleft()

            # Compute instantaneous window TPS (or turn TPS during first 500 ms)
            if len(self._token_window) >= 2:
                span = now - self._token_window[0][0]
                if span >= 0.025:
                    win_tokens = sum(cnt for _, cnt in self._token_window) - self._token_window[0][1]
                    inst_tps = max(0.0, win_tokens / span)
                    self._live_decode_tps = round(inst_tps, 2)
                    if self._ema_decode_tps <= 0.01:
                        self._ema_decode_tps = self._live_decode_tps
                    else:
                        self._ema_decode_tps = round(0.28 * self._live_decode_tps + 0.72 * self._ema_decode_tps, 2)
                    if self._live_decode_tps > self._peak_decode_tps:
                        self._peak_decode_tps = self._live_decode_tps
            elif self._tokens_this_turn > 1 and self._first_token_perf > 0:
                span = now - self._first_token_perf
                if span >= 0.02:
                    inst_tps = (self._tokens_this_turn - 1) / span
                    self._live_decode_tps = round(inst_tps, 2)
                    self._ema_decode_tps = self._live_decode_tps
                    if self._live_decode_tps > self._peak_decode_tps:
                        self._peak_decode_tps = self._live_decode_tps

    def on_request_done(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttft_ms: Optional[float] = None,
        decode_tps: Optional[float] = None,
        prefill_tps: Optional[float] = None,
        prefix_fork_kind: Optional[str] = None,
    ) -> None:
        now = time.perf_counter()
        with self._lock:
            if ttft_ms is not None and ttft_ms > 0:
                self._last_ttft_ms = round(float(ttft_ms), 2)
            if prefill_tps is not None and prefill_tps > 0:
                self._last_prefill_tps = round(float(prefill_tps), 2)
            elif prompt_tokens > 0 and self._last_ttft_ms > 0.1:
                self._last_prefill_tps = round(prompt_tokens / (self._last_ttft_ms / 1000.0), 2)

            if decode_tps is not None and decode_tps > 0:
                final_tps = round(float(decode_tps), 2)
            elif self._first_token_perf > 0 and now > self._first_token_perf:
                eff_toks = max(1, (completion_tokens or self._tokens_this_turn) - 1)
                final_tps = round(eff_toks / max(0.005, now - self._first_token_perf), 2)
            else:
                final_tps = self._live_decode_tps

            self._last_turn_decode_tps = final_tps
            self._last_turn_tokens = completion_tokens or self._tokens_this_turn
            if final_tps > self._peak_decode_tps:
                self._peak_decode_tps = final_tps
            if final_tps > 0:
                self._ema_decode_tps = round(0.4 * final_tps + 0.6 * (self._ema_decode_tps or final_tps), 2)
            if prefix_fork_kind:
                self._last_prefix_fork_kind = prefix_fork_kind

            self._is_generating = False
            self._active_request_id = None
            self._live_decode_tps = 0.0
            self._token_window.clear()

    # -----------------------------------------------------------------------
    # Log Ring Buffer & Engine Metric Extraction
    # -----------------------------------------------------------------------

    def push_log(self, line: str, source: str = "engine", level: str = "INFO") -> None:
        text = (line or "").rstrip("\r\n")
        if not text:
            return

        # Opportunistically parse native engine metrics from stderr lines
        if source == "engine":
            m_pf = self._re_prefill.search(text)
            if m_pf:
                try:
                    with self._lock:
                        self._last_prefill_tps = round(float(m_pf.group(3)), 2)
                except ValueError:
                    pass
            m_hits = self._re_moe_hits.search(text)
            if m_hits:
                try:
                    with self._lock:
                        self._moe_tier0_vram_hits = int(m_hits.group(1))
                        self._moe_tier1_ram_hits = int(m_hits.group(2))
                        self._moe_tier2_nvme_misses = int(m_hits.group(3))
                except ValueError:
                    pass
            m_dk = self._re_dyn_k.search(text)
            if m_dk:
                try:
                    with self._lock:
                        self._avg_dynamic_k = round(float(m_dk.group(1)), 2)
                except ValueError:
                    pass

        with self._lock:
            self._log_seq += 1
            entry = {
                "seq": self._log_seq,
                "ts": round(time.time(), 3),
                "source": source,
                "level": level,
                "line": text,
            }
            self._logs.append(entry)
            subs = list(self._log_subscribers)

        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(self._enqueue_nonblock, q, entry)
            except RuntimeError:
                pass

    @staticmethod
    def _enqueue_nonblock(q: asyncio.Queue, item: Dict[str, Any]) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def get_recent_logs(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._logs)
        return items[-max(1, min(limit, 500)):]

    def subscribe_logs(self) -> Tuple[asyncio.AbstractEventLoop, asyncio.Queue]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._log_subscribers.append((loop, q))
        return loop, q

    def unsubscribe_logs(self, sub: Tuple[asyncio.AbstractEventLoop, asyncio.Queue]) -> None:
        with self._lock:
            if sub in self._log_subscribers:
                self._log_subscribers.remove(sub)

    # -----------------------------------------------------------------------
    # 1 Hz Cached System Telemetry (GPU / RAM / CPU)
    # -----------------------------------------------------------------------

    def _poll_system_metrics_sync(self) -> None:
        gpu_used, gpu_total, gpu_util, gpu_temp, gpu_name = 0.0, 0.0, 0.0, 0.0, ""
        # 1. GPU via nvidia-smi (fast, 1 Hz cached)
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.2,
            ).strip()
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                if len(parts) >= 5:
                    gpu_name = parts[0]
                    gpu_used = float(parts[1])
                    gpu_total = float(parts[2])
                    gpu_util = float(parts[3]) if parts[3] not in ("", "N/A", "[N/A]") else 0.0
                    gpu_temp = float(parts[4]) if parts[4] not in ("", "N/A", "[N/A]") else 0.0
        except Exception:
            pass

        # 2. RAM & CPU via psutil or OS-native fallback (/proc/meminfo or GlobalMemoryStatusEx)
        ram_total, ram_avail, ram_used, cpu_pct = 0.0, 0.0, 0.0, 0.0
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram_total = vm.total / (1024.0 ** 3)
            ram_avail = vm.available / (1024.0 ** 3)
            ram_used = (vm.total - vm.available) / (1024.0 ** 3)
            cpu_pct = float(psutil.cpu_percent(interval=None))
        except Exception:
            if os.name == "nt":
                try:
                    class _MEMSTAT(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    st = _MEMSTAT()
                    st.dwLength = ctypes.sizeof(_MEMSTAT)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                        ram_total = st.ullTotalPhys / (1024.0 ** 3)
                        ram_avail = st.ullAvailPhys / (1024.0 ** 3)
                        ram_used = ram_total - ram_avail
                except Exception:
                    pass
            else:
                try:
                    meminfo: Dict[str, int] = {}
                    with open("/proc/meminfo", "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2:
                                meminfo[parts[0].rstrip(":")] = int(parts[1])
                    ram_total = meminfo.get("MemTotal", 0) / (1024.0 ** 2)
                    ram_avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) / (1024.0 ** 2)
                    ram_used = max(0.0, ram_total - ram_avail)
                except Exception:
                    pass

        with self._lock:
            if gpu_name:
                self._gpu_name = gpu_name
            if gpu_total > 0:
                self._gpu_used_mb = round(gpu_used, 1)
                self._gpu_total_mb = round(gpu_total, 1)
                self._gpu_peak_mb = round(max(self._gpu_peak_mb, gpu_used), 1)
                self._gpu_util_pct = round(gpu_util, 1)
                self._gpu_temp_c = round(gpu_temp, 1)
            if ram_total > 0:
                self._ram_total_gb = round(ram_total, 2)
                self._ram_avail_gb = round(ram_avail, 2)
                self._ram_used_gb = round(ram_used, 2)
            self._cpu_util_pct = round(cpu_pct, 1)
            self._sys_cache_ts = time.monotonic()
            self._sys_refreshing = False

    def reset_gpu_peak(self) -> None:
        with self._lock:
            self._gpu_peak_mb = 0.0
            self._sys_cache_ts = 0.0

    def maybe_refresh_system_metrics(self, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._sys_cache_ts) < 1.0:
                return
            if not force and self._sys_refreshing:
                return
            self._sys_refreshing = True
        if force:
            self._poll_system_metrics_sync()
        else:
            t = threading.Thread(target=self._poll_system_metrics_sync, daemon=True)
            t.start()

    # -----------------------------------------------------------------------
    # Unified 4 Hz Snapshot for SSE & REST
    # -----------------------------------------------------------------------

    def snapshot(
        self,
        backend: Any = None,
        engine_status: str = "ready",
        active_http_requests: int = 0,
        record_history: bool = True,
    ) -> Dict[str, Any]:
        self.maybe_refresh_system_metrics(force=(self._sys_cache_ts == 0.0))
        now = time.perf_counter()

        # Pull Swarm & PrefixCache live state from backend
        gate_state: Dict[str, Any] = {
            "queue_depth": 0,
            "active_session_id": None,
            "last_session_id": None,
            "waiters": [],
        }
        prefix_forks = {"root_anchor": 0, "turn_boundary": 0, "exact": 0, "cold": 0}
        if backend is not None:
            gate = getattr(backend, "engine_gate", None)
            if gate is not None and hasattr(gate, "debug_state"):
                try:
                    gate_state = gate.debug_state()
                except Exception:
                    pass
            pf = getattr(backend, "prefix_fork_counts", None)
            if isinstance(pf, dict):
                prefix_forks = dict(pf)

        with self._lock:
            # Decay stale live TPS if no token arrived in > 1.5s while flagged generating
            live_tps = self._live_decode_tps
            if self._is_generating and self._last_token_perf > 0 and (now - self._last_token_perf) > 1.5:
                live_tps = 0.0

            if record_history and (now - self._last_history_push) >= 0.22:
                self._last_history_push = now
                self._tps_history.append({
                    "ts": round(time.time(), 2),
                    "tps": live_tps if self._is_generating else 0.0,
                    "vram_mb": self._gpu_used_mb,
                    "generating": self._is_generating,
                })

            total_moe = self._moe_tier0_vram_hits + self._moe_tier1_ram_hits + self._moe_tier2_nvme_misses
            if total_moe > 0:
                t0_pct = round(100.0 * self._moe_tier0_vram_hits / total_moe, 1)
                t1_pct = round(100.0 * self._moe_tier1_ram_hits / total_moe, 1)
                t2_pct = round(100.0 * self._moe_tier2_nvme_misses / total_moe, 1)
            else:
                t0_pct = t1_pct = t2_pct = 0.0

            prefix_total = sum(prefix_forks.values())
            prefix_hits = prefix_forks.get("exact", 0) + prefix_forks.get("turn_boundary", 0) + prefix_forks.get("root_anchor", 0)
            prefix_hit_pct = round(100.0 * prefix_hits / prefix_total, 1) if prefix_total > 0 else 0.0
            slots_used = min(self._prefix_cache_max_slots, max(self._prefix_cache_slots_used, min(prefix_total, self._prefix_cache_max_slots)))

            return {
                "ts": round(time.time(), 3),
                "engine_status": engine_status,
                "tps": {
                    "live_decode_tps": live_tps,
                    "ema_decode_tps": self._ema_decode_tps,
                    "peak_decode_tps": self._peak_decode_tps,
                    "last_prefill_tps": self._last_prefill_tps,
                    "last_ttft_ms": self._last_ttft_ms,
                    "last_turn_decode_tps": self._last_turn_decode_tps,
                    "last_turn_tokens": self._last_turn_tokens,
                    "is_generating": self._is_generating,
                    "tokens_this_turn": self._tokens_this_turn,
                    "tokens_session_total": self._tokens_session_total,
                    "requests_total": self._requests_total,
                    "active_request_id": self._active_request_id,
                    "last_prefix_fork_kind": self._last_prefix_fork_kind,
                    "tps_history": list(self._tps_history),
                },
                "gpu": {
                    "name": self._gpu_name,
                    "used_mb": self._gpu_used_mb,
                    "total_mb": self._gpu_total_mb,
                    "peak_mb": self._gpu_peak_mb,
                    "util_pct": self._gpu_util_pct,
                    "temp_c": self._gpu_temp_c,
                },
                "system": {
                    "ram_used_gb": self._ram_used_gb,
                    "ram_available_gb": self._ram_avail_gb,
                    "ram_total_gb": self._ram_total_gb,
                    "cpu_util_pct": self._cpu_util_pct,
                },
                "moe_cache": {
                    "tier0_vram_hits": self._moe_tier0_vram_hits,
                    "tier1_ram_hits": self._moe_tier1_ram_hits,
                    "tier2_nvme_misses": self._moe_tier2_nvme_misses,
                    "tier0_vram_pct": t0_pct,
                    "tier1_ram_pct": t1_pct,
                    "tier2_nvme_pct": t2_pct,
                    "avg_dynamic_k": self._avg_dynamic_k,
                },
                "prefix_cache": {
                    "max_slots": self._prefix_cache_max_slots,
                    "slots_used": slots_used,
                    "hit_rate_pct": prefix_hit_pct,
                    "fork_counts": prefix_forks,
                },
                "swarm": {
                    "active_http_requests": active_http_requests,
                    "queue_depth": gate_state.get("queue_depth", 0),
                    "active_session_id": gate_state.get("active_session_id"),
                    "last_session_id": gate_state.get("last_session_id"),
                    "waiters": gate_state.get("waiters", []),
                },
            }


class TelemetryLogHandler(logging.Handler):
    """Forwards FastAPI/atlas.server logs into TelemetryHub's ring buffer."""

    def __init__(self, hub: TelemetryHub):
        super().__init__()
        self.hub = hub

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.hub.push_log(msg, source="server", level=record.levelname)
        except Exception:
            pass
