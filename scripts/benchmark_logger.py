"""
tendou launcher — Continuous Benchmark Logger.

Provides append-only structured reporting for benchmark runs across short and large
contexts (2k, 32k, 64k, 128k), ensuring cache activation, detailed metric extraction,
and full recording of AI generated outputs.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import platform
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_REPORT_MD = Path(__file__).resolve().parents[1] / "docs" / "BENCHMARK_LOG.md"
DEFAULT_REPORT_JSONL = Path(__file__).resolve().parents[1] / "experiments" / "benchmark_history.jsonl"


@dataclass
class BenchmarkItemResult:
    task_id: str
    category: str
    workload: str
    context_size: int
    prompt: str
    generated_output: str
    expected: Optional[str] = None
    task_pass: Optional[bool] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    prompt_cache_bytes: int = 0
    cache_hit: bool = False
    ttft_ms: float = 0.0
    decode_tps: float = 0.0
    effective_tps: float = 0.0
    prefill_compute_ms: float = 0.0
    prefill_io_ms: float = 0.0
    gpu_compute_ms: float = 0.0
    checkpoint_ms: float = 0.0
    elapsed_ms: float = 0.0
    finish_reason: str = "stop"
    repetition: int = 0
    turn: int = 0
    kv_quant: str = "q8_0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRunSummary:
    run_id: str
    timestamp: str
    model_name: str
    threads: int
    cache_mb: int
    hardware_info: Dict[str, Any]
    items: List[BenchmarkItemResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["items"] = [item.to_dict() if hasattr(item, "to_dict") else item for item in self.items]
        return d


def detect_hardware() -> Dict[str, Any]:
    """Detect available system hardware using standard library and system utilities."""
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": "AMD Ryzen 7 260 (8 Physical Zen 4 Cores)",
        "ram_gb": 15.29,
        "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB VRAM)",
    }

    # Detect RAM via ctypes (Windows)
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                info["ram_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            pass

    # Detect CPU name via winreg (Windows)
    if os.name == "nt":
        try:
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
            cores = os.cpu_count() or 8
            info["cpu"] = f"{cpu_name} ({cores} threads)"
        except Exception:
            pass

    # Detect GPU via nvidia-smi
    try:
        import subprocess

        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=2.0,
            stderr=subprocess.DEVNULL,
        ).strip()
        if gpu_out:
            info["gpu"] = gpu_out
    except Exception:
        pass

    return info


def _append_to_file(path: Path, text: str) -> None:
    """Appends text to file, ensuring clean newline separation even if trailing newline was missing."""
    needs_leading_newline = False
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    needs_leading_newline = True
        except Exception:
            pass

    with open(path, "a", encoding="utf-8") as f:
        if needs_leading_newline:
            f.write("\n")
        f.write(text)


class BenchmarkLogger:
    """
    Manages continuous append-only reporting for benchmarks.
    Appends markdown formatted logs to docs/BENCHMARK_LOG.md and
    structured lines to experiments/benchmark_history.jsonl.
    """

    def __init__(
        self,
        report_md_path: Optional[Path] = None,
        report_jsonl_path: Optional[Path] = None,
    ):
        self.report_md_path = Path(report_md_path or DEFAULT_REPORT_MD).resolve()
        self.report_jsonl_path = Path(report_jsonl_path or DEFAULT_REPORT_JSONL).resolve()

        # Ensure parent directories exist
        self.report_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_md_header(self) -> None:
        """If markdown report does not exist or is empty, write header banner."""
        if not self.report_md_path.exists() or self.report_md_path.stat().st_size == 0:
            header = (
                "# tendou launcher — Sürekli Benchmark Rapor Günlüğü (Continuous Benchmark Log)\n\n"
                "Bu dosya her yeni benchmark çalıştırıldığında append-only (sürekli son satırdan eklemeli)\n"
                "olarak güncellenir. Her test koşusunda metrikler (TTFT, prefill compute, decode TPS,\n"
                "efektif TPS, prompt cache kullanımı) ve yapay zeka tarafından üretilen tam metin çıktısı\n"
                "(AI generated output) eksiksiz olarak kaydedilir.\n\n"
            )
            self.report_md_path.write_text(header, encoding="utf-8")

    def append_run(self, run: BenchmarkRunSummary) -> None:
        """
        Appends the benchmark run summary to the end of both docs/BENCHMARK_LOG.md
        and experiments/benchmark_history.jsonl.
        """
        self._ensure_md_header()

        # 1. Append structured JSONL
        jsonl_line = json.dumps(run.to_dict(), ensure_ascii=False) + "\n"
        _append_to_file(self.report_jsonl_path, jsonl_line)

        # 2. Format Markdown section
        lines: List[str] = [
            "---",
            f"## Benchmark Koşusu: `{run.run_id}` ({run.timestamp})",
            "",
            "### 1. Çalışma Konfigürasyonu ve Donanım",
            f"- **Zaman**: `{run.timestamp}`",
            f"- **Model**: `{run.model_name}`",
            f"- **Donanım**: CPU: `{run.hardware_info.get('cpu', 'N/A')}` | GPU: `{run.hardware_info.get('gpu', 'N/A')}` | RAM: `{run.hardware_info.get('ram_gb', 'N/A')} GB`",
            f"- **Parametreler**: Threads={run.threads}, Prompt Cache={run.cache_mb} MB",
            f"- **Toplam Koşu Süresi**: `{run.total_duration_s:.2f} s`",
        ]
        if run.notes:
            lines.append(f"- **Notlar**: {run.notes}")

        lines.extend([
            "",
            "### 2. Metrik Özeti Tablosu",
            "| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ])

        for item in run.items:
            cache_detail = f"HIT ({item.cached_prompt_tokens})" if item.cache_hit else "COLD (0)"
            if item.prompt_cache_bytes > 0:
                cache_detail += f" [{item.prompt_cache_bytes / (1024 * 1024):.1f}MB]"
            pass_str = "PASS" if item.task_pass is True else ("FAIL" if item.task_pass is False else "N/A")
            lines.append(
                f"| `{item.task_id}` | {item.context_size} | `{item.kv_quant}` | {item.prompt_tokens} | "
                f"{item.completion_tokens} | {item.cached_prompt_tokens} | {item.ttft_ms:.1f} | "
                f"{item.decode_tps:.2f} | {item.effective_tps:.2f} | {cache_detail} | {pass_str} |"
            )

        lines.extend([
            "",
            "### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)",
            "",
        ])

        for idx, item in enumerate(run.items, 1):
            cache_status = f"CACHE HIT ({item.cached_prompt_tokens} tokens reused)" if item.cache_hit else "COLD PREFILL"
            cache_bytes_info = f", prompt_cache={item.prompt_cache_bytes / (1024 * 1024):.2f} MB" if item.prompt_cache_bytes > 0 else ""

            # Safely format prompt summary on a single line so markdown list indentation is not broken
            clean_prompt = " ".join(item.prompt.strip().split())
            prompt_summary = clean_prompt[:180] + ("..." if len(clean_prompt) > 180 else "")

            # Safely calculate fence length to avoid collisions with backticks in generated code
            max_backticks = max([len(m) for m in re.findall(r"`+", item.generated_output)] or [2])
            fence = "`" * max(3, max_backticks + 1)

            lines.extend([
                f"#### [{idx}] Görev: `{item.task_id}` | Context: {item.context_size} | Turn: {item.turn}",
                f"- **Kategori / İş Yükü**: `{item.category}` / `{item.workload}`",
                f"- **Context & KV Belleği**: `{item.context_size}` tokens (KV: `{item.kv_quant}`)",
                f"- **Token Sayaçları**: Prompt={item.prompt_tokens}, Üretilen={item.completion_tokens}, Cache Reused={item.cached_prompt_tokens} ({cache_status}{cache_bytes_info})",
                f"- **Performans**: TTFT=`{item.ttft_ms:.2f} ms`, Decode=`{item.decode_tps:.2f} TPS`, Efektif=`{item.effective_tps:.2f} TPS`, Toplam=`{item.elapsed_ms:.2f} ms`",
                f"- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`{item.prefill_compute_ms:.2f} ms`, gpu_compute=`{item.gpu_compute_ms:.2f} ms`, prefill_io=`{item.prefill_io_ms:.2f} ms`, checkpoint=`{item.checkpoint_ms:.2f} ms`",
                f"- **Bitiş Nedeni**: `{item.finish_reason}`",
                f"- **Prompt**: `{prompt_summary}`",
                "- **Modelin Ürettiği Çıktı (AI Generated Output)**:",
                f"{fence}text",
                item.generated_output.strip(),
                f"{fence}",
                "",
            ])

        lines.append("")  # trailing newline
        md_text = "\n".join(lines) + "\n"
        _append_to_file(self.report_md_path, md_text)
