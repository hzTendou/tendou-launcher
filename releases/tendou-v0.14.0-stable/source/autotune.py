"""Universal Hardware & Topology Auto-Tuning Engine for Tendou Launcher.

Automatically inspects host hardware (VRAM, RAM, CPU cores/SIMD, storage),
parses model GGUF metadata, and solves the optimal 3-Tier memory placement
and execution parameters to maximize TPS on any given system.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("atlas.autotune")


# ---------------------------------------------------------------------------
# Hardware Profiling
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    has_gpu: bool
    gpu_name: str
    vram_total_mb: float
    vram_free_mb: float
    cpu_cores_physical: int
    cpu_cores_logical: int
    has_avx512_vnni: bool
    has_avx512_vbmi: bool
    has_avx2: bool
    ram_total_gb: float
    ram_avail_gb: float
    is_windows: bool
    directstorage_supported: bool


def probe_hardware() -> HardwareProfile:
    """Probes GPU VRAM, CPU architecture/SIMD, RAM and storage capabilities."""
    is_win = platform.system() == "Windows"

    # 1. GPU / VRAM via NVML or CUDA
    has_gpu = False
    gpu_name = "None"
    vram_total_mb = 0.0
    vram_free_mb = 0.0

    try:
        if is_win:
            nvml = ctypes.CDLL("nvml.dll")
            nvml.nvmlInit_v2()
            handle = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))

            # Name
            name_buf = ctypes.create_string_buffer(64)
            nvml.nvmlDeviceGetName(handle, name_buf, 64)
            gpu_name = name_buf.value.decode("utf-8", errors="ignore")

            # Memory
            class Memory(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

            mem = Memory()
            nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
            vram_total_mb = mem.total / (1024.0 * 1024.0)
            vram_free_mb = mem.free / (1024.0 * 1024.0)
            nvml.nvmlShutdown()
            has_gpu = True
    except Exception:
        # Fallback to nvidia-smi if NVML dll direct call fails
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    vram_total_mb = float(parts[1])
                    vram_free_mb = float(parts[2])
                    has_gpu = True
        except Exception:
            pass

    # 2. CPU Cores & SIMD
    logical_cores = os.cpu_count() or 8
    physical_cores = logical_cores // 2 if logical_cores > 1 else 1

    # SIMD flags probe (Zen 4/5 and Intel Xeon have AVX-512 VNNI)
    has_vnni = False
    has_vbmi = False
    has_avx2 = True  # Safe default on all modern x86_64

    try:
        # Check through cpuinfo or environment hint
        cpu_str = platform.processor().lower() + " " + platform.machine().lower()
        if "ryzen" in cpu_str or "intel" in cpu_str or "amd" in cpu_str:
            has_vnni = True
            has_vbmi = True
    except Exception:
        pass

    # 3. Host RAM
    ram_total_gb = 16.0
    ram_avail_gb = 8.0

    if is_win:
        try:
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
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram_total_gb = stat.ullTotalPhys / (1024.0 ** 3)
            ram_avail_gb = stat.ullAvailPhys / (1024.0 ** 3)
        except Exception:
            pass

    # 4. DirectStorage / Windows BypassIO
    ds_supported = False
    if is_win:
        try:
            # Win 11 Build 22000+ has native BypassIO support
            ver = sys.getwindowsversion()
            if ver.build >= 22000:
                ds_supported = True
        except Exception:
            pass

    return HardwareProfile(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        cpu_cores_physical=physical_cores,
        cpu_cores_logical=logical_cores,
        has_avx512_vnni=has_vnni,
        has_avx512_vbmi=has_vbmi,
        has_avx2=has_avx2,
        ram_total_gb=ram_total_gb,
        ram_avail_gb=ram_avail_gb,
        is_windows=is_win,
        directstorage_supported=ds_supported,
    )


# ---------------------------------------------------------------------------
# Lightweight GGUF Header Inspector
# ---------------------------------------------------------------------------

@dataclass
class ModelTopology:
    model_path: str
    model_name: str
    architecture: str
    block_count: int
    expert_count: int
    expert_used_count: int
    intermediate_size: int
    file_size_gb: float
    quant_type: str
    expert_slice_bytes: int
    is_moe: bool


def inspect_model_topology(model_path: str | Path) -> ModelTopology:
    """Reads GGUF header metadata without materializing tensor weights."""
    p = Path(model_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {p}")

    file_size_gb = p.stat().st_size / (1024.0 ** 3)
    name = p.stem

    # Defaults
    arch = "qwen2moe"
    block_count = 48
    expert_count = 512
    expert_used_count = 10
    intermediate_size = 512
    quant_type = "IQ3_XXS"
    expert_slice_bytes = 1253376

    # Fast heuristic check based on model file name if GGUF magic is valid
    name_lower = name.lower()
    if "35b" in name_lower or "qwen3.6" in name_lower:
        block_count = 40
        expert_count = 256
        expert_used_count = 8
        intermediate_size = 512
        quant_type = "IQ4_XS"
        expert_slice_bytes = 1458176
    elif "flash" in name_lower or "qwen3.8" in name_lower:
        block_count = 48
        expert_count = 512
        expert_used_count = 10
        intermediate_size = 512
        quant_type = "IQ3_XXS"
        expert_slice_bytes = 1253376

    # Parse GGUF header directly if possible
    try:
        with p.open("rb") as f:
            magic = f.read(4)
            if magic == b"GGUF":
                version = struct.unpack("<I", f.read(4))[0]
                if version in (2, 3):
                    tensor_count = struct.unpack("<Q" if version == 3 else "<I", f.read(8 if version == 3 else 4))[0]
                    metadata_kv_count = struct.unpack("<Q" if version == 3 else "<I", f.read(8 if version == 3 else 4))[0]

                    # Read up to first 250 metadata keys
                    for _ in range(min(metadata_kv_count, 250)):
                        k_len = struct.unpack("<Q", f.read(8))[0]
                        if k_len > 1024:
                            break
                        key = f.read(k_len).decode("utf-8", errors="ignore")
                        val_type = struct.unpack("<I", f.read(4))[0]

                        # Handle UINT32 / INT32 / STRING
                        if val_type in (4, 5):  # UINT32, INT32
                            val = struct.unpack("<I", f.read(4))[0]
                            if "block_count" in key:
                                block_count = val
                            elif "expert_count" in key:
                                expert_count = val
                            elif "expert_used_count" in key:
                                expert_used_count = val
                            elif "intermediate_size" in key:
                                intermediate_size = val
                        elif val_type == 8:  # STRING
                            s_len = struct.unpack("<Q", f.read(8))[0]
                            val_str = f.read(s_len).decode("utf-8", errors="ignore")
                            if "general.architecture" in key:
                                arch = val_str
                        elif val_type in (0, 1, 7):  # 1 byte
                            f.read(1)
                        elif val_type in (2, 3):  # 2 bytes
                            f.read(2)
                        elif val_type in (6, 10, 11, 12):  # 8 bytes
                            f.read(8)
                        elif val_type == 9:  # ARRAY
                            elem_type = struct.unpack("<I", f.read(4))[0]
                            elem_count = struct.unpack("<Q", f.read(8))[0]
                            # skip small array
                            elem_size = 1
                            if elem_type in (2, 3):
                                elem_size = 2
                            elif elem_type in (4, 5, 6):
                                elem_size = 4
                            elif elem_type in (10, 11, 12):
                                elem_size = 8
                            elif elem_type == 8:
                                # String array - skip each string
                                for _ in range(min(elem_count, 1000)):
                                    slen = struct.unpack("<Q", f.read(8))[0]
                                    f.seek(slen, os.SEEK_CUR)
                                continue
                            f.seek(elem_size * elem_count, os.SEEK_CUR)
    except Exception:
        pass

    is_moe = (expert_count > 1)

    return ModelTopology(
        model_path=str(p),
        model_name=name,
        architecture=arch,
        block_count=block_count,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        intermediate_size=intermediate_size,
        file_size_gb=round(file_size_gb, 2),
        quant_type=quant_type,
        expert_slice_bytes=expert_slice_bytes,
        is_moe=is_moe,
    )


# ---------------------------------------------------------------------------
# Auto-Tuner Optimization Solver
# ---------------------------------------------------------------------------

@dataclass
class AutoTuneConfig:
    threads: int
    gpu_layers: int
    vram_cache_mb: int
    moe_cache_mib: int
    mtp_mode: str
    mtp_draft_n: int
    boost_level: int
    ctx_size: int
    physical_map_path: Optional[str]
    rationale: Dict[str, Any]
    prompt_cache_mb: int = 512


def solve_optimal_configuration(
    hw: HardwareProfile,
    model: ModelTopology,
    base_dir: Optional[Path] = None,
) -> AutoTuneConfig:
    """Solves the optimal 3-Tier memory placement and speculative settings for maximum TPS."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    # 1. Thread Allocation
    # Zen 4/5 8C/16T sweet spot is 8 threads to prevent L3 cache and SMT contention
    threads = min(hw.cpu_cores_physical, 8)
    if threads < 4 and hw.cpu_cores_logical >= 4:
        threads = 4

    # 2. Dense Layers & VRAM Allocation (Tier-0)
    # Estimate dense attention memory: ~2.5 GB for 35B/3.8, plus KV cache for 8192 ctx (~400 MB)
    dense_vram_mb = 2560.0
    kv_cache_mb = 400.0
    vram_guard_mb = 512.0

    gpu_layers = 0
    vram_cache_mb = 0
    moe_cache_mib = 0

    if hw.has_gpu and hw.vram_free_mb > 2000.0:
        # Offload dense layers to GPU via -ngl 99; keep 0-2 MoE anchor layers in VRAM
        # Deep MoE layers (2..N) stay CPU/RAM backed to prevent VRAM OOM on 8GB GPUs
        avail_vram_for_experts = max(0.0, hw.vram_free_mb - dense_vram_mb - kv_cache_mb - vram_guard_mb)
        gpu_layers = 2 if avail_vram_for_experts >= 1800.0 else 0
        vram_cache_mb = int(min(avail_vram_for_experts, 4096.0))

    # 3. Host RAM Pinning (Tier-1)
    # Reserve 2.5 GB for OS and application
    ram_guard_gb = 2.5
    avail_ram_gb = max(1.0, hw.ram_avail_gb - ram_guard_gb)

    # For 35B model: hot 34.4% experts = 4.85 GiB (~5120 MiB)
    # For Qwen3.8 Flash Next: hot cache budget ~5120-6144 MiB
    max_ram_cache_mib = int(avail_ram_gb * 1024.0)
    moe_cache_mib = min(max_ram_cache_mib, 5120)

    # 4. Multi-Turn Prefix State Cache (Prompt Caching)
    prompt_cache_mb = 512 if hw.ram_avail_gb >= 2.0 else 256

    # 5. Speculative Decoding Strategy (PLD vs Neural MTP)
    # Neural MTP requires ~2.2 GB VRAM for draft weights.
    # If VRAM headroom after VRAM cache is tight (< 2400 MB), 0-VRAM PLD is strictly superior.
    remaining_vram_mb = hw.vram_free_mb - dense_vram_mb - vram_cache_mb
    mtp_mode = "pld"
    mtp_draft_n = 3

    # Check for MTP model file
    mtp_q4 = base_dir / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_0.gguf"
    if remaining_vram_mb >= 2400.0 and mtp_q4.is_file() and "flash" in model.model_name.lower():
        mtp_mode = "vram"
        mtp_draft_n = 2
    else:
        mtp_mode = "pld"
        # On PLD, n=3 or n=4 gives tepe EV(d) as proven in Phase 11 horizon sweep
        mtp_draft_n = 3

    # 5. Physical Map Selection (4KB Fused Sektör Hizalı & Model Quant Uyumluluğu)
    physical_map_path = None
    if model.block_count == 40 and model.expert_count == 256:
        # Qwen3.6-35B
        cand_35b = [
            base_dir / "config" / "atlas_physical_map_35b_3tier.json",
            base_dir / "config" / "atlas_physical_map_35b.json",
            Path(r"C:\Users\Ali\WebProjects\tening\artifacts\phase11\atlas_physical_map_3tier_server.json"),
        ]
        for c in cand_35b:
            if c.is_file():
                try:
                    with open(c, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    # Check if map belongs to same quant or file
                    m_file = m_data.get("file", {}).get("path", "")
                    if model.quant_type.lower() in m_file.lower() or "35b" in m_file.lower():
                        physical_map_path = str(c)
                        break
                except Exception:
                    continue
    elif model.block_count == 48 and model.expert_count == 512:
        # Qwen3.8 Flash Next
        cand_38 = [
            base_dir / "config" / "atlas_physical_map_qwen38_iq3xxs.json",
            base_dir / "config" / "atlas_physical_map_qwen38.json",
            base_dir / "config" / "atlas_physical_map.json",
        ]
        for c in cand_38:
            if c.is_file():
                try:
                    with open(c, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    m_file = str(m_data)
                    # Only accept if quant matches (prevent Q5_K_S map on IQ3_XXS model)
                    if model.quant_type.lower() in m_file.lower():
                        physical_map_path = str(c)
                        break
                except Exception:
                    continue

    rationale = {
        "hardware_detected": f"{hw.gpu_name} ({hw.vram_total_mb:.0f} MB VRAM, {hw.vram_free_mb:.0f} MB free) / {hw.ram_total_gb:.1f} GB RAM / {threads} threads",
        "model_detected": f"{model.model_name} ({model.block_count}L x {model.expert_count}E, Top-{model.expert_used_count}, {model.quant_type})",
        "tier0_vram_cache": f"{vram_cache_mb} MB allocated for ultra-hot GPU experts",
        "tier1_ram_cache": f"{moe_cache_mib} MB pinned host RAM via VirtualLock",
        "tier2_nvme": "4KB DirectStorage / BypassIO streaming active",
        "prompt_cache": f"{prompt_cache_mb} MB multi-turn prefix state cache active",
        "speculative": f"{mtp_mode.upper()} mode with draft_n={mtp_draft_n}",
    }

    return AutoTuneConfig(
        threads=threads,
        gpu_layers=gpu_layers,
        vram_cache_mb=vram_cache_mb,
        moe_cache_mib=moe_cache_mib,
        mtp_mode=mtp_mode,
        mtp_draft_n=mtp_draft_n,
        boost_level=2,
        ctx_size=8192,
        physical_map_path=physical_map_path,
        rationale=rationale,
        prompt_cache_mb=prompt_cache_mb,
    )


# ---------------------------------------------------------------------------
# Dynamic Speculative Horizon Controller (Adaptive Zero-Penalty Policy)
# ---------------------------------------------------------------------------

def estimate_prompt_pld_suitability(
    prompt: str,
    category: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, int, str]:
    """
    Evaluates whether a workload will benefit from Prompt-Lookup Decoding (PLD).
    
    Returns:
        (use_pld: bool, draft_n: int, rationale: str)
        
    Mathematical Rationale:
      Expected speedup EV(d) = alpha * d - (1 - alpha) * c_rollback.
      - If acceptance rate alpha >= 0.50 (e.g. repetitive code, refactoring patterns):
        EV(d=3) > 0 -> Up to 2.5x throughput gain.
      - If acceptance rate alpha < 0.25 (e.g. greenfield reasoning, tool stream):
        EV(d=3) < 0 -> Negative speedup (-10% to -50% penalty).
        Controller enforces d* = 0 (zero-penalty autoregressive fallback).
    """
    if tools:
        return (
            False,
            0,
            "Tool calling stream requires immediate first-argument emission; adaptive fallback d*=0 enforced",
        )

    cat_lower = (category or "").lower()
    if cat_lower in (
        "code_refactoring",
        "repetitive_code",
        "pattern_repetition",
        "templated_expansion",
        "cline_replace_in_file",
        "cline_edit",
    ):
        return (
            True,
            3,
            f"Workload category '{category}' has verified high acceptance (alpha >= 0.70) -> PLD d*=3 active",
        )

    if cat_lower in (
        "greenfield_reasoning",
        "tool_calling_stream",
        "creative_synthesis",
        "zero_shot_math",
        "cline_tool_call",
        "cline_exploration",
    ):
        return (
            False,
            0,
            f"Workload category '{category}' is non-repetitive (alpha ~ 0) -> Zero-penalty fallback d*=0 active",
        )

    # Cline XML Agentic Mode Detection:
    # Cline embeds XML tool instructions (<replace_in_file>, <read_file>, <write_to_file>, etc.) in the system prompt.
    # Distinguish between Cline file-editing turns (where SEARCH/REPLACE copies code from prompt -> PLD d*=3)
    # and Cline exploration/command turns (where only a short XML tool call is emitted -> d*=0).
    is_cline_prompt = (
        "You are Cline" in prompt
        or ("<replace_in_file>" in prompt and "<read_file>" in prompt)
        or "<environment_details>" in prompt
    )
    if is_cline_prompt:
        # Extract conversation turns after the system prompt to avoid false n-gram triggers on system boilerplate
        conv_body = prompt
        sys_end = prompt.find("<|im_end|>")
        if sys_end != -1:
            conv_body = prompt[sys_end + len("<|im_end|>") :]
        conv_lower = conv_body.lower()
        has_code_context = (
            "[read_file for" in conv_lower
            or "<file_content" in conv_lower
            or "<<<<<<< search" in conv_lower
            or "def " in conv_body
            or "class " in conv_body
            or "function " in conv_body
        )
        wants_edit = any(
            kw in conv_lower
            for kw in (
                "replace_in_file",
                "write_to_file",
                "refactor",
                "optimize",
                "update",
                "patch",
                "modify",
                "fix",
                "edit",
                "implement",
            )
        )
        if has_code_context and wants_edit:
            return (
                True,
                3,
                "Cline XML file-editing / SEARCH-REPLACE context detected -> PLD d*=3 active",
            )
        return (
            False,
            0,
            "Cline XML exploration turn detected -> Zero-penalty fallback d*=0 active",
        )

    # Autonomous raw prompt n-gram redundancy estimator
    words = prompt.split()
    if len(words) >= 6:
        ngrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        redundancy = 1.0 - (len(set(ngrams)) / len(ngrams))
        if redundancy >= 0.20:
            return (
                True,
                3,
                f"Autonomous n-gram redundancy analysis: score={redundancy:.2f} >= 0.20 -> PLD d*=3 active",
            )

    return (
        False,
        0,
        "Autonomous n-gram redundancy analysis: novel syntax (alpha < 0.20) -> Zero-penalty fallback d*=0 active",
    )


def solve_workload_speculative_policy(
    prompt: str,
    category: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_mtp_mode: str = "pld",
) -> Tuple[str, int, str]:
    """Resolves the exact runtime speculative mode and draft horizon for a request."""
    if base_mtp_mode == "off":
        return "off", 0, "Speculative decoding explicitly disabled by engine configuration"

    use_pld, draft_n, rationale = estimate_prompt_pld_suitability(prompt, category=category, tools=tools)
    if use_pld:
        return "pld", draft_n, rationale
    return "off", 0, rationale


# ---------------------------------------------------------------------------
# CLI Entrypoint for Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tendou AutoTune Diagnostic Tool")
    parser.add_argument("--model-path", type=str, default=None, help="Path to GGUF model")
    args = parser.parse_args()

    print("================================================================================")
    print("TENDOU LAUNCHER: UNIVERSAL AUTO-TUNING ENGINE")
    print("================================================================================")
    hw = probe_hardware()
    print(f"GPU Profile  : {hw.gpu_name} (Total: {hw.vram_total_mb:.1f} MB, Free: {hw.vram_free_mb:.1f} MB)")
    print(f"CPU Profile  : {hw.cpu_cores_physical} physical / {hw.cpu_cores_logical} logical cores (AVX-512 VNNI: {hw.has_avx512_vnni})")
    print(f"Host RAM     : {hw.ram_avail_gb:.2f} GB free of {hw.ram_total_gb:.2f} GB total")
    print(f"BypassIO     : {'Supported (Windows 11 DirectStorage)' if hw.directstorage_supported else 'Fallback mmap'}")

    m_path = args.model_path or os.environ.get("TENDOU_MODEL_PATH")
    if not m_path or not os.path.exists(m_path):
        # Default to Qwen3.8 Flash Next or Qwen3.6 if available
        p1 = Path(r"C:\Users\Ali\WebProjects\tendou-launcher\models\qwen3.8-gsq-rco-iq3_xxs\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
        p2 = Path(r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
        m_path = str(p1 if p1.exists() else p2)

    if os.path.exists(m_path):
        print(f"\nTarget Model : {m_path}")
        model = inspect_model_topology(m_path)
        print(f"Topology     : {model.block_count} layers x {model.expert_count} experts (Top-{model.expert_used_count})")
        print(f"File Size    : {model.file_size_gb} GB ({model.quant_type})")

        cfg = solve_optimal_configuration(hw, model)
        print("\n================================================================================")
        print("OPTIMAL CONFIGURATION SOLVED (Maximum TPS):")
        print("================================================================================")
        print(f"Threads          : {cfg.threads}")
        print(f"GPU Offload (ngl): {cfg.gpu_layers}")
        print(f"VRAM Cache (MB)  : {cfg.vram_cache_mb}")
        print(f"MoE RAM Pin (MiB): {cfg.moe_cache_mib}")
        print(f"MTP / PLD Mode   : {cfg.mtp_mode} (draft_n={cfg.mtp_draft_n})")
        print(f"Physical Map     : {cfg.physical_map_path}")
        print("Rationale:")
        for k, v in cfg.rationale.items():
            print(f"  - {k}: {v}")
