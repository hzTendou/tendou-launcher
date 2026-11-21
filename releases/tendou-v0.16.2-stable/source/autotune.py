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
        nvml = ctypes.CDLL("nvml.dll" if is_win else "libnvidia-ml.so.1")
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
        if vram_total_mb > 0:
            has_gpu = True
    except Exception:
        pass

    if not has_gpu:
        # Fallback to nvidia-smi if NVML library direct call fails
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
        cpu_str = platform.processor().lower() + " " + platform.machine().lower()
        if not is_win and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                cpuinfo = f.read().lower()
            cpu_str += " " + cpuinfo
            core_ids = set()
            for line in cpuinfo.splitlines():
                if line.startswith("core id"):
                    core_ids.add(line.split(":", 1)[-1].strip())
            if core_ids:
                physical_cores = len(core_ids)
        if "avx512_vnni" in cpu_str or "ryzen" in cpu_str or "intel" in cpu_str or "amd" in cpu_str:
            has_vnni = True
        if "avx512vbmi" in cpu_str or "avx512_vbmi" in cpu_str or "ryzen" in cpu_str or "intel" in cpu_str or "amd" in cpu_str:
            has_vbmi = True
        if "avx2" in cpu_str:
            has_avx2 = True
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
    elif os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_total_gb = float(line.split()[1]) / (1024.0 * 1024.0)
                    elif line.startswith("MemAvailable:"):
                        ram_avail_gb = float(line.split()[1]) / (1024.0 * 1024.0)
        except Exception:
            pass

    # 4. DirectStorage / Windows BypassIO / Linux NVMe Direct I/O
    ds_supported = False
    if is_win:
        try:
            # Win 11 Build 22000+ has native BypassIO support
            ver = sys.getwindowsversion()
            if ver.build >= 22000:
                ds_supported = True
        except Exception:
            pass
    else:
        try:
            rel = platform.release().split(".", 2)
            if int(rel[0]) >= 5:
                ds_supported = True
        except Exception:
            ds_supported = True

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
    if "-00001-of-" in p.name:
        prefix = p.name.split("-00001-of-")[0]
        shards = list(p.parent.glob(f"{prefix}-*-of-*.gguf"))
        if shards:
            file_size_gb = sum(s.stat().st_size for s in shards) / (1024.0 ** 3)
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
    if "122b" in name_lower or "a10b" in name_lower:
        block_count = 48
        expert_count = 256
        expert_used_count = 8
        intermediate_size = 1024
        if "mxfp4" in name_lower:
            quant_type = "MXFP4_MOE"
            expert_slice_bytes = 5505024
        else:
            quant_type = "Q6_K_P"
            expert_slice_bytes = 7741440
    elif "35b" in name_lower or "qwen3.6" in name_lower:
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

                        # Handle UINT32 / INT32 / UINT64 / INT64 / STRING
                        if val_type in (4, 5):  # UINT32, INT32
                            val = struct.unpack("<I" if val_type == 4 else "<i", f.read(4))[0]
                            if "block_count" in key:
                                block_count = val
                            elif "expert_count" in key:
                                expert_count = val
                            elif "expert_used_count" in key:
                                expert_used_count = val
                            elif "intermediate_size" in key:
                                intermediate_size = val
                        elif val_type in (10, 11):  # UINT64, INT64
                            val = struct.unpack("<Q" if val_type == 10 else "<q", f.read(8))[0]
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
                        elif val_type == 6:  # 4 bytes (FLOAT32)
                            f.read(4)
                        elif val_type == 12:  # 8 bytes (FLOAT64)
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
    moe_cache_cpu_share: str = "auto"
    ffn_chunk_size: int = 512
    cache_type_k: str = "f8_e4m3"
    cache_type_v: str = "f8_e4m3"
    vram_layer_cap: str = "auto"
    ram_lock_policy: str = "auto"


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
    # Estimate dense attention memory: ~6.2 GB for 122B MXFP4 (Q8_0 dense), ~4.9 GB for 122B Q6_K, ~2.5 GB for 35B/3.8, plus KV cache for 8192 ctx
    is_122b = (model.block_count == 48 and model.expert_count == 256) or ("122b" in model.model_name.lower())
    dense_vram_mb = 6200.0 if (is_122b and "mxfp4" in model.quant_type.lower()) else (4900.0 if is_122b else 2560.0)
    layer_expert_mb = 2372.0 if is_122b else 900.0
    kv_cache_mb = 400.0
    vram_guard_mb = 512.0

    gpu_layers = 0
    vram_cache_mb = 0
    moe_cache_mib = 0

    if hw.has_gpu and hw.vram_free_mb > 2000.0:
        # Offload dense layers to GPU via -ngl 99; keep 0-2 MoE anchor layers in VRAM only if capacity permits
        # Deep MoE layers stay CPU/RAM backed to prevent VRAM OOM / WDDM PCIe thrashing on 8GB GPUs
        avail_vram_for_experts = max(0.0, hw.vram_free_mb - dense_vram_mb - kv_cache_mb - vram_guard_mb)
        gpu_layers = 2 if avail_vram_for_experts >= (2.0 * layer_expert_mb) else 0
        vram_cache_mb = int(min(avail_vram_for_experts, 4096.0)) if gpu_layers > 0 else 0

    # 3. Host RAM Pinning (Tier-1)
    # Reserve 2.5 GB for OS and application
    ram_guard_gb = 2.5
    avail_ram_gb = max(1.0, hw.ram_avail_gb - ram_guard_gb)

    # For 35B model: hot 34.4% experts = 4.85 GiB (~5120 MiB)
    # For Qwen3.8 Flash Next: hot cache budget ~5120-6144 MiB
    max_ram_cache_mib = int(avail_ram_gb * 1024.0)
    moe_cache_mib = 2048 if is_122b else min(max_ram_cache_mib, 5120)

    # 4. Multi-Turn Prefix State Cache (Prompt Caching)
    prompt_cache_mb = 512 if hw.ram_avail_gb >= 2.0 else 256

    # 5. Speculative Decoding Strategy (PLD vs Neural MTP)
    # Neural MTP requires ~2.2 GB VRAM for draft weights.
    # On 122B+ models on laptop CPU/RAM, batch verification of drafts creates severe memory latency spikes;
    # pure nominal autoregressive decoding (d*=0) gives 3.8+ TPS without batch latency penalties.
    remaining_vram_mb = hw.vram_free_mb - dense_vram_mb - vram_cache_mb
    if is_122b:
        mtp_mode = "off"
        mtp_draft_n = 0
    else:
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
        if "iq4_xs" in model.quant_type.lower() or "iq4_xs" in model.model_name.lower() or "iq4xs" in model.model_name.lower():
            cand_35b = [
                base_dir / "config" / "atlas_physical_map_35b_iq4xs.json",
                base_dir / "config" / "atlas_physical_map_35b_3tier.json",
                base_dir / "config" / "atlas_physical_map_35b.json",
            ]
        else:
            cand_35b = [
                base_dir / "config" / "atlas_physical_map_35b_3tier.json",
                base_dir / "config" / "atlas_physical_map_35b.json",
                base_dir / "config" / "atlas_physical_map_35b_iq4xs.json",
            ]
        fallback_35b = None
        for c in cand_35b:
            if c.is_file():
                if fallback_35b is None:
                    fallback_35b = str(c)
                try:
                    with open(c, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    # Check if map belongs to same quant
                    m_file = m_data.get("file", {}).get("path", "") or m_data.get("source", "")
                    if model.quant_type.lower() in m_file.lower():
                        physical_map_path = str(c)
                        break
                except Exception:
                    continue
        if not physical_map_path and fallback_35b:
            physical_map_path = fallback_35b
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
    elif model.block_count == 48 and model.expert_count == 256:
        # Qwen3.5-122B-A10B
        if "mxfp4" in model.quant_type.lower() or "mxfp4" in model.model_name.lower():
            cand_122b = [
                base_dir / "config" / "atlas_physical_map_122b_mxfp4.json",
                base_dir / "config" / "atlas_physical_map_122b.json",
            ]
        else:
            cand_122b = [
                base_dir / "config" / "atlas_physical_map_122b_q6k.json",
                base_dir / "config" / "atlas_physical_map_122b.json",
            ]
        for c in cand_122b:
            if c.is_file():
                try:
                    with open(c, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    m_file = str(m_data)
                    if model.quant_type.lower() in m_file.lower() or "122b" in m_file.lower():
                        physical_map_path = str(c)
                        break
                except Exception:
                    continue

    rationale = {
        "hardware_detected": f"{hw.gpu_name} ({hw.vram_total_mb:.0f} MB VRAM, {hw.vram_free_mb:.0f} MB free) / {hw.ram_total_gb:.1f} GB RAM / {threads} threads",
        "model_detected": f"{model.model_name} ({model.block_count}L x {model.expert_count}E, Top-{model.expert_used_count}, {model.quant_type})",
        "tier0_vram_cache": f"{vram_cache_mb} MB allocated for ultra-hot GPU experts",
        "tier1_ram_cache": f"{moe_cache_mib} MB locked host RAM via ClassLRU",
        "tier2_nvme": "Multi-worker DirectStorage / io_uring streaming active",
        "prompt_cache": f"{prompt_cache_mb} MB multi-turn prefix state cache active",
        "kv_cache": "f8_e4m3 (FP8 E4M3) up to 32k context",
        "cpu_share": "auto (closed-loop cost-model adaptive)",
        "ffn_chunk_size": "512 tokens (enables ubatch >= 1024 prefill)",
        "vram_layer_cap": "auto (Phase 18 adaptive VRAM MoE layer residency cap)",
        "ram_lock_policy": "auto (Phase 18 adaptive host RAM lock policy)",
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
        moe_cache_cpu_share="auto",
        ffn_chunk_size=512,
        cache_type_k="f8_e4m3",
        cache_type_v="f8_e4m3",
        vram_layer_cap="auto",
        ram_lock_policy="auto",
    )


def _estimate_kv_budget_mb(model: ModelTopology, ctx_size: int, cache_k: str, cache_v: str) -> float:
    """Estimate KV cache VRAM footprint in MiB for hybrid MoE models."""
    def _bpe(q: str) -> float:
        ql = (q or "f8_e4m3").lower()
        if ql in ("f16", "bf16"):
            return 2.0
        if ql in ("q8_0", "f8_e4m3", "fp8", "f8_e5m2"):
            return 1.05
        if ql in ("q4_0", "q4_1"):
            return 0.56
        return 1.05

    # Hybrid DeltaNet/GQA models keep full attention KV on ~1/4 of layers (e.g. 10-12 layers)
    # with GQA kv_heads=2..4, head_dim=256. Empirical RTX 5060 calibration:
    # 35B/38B @ 8K q8_0 ~ 320 MiB, 16K f8_e4m3 ~ 620 MiB, 64K f8_e4m3 ~ 1450 MiB.
    scale_factor = (_bpe(cache_k) + _bpe(cache_v)) / 2.1
    base_per_8k_mb = 320.0 if model.block_count <= 40 else 380.0
    if "122b" in model.model_name.lower() or (model.block_count == 48 and model.expert_count == 256):
        base_per_8k_mb = 480.0
    chunks = max(0.5, float(ctx_size) / 8192.0)
    # Sublinear scaling past 16k due to hybrid recurrent layers occupying fixed state size
    if chunks <= 2.0:
        return round(base_per_8k_mb * chunks * scale_factor, 1)
    return round(base_per_8k_mb * (2.0 + (chunks - 2.0) * 0.45) * scale_factor, 1)


def compute_auto_plan(
    hw: HardwareProfile,
    model: ModelTopology,
    ctx_size: int = 8192,
    base_dir: Optional[Path] = None,
    explicit_gpu_layers: Optional[int] = None,
    explicit_vram_layer_cap: Optional[str] = None,
    explicit_cache_type_k: Optional[str] = None,
    explicit_cache_type_v: Optional[str] = None,
    explicit_fp8_kv: Optional[bool] = None,
) -> AutoTuneConfig:
    """
    v0.16.2 Model-Aware AutoTune Planner:
      - Adaptive KV Cache: q8_0 for ctx <= 8192 (fastest short-ctx FlashAttn),
        f8_e4m3 for ctx >= 16384 (~50% KV VRAM reduction without Q4_0 quality loss).
      - Model-Aware GPU MoE Layer Placement:
        * Computes actual expert_layer_vram_mb from GGUF metadata and available VRAM
          after dense layers, KV budget, and 1024 MiB safety headroom.
        * 35B-class (<= 45 GB, <= 40 layers): automatically places 6..8 MoE layers in VRAM
          when headroom permits (utilizing idle ~3.5 GiB on 8 GiB GPUs).
        * 38B/122B-class (> 45 GB or 48 layers): preserves safe <= 2 layer cap to avoid OOM.
      - Never overrides explicit user --gpu-layers / --vram-layer-cap / KV flags.
      - Neural MTP stays strictly OFF ('off', draft_n=0) by default.
    """
    base_cfg = solve_optimal_configuration(hw, model, base_dir=base_dir)
    name_lower = model.model_name.lower()

    is_122b = (model.block_count == 48 and model.expert_count == 256) or ("122b" in name_lower)
    is_38b = (
        (model.block_count >= 48 and model.expert_count >= 512)
        or model.file_size_gb > 45.0
        or "3.8" in name_lower
        or "qwen38" in name_lower
        or "flash-next" in name_lower
    )
    is_35b_class = (
        not is_122b
        and not is_38b
        and model.is_moe
        and model.file_size_gb <= 45.0
        and model.block_count <= 40
    )

    # 1. Context-Based Adaptive KV Cache (Stage 0.3)
    if explicit_cache_type_k is not None or explicit_cache_type_v is not None:
        kv_k = explicit_cache_type_k or explicit_cache_type_v or ("q8_0" if ctx_size <= 8192 else "f8_e4m3")
        kv_v = explicit_cache_type_v or explicit_cache_type_k or kv_k
        kv_policy = "explicit"
    elif explicit_fp8_kv is True:
        kv_k = "f8_e4m3"
        kv_v = "f8_e4m3"
        kv_policy = "explicit_fp8"
    elif explicit_fp8_kv is False:
        kv_k = "q8_0" if ctx_size <= 32768 else "q4_0"
        kv_v = "q8_0" if ctx_size <= 8192 else "q4_0"
        kv_policy = "explicit_non_fp8"
    elif ctx_size <= 8192:
        kv_k = "q8_0"
        kv_v = "q8_0"
        kv_policy = "adaptive_short_q8_0"
    else:
        kv_k = "f8_e4m3"
        kv_v = "f8_e4m3"
        kv_policy = "adaptive_long_f8_e4m3"

    # 2. Compute per-layer MoE VRAM size & safe headroom (1024 MiB)
    dense_vram_mb = (
        6200.0 if (is_122b and "mxfp4" in model.quant_type.lower())
        else (4900.0 if is_122b else (3100.0 if is_38b else 2560.0))
    )
    kv_budget_mb = _estimate_kv_budget_mb(model, ctx_size, kv_k, kv_v)
    safety_headroom_mb = 1024.0

    meta_layer_mb = (float(model.expert_slice_bytes) * float(max(1, model.expert_count))) / (1024.0 * 1024.0)
    size_layer_mb = max(0.0, (model.file_size_gb * 1024.0) - dense_vram_mb) / float(max(1, model.block_count))
    if is_122b:
        expert_layer_vram_mb = max(meta_layer_mb, 2372.0)
    elif is_38b:
        expert_layer_vram_mb = max(meta_layer_mb, size_layer_mb, 1450.0)
    else:
        expert_layer_vram_mb = max(256.0, max(meta_layer_mb, size_layer_mb))

    avail_moe_vram_mb = max(0.0, hw.vram_free_mb - dense_vram_mb - kv_budget_mb - safety_headroom_mb)

    # 3. Model-Based GPU Layer Placement (Stage 0.1)
    explicit_cap_int: Optional[int] = None
    if explicit_vram_layer_cap is not None:
        cap_s = str(explicit_vram_layer_cap).strip().lower()
        if cap_s.isdigit():
            explicit_cap_int = int(cap_s)

    if explicit_gpu_layers is not None:
        gpu_layers = max(0, int(explicit_gpu_layers))
        layer_policy = "explicit_user_override"
    elif not hw.has_gpu or hw.vram_free_mb <= 2000.0:
        gpu_layers = 0
        layer_policy = "cpu_only_or_low_vram"
    elif is_35b_class:
        fitted = int(avail_moe_vram_mb // expert_layer_vram_mb)
        # On 8 GiB class GPUs (~7000 MiB free), 35B (~356 MiB/layer) fits 6..8 MoE layers safely
        if fitted >= 6:
            gpu_layers = min(8, fitted)
        elif fitted >= 2:
            gpu_layers = fitted
        else:
            gpu_layers = max(0, fitted)
        layer_policy = f"auto_35b_fit_{gpu_layers}L"
    else:
        # 38B (75.8 GB, 48L) & 122B: protect 8 GiB VRAM from OOM by capping at <= 2 layers
        fitted = int(avail_moe_vram_mb // expert_layer_vram_mb)
        if is_38b and hw.vram_free_mb >= 6400.0:
            gpu_layers = 2
        else:
            gpu_layers = min(2, max(0, fitted))
        layer_policy = f"auto_large_safe_cap_{gpu_layers}L"

    if explicit_gpu_layers is None and explicit_cap_int is not None:
        gpu_layers = min(gpu_layers, explicit_cap_int)
        layer_policy = f"explicit_vram_cap_{gpu_layers}L"

    if explicit_vram_layer_cap is not None and str(explicit_vram_layer_cap).strip() != "":
        vram_layer_cap = str(explicit_vram_layer_cap).strip()
    elif is_38b:
        vram_layer_cap = "2"
    else:
        vram_layer_cap = "auto"

    estimated_vram_peak_mb = round(
        dense_vram_mb + kv_budget_mb + (gpu_layers * expert_layer_vram_mb) if hw.has_gpu else 0.0,
        1,
    )
    vram_cache_mb = (
        int(max(0.0, min(avail_moe_vram_mb - gpu_layers * expert_layer_vram_mb, 3200.0)))
        if (hw.has_gpu and gpu_layers > 0)
        else 0
    )

    # Heuristic TPS projection grounded in verified v0.16.1/v0.16.2 RTX 5060 + Ryzen 7 260 runs
    if not hw.has_gpu:
        est_tps = 18.0 if is_35b_class else 5.5
    elif is_35b_class:
        est_tps = round(82.0 + max(0, gpu_layers - 2) * 2.1, 1)
    elif is_38b:
        est_tps = 36.0 if gpu_layers >= 2 else 28.0
    elif is_122b:
        est_tps = 11.0
    else:
        est_tps = 45.0

    rationale = dict(base_cfg.rationale)
    rationale.update({
        "autotune_version": "0.16.2",
        "expert_layer_vram_mb": round(expert_layer_vram_mb, 1),
        "dense_vram_mb": round(dense_vram_mb, 1),
        "kv_budget_mb": round(kv_budget_mb, 1),
        "safety_headroom_mb": safety_headroom_mb,
        "avail_moe_vram_mb": round(avail_moe_vram_mb, 1),
        "estimated_vram_peak_mb": estimated_vram_peak_mb,
        "estimated_decode_tps": est_tps,
        "gpu_layer_policy": layer_policy,
        "kv_policy": kv_policy,
        "kv_cache": f"{kv_k}/{kv_v} ({kv_policy}, ctx={ctx_size})",
        "vram_layer_cap": vram_layer_cap,
        "speculative": "OFF (MTP disabled by default; opt-in only)",
    })

    return AutoTuneConfig(
        threads=base_cfg.threads,
        gpu_layers=gpu_layers,
        vram_cache_mb=vram_cache_mb if vram_cache_mb > 0 else base_cfg.vram_cache_mb,
        moe_cache_mib=base_cfg.moe_cache_mib,
        mtp_mode="off",
        mtp_draft_n=0,
        boost_level=base_cfg.boost_level,
        ctx_size=ctx_size,
        physical_map_path=base_cfg.physical_map_path,
        rationale=rationale,
        prompt_cache_mb=base_cfg.prompt_cache_mb,
        moe_cache_cpu_share=base_cfg.moe_cache_cpu_share,
        ffn_chunk_size=base_cfg.ffn_chunk_size,
        cache_type_k=kv_k,
        cache_type_v=kv_v,
        vram_layer_cap=vram_layer_cap,
        ram_lock_policy=base_cfg.ram_lock_policy,
    )


def scan_local_models(
    base_dir: Optional[Path] = None,
    extra_dirs: Optional[List[str]] = None,
    hw: Optional[HardwareProfile] = None,
    ctx_size: int = 8192,
) -> List[Dict[str, Any]]:
    """
    Discovers local GGUF models across repo models/, HuggingFace cache (Linux & Windows),
    shared NTFS /mnt/Ortak/huggingface/hub, and optional user-specified directories.
    Deduplicates multi-shard GGUF files (reporting shard 1 with aggregated size).
    """
    import re as _re

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    search_dirs: List[Path] = [
        root / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    if os.name == "nt":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            search_dirs.append(Path(userprofile) / ".cache" / "huggingface" / "hub")
    else:
        search_dirs.append(Path("/mnt/Ortak/huggingface/hub"))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        search_dirs.append(Path(hf_home) / "hub")
        search_dirs.append(Path(hf_home))

    if extra_dirs:
        for d in extra_dirs:
            if d:
                search_dirs.append(Path(d))

    # Deduplicate search dirs while preserving order
    unique_dirs: List[Path] = []
    seen_dirs: set[str] = set()
    for d in search_dirs:
        try:
            key = str(d.resolve()) if d.exists() else str(d)
        except OSError:
            key = str(d)
        if key not in seen_dirs and d.is_dir():
            seen_dirs.add(key)
            unique_dirs.append(d)

    if hw is None:
        hw = probe_hardware()

    shard_re = _re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", _re.IGNORECASE)
    discovered: List[Dict[str, Any]] = []
    seen_realpaths: set[str] = set()

    for sdir in unique_dirs:
        for dirpath, _dirnames, filenames in os.walk(sdir):
            for fname in sorted(filenames):
                if not fname.lower().endswith(".gguf"):
                    continue
                m_shard = shard_re.search(fname)
                if m_shard and int(m_shard.group(1)) != 1:
                    continue  # Skip non-first shards
                fpath = Path(dirpath) / fname
                try:
                    real_key = str(fpath.resolve())
                except OSError:
                    real_key = str(fpath)
                if real_key in seen_realpaths:
                    continue
                seen_realpaths.add(real_key)
                try:
                    topo = inspect_model_topology(fpath)
                    plan = compute_auto_plan(hw, topo, ctx_size=ctx_size, base_dir=root)
                    is_mtp_head = "mtp" in fname.lower() or "mtp" in str(fpath.parent).lower()
                    discovered.append({
                        "path": str(fpath),
                        "name": topo.model_name,
                        "filename": fname,
                        "architecture": topo.architecture,
                        "block_count": topo.block_count,
                        "expert_count": topo.expert_count,
                        "expert_used_count": topo.expert_used_count,
                        "file_size_gb": topo.file_size_gb,
                        "quant_type": topo.quant_type,
                        "is_moe": topo.is_moe,
                        "is_mtp_head": is_mtp_head,
                        "shards": int(m_shard.group(2)) if m_shard else 1,
                        "autotune": {
                            "threads": plan.threads,
                            "gpu_layers": plan.gpu_layers,
                            "vram_cache_mb": plan.vram_cache_mb,
                            "moe_cache_mib": plan.moe_cache_mib,
                            "cache_type_k": plan.cache_type_k,
                            "cache_type_v": plan.cache_type_v,
                            "vram_layer_cap": plan.vram_layer_cap,
                            "ram_lock_policy": plan.ram_lock_policy,
                            "mtp_mode": plan.mtp_mode,
                            "estimated_vram_peak_mb": plan.rationale.get("estimated_vram_peak_mb"),
                            "estimated_decode_tps": plan.rationale.get("estimated_decode_tps"),
                            "expert_layer_vram_mb": plan.rationale.get("expert_layer_vram_mb"),
                            "gpu_layer_policy": plan.rationale.get("gpu_layer_policy"),
                            "kv_policy": plan.rationale.get("kv_policy"),
                        },
                    })
                except Exception:
                    continue

    # Primary non-MTP models first, sorted by size descending
    discovered.sort(key=lambda item: (item["is_mtp_head"], -item["file_size_gb"], item["name"]))
    return discovered



def calibrate_moe_cache(
    engine_exe: str,
    model_path: str,
    hw: HardwareProfile,
    model: ModelTopology,
    cache_candidates_mb: Optional[List[int]] = None,
    share_candidates: Optional[List[float]] = None,
    n_predict: int = 32,
) -> Dict[str, Any]:
    """PR #30966-inspired calibration sweep across (vram_cache_mb, cpu_share) pairs."""
    import re, subprocess, time
    if cache_candidates_mb is None:
        max_safe = max(512, int(hw.vram_free_mb - (4900 if model.expert_count == 256 and model.block_count == 48 else 2900) - 768))
        cache_candidates_mb = sorted({min(max_safe, v) for v in (512, 1024, 2048, 3200) if v <= max_safe or v == 512})
    if share_candidates is None:
        share_candidates = [0.0, 0.25, 0.5, 0.75, 0.95]

    results = []
    best = None
    for cache_mb in cache_candidates_mb:
        for share in share_candidates:
            cmd = [
                engine_exe,
                "-m", model_path,
                "-ngl", "99",
                "--atlas-gpu-first",
                "--atlas-gpu-layers", "0",
                "--atlas-vram-cache", str(cache_mb),
                "--atlas-moe-cache-cpu-share", f"{share:.2f}",
                "-n", str(n_predict),
                "-p", "Write a short Python binary search function.",
            ]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                dt = time.perf_counter() - t0
                combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
                m_gen = re.search(r"\[ATLAS\]\s+Generate:\s+\d+\s+tokens\s+/\s+[0-9.]+\s+s\s+=\s+([0-9.]+)\s+TPS", combined)
                tps = float(m_gen.group(1)) if m_gen else ((n_predict / dt) if (proc.returncode == 0 and dt > 0) else 0.0)
                entry = {"vram_cache_mb": cache_mb, "cpu_share": share, "tps": round(tps, 3), "ok": proc.returncode == 0}
            except Exception as exc:
                entry = {"vram_cache_mb": cache_mb, "cpu_share": share, "tps": 0.0, "ok": False, "error": str(exc)}
            results.append(entry)
            if entry["ok"] and (best is None or entry["tps"] > best["tps"]):
                best = entry

    return {"best": best, "trials": results}


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
    parser.add_argument("--calibrate-moe-cache", action="store_true", help="Run PR #30966-style (vram_cache, cpu_share) sweep")
    _exe_name = "tendou-engine.exe" if os.name == "nt" else "tendou-engine"
    parser.add_argument("--engine-exe", type=str, default=str(Path(__file__).resolve().parents[2] / "releases" / "tendou-v0.16.2-stable" / "bin" / _exe_name), help="Path to tendou-engine binary")
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
        repo_root = Path(__file__).resolve().parents[2]
        p_linux = repo_root / "models" / "qwen3.8-gsq-rco-iq3_xxs" / "IQ3_XXS" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
        p1 = Path(r"C:\Users\Ali\WebProjects\tendou-launcher\models\qwen3.8-gsq-rco-iq3_xxs\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
        p2 = Path(r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
        m_path = str(p_linux if p_linux.exists() else (p1 if p1.exists() else p2))

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
        print(f"KV Cache Type    : K={cfg.cache_type_k}, V={cfg.cache_type_v}")
        print(f"CPU Share Policy : {cfg.moe_cache_cpu_share}")
        print(f"FFN Chunk Size   : {cfg.ffn_chunk_size}")
        print(f"VRAM Layer Cap   : {cfg.vram_layer_cap}")
        print(f"RAM Lock Policy  : {cfg.ram_lock_policy}")
        if args.calibrate_moe_cache:
            print("\nRunning PR #30966-style MoE Cache + CPU Share calibration sweep...")
            sweep = calibrate_moe_cache(args.engine_exe, m_path, hw, model)
            import json
            print(json.dumps(sweep, indent=2))
