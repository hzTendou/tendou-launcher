"""
Tendou Studio — Configuration Schema, CLI Flag Serializer & Profile Store.

Covers every CLI flag exposed by src/atlas/server.py (create_parser) with
typed Pydantic v2 validation, dynamic UI metadata, hardware safety guardrails,
Linux/Windows CLI preview serialization, and atomic JSON profile persistence.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = REPO_ROOT / "config" / "studio_profile.json"


class EngineSettingsModel(BaseModel):
    """Complete typed representation of all 50+ Tendou Launcher CLI flags."""

    # 1. Core & Server
    host: str = Field(default="127.0.0.1", description="Bind host address")
    port: int = Field(default=8000, ge=1, le=65535, description="HTTP server port")
    force: bool = Field(default=False, description="Force terminate process occupying target port")
    exe_path: Optional[str] = Field(default=None, description="Path to tendou-engine binary (auto-resolved if empty)")
    model: Optional[str] = Field(default=None, description="Preset alias ('35b' or '38b')")
    model_path: Optional[str] = Field(default=None, description="Absolute path to GGUF model file")
    threads: int = Field(default=8, ge=1, le=64, description="CPU worker threads (8 optimal on Zen 4/5)")
    ctx_size: int = Field(default=65536, ge=2048, le=262144, description="Context window size in tokens")
    mock: bool = Field(default=False, description="Run with mock backend (no model weights)")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum probability sampling threshold")

    # 2. GPU, VRAM & KV Cache
    gpu: bool = Field(default=True, description="Enable GPU-first CUDA acceleration (False = --cpu-only)")
    gpu_layers: Optional[int] = Field(default=None, ge=0, le=64, description="Full MoE layers in GPU VRAM (None = AutoTune 6-8 for 35B, 2 for 38B)")
    ngl: int = Field(default=99, ge=0, le=256, description="Attention/shared expert layers offloaded to GPU")
    vram_cache: int = Field(default=3200, ge=0, le=49152, description="Dynamic VRAM expert cache budget (MiB)")
    vram_layer_cap: Optional[str] = Field(default="auto", description="VRAM MoE layer residency cap ('auto', 'all', or integer)")
    ram_lock_policy: Optional[Literal["auto", "on", "off"]] = Field(default="auto", description="Host RAM mlock/VirtualLock policy")
    prompt_cache_mb: int = Field(default=768, ge=0, le=16384, description="Multi-turn PrefixCache RAM budget (MB)")
    fp8_kv: Optional[bool] = Field(default=None, description="Force FP8 E4M3 KV cache (None = Adaptive: q8_0 <=8K, f8_e4m3 >=16K)")
    cache_type_k: Optional[str] = Field(default=None, description="KV Cache K quant ('auto'/None, q8_0, f8_e4m3, q4_0, f16)")
    cache_type_v: Optional[str] = Field(default=None, description="KV Cache V quant ('auto'/None, q8_0, f8_e4m3, q4_0, f16)")
    cuda_sched: Literal["yield", "blocking", "spin", "auto"] = Field(default="yield", description="CUDA primary context scheduling mode")

    # 3. MoE Cache & Dynamic-K
    boost: bool = Field(default=True, description="Enable Dynamic-K boost pruning for higher decode TPS")
    boost_level: int = Field(default=2, ge=1, le=3, description="Boost level: 1 (Quality-safe), 2 (Turbo), 3 (Ultra K=1)")
    expert_k: int = Field(default=5, ge=1, le=12, description="Active MoE experts per token (--atlas-k)")
    moe_cache_mib: int = Field(default=2048, ge=0, le=65536, description="Tier-1 Split-MoE RAM/GPU cache budget (MiB)")
    moe_cache_cpu_share: Optional[str] = Field(default="auto", description="Adaptive MoE miss CPU share ('auto' or 0.0..1.0)")
    no_sync_cache: bool = Field(default=True, description="Zero-sync mapped router mailbox (--zero-sync-cache)")
    cascading_migration: bool = Field(default=True, description="Dynamic cascading 3-tier MoE working set migration")
    ffn_chunk_size: Optional[int] = Field(default=512, ge=0, le=4096, description="FFN/MoE prefill chunk size in tokens (0 = disabled)")

    # 4. Prefetch & Async I/O
    odmoe_lead: int = Field(default=2, ge=0, le=12, description="OD-MoE advance prefetch lead in layers")
    odmoe_quick_evict: bool = Field(default=True, description="OD-MoE rapid post-layer eviction")
    spice_conf_high: float = Field(default=0.65, ge=0.0, le=1.0, description="SPICE high confidence threshold for VRAM prefetch")
    spice_conf_mid: float = Field(default=0.30, ge=0.0, le=1.0, description="SPICE mid confidence threshold for RAM staging")
    prefetch_candidates: int = Field(default=10, ge=0, le=64, description="Maximum predicted experts per layer")
    tutti_async_io: bool = Field(default=False, description="Asynchronous expert page warming")
    tutti_queue_depth: int = Field(default=64, ge=1, le=512, description="Maximum queued expert chunks")
    readback_interval: int = Field(default=8, ge=1, le=256, description="Router readback interval in decode tokens")
    aggressive_prefetch: bool = Field(default=False, description="Experimental lead=4, 16-candidate async page-warming profile")
    p2_gpu_binding: bool = Field(default=False, description="Enable P2 dynamic GPU expert buffer binding")
    p2_vram_budget_mib: int = Field(default=0, ge=0, le=32768, description="P2 VRAM cache budget (MiB, 0 = default)")
    p3_async_transfer: bool = Field(default=False, description="Enable P3 async H2D transfer pipeline")
    p3_staging_slots: int = Field(default=4, ge=1, le=32, description="P3 pinned staging buffer slots")
    p3_queue_depth: int = Field(default=8, ge=1, le=64, description="P3 H2D queue depth")

    # 5. Speculative (MTP) & TLEP (Opt-In — MTP strictly OFF by default)
    mtp: Literal["off", "ram", "vram", "atlas", "auto", "hybrid", "pld"] = Field(
        default="off", description="Speculative decoding mode (default: off; opt-in only)"
    )
    mtp_mode: Optional[Literal["off", "ram", "vram", "atlas", "auto", "hybrid", "pld"]] = Field(
        default=None, description="Optional MTP mode override alias"
    )
    mtp_loc: Optional[Literal["ram", "vram", "hybrid", "auto"]] = Field(
        default=None, description="MTP draft tensor placement location"
    )
    mtp_draft_n: int = Field(default=2, ge=0, le=8, description="MTP speculative draft token horizon")
    mtp_path: Optional[str] = Field(default=None, description="Path to separate MTP draft GGUF model")
    tlep: bool = Field(default=False, description="Enable TLEP (Tendou Lookahead Execution Pipeline)")
    tlep_depth: int = Field(default=4, ge=1, le=16, description="TLEP maximum lookahead horizon depth")
    tlep_guard: float = Field(default=1.5, ge=0.0, le=20.0, description="TLEP minimum slack guard margin (ms)")
    tlep_bw: int = Field(default=128, ge=16, le=2048, description="TLEP predictive prefetch bandwidth budget (MB)")

    def normalized_kv(self) -> tuple[Optional[str], Optional[str]]:
        k = None if (self.cache_type_k in (None, "", "auto")) else self.cache_type_k
        v = None if (self.cache_type_v in (None, "", "auto")) else self.cache_type_v
        return k, v


SETTINGS_GROUPS: List[Dict[str, str]] = [
    {"id": "core", "label": "Core & Server", "description": "Model path, threading, context window, and HTTP binding"},
    {"id": "gpu_kv", "label": "GPU, VRAM & KV Cache", "description": "GPU layer placement, adaptive FP8/Q8 KV cache, and CUDA scheduling"},
    {"id": "moe_cache", "label": "MoE Cache & Dynamic-K", "description": "3-tier MoE residency, Boost Dynamic-K pruning, and FFN chunking"},
    {"id": "prefetch_io", "label": "Prefetch & Async I/O", "description": "OD-MoE lead, SPICE confidence thresholds, P2/P3 async pipelines"},
    {"id": "speculative_tlep", "label": "Speculative (MTP) & TLEP", "description": "Opt-in MTP drafting (default OFF) and TLEP lookahead controls"},
    {"id": "cli_profiles", "label": "CLI Preview & Profiles", "description": "Live Linux/Windows command serializer and profile management"},
]


def get_settings_schema(ram_total_gb: float = 16.0, vram_total_mb: float = 8151.0) -> Dict[str, Any]:
    """Returns dynamic UI metadata covering every EngineSettingsModel field."""
    low_ram_warn = (
        f"System RAM is {ram_total_gb:.1f} GiB (<= 16 GiB). Enabling Neural MTP adds ~2.36 GB draft weights "
        "and can cause page-cache thrashing or OOM. Keep 'off' unless benchmarking opt-in MTP."
        if ram_total_gb <= 16.5
        else None
    )

    fields: List[Dict[str, Any]] = [
        # Core & Server
        {"key": "model_path", "label": "Model GGUF Path", "cli_flag": "--model-path", "group": "core", "type": "string", "nullable": True, "description": "Absolute path to primary .gguf model file"},
        {"key": "model", "label": "Model Alias Preset", "cli_flag": "--model", "group": "core", "type": "select", "choices": ["", "35b", "38b"], "nullable": True, "description": "Short alias ('35b' or '38b')"},
        {"key": "threads", "label": "CPU Threads", "cli_flag": "--threads", "group": "core", "type": "slider", "min": 1, "max": 32, "step": 1, "description": "Worker threads (8 recommended on 8C/16T Zen 4/5)"},
        {"key": "ctx_size", "label": "Context Window", "cli_flag": "--ctx-size", "group": "core", "type": "number", "min": 2048, "max": 262144, "step": 2048, "quick_values": [4096, 8192, 16384, 32768, 65536, 131072, 262144], "description": "Total KV context tokens (<=8K uses q8_0, >=16K uses f8_e4m3 automatically)"},
        {"key": "host", "label": "Bind Host", "cli_flag": "--host", "group": "core", "type": "string", "description": "Server bind address (127.0.0.1 or 0.0.0.0)"},
        {"key": "port", "label": "Server Port", "cli_flag": "--port", "group": "core", "type": "number", "min": 1, "max": 65535, "step": 1, "description": "HTTP port for Studio Web UI and OpenAI/Anthropic API"},
        {"key": "min_p", "label": "Min-P Threshold", "cli_flag": "--min-p", "group": "core", "type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "description": "Minimum probability sampling cutoff"},
        {"key": "exe_path", "label": "Engine Binary Path", "cli_flag": "--exe-path", "group": "core", "type": "string", "nullable": True, "description": "Override path to tendou-engine / tendou-engine.exe"},
        {"key": "force", "label": "Force Takeover Port", "cli_flag": "--force", "group": "core", "type": "boolean", "description": "Terminate existing process listening on target port"},
        {"key": "mock", "label": "Mock Backend Mode", "cli_flag": "--mock", "group": "core", "type": "boolean", "description": "Run without native engine binary or model weights"},

        # GPU, VRAM & KV Cache
        {"key": "gpu", "label": "GPU Acceleration", "cli_flag": "--gpu / --cpu-only", "group": "gpu_kv", "type": "boolean", "description": "Enable CUDA GPU-first acceleration (disable for --cpu-only)"},
        {"key": "gpu_layers", "label": "GPU MoE Layers", "cli_flag": "--gpu-layers", "group": "gpu_kv", "type": "number", "min": 0, "max": 48, "step": 1, "nullable": True, "placeholder": "AutoTune (35B: 6-8, 38B: 2)", "description": "Leave empty for v0.16.2 AutoTune, or set explicit integer override"},
        {"key": "ngl", "label": "Dense GPU Layers (-ngl)", "cli_flag": "--ngl", "group": "gpu_kv", "type": "number", "min": 0, "max": 256, "step": 1, "description": "Attention & shared expert layers offloaded to GPU (99 = all)"},
        {"key": "vram_cache", "label": "Dynamic VRAM Cache (MiB)", "cli_flag": "--vram-cache", "group": "gpu_kv", "type": "slider", "min": 0, "max": max(4096, int(vram_total_mb)), "step": 128, "description": "Tier-0 dynamic VRAM expert cache budget"},
        {"key": "vram_layer_cap", "label": "VRAM Layer Residency Cap", "cli_flag": "--vram-layer-cap", "group": "gpu_kv", "type": "string", "nullable": True, "quick_values": ["auto", "0", "2", "6", "8", "all"], "description": "'auto', 'all', or integer cap on Tier-0 MoE cached layers"},
        {"key": "ram_lock_policy", "label": "RAM Pinning Policy", "cli_flag": "--ram-lock-policy", "group": "gpu_kv", "type": "select", "choices": ["auto", "on", "off"], "nullable": True, "description": "Host RAM mlock/VirtualLock policy ('auto' protects mmap page cache on <=16GB RAM)"},
        {"key": "prompt_cache_mb", "label": "PrefixCache Budget (MB)", "cli_flag": "--prompt-cache-mb", "group": "gpu_kv", "type": "slider", "min": 0, "max": 4096, "step": 64, "description": "16-slot hierarchical KV tree prefix cache budget"},
        {"key": "fp8_kv", "label": "FP8 E4M3 KV Override", "cli_flag": "--fp8-kv", "group": "gpu_kv", "type": "tribool", "nullable": True, "description": "Auto (None): q8_0 <=8K, f8_e4m3 >=16K; True/False forces override"},
        {"key": "cache_type_k", "label": "KV Cache K Quant", "cli_flag": "--cache-type-k", "group": "gpu_kv", "type": "select", "choices": ["auto", "q8_0", "f8_e4m3", "q4_0", "f16"], "nullable": True, "description": "Key cache quantization type"},
        {"key": "cache_type_v", "label": "KV Cache V Quant", "cli_flag": "--cache-type-v", "group": "gpu_kv", "type": "select", "choices": ["auto", "q8_0", "f8_e4m3", "q4_0", "f16"], "nullable": True, "description": "Value cache quantization type"},
        {"key": "cuda_sched", "label": "CUDA Context Scheduler", "cli_flag": "--cuda-sched", "group": "gpu_kv", "type": "select", "choices": ["yield", "blocking", "spin", "auto"], "description": "CUDA primary context synchronization mode"},

        # MoE Cache & Dynamic-K
        {"key": "boost", "label": "Dynamic-K Boost Mode", "cli_flag": "--boost / --no-boost", "group": "moe_cache", "type": "boolean", "description": "Enable synchronized Dynamic-K expert pruning for peak decode TPS"},
        {"key": "boost_level", "label": "Boost Level", "cli_flag": "--boost-level", "group": "moe_cache", "type": "select", "choices": [1, 2, 3], "description": "1 = Quality-Safe (4/3/2), 2 = Turbo (4/3/2 + outer K=1), 3 = Ultra Max TPS"},
        {"key": "expert_k", "label": "Active Experts (--atlas-k)", "cli_flag": "--atlas-k", "group": "moe_cache", "type": "slider", "min": 1, "max": 10, "step": 1, "description": "Active MoE experts evaluated per token"},
        {"key": "moe_cache_mib", "label": "MoE Host Cache (MiB)", "cli_flag": "--moe-cache-mib", "group": "moe_cache", "type": "slider", "min": 0, "max": 16384, "step": 256, "description": "Tier-1 Split-MoE RAM cache budget"},
        {"key": "moe_cache_cpu_share", "label": "MoE Miss CPU Share", "cli_flag": "--moe-cache-cpu-share", "group": "moe_cache", "type": "string", "nullable": True, "quick_values": ["auto", "0.0", "0.25", "0.50", "0.75", "1.0"], "description": "'auto' (closed-loop cost model) or fixed 0.0..1.0"},
        {"key": "no_sync_cache", "label": "Zero-Sync Router Mailbox", "cli_flag": "--zero-sync-cache", "group": "moe_cache", "type": "boolean", "description": "Lockless mapped router mailbox avoiding CUDA stream stalls"},
        {"key": "cascading_migration", "label": "Cascading 3-Tier Migration", "cli_flag": "--cascading-migration", "group": "moe_cache", "type": "boolean", "description": "Dynamic working-set migration across VRAM, RAM, and NVMe"},
        {"key": "ffn_chunk_size", "label": "FFN Prefill Chunk Size", "cli_flag": "--ffn-chunk-size", "group": "moe_cache", "type": "select", "choices": [0, 256, 512, 1024], "nullable": True, "description": "Token chunk size for MoE prefill memory efficiency (0 = off)"},

        # Prefetch & Async I/O
        {"key": "odmoe_lead", "label": "OD-MoE Prefetch Lead", "cli_flag": "--odmoe-lead", "group": "prefetch_io", "type": "slider", "min": 0, "max": 8, "step": 1, "description": "Advance expert prefetch lead in layers"},
        {"key": "odmoe_quick_evict", "label": "OD-MoE Quick Evict", "cli_flag": "--odmoe-quick-evict", "group": "prefetch_io", "type": "boolean", "description": "Rapid post-layer cache eviction"},
        {"key": "spice_conf_high", "label": "SPICE High Confidence", "cli_flag": "--spice-conf-high", "group": "prefetch_io", "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "description": "Threshold for VRAM prefetch promotion"},
        {"key": "spice_conf_mid", "label": "SPICE Mid Confidence", "cli_flag": "--spice-conf-mid", "group": "prefetch_io", "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "description": "Threshold for RAM staging"},
        {"key": "prefetch_candidates", "label": "Prefetch Candidates", "cli_flag": "--prefetch-candidates", "group": "prefetch_io", "type": "slider", "min": 0, "max": 32, "step": 1, "description": "Max predicted experts per layer"},
        {"key": "tutti_async_io", "label": "Tutti Async Page Warm", "cli_flag": "--tutti-async-io", "group": "prefetch_io", "type": "boolean", "description": "Asynchronous expert page warming"},
        {"key": "tutti_queue_depth", "label": "Tutti Queue Depth", "cli_flag": "--tutti-queue-depth", "group": "prefetch_io", "type": "slider", "min": 8, "max": 256, "step": 8, "description": "Max queued expert prefetch chunks"},
        {"key": "readback_interval", "label": "Router Readback Interval", "cli_flag": "--readback-interval", "group": "prefetch_io", "type": "slider", "min": 1, "max": 128, "step": 1, "description": "Decode tokens between router statistics readbacks"},
        {"key": "aggressive_prefetch", "label": "Aggressive Prefetch Profile", "cli_flag": "--aggressive-prefetch", "group": "prefetch_io", "type": "boolean", "description": "Experimental lead=4, 16-candidate async warming"},
        {"key": "p2_gpu_binding", "label": "P2 Dynamic GPU Binding", "cli_flag": "--p2-gpu-binding", "group": "prefetch_io", "type": "boolean", "description": "Enable P2 dynamic GPU expert buffer binding"},
        {"key": "p2_vram_budget_mib", "label": "P2 VRAM Budget (MiB)", "cli_flag": "--p2-vram-budget-mib", "group": "prefetch_io", "type": "number", "min": 0, "max": 16384, "step": 128, "description": "P2 VRAM cache budget (0 = default)"},
        {"key": "p3_async_transfer", "label": "P3 Async H2D Pipeline", "cli_flag": "--p3-async-transfer", "group": "prefetch_io", "type": "boolean", "description": "Enable P3 pinned staging async H2D transfers"},
        {"key": "p3_staging_slots", "label": "P3 Staging Slots", "cli_flag": "--p3-staging-slots", "group": "prefetch_io", "type": "slider", "min": 1, "max": 16, "step": 1, "description": "Pinned host staging slots"},
        {"key": "p3_queue_depth", "label": "P3 H2D Queue Depth", "cli_flag": "--p3-queue-depth", "group": "prefetch_io", "type": "slider", "min": 1, "max": 32, "step": 1, "description": "Max in-flight H2D transfers"},

        # Speculative (MTP) & TLEP
        {"key": "mtp", "label": "MTP Speculative Mode", "cli_flag": "--mtp", "group": "speculative_tlep", "type": "select", "choices": ["off", "pld", "ram", "vram", "hybrid", "atlas", "auto"], "safety_warning": low_ram_warn, "description": "Default OFF. Opt-in PLD (zero VRAM) or Neural MTP drafting"},
        {"key": "mtp_mode", "label": "MTP Mode Alias", "cli_flag": "--mtp-mode", "group": "speculative_tlep", "type": "select", "choices": ["", "off", "pld", "ram", "vram", "hybrid", "atlas", "auto"], "nullable": True, "description": "Optional alias overriding --mtp"},
        {"key": "mtp_loc", "label": "MTP Placement Location", "cli_flag": "--mtp-loc", "group": "speculative_tlep", "type": "select", "choices": ["", "ram", "vram", "hybrid", "auto"], "nullable": True, "description": "Tensor placement for Neural MTP head"},
        {"key": "mtp_draft_n", "label": "MTP Draft Horizon", "cli_flag": "--mtp-draft-n", "group": "speculative_tlep", "type": "slider", "min": 0, "max": 6, "step": 1, "description": "Draft tokens per verification step"},
        {"key": "mtp_path", "label": "MTP Draft GGUF Path", "cli_flag": "--mtp-path", "group": "speculative_tlep", "type": "string", "nullable": True, "description": "Optional separate draft model GGUF path"},
        {"key": "tlep", "label": "TLEP Lookahead Pipeline", "cli_flag": "--tlep", "group": "speculative_tlep", "type": "boolean", "description": "Enable Tendou Lookahead Execution Pipeline"},
        {"key": "tlep_depth", "label": "TLEP Max Depth", "cli_flag": "--tlep-depth", "group": "speculative_tlep", "type": "slider", "min": 1, "max": 12, "step": 1, "description": "Maximum lookahead horizon"},
        {"key": "tlep_guard", "label": "TLEP Slack Guard (ms)", "cli_flag": "--tlep-guard", "group": "speculative_tlep", "type": "slider", "min": 0.0, "max": 10.0, "step": 0.25, "description": "Minimum slack guard margin"},
        {"key": "tlep_bw", "label": "TLEP Bandwidth Budget (MB)", "cli_flag": "--tlep-bw", "group": "speculative_tlep", "type": "slider", "min": 32, "max": 1024, "step": 32, "description": "Predictive prefetch bandwidth budget"},
    ]

    return {
        "groups": SETTINGS_GROUPS,
        "fields": fields,
        "total_flags": len(fields),
        "ram_total_gb": round(ram_total_gb, 2),
        "vram_total_mb": round(vram_total_mb, 1),
    }


def get_presets() -> List[Dict[str, Any]]:
    """Returns built-in Enterprise Studio presets (MTP always OFF by default)."""
    return [
        {
            "id": "autotune_optimal",
            "name": "AutoTune Optimal",
            "badge": "Recommended",
            "description": "v0.16.2 Model-Aware AutoTune: 6-8 GPU layers on 35B, safe 2 layers on 38B, adaptive Q8_0/FP8 KV, Boost L2, MTP OFF.",
            "overrides": {
                "threads": 8,
                "gpu": True,
                "gpu_layers": None,
                "vram_layer_cap": "auto",
                "ram_lock_policy": "auto",
                "ctx_size": 65536,
                "fp8_kv": None,
                "cache_type_k": None,
                "cache_type_v": None,
                "boost": True,
                "boost_level": 2,
                "expert_k": 5,
                "ffn_chunk_size": 512,
                "no_sync_cache": True,
                "mtp": "off",
            },
        },
        {
            "id": "max_decode_speed",
            "name": "Max Decode Speed",
            "badge": "Peak TPS",
            "description": "8K context with Q8_0 FlashAttention, Boost L2, AutoTune GPU layers, zero-sync cache for maximum decode tok/s.",
            "overrides": {
                "threads": 8,
                "gpu": True,
                "gpu_layers": None,
                "ctx_size": 8192,
                "fp8_kv": None,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "boost": True,
                "boost_level": 2,
                "expert_k": 5,
                "cuda_sched": "yield",
                "no_sync_cache": True,
                "mtp": "off",
            },
        },
        {
            "id": "long_context",
            "name": "Long Context (64K–256K)",
            "badge": "FP8 E4M3 KV",
            "description": "131K context window with FP8 E4M3 KV cache, FFN 512 chunking, and 1024 MB PrefixCache for large multi-turn repos.",
            "overrides": {
                "threads": 8,
                "gpu": True,
                "gpu_layers": None,
                "ctx_size": 131072,
                "fp8_kv": True,
                "cache_type_k": "f8_e4m3",
                "cache_type_v": "f8_e4m3",
                "prompt_cache_mb": 1024,
                "ffn_chunk_size": 512,
                "boost": True,
                "boost_level": 2,
                "mtp": "off",
            },
        },
        {
            "id": "low_vram_safe",
            "name": "Ultra-Low VRAM Safe",
            "badge": "8GB Guard",
            "description": "Conservative 1-2 GPU layers, FP8 KV, ram-lock auto, zero risk of desktop compositor OOM on 8 GB GPUs.",
            "overrides": {
                "threads": 8,
                "gpu": True,
                "gpu_layers": 1,
                "vram_cache": 1536,
                "vram_layer_cap": "1",
                "ram_lock_policy": "auto",
                "ctx_size": 32768,
                "fp8_kv": True,
                "cache_type_k": "f8_e4m3",
                "cache_type_v": "f8_e4m3",
                "boost": True,
                "boost_level": 1,
                "expert_k": 5,
                "mtp": "off",
            },
        },
    ]


def serialize_to_cli_args(
    cfg: EngineSettingsModel,
    platform: Literal["linux", "windows"] = "linux",
) -> Dict[str, Any]:
    """Converts EngineSettingsModel into CLI arg list, copyable shell command, and backend kwargs."""
    args: List[str] = [
        "--host", str(cfg.host),
        "--port", str(cfg.port),
    ]
    if cfg.force:
        args.append("--force")
    if cfg.mock:
        args.append("--mock")
    if cfg.exe_path:
        args.extend(["--exe-path", str(cfg.exe_path)])
    if cfg.model:
        args.extend(["--model", str(cfg.model)])
    elif cfg.model_path:
        args.extend(["--model-path", str(cfg.model_path)])

    args.extend([
        "--threads", str(cfg.threads),
        "--ctx-size", str(cfg.ctx_size),
        "--prompt-cache-mb", str(cfg.prompt_cache_mb),
        "--atlas-k", str(cfg.expert_k),
        "--boost" if cfg.boost else "--no-boost",
        "--boost-level", str(cfg.boost_level),
    ])

    if cfg.gpu:
        args.append("--gpu")
        if cfg.gpu_layers is not None:
            args.extend(["--gpu-layers", str(cfg.gpu_layers)])
        args.extend([
            "--ngl", str(cfg.ngl),
            "--vram-cache", str(cfg.vram_cache),
            "--cuda-sched", str(cfg.cuda_sched),
        ])
    else:
        args.append("--cpu-only")

    ck, cv = cfg.normalized_kv()
    if cfg.fp8_kv is True:
        args.append("--fp8-kv")
    elif cfg.fp8_kv is False:
        args.append("--no-fp8-kv")
    if ck:
        args.extend(["--cache-type-k", ck])
    if cv:
        args.extend(["--cache-type-v", cv])

    if cfg.vram_layer_cap not in (None, ""):
        args.extend(["--vram-layer-cap", str(cfg.vram_layer_cap)])
    if cfg.ram_lock_policy not in (None, ""):
        args.extend(["--ram-lock-policy", str(cfg.ram_lock_policy)])

    args.extend([
        "--moe-cache-mib", str(cfg.moe_cache_mib),
        "--min-p", str(cfg.min_p),
        "--zero-sync-cache" if cfg.no_sync_cache else "--no-zero-sync-cache",
        "--cascading-migration" if cfg.cascading_migration else "--no-cascading-migration",
    ])
    if cfg.moe_cache_cpu_share not in (None, ""):
        args.extend(["--moe-cache-cpu-share", str(cfg.moe_cache_cpu_share)])
    if cfg.ffn_chunk_size is not None:
        args.extend(["--ffn-chunk-size", str(cfg.ffn_chunk_size)])

    args.extend([
        "--odmoe-lead", str(cfg.odmoe_lead),
        "--odmoe-quick-evict" if cfg.odmoe_quick_evict else "--no-odmoe-quick-evict",
        "--spice-conf-high", str(cfg.spice_conf_high),
        "--spice-conf-mid", str(cfg.spice_conf_mid),
        "--tutti-async-io" if cfg.tutti_async_io else "--no-tutti-async-io",
        "--readback-interval", str(cfg.readback_interval),
        "--prefetch-candidates", str(cfg.prefetch_candidates),
        "--tutti-queue-depth", str(cfg.tutti_queue_depth),
    ])
    if cfg.aggressive_prefetch:
        args.append("--aggressive-prefetch")
    if cfg.p2_gpu_binding:
        args.append("--p2-gpu-binding")
        if cfg.p2_vram_budget_mib > 0:
            args.extend(["--p2-vram-budget-mib", str(cfg.p2_vram_budget_mib)])
    if cfg.p3_async_transfer:
        args.extend([
            "--p3-async-transfer",
            "--p3-staging-slots", str(cfg.p3_staging_slots),
            "--p3-queue-depth", str(cfg.p3_queue_depth),
        ])

    eff_mtp = cfg.mtp_mode if cfg.mtp_mode not in (None, "") else cfg.mtp
    args.extend(["--mtp", str(eff_mtp)])
    if eff_mtp != "off":
        args.extend(["--mtp-draft-n", str(cfg.mtp_draft_n)])
        if cfg.mtp_loc:
            args.extend(["--mtp-loc", str(cfg.mtp_loc)])
        if cfg.mtp_path:
            args.extend(["--mtp-path", str(cfg.mtp_path)])

    if cfg.tlep:
        args.extend([
            "--tlep",
            "--tlep-depth", str(cfg.tlep_depth),
            "--tlep-guard", str(cfg.tlep_guard),
            "--tlep-bw", str(cfg.tlep_bw),
        ])

    if platform == "windows":
        def _win_q(s: str) -> str:
            return f'"{s}"' if (" " in s or "\\" in s) else s
        cmd_str = "start_server.cmd " + " ".join(_win_q(a) for a in args)
    else:
        cmd_str = "./start_server.sh " + " ".join(shlex.quote(a) for a in args)

    backend_kwargs: Dict[str, Any] = {
        "threads": cfg.threads,
        "boost": cfg.boost,
        "boost_level": cfg.boost_level,
        "expert_k": cfg.expert_k,
        "gpu": cfg.gpu,
        "gpu_layers": cfg.gpu_layers,
        "ngl": cfg.ngl,
        "vram_cache": cfg.vram_cache,
        "ctx_size": cfg.ctx_size,
        "odmoe_lead": 4 if cfg.aggressive_prefetch else cfg.odmoe_lead,
        "odmoe_quick_evict": cfg.odmoe_quick_evict,
        "spice_conf_high": cfg.spice_conf_high,
        "spice_conf_mid": cfg.spice_conf_mid,
        "tutti_async_io": True if cfg.aggressive_prefetch else cfg.tutti_async_io,
        "readback_interval": cfg.readback_interval,
        "prefetch_candidates": 16 if cfg.aggressive_prefetch else cfg.prefetch_candidates,
        "tutti_queue_depth": cfg.tutti_queue_depth,
        "p2_gpu_binding": cfg.p2_gpu_binding,
        "p2_vram_budget_mib": cfg.p2_vram_budget_mib,
        "p3_async_transfer": cfg.p3_async_transfer,
        "p3_staging_slots": cfg.p3_staging_slots,
        "p3_queue_depth": cfg.p3_queue_depth,
        "prompt_cache_mb": cfg.prompt_cache_mb,
        "cuda_sched": cfg.cuda_sched,
        "cache_type_k": ck,
        "cache_type_v": cv,
        "mtp": eff_mtp,
        "mtp_draft_n": cfg.mtp_draft_n,
        "mtp_path": cfg.mtp_path or None,
        "mtp_loc": cfg.mtp_loc or None,
        "moe_cache_mib": cfg.moe_cache_mib,
        "min_p": cfg.min_p,
        "tlep": cfg.tlep,
        "tlep_max_depth": cfg.tlep_depth,
        "tlep_min_guard_ms": cfg.tlep_guard,
        "tlep_pf_bw_budget_mb": cfg.tlep_bw,
        "cascading_migration": cfg.cascading_migration,
        "moe_cache_cpu_share": cfg.moe_cache_cpu_share or None,
        "ffn_chunk_size": cfg.ffn_chunk_size,
        "no_sync_cache": cfg.no_sync_cache,
        "fp8_kv": cfg.fp8_kv,
        "vram_layer_cap": cfg.vram_layer_cap or None,
        "ram_lock_policy": cfg.ram_lock_policy or None,
    }
    if cfg.exe_path:
        backend_kwargs["exe_path"] = cfg.exe_path
    if cfg.model_path:
        backend_kwargs["model_path"] = cfg.model_path

    return {
        "args": args,
        "command": cmd_str,
        "platform": platform,
        "backend_kwargs": backend_kwargs,
    }


def settings_from_backend(backend: Any, host: str = "127.0.0.1", port: int = 8000) -> EngineSettingsModel:
    """Extracts an EngineSettingsModel snapshot from a running backend instance."""
    is_mock = backend.__class__.__name__ == "MockAtlasBackend"
    data: Dict[str, Any] = {"host": host, "port": port, "mock": is_mock}
    mapping = [
        "exe_path", "model_path", "threads", "boost", "boost_level", "expert_k",
        "gpu", "gpu_layers", "ngl", "vram_cache", "ctx_size", "odmoe_lead",
        "odmoe_quick_evict", "spice_conf_high", "spice_conf_mid", "tutti_async_io",
        "readback_interval", "prefetch_candidates", "tutti_queue_depth",
        "p2_gpu_binding", "p2_vram_budget_mib", "p3_async_transfer",
        "p3_staging_slots", "p3_queue_depth", "prompt_cache_mb", "cuda_sched",
        "cache_type_k", "cache_type_v", "mtp", "mtp_draft_n", "mtp_path",
        "mtp_loc", "moe_cache_mib", "min_p", "tlep", "cascading_migration",
        "moe_cache_cpu_share", "ffn_chunk_size", "no_sync_cache", "fp8_kv",
        "vram_layer_cap", "ram_lock_policy",
    ]
    for attr in mapping:
        if hasattr(backend, attr):
            val = getattr(backend, attr)
            if val is not None or attr in ("gpu_layers", "fp8_kv", "cache_type_k", "cache_type_v", "mtp_path", "mtp_loc", "vram_layer_cap", "ram_lock_policy", "moe_cache_cpu_share", "ffn_chunk_size"):
                data[attr] = val
    if hasattr(backend, "tlep_max_depth"):
        data["tlep_depth"] = getattr(backend, "tlep_max_depth")
    if hasattr(backend, "tlep_min_guard_ms"):
        data["tlep_guard"] = getattr(backend, "tlep_min_guard_ms")
    if hasattr(backend, "tlep_pf_bw_budget_mb"):
        data["tlep_bw"] = getattr(backend, "tlep_pf_bw_budget_mb")
    return EngineSettingsModel(**data)


def compute_settings_diff(old: EngineSettingsModel, new: EngineSettingsModel) -> List[Dict[str, Any]]:
    """Computes field-level diff between two settings models for Hot-Swap UI confirmation."""
    d_old = old.model_dump()
    d_new = new.model_dump()
    diffs: List[Dict[str, Any]] = []
    for k, v_new in d_new.items():
        v_old = d_old.get(k)
        if v_old != v_new:
            diffs.append({"key": k, "old": v_old, "new": v_new})
    return diffs


class ProfileStore:
    """Atomic JSON persistence for Tendou Studio engine profiles."""

    def __init__(self, profile_path: Path = DEFAULT_PROFILE_PATH):
        self.profile_path = Path(profile_path)

    def load(self, fallback: Optional[EngineSettingsModel] = None) -> EngineSettingsModel:
        if self.profile_path.is_file():
            try:
                raw = json.loads(self.profile_path.read_text(encoding="utf-8"))
                return EngineSettingsModel(**raw)
            except Exception:
                pass
        return fallback if fallback is not None else EngineSettingsModel()

    def save(self, cfg: EngineSettingsModel) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.profile_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(cfg.model_dump(), indent=2), encoding="utf-8")
        os.replace(tmp_path, self.profile_path)
