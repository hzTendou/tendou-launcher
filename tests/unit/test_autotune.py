"""Unit tests for Universal Hardware & Topology Auto-Tuning Engine (src/atlas/autotune.py)."""

import pytest
from pathlib import Path
from src.atlas.autotune import (
    HardwareProfile,
    ModelTopology,
    probe_hardware,
    inspect_model_topology,
    solve_optimal_configuration,
)


def test_probe_hardware():
    hw = probe_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.cpu_cores_physical >= 1
    assert hw.cpu_cores_logical >= hw.cpu_cores_physical
    assert hw.ram_total_gb > 0.0


def test_solve_optimal_configuration_35b():
    hw = HardwareProfile(
        has_gpu=True,
        gpu_name="NVIDIA RTX 5060 Laptop",
        vram_total_mb=8151.0,
        vram_free_mb=7000.0,
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        has_avx512_vnni=True,
        has_avx512_vbmi=True,
        has_avx2=True,
        ram_total_gb=16.0,
        ram_avail_gb=8.0,
        is_windows=True,
        directstorage_supported=True,
    )

    model_35b = ModelTopology(
        model_path="dummy_35b.gguf",
        model_name="Qwen3.6-35B-A3B-UD-IQ4_XS",
        architecture="qwen2moe",
        block_count=40,
        expert_count=256,
        expert_used_count=8,
        intermediate_size=512,
        file_size_gb=16.51,
        quant_type="IQ4_XS",
        expert_slice_bytes=1458176,
        is_moe=True,
    )

    cfg = solve_optimal_configuration(hw, model_35b)
    assert cfg.threads == 8
    assert cfg.gpu_layers in (0, 2)
    assert cfg.vram_cache_mb > 0
    assert cfg.moe_cache_mib > 0
    assert cfg.mtp_mode == "pld"
    assert cfg.mtp_draft_n in (3, 4)


def test_solve_optimal_configuration_qwen38():
    hw = HardwareProfile(
        has_gpu=True,
        gpu_name="NVIDIA RTX 5060 Laptop",
        vram_total_mb=8151.0,
        vram_free_mb=7200.0,
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        has_avx512_vnni=True,
        has_avx512_vbmi=True,
        has_avx2=True,
        ram_total_gb=16.0,
        ram_avail_gb=8.0,
        is_windows=True,
        directstorage_supported=True,
    )

    model_38 = ModelTopology(
        model_path="dummy_qwen38.gguf",
        model_name="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS",
        architecture="qwen2moe",
        block_count=48,
        expert_count=512,
        expert_used_count=10,
        intermediate_size=512,
        file_size_gb=43.81,
        quant_type="IQ3_XXS",
        expert_slice_bytes=1253376,
        is_moe=True,
    )

    cfg = solve_optimal_configuration(hw, model_38)
    assert cfg.threads == 8
    assert cfg.gpu_layers in (0, 2)
    assert cfg.vram_cache_mb > 0
    assert cfg.moe_cache_mib > 0
    assert cfg.mtp_mode == "pld"
