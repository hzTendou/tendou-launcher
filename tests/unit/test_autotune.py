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
    assert cfg.moe_cache_cpu_share == "auto"
    assert cfg.ffn_chunk_size == 512
    assert cfg.cache_type_k == "f8_e4m3"
    assert cfg.cache_type_v == "f8_e4m3"
    assert cfg.vram_layer_cap == "auto"
    assert cfg.ram_lock_policy == "auto"
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
    assert cfg.moe_cache_cpu_share == "auto"
    assert cfg.ffn_chunk_size == 512
    assert cfg.cache_type_k == "f8_e4m3"
    assert cfg.cache_type_v == "f8_e4m3"
    assert cfg.vram_layer_cap == "auto"
    assert cfg.ram_lock_policy == "auto"


def test_phase18_server_defaults_and_cli_knobs(tmp_path, monkeypatch):
    from src.atlas.server import AtlasSubprocessBackend, create_parser

    monkeypatch.delenv("TENDOU_MTP_MODE", raising=False)
    monkeypatch.delenv("TENDOU_MTP_PATH", raising=False)
    monkeypatch.delenv("ATLAS_MTP_PATH", raising=False)
    monkeypatch.delenv("TENDOU_VRAM_LAYER_CAP", raising=False)
    monkeypatch.delenv("TENDOU_RAM_LOCK_POLICY", raising=False)

    model_35b = tmp_path / "Qwen3.6-35B-A3B.gguf"
    model_35b.touch()

    # 1. Default MTP MUST stay OFF (never auto-enabled)
    backend_default = AtlasSubprocessBackend(exe_path="/bin/true", model_path=str(model_35b))
    assert backend_default.mtp == "off"
    cmd_default = backend_default._build_cmd()
    idx_mtp = cmd_default.index("--atlas-mtp")
    assert cmd_default[idx_mtp + 1] == "off"
    assert "--atlas-mtp-path" not in cmd_default

    parsed_default = create_parser().parse_args([])
    assert parsed_default.mtp == "off"

    # 2. Explicit mtp="ram" with mtp_path emits --atlas-mtp ram --atlas-mtp-path ...
    fake_mtp = tmp_path / "draft-mtp.gguf"
    fake_mtp.touch()
    backend_ram = AtlasSubprocessBackend(
        exe_path="/bin/true",
        model_path=str(model_35b),
        mtp="ram",
        mtp_path=str(fake_mtp),
    )
    cmd_ram = backend_ram._build_cmd()
    assert cmd_ram[cmd_ram.index("--atlas-mtp") + 1] == "ram"
    assert cmd_ram[cmd_ram.index("--atlas-mtp-path") + 1] == str(fake_mtp)

    # Verify all MTP modes forward --atlas-mtp-path
    for mode in ("ram", "vram", "hybrid", "atlas", "auto"):
        b = AtlasSubprocessBackend(
            exe_path="/bin/true",
            model_path=str(model_35b),
            mtp=mode,
            mtp_path=str(fake_mtp),
        )
        c = b._build_cmd()
        assert c[c.index("--atlas-mtp") + 1] == mode
        assert c[c.index("--atlas-mtp-path") + 1] == str(fake_mtp)

    # 3. vram_layer_cap and ram_lock_policy forward --atlas-vram-layer-cap & --atlas-ram-lock-policy
    backend_knobs = AtlasSubprocessBackend(
        exe_path="/bin/true",
        model_path=str(model_35b),
        vram_layer_cap="2",
        ram_lock_policy="on",
    )
    cmd_knobs = backend_knobs._build_cmd()
    assert cmd_knobs[cmd_knobs.index("--atlas-vram-layer-cap") + 1] == "2"
    assert cmd_knobs[cmd_knobs.index("--atlas-ram-lock-policy") + 1] == "on"

    # CLI parser & env var support
    monkeypatch.setenv("TENDOU_MTP_MODE", "ram")
    monkeypatch.setenv("TENDOU_VRAM_LAYER_CAP", "auto")
    monkeypatch.setenv("TENDOU_RAM_LOCK_POLICY", "off")
    parsed_env = create_parser().parse_args([])
    assert parsed_env.mtp == "ram"
    assert parsed_env.vram_layer_cap == "auto"
    assert parsed_env.ram_lock_policy == "off"

    # 4. Qwen3.8 MTP auto-resolution when mtp enabled and mtp_path omitted
    repo_root = Path(__file__).resolve().parents[2]
    qwen38_mtp = repo_root / "models" / "mtp" / "Qwen3.8-Flash-Next-MTP-Q4_0.gguf"
    model_q38 = tmp_path / "Qwen3.8-Flash-Next-IQ3_XXS.gguf"
    model_q38.touch()
    backend_q38_off = AtlasSubprocessBackend(exe_path="/bin/true", model_path=str(model_q38))
    assert backend_q38_off.mtp == "off"
    assert "--atlas-mtp-path" not in backend_q38_off._build_cmd()
    if qwen38_mtp.exists():
        backend_q38_ram = AtlasSubprocessBackend(exe_path="/bin/true", model_path=str(model_q38), mtp="ram")
        assert backend_q38_ram.mtp_path == str(qwen38_mtp)
        cmd_q38 = backend_q38_ram._build_cmd()
        assert cmd_q38[cmd_q38.index("--atlas-mtp-path") + 1] == str(qwen38_mtp)
