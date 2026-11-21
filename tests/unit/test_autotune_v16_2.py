"""Unit tests for v0.16.2-stable Model-Aware AutoTune, Adaptive KV Cache, and Windows Integration."""

from pathlib import Path
from src.atlas.autotune import (
    HardwareProfile,
    ModelTopology,
    compute_auto_plan,
    scan_local_models,
)
from src.atlas.server import AtlasSubprocessBackend, create_parser


def _rtx5060_hw() -> HardwareProfile:
    return HardwareProfile(
        has_gpu=True,
        gpu_name="NVIDIA GeForce RTX 5060 Laptop GPU",
        vram_total_mb=8151.0,
        vram_free_mb=7000.0,
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        has_avx512_vnni=True,
        has_avx512_vbmi=True,
        has_avx2=True,
        ram_total_gb=16.0,
        ram_avail_gb=9.5,
        is_windows=False,
        directstorage_supported=False,
    )


def _model_35b() -> ModelTopology:
    return ModelTopology(
        model_path="Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        model_name="Qwen3.6-35B-A3B-UD-IQ4_XS",
        architecture="qwen35moe",
        block_count=40,
        expert_count=256,
        expert_used_count=8,
        intermediate_size=512,
        file_size_gb=16.51,
        quant_type="IQ4_XS",
        expert_slice_bytes=1458176,
        is_moe=True,
    )


def _model_38b() -> ModelTopology:
    return ModelTopology(
        model_path="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        model_name="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002",
        architecture="qwen4exp",
        block_count=48,
        expert_count=512,
        expert_used_count=10,
        intermediate_size=512,
        file_size_gb=75.8,
        quant_type="IQ3_XXS",
        expert_slice_bytes=1253376,
        is_moe=True,
    )


def test_v16_2_autotune_35b_places_6_to_8_gpu_layers():
    hw = _rtx5060_hw()
    m35 = _model_35b()

    plan_8k = compute_auto_plan(hw, m35, ctx_size=8192)
    assert 6 <= plan_8k.gpu_layers <= 8
    assert plan_8k.cache_type_k == "q8_0"
    assert plan_8k.cache_type_v == "q8_0"
    assert plan_8k.mtp_mode == "off"
    assert plan_8k.mtp_draft_n == 0
    assert plan_8k.rationale["estimated_vram_peak_mb"] < hw.vram_free_mb

    plan_64k = compute_auto_plan(hw, m35, ctx_size=65536)
    assert 5 <= plan_64k.gpu_layers <= 8
    assert plan_64k.cache_type_k == "f8_e4m3"
    assert plan_64k.cache_type_v == "f8_e4m3"
    assert plan_64k.mtp_mode == "off"


def test_v16_2_autotune_38b_keeps_safe_cap_2():
    hw = _rtx5060_hw()
    m38 = _model_38b()

    plan = compute_auto_plan(hw, m38, ctx_size=65536)
    assert plan.gpu_layers <= 2
    assert plan.vram_layer_cap == "2"
    assert plan.cache_type_k == "f8_e4m3"
    assert plan.cache_type_v == "f8_e4m3"
    assert plan.mtp_mode == "off"


def test_v16_2_explicit_user_overrides_never_clobbered():
    hw = _rtx5060_hw()
    m35 = _model_35b()

    plan = compute_auto_plan(
        hw,
        m35,
        ctx_size=65536,
        explicit_gpu_layers=3,
        explicit_vram_layer_cap="1",
        explicit_cache_type_k="q4_0",
        explicit_cache_type_v="q4_0",
    )
    assert plan.gpu_layers == 3
    assert plan.vram_layer_cap == "1"
    assert plan.cache_type_k == "q4_0"
    assert plan.cache_type_v == "q4_0"


def test_v16_2_server_adaptive_kv_and_explicit_gpu_layers(tmp_path, monkeypatch):
    monkeypatch.delenv("TENDOU_FP8_KV", raising=False)
    dummy = tmp_path / "model.gguf"
    dummy.write_bytes(b"GGUF")

    # ctx <= 8192 -> q8_0
    b8k = AtlasSubprocessBackend(exe_path="/bin/true", model_path=str(dummy), ctx_size=8192, gpu_layers=4)
    cmd8 = b8k._build_cmd()
    assert cmd8[cmd8.index("-ctk") + 1] == "q8_0"
    assert cmd8[cmd8.index("-ctv") + 1] == "q8_0"
    assert cmd8[cmd8.index("--atlas-gpu-layers") + 1] == "4"

    # ctx >= 16384 -> f8_e4m3
    b16k = AtlasSubprocessBackend(exe_path="/bin/true", model_path=str(dummy), ctx_size=16384, gpu_layers=2)
    cmd16 = b16k._build_cmd()
    assert cmd16[cmd16.index("-ctk") + 1] == "f8_e4m3"
    assert cmd16[cmd16.index("-ctv") + 1] == "f8_e4m3"

    # Explicit override respected
    b_exp = AtlasSubprocessBackend(exe_path="/bin/true", model_path=str(dummy), ctx_size=8192, cache_type_k="f16", cache_type_v="q4_0")
    cmd_exp = b_exp._build_cmd()
    assert cmd_exp[cmd_exp.index("-ctk") + 1] == "f16"
    assert cmd_exp[cmd_exp.index("-ctv") + 1] == "q4_0"

    # Parser defaults: mtp=off, gpu_layers=None (AutoTune)
    parsed = create_parser().parse_args([])
    assert parsed.mtp == "off"
    assert parsed.gpu_layers is None


def test_scan_local_models_deduplicates_shards(tmp_path):
    m_dir = tmp_path / "models"
    m_dir.mkdir()
    (m_dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf").write_bytes(b"GGUF" + b"\x00" * 1024)
    (m_dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf").write_bytes(b"\x00" * 1024)
    (m_dir / "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf").write_bytes(b"GGUF" + b"\x00" * 2048)

    found = scan_local_models(base_dir=tmp_path, hw=_rtx5060_hw())
    names = [m["filename"] for m in found]
    assert "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" in names
    assert "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf" not in names
    assert "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" in names
    m38 = next(m for m in found if "Qwen3.8" in m["filename"])
    assert m38["shards"] == 2


def test_vram_layer_cap_auto_vs_numeric_in_autotune_and_build_cmd(monkeypatch, tmp_path):
    hw = _rtx5060_hw()
    m35 = _model_35b()

    # Numeric vram_layer_cap='4' with explicit_gpu_layers=None MUST cap gpu_layers to 4
    plan_capped = compute_auto_plan(hw, m35, ctx_size=8192, explicit_vram_layer_cap="4")
    assert plan_capped.gpu_layers == 4
    assert plan_capped.vram_layer_cap == "4"

    # 'auto' vram_layer_cap MUST still allow full 6..8 AutoTune layers on 35B
    plan_auto = compute_auto_plan(hw, m35, ctx_size=8192, explicit_vram_layer_cap="auto")
    assert 6 <= plan_auto.gpu_layers <= 8

    # AtlasSubprocessBackend._build_cmd with Studio default vram_layer_cap='auto' must NOT drop to 2
    from src.atlas import server as srv
    fake_gguf = tmp_path / "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
    fake_gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    monkeypatch.setattr(srv, "probe_hardware", lambda: hw)
    monkeypatch.setattr(srv, "inspect_model_topology", lambda _p: m35)

    backend_auto = srv.AtlasSubprocessBackend(model_path=str(fake_gguf), ctx_size=8192, vram_layer_cap="auto")
    cmd_auto = backend_auto._build_cmd()
    idx = cmd_auto.index("--atlas-gpu-layers")
    assert 6 <= int(cmd_auto[idx + 1]) <= 8

    backend_cap3 = srv.AtlasSubprocessBackend(model_path=str(fake_gguf), ctx_size=8192, vram_layer_cap="3")
    cmd_cap3 = backend_cap3._build_cmd()
    idx3 = cmd_cap3.index("--atlas-gpu-layers")
    assert int(cmd_cap3[idx3 + 1]) == 3

