"""Unit & integration tests for Tendou Studio Control Plane (/api/ui/*), Live TPS TelemetryHub, and EngineSettingsModel."""

import time
from pathlib import Path
from fastapi.testclient import TestClient

from src.atlas.server import MockAtlasBackend, create_app
from src.atlas.ui_config import (
    EngineSettingsModel,
    ProfileStore,
    compute_settings_diff,
    get_presets,
    get_settings_schema,
    serialize_to_cli_args,
)
from src.atlas.ui_telemetry import TelemetryHub


def test_engine_settings_model_covers_all_cli_flags_and_serializers():
    cfg = EngineSettingsModel(
        threads=8,
        ctx_size=65536,
        gpu=True,
        gpu_layers=6,
        boost=True,
        boost_level=2,
        mtp="off",
        tlep=True,
        p2_gpu_binding=True,
        p3_async_transfer=True,
    )
    linux_ser = serialize_to_cli_args(cfg, platform="linux")
    win_ser = serialize_to_cli_args(cfg, platform="windows")

    assert linux_ser["command"].startswith("./start_server.sh ")
    assert win_ser["command"].startswith("start_server.cmd ")
    assert "--gpu-layers" in linux_ser["args"]
    assert "--mtp" in linux_ser["args"]
    assert linux_ser["args"][linux_ser["args"].index("--mtp") + 1] == "off"
    assert "--tlep" in linux_ser["args"]
    assert "--p2-gpu-binding" in linux_ser["args"]
    assert "--p3-async-transfer" in linux_ser["args"]
    assert linux_ser["backend_kwargs"]["mtp"] == "off"
    assert linux_ser["backend_kwargs"]["gpu_layers"] == 6

    schema = get_settings_schema(ram_total_gb=14.9, vram_total_mb=8151.0)
    assert schema["total_flags"] >= 49
    assert len(schema["groups"]) == 6
    mtp_field = next(f for f in schema["fields"] if f["key"] == "mtp")
    assert mtp_field["safety_warning"] is not None

    presets = get_presets()
    assert len(presets) >= 4
    for p in presets:
        assert p["overrides"].get("mtp", "off") == "off"


def test_profile_store_atomic_roundtrip_and_diff(tmp_path):
    store = ProfileStore(tmp_path / "studio_profile.json")
    cfg1 = EngineSettingsModel(threads=8, ctx_size=8192, boost=True)
    store.save(cfg1)
    loaded = store.load()
    assert loaded.threads == 8
    assert loaded.ctx_size == 8192
    assert loaded.mtp == "off"

    cfg2 = EngineSettingsModel(threads=12, ctx_size=65536, boost=False)
    diff = compute_settings_diff(cfg1, cfg2)
    keys = {d["key"] for d in diff}
    assert {"threads", "ctx_size", "boost"}.issubset(keys)


def test_telemetry_hub_live_tps_sliding_window_and_logs():
    hub = TelemetryHub()
    hub.on_request_start("req-1", prompt_tokens=100)
    hub.on_prefill_done(ttft_ms=120.5, prompt_tokens=100, prefix_fork_kind="root_anchor")
    for _ in range(12):
        hub.on_token(1)
        time.sleep(0.01)

    snap_active = hub.snapshot(engine_status="ready", record_history=True)
    assert snap_active["tps"]["is_generating"] is True
    assert snap_active["tps"]["tokens_this_turn"] == 12
    assert snap_active["tps"]["live_decode_tps"] > 10.0
    assert snap_active["tps"]["peak_decode_tps"] >= snap_active["tps"]["live_decode_tps"]
    assert snap_active["tps"]["last_ttft_ms"] == 120.5
    assert snap_active["tps"]["last_prefix_fork_kind"] == "root_anchor"

    hub.on_request_done(prompt_tokens=100, completion_tokens=12, prefix_fork_kind="root_anchor")
    snap_done = hub.snapshot(engine_status="ready", record_history=True)
    assert snap_done["tps"]["is_generating"] is False
    assert snap_done["tps"]["tokens_session_total"] == 12
    assert snap_done["tps"]["last_turn_decode_tps"] > 10.0

    hub.push_log("[ATLAS] Prefill: 256 tokens / 0.50 s = 512.0 TPS", source="engine")
    hub.push_log("[ATLAS] vram_hits=42 ram_hits=38 misses=20 avg_k=2.45", source="engine")
    snap_moe = hub.snapshot(engine_status="ready", record_history=False)
    assert snap_moe["tps"]["last_prefill_tps"] == 512.0
    assert snap_moe["moe_cache"]["tier0_vram_pct"] == 42.0
    assert snap_moe["moe_cache"]["tier1_ram_pct"] == 38.0
    assert snap_moe["moe_cache"]["tier2_nvme_pct"] == 20.0
    assert snap_moe["moe_cache"]["avg_dynamic_k"] == 2.45
    assert len(hub.get_recent_logs()) >= 2


def test_ui_api_endpoints_end_to_end(tmp_path):
    backend = MockAtlasBackend(model_id="qwen3.6-35b")
    app = create_app(backend=backend, host="127.0.0.1", port=18123)
    # Isolate profile and chat writes inside tmp_path so unit tests never touch repo config/data
    from src.atlas.ui_chats import ChatStore
    app.state.studio.profile_store = ProfileStore(tmp_path / "test_profile.json")
    app.state.studio.chat_store = ChatStore(tmp_path / "chats")

    with TestClient(app) as client:
        # 1. /api/ui/state
        r_state = client.get("/api/ui/state")
        assert r_state.status_code == 200
        st = r_state.json()
        assert st["status"] == "ready"
        assert st["version"] == "0.16.3-stable"
        assert st["os"] in ("linux", "windows")
        assert "telemetry" in st

        # 2. /api/ui/config/schema & /presets
        r_schema = client.get("/api/ui/config/schema")
        assert r_schema.status_code == 200
        assert r_schema.json()["total_flags"] >= 49

        r_presets = client.get("/api/ui/config/presets")
        assert r_presets.status_code == 200
        assert len(r_presets.json()["presets"]) >= 4

        # 3. GET / PUT /api/ui/config
        r_cfg = client.get("/api/ui/config")
        assert r_cfg.status_code == 200
        assert "./start_server.sh" in r_cfg.json()["cli"]["linux"]
        assert "start_server.cmd" in r_cfg.json()["cli"]["windows"]

        r_put = client.put("/api/ui/config", json={"threads": 12, "ctx_size": 32768, "mock": True})
        assert r_put.status_code == 200
        assert r_put.json()["settings"]["threads"] == 12
        assert "--threads 12" in r_put.json()["cli"]["linux"]
        assert any(d["key"] == "threads" for d in r_put.json()["diff_from_running"])

        # 4. /api/ui/models & /models/scan
        scan_dir = tmp_path / "custom_models"
        scan_dir.mkdir()
        (scan_dir / "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf").write_bytes(b"GGUF" + b"\x00" * 512)
        r_scan = client.post("/api/ui/models/scan", json={"directory": str(scan_dir)})
        assert r_scan.status_code == 200
        assert any(m["filename"] == "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" for m in r_scan.json()["models"])

        # Bad scan directory returns 400
        assert client.post("/api/ui/models/scan", json={"directory": str(tmp_path / "missing")}).status_code == 400

        # 5. /api/ui/agents/configs
        r_ag = client.get("/api/ui/agents/configs")
        assert r_ag.status_code == 200
        ag = r_ag.json()
        assert "127.0.0.1:18123/v1" in ag["openai_url"]
        assert "ANTHROPIC_BASE_URL" in ag["claude_code"]["linux"]
        assert "opencode.ai" in ag["opencode"]["content"]

        # 6. /api/ui/benchmark/run & /history
        r_bench = client.post("/api/ui/benchmark/run", json={"max_tokens": 16})
        assert r_bench.status_code == 200
        br = r_bench.json()
        assert br["cold"]["tokens"] > 0
        assert br["warm"]["decode_tps"] > 0

        r_hist = client.get("/api/ui/benchmark/history")
        assert r_hist.status_code == 200
        assert r_hist.json()["last_live_run"] is not None

        # 7. /api/ui/engine/stop -> /start -> /restart (Hot-Swap)
        r_stop = client.post("/api/ui/engine/stop")
        assert r_stop.status_code == 200
        assert client.get("/api/ui/state").json()["status"] == "stopped"

        r_start = client.post("/api/ui/engine/start", json={"settings": {"mock": True, "threads": 8}, "save_profile": True})
        assert r_start.status_code == 200
        assert client.get("/api/ui/state").json()["status"] == "ready"

        r_restart = client.post("/api/ui/engine/restart", json={"settings": {"mock": True, "threads": 6}, "save_profile": True})
        assert r_restart.status_code == 200
        assert r_restart.json()["status"] == "ready"
        assert any(d["key"] == "threads" for d in r_restart.json()["applied_diff"])

        # 8. Unknown API routes still return 404 (not SPA index.html), while / serves built SPA
        assert client.get("/v1/nonexistent-endpoint").status_code == 404
        assert client.get("/api/ui/nonexistent").status_code == 404
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "text/html" in r_root.headers.get("content-type", "")
        assert "Tendou Studio" in r_root.text


def test_benchmark_no_double_counting_and_config_put_contract(tmp_path):
    from src.atlas.server import MockAtlasBackend, create_app, create_parser
    from src.atlas.ui_config import EngineSettingsModel, serialize_to_cli_args

    # 1. --no-odmoe-quick-evict round-trip through serialize_to_cli_args -> create_parser
    cfg = EngineSettingsModel(odmoe_quick_evict=False, threads=6, boost=False)
    ser = serialize_to_cli_args(cfg, platform="linux")
    assert "--no-odmoe-quick-evict" in ser["args"]
    parsed = create_parser().parse_args(ser["args"])
    assert parsed.odmoe_quick_evict is False
    assert parsed.threads == 6
    assert parsed.boost is False

    # 2. /api/ui/benchmark/run must NOT double-count tokens or requests in TelemetryHub
    app = create_app(backend=MockAtlasBackend(), host="127.0.0.1", port=18125, auto_start_engine=True)
    app.state.studio.profile_store.profile_path = tmp_path / "studio_profile.json"
    with TestClient(app) as client:
        r_b = client.post("/api/ui/benchmark/run", json={"prompt": "Write binary search", "max_tokens": 20, "temperature": 0.2})
        assert r_b.status_code == 200
        bdata = r_b.json()
        expected_toks = bdata["cold"]["tokens"] + bdata["warm"]["tokens"]
        st = client.get("/api/ui/state").json()
        assert st["telemetry"]["tps"]["tokens_session_total"] == expected_toks
        assert st["telemetry"]["tps"]["requests_total"] == 2

        # 3. PUT /api/ui/config MUST return running_settings & profile_path so UI keeps pending Hot-Swap diff
        r_put = client.put("/api/ui/config", json={"threads": 4})
        assert r_put.status_code == 200
        pdata = r_put.json()
        assert "running_settings" in pdata
        assert "profile_path" in pdata
        assert pdata["settings"]["threads"] == 4
        assert pdata["running_settings"]["threads"] != 4
        assert any(d["key"] == "threads" for d in pdata["diff_from_running"])

        # 4. Path traversal attempt cannot escape ui/dist/ (falls back safely to SPA index.html, never leaks package.json)
        r_trav = client.get("/../package.json")
        assert r_trav.status_code in (200, 404)
        assert '"devDependencies"' not in r_trav.text


def test_persistent_chats_api_and_context_curve(tmp_path):
    from src.atlas.ui_chats import ChatStore

    backend = MockAtlasBackend(model_id="qwen3.6-35b")
    app = create_app(backend=backend, host="127.0.0.1", port=18127)
    app.state.studio.profile_store = ProfileStore(tmp_path / "studio_profile.json")
    app.state.studio.chat_store = ChatStore(tmp_path / "chats")

    with TestClient(app) as client:
        # Initially empty
        r_empty = client.get("/api/ui/chats")
        assert r_empty.status_code == 200
        assert r_empty.json()["count"] == 0

        # Create a chat
        r_new = client.post("/api/ui/chats", json={"system_prompt": "You are a systems engineer."})
        assert r_new.status_code == 200
        chat = r_new.json()
        cid = chat["id"]
        assert chat["title"] == "New Chat"

        # Upsert Turn 1 user + assistant (auto-derives title from first user prompt)
        r_u1 = client.post(
            f"/api/ui/chats/{cid}/messages",
            json={"id": "u-1", "role": "user", "content": "Explain zero-copy mmap THP pinning in Linux"},
        )
        assert r_u1.status_code == 200
        assert "Explain zero-copy mmap" in r_u1.json()["title"]

        r_a1 = client.post(
            f"/api/ui/chats/{cid}/messages",
            json={
                "id": "a-1",
                "role": "assistant",
                "content": "Zero-copy mlock pins file-backed pages directly.",
                "thinking": "Check page alignment first.",
                "model_id": "qwen3.6-35b",
                "metrics": {
                    "ttft_ms": 145.2,
                    "decode_tps": 92.4,
                    "prefill_tps": 680.0,
                    "prompt_tokens": 120,
                    "completion_tokens": 64,
                    "total_tokens": 184,
                    "duration_ms": 840.0,
                    "context_limit": 65536,
                    "prefix_fork_kind": "cold",
                },
            },
        )
        assert r_a1.status_code == 200
        assert len(r_a1.json()["context_curve"]) == 1

        # Upsert Turn 2 with larger growing context
        client.post(f"/api/ui/chats/{cid}/messages", json={"id": "u-2", "role": "user", "content": "Now benchmark 16K context"})
        r_a2 = client.post(
            f"/api/ui/chats/{cid}/messages",
            json={
                "id": "a-2",
                "role": "assistant",
                "content": "At 16K context, FP8 KV preserves VRAM headroom.",
                "metrics": {
                    "ttft_ms": 210.0,
                    "decode_tps": 88.1,
                    "prompt_tokens": 4200,
                    "completion_tokens": 96,
                    "total_tokens": 4296,
                    "duration_ms": 1300.0,
                    "context_limit": 65536,
                    "prefix_fork_kind": "turn_boundary",
                },
            },
        )
        assert r_a2.status_code == 200
        curve = r_a2.json()["context_curve"]
        assert len(curve) == 2
        assert curve[0]["total_tokens"] == 184
        assert curve[1]["total_tokens"] == 4296
        assert curve[1]["decode_tps"] == 88.1

        # Rename, pin, search, export
        r_upd = client.put(f"/api/ui/chats/{cid}", json={"title": "Hot-RAM Architecture", "pinned": True})
        assert r_upd.status_code == 200
        assert r_upd.json()["title"] == "Hot-RAM Architecture"
        assert r_upd.json()["pinned"] is True

        r_search = client.get("/api/ui/chats?q=FP8")
        assert r_search.status_code == 200
        assert r_search.json()["count"] == 1
        assert r_search.json()["chats"][0]["total_tokens"] == 4296

        r_exp = client.get(f"/api/ui/chats/{cid}/export")
        assert r_exp.status_code == 200
        assert "# Hot-RAM Architecture" in r_exp.json()["markdown"]
        assert "4296 ctx" in r_exp.json()["markdown"] or "Context 4296 tok" in r_exp.json()["markdown"]

        # Reject invalid/path-traversal chat IDs
        assert client.get("/api/ui/chats/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
        assert client.get("/api/ui/chats/bad..id").status_code == 400
        assert client.get("/api/ui/chats/nonexistent_chat_1234").status_code == 404

        # Delete
        r_del = client.delete(f"/api/ui/chats/{cid}")
        assert r_del.status_code == 200
        assert client.get("/api/ui/chats").json()["count"] == 0


def test_hot_ram_telemetry_parsing():
    hub = TelemetryHub()
    hub.push_log(
        "[ATLAS-HOTRAM] Zero-Copy Hot-RAM: pinned=9478.0 MiB (26/32 CPU MoE layers, total_cpu_moe=11614.0 MiB, budget=9563.0 MiB, mode=zero-copy-prewarm, mlock_ok=reclaimable, elapsed=2140.5 ms)",
        source="engine",
    )
    snap = hub.snapshot(engine_status="ready", record_history=False)
    assert snap["system"]["hot_ram_mb"] >= 9478.0
    assert snap["system"]["hot_ram_mode"] == "zero-copy-prewarm"
    assert "ram_cached_gb" in snap["system"]
    assert "engine_cpu_pct" in snap["system"]
    # Calling snapshot(backend=None) again must NOT wipe parsed hot_ram_mb/mode
    snap2 = hub.snapshot(backend=None, engine_status="ready", record_history=False)
    assert snap2["system"]["hot_ram_mb"] >= 9478.0
    assert snap2["system"]["hot_ram_mode"] == "zero-copy-prewarm"


def test_chat_title_multiline_and_export_and_completion_timings(tmp_path):
    from src.atlas.autotune import inspect_model_topology
    from src.atlas.ui_chats import ChatStore, derive_chat_title, export_conversation_markdown

    assert derive_chat_title("# Heading Line\nSecond line details here") == "Heading Line"
    assert derive_chat_title("\n\n  First real line  \nSecond") == "First real line"

    store = ChatStore(tmp_path / "chats")
    conv = store.create_chat()
    conv = store.update_chat(conv.id, type("Req", (), {
        "title": "X" * 300,
        "model_id": None,
        "system_prompt": "Line A\nLine B",
        "pinned": None,
        "params": None,
        "messages": None,
    })())
    assert conv is not None
    assert len(conv.title) <= 160
    md = export_conversation_markdown(conv)
    assert "> Line A\n> Line B" in md

    # Verify MTP Q4_0 quant filename detection (not mislabeled as IQ3_XXS)
    fake_mtp = tmp_path / "Qwen3.8-Flash-Next-MTP-Q4_0.gguf"
    fake_mtp.write_bytes(b"NOTGGUF")
    topo = inspect_model_topology(fake_mtp)
    assert topo.quant_type == "Q4_0"

    # Verify /v1/chat/completions emits real timings on both non-stream and SSE stream
    backend = MockAtlasBackend(model_id="qwen3.6-35b")
    app = create_app(backend=backend, host="127.0.0.1", port=18128)
    with TestClient(app) as client:
        r_ns = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.6-35b", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert r_ns.status_code == 200
        body = r_ns.json()
        assert "timings" in body
        assert body["timings"]["ttft_ms"] >= 0
        assert body["timings"]["prefix_fork_kind"] in ("cold", "exact", "turn_boundary", "root_anchor")

        r_st = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.6-35b", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r_st.status_code == 200
        assert '"timings"' in r_st.text
        assert '"prefix_fork_kind"' in r_st.text


def test_unlimited_max_tokens_and_512e_boost_level(tmp_path):
    from src.atlas.autotune import HardwareProfile, ModelTopology, compute_auto_plan
    from src.atlas.ui_chats import ChatCreateRequest, ChatParams, ChatStore

    # 1) ChatParams allows 0 (Unlimited) and defaults to 4096
    p_default = ChatParams()
    assert p_default.max_tokens == 4096
    store = ChatStore(tmp_path / "chats_unlim")
    conv = store.create_chat(ChatCreateRequest(params=ChatParams(max_tokens=0)))
    assert conv.params.max_tokens == 0

    # 2) Server accepts max_tokens=0 (unlimited) and routes n_predict=0 without fast-lane demotion
    captured = {}

    class CapturingBackend(MockAtlasBackend):
        async def generate(self, prompt, n_predict=512, temp=0.7, top_p=0.9, min_p=None, stop=None, req_id=None, mtp_draft_n=None, session_id=None, priority=None, top_k=None, repeat_penalty=None):
            captured["n_predict"] = n_predict
            captured["top_k"] = top_k
            captured["repeat_penalty"] = repeat_penalty
            captured["priority"] = priority
            async for ev in super().generate(prompt, n_predict=n_predict, temp=temp, top_p=top_p, min_p=min_p, stop=stop, req_id=req_id, mtp_draft_n=mtp_draft_n, session_id=session_id, priority=priority):
                yield ev

    backend = CapturingBackend(model_id="qwen3.8-flash-next")
    app = create_app(backend=backend, host="127.0.0.1", port=18129)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0, "stream": False},
        )
        assert r.status_code == 200
        assert captured["n_predict"] == 0
        assert captured["top_k"] == 40
        assert captured["repeat_penalty"] == 1.08

    # 3) AutoTune recommends Quality-Safe boost_level=1 for 512-expert Qwen3.8, level=2 for 35B
    hw = HardwareProfile(
        has_gpu=True, gpu_name="RTX 5060", vram_total_mb=8151.0, vram_free_mb=7200.0,
        cpu_cores_physical=8, cpu_cores_logical=16, has_avx512_vnni=True, has_avx512_vbmi=True,
        has_avx2=True, ram_total_gb=15.0, ram_avail_gb=11.0, is_windows=False, directstorage_supported=False,
    )
    topo_38b = ModelTopology(
        model_path="/tmp/qwen38.gguf", model_name="Qwen3.8-Flash-Next-IQ3_XXS", architecture="qwen4exp",
        block_count=48, expert_count=512, expert_used_count=10, intermediate_size=640,
        file_size_gb=70.6, quant_type="IQ3_XXS", expert_slice_bytes=3 * 1024 * 1024, is_moe=True,
    )
    topo_35b = ModelTopology(
        model_path="/tmp/qwen35.gguf", model_name="Qwen3.6-35B-A3B-IQ4_XS", architecture="qwen35moe",
        block_count=40, expert_count=256, expert_used_count=8, intermediate_size=1024,
        file_size_gb=16.5, quant_type="IQ4_XS", expert_slice_bytes=1400 * 1024, is_moe=True,
    )
    assert compute_auto_plan(hw, topo_38b, ctx_size=16384).boost_level == 1
    assert compute_auto_plan(hw, topo_35b, ctx_size=16384).boost_level == 2



