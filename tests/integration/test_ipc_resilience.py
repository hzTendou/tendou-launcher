import asyncio
import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.atlas.server import (
    TENDOU_VERSION,
    AtlasSubprocessBackend,
    MockAtlasBackend,
    _PriorityEngineGate,
    _resolve_default_engine_path,
    _resolve_version,
    create_app,
)

@pytest.mark.asyncio
async def test_oversized_done_line_does_not_kill_reader(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    backend = AtlasSubprocessBackend(exe_path=sys.executable, model_path=str(model))
    spawn = asyncio.create_subprocess_exec

    fake_engine_code = (
        "import sys, json\n"
        "print('[ATLAS_READY]', flush=True)\n"
        "while True:\n"
        "    line = sys.stdin.readline()\n"
        "    if not line: break\n"
        "    req = json.loads(line)\n"
        "    rid = req.get('id', '')\n"
        "    cmd = req.get('cmd', '')\n"
        "    if cmd == 'generate':\n"
        "        big_text = 'A' * 100000\n"
        "        print(json.dumps({'event': 'token', 'id': rid, 'token': 'ok'}), flush=True)\n"
        "        print(json.dumps({'event': 'done', 'id': rid, 'text': big_text}), flush=True)\n"
        "    elif cmd == 'exit': break\n"
    )

    async def fake_child(*args, **kwargs):
        return await spawn(sys.executable, "-c", fake_engine_code, **kwargs)

    with patch("asyncio.create_subprocess_exec", fake_child):
        await asyncio.wait_for(backend.start(), 5)
        assert backend.is_healthy()

        events1 = []
        async for ev in backend.generate("hello 1"):
            events1.append(ev)
        assert any(e.get("event") == "token" for e in events1)
        assert any(e.get("event") == "done" for e in events1)

        assert backend.is_healthy()
        assert backend._reader_task is not None and not backend._reader_task.done()

        events2 = []
        async for ev in backend.generate("hello 2"):
            events2.append(ev)
        assert any(e.get("event") == "done" for e in events2)

        await backend.stop()


@pytest.mark.asyncio
async def test_stderr_lines_deque_limit(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    backend = AtlasSubprocessBackend(exe_path=sys.executable, model_path=str(model))
    spawn = asyncio.create_subprocess_exec

    fake_engine_code = (
        "import sys\n"
        "print('[ATLAS_READY]', flush=True)\n"
        "for i in range(5000):\n"
        "    sys.stderr.write(f'stderr log line {i}\\n')\n"
        "sys.stderr.flush()\n"
        "sys.stdin.readline()\n"
    )

    async def fake_child(*args, **kwargs):
        return await spawn(sys.executable, "-c", fake_engine_code, **kwargs)

    with patch("asyncio.create_subprocess_exec", fake_child):
        await asyncio.wait_for(backend.start(), 5)
        await asyncio.sleep(0.2)
        assert len(backend.stderr_lines) <= 2000
        await backend.stop()


@pytest.mark.asyncio
async def test_timeout_sends_cancel_to_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("TENDOU_INFERENCE_TIMEOUT", "2")
    model = tmp_path / "model.gguf"
    model.touch()
    backend = AtlasSubprocessBackend(exe_path=sys.executable, model_path=str(model))
    spawn = asyncio.create_subprocess_exec

    # Fake engine logs commands received on stdin to a file
    log_file = tmp_path / "engine_cmds.log"
    fake_engine_code = (
        f"import sys, json\n"
        f"print('[ATLAS_READY]', flush=True)\n"
        f"with open(r'{log_file}', 'a') as f:\n"
        f"    while True:\n"
        f"        line = sys.stdin.readline()\n"
        f"        if not line: break\n"
        f"        f.write(line)\n"
        f"        f.flush()\n"
    )

    async def fake_child(*args, **kwargs):
        return await spawn(sys.executable, "-c", fake_engine_code, **kwargs)

    with patch("asyncio.create_subprocess_exec", fake_child):
        await asyncio.wait_for(backend.start(), 5)
        events = []
        async for ev in backend.generate("hang prompt"):
            events.append(ev)
        
        # Verify error event produced
        assert any(e.get("event") == "error" and "timed out" in e.get("message", "") for e in events)
        
        # Verify cancel command was delivered to stdin of engine
        await asyncio.sleep(0.1)
        assert log_file.exists()
        logged = log_file.read_text()
        assert '"cmd": "cancel"' in logged
        
        await backend.stop()


def test_count_tokens_and_debug_state_endpoints():
    app = create_app(MockAtlasBackend())
    client = TestClient(app)

    # Test /debug/state
    resp = client.get("/v1/debug/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "healthy" in data
    assert "active_requests_http" in data

    # Test /v1/messages/count_tokens
    payload = {
        "messages": [{"role": "user", "content": "Write a geometry document for high school."}]
    }
    resp = client.post("/v1/messages/count_tokens", json=payload)
    assert resp.status_code == 200
    tok_data = resp.json()
    assert "input_tokens" in tok_data
    assert tok_data["input_tokens"] > 0

    # Test /api/hello GET and HEAD preflight routes used by Claude Code
    assert client.get("/api/hello").status_code == 200
    assert client.head("/api/hello").status_code == 200


@pytest.mark.asyncio
async def test_queued_requests_emit_keepalive_and_abort_cleanly(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    backend = AtlasSubprocessBackend(exe_path=sys.executable, model_path=str(model))
    spawn = asyncio.create_subprocess_exec

    # Fake engine sleeps briefly on the first request and records generated ids
    fake_engine_code = (
        "import sys, json, time\n"
        "print('[ATLAS_READY]', flush=True)\n"
        "while True:\n"
        "    line = sys.stdin.readline()\n"
        "    if not line: break\n"
        "    req = json.loads(line)\n"
        "    rid = req.get('id', '')\n"
        "    cmd = req.get('cmd', '')\n"
        "    if cmd == 'generate':\n"
        "        print(json.dumps({'event': 'hb', 'id': rid, 'phase': 1, 'prefill_pos': 64, 'prefill_total': 128, 'generated': 0}), flush=True)\n"
        "        if req.get('prompt') == 'slow':\n"
        "            time.sleep(2.8)\n"
        "        print(json.dumps({'event': 'token', 'id': rid, 'token': 'ok:' + req.get('prompt', '')}), flush=True)\n"
        "        print(json.dumps({'event': 'done', 'id': rid, 'finish_reason': 'stop', 'prompt_tokens': 5, 'completion_tokens': 1}), flush=True)\n"
        "    elif cmd == 'exit': break\n"
    )

    async def fake_child(*args, **kwargs):
        return await spawn(sys.executable, "-c", fake_engine_code, **kwargs)

    with patch("asyncio.create_subprocess_exec", fake_child):
        await asyncio.wait_for(backend.start(), 5)
        app = create_app(backend)

        async def collect(prompt, rid):
            evs = []
            async for ev in backend.generate(prompt, req_id=rid):
                evs.append(ev)
            return evs

        t1 = asyncio.create_task(collect("slow", "r1"))
        await asyncio.sleep(0.15)  # Let r1 acquire lock and enter engine

        # Launch queued r2 and r3 concurrently behind r1
        t2 = asyncio.create_task(collect("second", "r2"))
        gen3 = backend.generate("aborted-queued", req_id="r3")
        ev3_first = await asyncio.wait_for(gen3.__anext__(), timeout=4.0)
        assert ev3_first.get("event") == "ping" and ev3_first.get("queued") is True
        await gen3.aclose()

        r2_events = await asyncio.wait_for(t2, timeout=10.0)
        r1_events = await asyncio.wait_for(t1, timeout=10.0)

        assert any(e.get("event") == "ping" and e.get("queued") is True for e in r2_events)
        assert any(e.get("event") == "token" and e.get("token") == "ok:second" for e in r2_events)
        assert any(e.get("event") == "done" for e in r1_events)

        client = TestClient(app)
        dbg = client.get("/v1/debug/state").json()
        assert dbg["last_heartbeat"] is not None
        assert dbg["last_heartbeat"]["phase"] == 1
        assert dbg["last_heartbeat_age_sec"] is not None
        assert dbg["lock_locked"] is False

        await backend.stop()


def test_anthropic_aux_fastlane_and_context_overflow():
    class OverflowBackend:
        mtp = "off"
        async def start(self): pass
        async def stop(self): pass
        async def cancel(self, rid): pass
        async def generate(self, prompt, n_predict, temp, top_p, stop, req_id, mtp_draft_n=None):
            yield {"event": "error", "code": "context_overflow", "message": "prompt is too long: 70000 tokens > 65536 maximum"}

    app = create_app(OverflowBackend())
    client = TestClient(app)

    # Aux fastlane intercepts title/topic probes without touching backend generate()
    r_topic = client.post("/v1/messages", json={
        "model": "claude-opus-5-5",
        "querySource": "topic_detection",
        "system": "Analyze if this message indicates a new conversation topic and set isNewTopic.",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r_topic.status_code == 200
    assert "isNewTopic" in r_topic.json()["content"][0]["text"]

    # repl_main_thread must NEVER be intercepted by aux fastlane; maps context_overflow -> HTTP 400
    r_main = client.post("/v1/messages", json={
        "model": "claude-opus-5-5",
        "querySource": "repl_main_thread",
        "messages": [{"role": "user", "content": "very long prompt"}],
        "stream": False,
    })
    assert r_main.status_code == 400

    # Streaming context_overflow emits invalid_request_error SSE event
    r_stream = client.post("/v1/messages", json={
        "model": "claude-opus-5-5",
        "querySource": "repl_main_thread",
        "messages": [{"role": "user", "content": "very long prompt"}],
        "stream": True,
    })
    assert r_stream.status_code == 200
    data_events = [json.loads(line[6:]) for line in r_stream.text.splitlines() if line.startswith("data: ")]
    errs = [d for d in data_events if d.get("type") == "error"]
    assert len(errs) == 1
    assert errs[0]["error"]["type"] == "invalid_request_error"


def test_release_version_and_default_engine_resolution(monkeypatch):
    monkeypatch.delenv("TENDOU_VERSION", raising=False)
    monkeypatch.delenv("TENDOU_ENGINE_PATH", raising=False)
    repo_root = Path(__file__).resolve().parents[2]
    version_txt = (repo_root / "VERSION.txt").read_text(encoding="utf-8").strip()
    assert version_txt == "0.16.3-stable"
    assert _resolve_version() == version_txt
    assert TENDOU_VERSION == version_txt

    resolved_exe = _resolve_default_engine_path()
    expected_v163_bin = repo_root / "releases" / "tendou-v0.16.3-stable" / "bin" / "tendou-engine"
    expected_v162_bin = repo_root / "releases" / "tendou-v0.16.2-stable" / "bin" / "tendou-engine"
    expected_v161_bin = repo_root / "releases" / "tendou-v0.16.1-stable" / "bin" / "tendou-engine"
    expected_v16_bin = repo_root / "releases" / "tendou-v0.16.0-stable" / "bin" / "tendou-engine"
    expected_v15_bin = repo_root / "releases" / "tendou-v0.15.0-stable" / "bin" / "tendou-engine"
    if os.name != "nt":
        if expected_v163_bin.exists():
            assert Path(resolved_exe) == expected_v163_bin
        elif expected_v162_bin.exists():
            assert Path(resolved_exe) == expected_v162_bin
        elif expected_v161_bin.exists():
            assert Path(resolved_exe) == expected_v161_bin
        elif expected_v16_bin.exists():
            assert Path(resolved_exe) == expected_v16_bin
        elif expected_v15_bin.exists():
            assert Path(resolved_exe) == expected_v15_bin

    backend = MockAtlasBackend()
    backend.exe_path = resolved_exe
    app = create_app(backend)
    assert app.version == version_txt
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["version"] == version_txt
    assert health["exe_path"] == resolved_exe

    dbg = client.get("/v1/debug/state").json()
    assert dbg["version"] == version_txt
    assert dbg["exe_path"] == resolved_exe
    assert dbg["queue_depth"] == 0
    assert "prefix_fork_counts" in dbg
    assert "active_session_id" in dbg
    assert "last_session_id" in dbg

    hello_head = client.head("/api/hello")
    assert hello_head.status_code == 200


@pytest.mark.asyncio
async def test_priority_engine_gate_ordering_affinity_aging_and_cancel(monkeypatch):
    # 1. priority=0 runs ahead of queued priority=2; warm session (priority=1) beats cold priority=2
    monkeypatch.setenv("TENDOU_QUEUE_AGING_S", "15.0")
    gate = _PriorityEngineGate()
    order = []

    await gate.acquire(priority=2, session_id="sess-warm")
    assert gate.locked() is True
    assert gate.active_session_id == "sess-warm"
    assert gate.last_session_id == "sess-warm"

    async def worker(label, *, priority=None, session_id=None, n_predict=512):
        await gate.acquire(priority=priority, session_id=session_id, n_predict=n_predict)
        try:
            order.append(label)
            await asyncio.sleep(0.01)
        finally:
            gate.release()

    t_cold = asyncio.create_task(worker("cold", priority=2, session_id="sess-cold"))
    await asyncio.sleep(0.01)
    t_warm = asyncio.create_task(worker("warm", session_id="sess-warm"))
    await asyncio.sleep(0.01)
    t_fast = asyncio.create_task(worker("fast", priority=0, session_id="sess-fast"))
    await asyncio.sleep(0.01)
    assert gate.queue_depth == 3

    gate.release()
    await asyncio.gather(t_cold, t_warm, t_fast)
    assert order == ["fast", "warm", "cold"]
    assert gate.queue_depth == 0
    assert gate.locked() is False

    # 2. Aging promotes long-waiting priority=2 ahead of newly queued priority=1
    monkeypatch.setenv("TENDOU_QUEUE_AGING_S", "0.08")
    gate_age = _PriorityEngineGate()
    age_order = []
    await gate_age.acquire(priority=2, session_id="sess-hot")

    async def age_worker(label, *, priority=None, session_id=None):
        await gate_age.acquire(priority=priority, session_id=session_id)
        try:
            age_order.append(label)
        finally:
            gate_age.release()

    t_old_cold = asyncio.create_task(age_worker("aged_cold", priority=2, session_id="sess-old"))
    await asyncio.sleep(0.12)  # Exceed aging threshold
    t_new_warm = asyncio.create_task(age_worker("new_warm", session_id="sess-hot"))
    await asyncio.sleep(0.01)
    gate_age.release()
    await asyncio.gather(t_old_cold, t_new_warm)
    assert age_order == ["aged_cold", "new_warm"]
    assert gate_age.locked() is False

    # 3. Cancelled waiter cleans up cleanly without leaking active slot
    gate_cancel = _PriorityEngineGate()
    await gate_cancel.acquire(priority=2, session_id="holder")
    t_victim = asyncio.create_task(gate_cancel.acquire(priority=2, session_id="victim"))
    t_survivor = asyncio.create_task(gate_cancel.acquire(priority=2, session_id="survivor"))
    await asyncio.sleep(0.01)
    assert gate_cancel.queue_depth == 2

    t_victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t_victim
    assert gate_cancel.queue_depth == 1
    assert gate_cancel.locked() is True

    gate_cancel.release()
    await asyncio.wait_for(t_survivor, timeout=1.0)
    assert gate_cancel.active_session_id == "survivor"
    gate_cancel.release()
    assert gate_cancel.locked() is False
    assert gate_cancel.queue_depth == 0


@pytest.mark.asyncio
async def test_ipc_prefix_fork_counts_session_payload_and_debug_state(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    backend = AtlasSubprocessBackend(exe_path=sys.executable, model_path=str(model))
    spawn = asyncio.create_subprocess_exec
    payload_log = tmp_path / "payloads.jsonl"

    fake_engine_code = (
        f"import sys, json\n"
        f"print('[ATLAS_READY]', flush=True)\n"
        f"fork_kinds = ['root_anchor', 'turn_boundary', 'exact', 'cold']\n"
        f"idx = 0\n"
        f"while True:\n"
        f"    line = sys.stdin.readline()\n"
        f"    if not line: break\n"
        f"    req = json.loads(line)\n"
        f"    cmd = req.get('cmd', '')\n"
        f"    rid = req.get('id', '')\n"
        f"    if cmd == 'generate':\n"
        f"        with open(r'{payload_log}', 'a') as f:\n"
        f"            f.write(json.dumps({{'session_id': req.get('session_id'), 'priority': req.get('priority')}}) + '\\n')\n"
        f"        fk = fork_kinds[idx % len(fork_kinds)]\n"
        f"        idx += 1\n"
        f"        print(json.dumps({{'event': 'token', 'id': rid, 'token': 'ok'}}), flush=True)\n"
        f"        print(json.dumps({{'event': 'done', 'id': rid, 'finish_reason': 'stop', 'prompt_tokens': 4, 'completion_tokens': 1, 'prefix_fork_kind': fk}}), flush=True)\n"
        f"    elif cmd == 'exit': break\n"
    )

    async def fake_child(*args, **kwargs):
        return await spawn(sys.executable, "-c", fake_engine_code, **kwargs)

    with patch("asyncio.create_subprocess_exec", fake_child):
        await asyncio.wait_for(backend.start(), 5)
        app = create_app(backend)

        # First call: cold session
        evs1 = [ev async for ev in backend.generate("long prompt", n_predict=128, session_id="sess-alpha")]
        assert any(e.get("event") == "done" for e in evs1)
        # Second call: warm session affinity -> priority 1
        evs2 = [ev async for ev in backend.generate("long follow-up", n_predict=128, session_id="sess-alpha")]
        assert any(e.get("event") == "done" for e in evs2)
        # Third call: fast-lane short request -> priority 0
        evs3 = [ev async for ev in backend.generate("ping", n_predict=32, session_id="sess-beta")]
        assert any(e.get("event") == "done" for e in evs3)

        recorded = [json.loads(line) for line in payload_log.read_text().splitlines() if line.strip()]
        assert recorded[0] == {"session_id": "sess-alpha", "priority": 2}
        assert recorded[1] == {"session_id": "sess-alpha", "priority": 1}
        assert recorded[2] == {"session_id": "sess-beta", "priority": 0}

        client = TestClient(app)
        dbg = client.get("/v1/debug/state").json()
        assert dbg["queue_depth"] == 0
        assert dbg["active_session_id"] is None
        assert dbg["last_session_id"] == "sess-beta"
        assert dbg["prefix_fork_counts"]["root_anchor"] == 1
        assert dbg["prefix_fork_counts"]["turn_boundary"] == 1
        assert dbg["prefix_fork_counts"]["exact"] == 1
        assert dbg["prefix_fork_counts"]["cold"] == 0

        await backend.stop()


def test_http_session_id_forwarding_openai_and_anthropic():
    seen_sessions = []

    class RecordingBackend(MockAtlasBackend):
        async def generate(
            self,
            prompt,
            n_predict=512,
            temp=0.7,
            top_p=0.9,
            min_p=None,
            stop=None,
            req_id=None,
            mtp_draft_n=None,
            session_id=None,
            priority=None,
        ):
            seen_sessions.append(session_id)
            async for ev in super().generate(
                prompt=prompt,
                n_predict=n_predict,
                temp=temp,
                top_p=top_p,
                min_p=min_p,
                stop=stop,
                req_id=req_id,
                mtp_draft_n=mtp_draft_n,
                session_id=session_id,
                priority=priority,
            ):
                yield ev

    app = create_app(RecordingBackend())
    client = TestClient(app)

    r1 = client.post(
        "/v1/chat/completions",
        headers={"X-Tendou-Session": "hdr-tendou-1"},
        json={"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.8-flash-next", "user": "openai-user-2", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert r2.status_code == 200

    r3 = client.post(
        "/v1/messages",
        headers={"X-Session-Id": "hdr-sess-3"},
        json={"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert r3.status_code == 200

    r4 = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-5-5",
            "metadata": {"user_id": "anth-meta-4"},
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
    )
    assert r4.status_code == 200

    assert seen_sessions == ["hdr-tendou-1", "openai-user-2", "hdr-sess-3", "anth-meta-4"]
    dbg = client.get("/v1/debug/state").json()
    assert dbg["last_session_id"] == "anth-meta-4"
    assert dbg["queue_depth"] == 0
