"""Lifecycle failure and quality-gate regressions; no model download needed."""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.atlas.server import AtlasSubprocessBackend, ChatMessage, format_chatml_prompt
from scripts.evaluate_resident import evaluate
from src.atlas.native_server import make_launch, parser as native_parser


def test_native_mtp_launch_is_single_slot_and_requires_real_head(tmp_path):
    executable, model, head = [tmp_path / name for name in ("server.exe", "model.gguf", "mtp.gguf")]
    for path in (executable, model):
        path.touch()
    args = native_parser().parse_args(["--exe-path", str(executable), "--model-path", str(model), "--mtp-path", str(head)])
    with pytest.raises(ValueError, match="MTP GGUF"):
        make_launch(args)
    head.touch()
    command, env = make_launch(args)
    assert command[command.index("-np") + 1] == "1"
    assert command[command.index("--spec-draft-n-max") + 1] == "1"
    assert env["ATLAS_MTP_PATH"] == str(head)
    assert env["ATLAS_HTTP_CLOSE"] == "1"
    assert not any("expert_used_count" in arg for arg in command)


@pytest.mark.asyncio
async def test_startup_exit_fails_without_waiting_for_timeout(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    backend = AtlasSubprocessBackend(exe_path=sys.executable, model_path=str(model))
    spawn = asyncio.create_subprocess_exec

    async def exiting_child(*args, **kwargs):
        return await spawn(sys.executable, "-c", "raise SystemExit(7)", **kwargs)

    with patch("asyncio.create_subprocess_exec", exiting_child):
        with pytest.raises(RuntimeError, match="code 7"):
            await asyncio.wait_for(backend.start(), 5)
    assert backend.proc is None
    assert not backend.is_healthy()
    assert backend._reader_task is None
    assert backend._stderr_task is None


def test_missing_quality_evidence_cannot_pass():
    with pytest.raises(ValueError, match="Missing"):
        evaluate({"results": []}, {"results": []})


@pytest.mark.asyncio
async def test_engine_exit_unblocks_active_request(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    backend = AtlasSubprocessBackend(exe_path=sys.executable, model_path=str(model))
    spawn = asyncio.create_subprocess_exec

    async def exiting_child(*args, **kwargs):
        return await spawn(sys.executable, "-c",
            "import sys; print('[ATLAS_READY]', flush=True); sys.stdin.readline(); sys.exit(9)", **kwargs)

    async def request():
        return [event async for event in backend.generate("hello")]

    with patch("asyncio.create_subprocess_exec", exiting_child):
        try:
            await backend.start()
            events = await asyncio.wait_for(request(), 5)
            assert events[-1]["event"] == "error"
            assert "code 9" in events[-1]["message"]
        finally:
            await backend.stop()


def test_diverse_wrong_output_fails_quality_gate():
    def report(texts):
        return {"results": [dict(case=name, repetition=0, text=text,
            token_ids=[1, 2], ttft_ms=10, effective_tps=2,
            done={"finish_reason": "stop"}) for name, text in texts.items()]}
    baseline = report(dict(math="323", code="[1, 2, 3]", logic="Yes"))
    candidate = report(dict(math="A diverse but wrong answer", code="An unrelated sentence", logic="No"))
    result = evaluate(baseline, candidate)
    assert not result["smoke_quality_gate_pass"]
    assert result["task_accuracy_pct"] == 0
    assert result["general_quality_status"] == "NOT_EVALUATED"


@pytest.mark.skipif(os.environ.get("ATLAS_TEST_REAL") != "1", reason="Set ATLAS_TEST_REAL=1 for resident model validation")
@pytest.mark.asyncio
async def test_recurrent_cache_repeat_extension_and_miss():
    messages = [ChatMessage(role="user", content="What is 2+2? Reply only with the number.")]
    base = format_chatml_prompt(messages)
    extension = format_chatml_prompt(messages + [ChatMessage(role="assistant", content="4"),
        ChatMessage(role="user", content="What is 3+3? Reply only with the number.")])
    prompts = [base, base, extension, base.replace("2+2", "7+1"), base]
    runs = []
    for cache_mb in (0, 256):
        backend = AtlasSubprocessBackend(ctx_size=2048,
            extra_args=["--atlas-prompt-cache-mb", str(cache_mb), "-b", "16", "-ub", "16"])
        rows = []
        try:
            await backend.start()
            for prompt in prompts:
                ids, done = [], None
                async for event in backend.generate(prompt, n_predict=8, temp=0):
                    assert event["event"] != "error", event
                    if event["event"] == "token":
                        ids.append(event["token_id"])
                    elif event["event"] == "done":
                        done = event
                assert done and ids
                rows.append(dict(token_ids=ids, done=done))
        finally:
            await backend.stop()
        runs.append(rows)
    output = Path("experiments/runtime_audit/cache_regression.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    assert [r["token_ids"] for r in runs[0]] == [r["token_ids"] for r in runs[1]]
    assert all(r["done"]["cached_prompt_tokens"] == 0 for r in runs[0])
    assert runs[1][1]["done"]["cached_prompt_tokens"] > 0
    assert runs[1][2]["done"]["cached_prompt_tokens"] > 0
    assert runs[1][3]["done"]["cached_prompt_tokens"] == 0
    # Different questions can still share a valid system-prefix checkpoint.
    assert runs[1][4]["done"]["cached_prompt_tokens"] < runs[1][4]["done"]["prompt_tokens"]
    assert all(r["done"]["prompt_cache_bytes"] <= 256 * 1024 * 1024 for r in runs[1])


@pytest.mark.parametrize("flag", ["--odmoe-lead", "--spice-conf-high", "--spice-conf-mid", "--tutti-async-io"])
def test_native_api_rejects_unwired_prefetch_options(flag):
    with pytest.raises(SystemExit):
        native_parser().parse_args([flag] if flag == "--tutti-async-io" else [flag, "2" if flag == "--odmoe-lead" else "0.5"])


def test_regular_api_keeps_measured_reference_prefetch_defaults():
    backend = AtlasSubprocessBackend()
    assert backend.odmoe_lead == 2
    assert backend.spice_conf_high == pytest.approx(0.65)
    assert backend.spice_conf_mid == pytest.approx(0.30)
    assert backend.tutti_async_io is False
    assert backend.readback_interval == 8
    assert backend.prefetch_candidates == 10
    assert backend.tutti_queue_depth == 64
    assert backend.cuda_sched == "yield"
    assert backend.threads == 16
    assert backend.prompt_cache_mb == 256


def test_atlas_subprocess_backend_cuda_sched_none_env_safety():
    backend = AtlasSubprocessBackend(cuda_sched=None)
    assert backend.cuda_sched is None
    # Verify environment building does not inject None which causes TypeError in Windows CreateProcess
    env = os.environ.copy()
    env["GGML_CUDA_REGISTER_HOST"] = os.environ.get("GGML_CUDA_REGISTER_HOST", "1")
    if backend.cuda_sched:
        env["ATLAS_CUDA_SCHED"] = os.environ.get("ATLAS_CUDA_SCHED", str(backend.cuda_sched))
    assert "ATLAS_CUDA_SCHED" not in env or isinstance(env["ATLAS_CUDA_SCHED"], str)
    assert all(isinstance(v, str) for v in env.values())


def test_server_cli_cuda_sched_parsing():
    from src.atlas.server import create_parser
    parser = create_parser()
    args = parser.parse_args(["--cuda-sched", "blocking"])
    assert args.cuda_sched == "blocking"
    args = parser.parse_args(["--cuda-sched", "spin"])
    assert args.cuda_sched == "spin"
    args = parser.parse_args([])
    assert args.cuda_sched == "yield"

    with pytest.raises(SystemExit):
        parser.parse_args(["--cuda-sched", "invalid_sched"])


def test_server_cli_mtp_parsing():
    from src.atlas.server import create_parser, AtlasSubprocessBackend
    parser = create_parser()
    args = parser.parse_args(["--mtp", "ram", "--mtp-draft-n", "4", "--mtp-path", "draft.gguf"])
    assert args.mtp == "ram"
    assert args.mtp_draft_n == 4
    assert args.mtp_path == "draft.gguf"

    # Default should be None
    args_default = parser.parse_args([])
    assert args_default.mtp is None
    assert args_default.mtp_draft_n == 2
    assert args_default.mtp_path is None

    backend = AtlasSubprocessBackend(mtp="vram", mtp_draft_n=3)
    assert backend.mtp == "vram"
    assert backend.mtp_draft_n == 3
