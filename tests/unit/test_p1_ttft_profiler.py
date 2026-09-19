import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.atlas.server import ChatMessage, format_chatml_prompt

# Assuming the profiler logic is refactored into testable functions or we test the main run loop.
import scripts.p1_ttft_profiler as profiler

def test_get_tasks_returns_three_tasks():
    tasks = profiler.get_tasks()
    assert len(tasks) == 3

def test_get_tasks_contains_short_workload():
    tasks = profiler.get_tasks()
    assert any(t["workload"] == "short_request" for t in tasks)

def test_get_tasks_contains_long_prompt():
    tasks = profiler.get_tasks()
    assert any(t["workload"] == "long_prompt" for t in tasks)

def test_get_tasks_contains_multi_turn():
    tasks = profiler.get_tasks()
    assert any(t["workload"] == "multi_turn" for t in tasks)

def test_short_request_has_correct_prompt():
    tasks = profiler.get_tasks()
    short = next(t for t in tasks if t["workload"] == "short_request")
    assert "17 kere 19 kac eder" in short["prompt"]

def test_long_prompt_length():
    tasks = profiler.get_tasks()
    long_task = next(t for t in tasks if t["workload"] == "long_prompt")
    assert len(long_task["prompt"]) > 1000

def test_multi_turn_has_three_turns():
    tasks = profiler.get_tasks()
    multi = next(t for t in tasks if t["workload"] == "multi_turn")
    assert len(multi["turns"]) == 3

@pytest.mark.asyncio
async def test_profiler_run_success(tmp_path):
    args = MagicMock()
    args.exe = Path("dummy.exe")
    args.model = Path("dummy.gguf")
    args.output_dir = tmp_path
    args.profile = "test_prof"
    args.repetitions = 1
    args.threads = 4

    backend = AsyncMock()
    
    async def mock_generate(*args, **kwargs):
        yield {"event": "token", "token": "hello"}
        yield {
            "event": "done",
            "ttft_ms": 100.0,
            "prefill_io_ms": 20.0,
            "prefill_compute_ms": 50.0,
            "checkpoint_ms": 10.0,
            "gpu_compute_ms": 15.0
        }
    
    backend.generate = mock_generate
    
    with patch("scripts.p1_ttft_profiler.AtlasSubprocessBackend", return_value=backend):
        await profiler.run(args)
    
    out_file = tmp_path / "ttft-breakdown-test_prof.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert len(data) == 5  # 1 short + 1 long + 3 turns
    assert data[0]["ttft_ms"] == 100.0
    assert data[0]["prefill_io_ms"] == 20.0

@pytest.mark.asyncio
async def test_profiler_run_error(tmp_path):
    args = MagicMock()
    args.exe = Path("dummy.exe")
    args.model = Path("dummy.gguf")
    args.output_dir = tmp_path
    args.profile = "test_err"
    args.repetitions = 1
    args.threads = 4

    backend = AsyncMock()
    
    async def mock_generate(*args, **kwargs):
        yield {"event": "error", "message": "fail"}
    
    backend.generate = mock_generate
    
    with patch("scripts.p1_ttft_profiler.AtlasSubprocessBackend", return_value=backend):
        with pytest.raises(RuntimeError):
            await profiler.run(args)

def test_format_chatml():
    prompt = format_chatml_prompt([ChatMessage(role="user", content="hello")])
    assert "<|im_start|>user" in prompt
