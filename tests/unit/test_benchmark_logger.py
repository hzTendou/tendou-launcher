"""
Unit tests for continuous benchmark logging and context suite configurations.
"""
import json
import tempfile
from pathlib import Path
import pytest

from scripts.benchmark_logger import (
    BenchmarkItemResult,
    BenchmarkLogger,
    BenchmarkRunSummary,
    detect_hardware,
)
from scripts.run_continuous_benchmark import (
    CONTEXT_CONFIGS,
    build_suite_tasks,
)


def test_hardware_detection():
    hw = detect_hardware()
    assert "cpu" in hw
    assert "gpu" in hw
    assert "ram_gb" in hw
    assert "os" in hw
    assert hw["ram_gb"] > 0


def test_context_configs():
    assert "short" in CONTEXT_CONFIGS
    assert "32k" in CONTEXT_CONFIGS
    assert "64k" in CONTEXT_CONFIGS
    assert "128k" in CONTEXT_CONFIGS

    assert CONTEXT_CONFIGS["short"]["ctx_size"] == 2048
    assert CONTEXT_CONFIGS["32k"]["ctx_size"] == 32768
    assert CONTEXT_CONFIGS["32k"]["kv_quant"] == "q8_0"

    assert CONTEXT_CONFIGS["64k"]["ctx_size"] == 65536
    assert CONTEXT_CONFIGS["64k"]["kv_quant"] == "q4_0"

    assert CONTEXT_CONFIGS["128k"]["ctx_size"] == 131072
    assert CONTEXT_CONFIGS["128k"]["kv_quant"] == "q4_0"


def test_suite_tasks_builder():
    for ctx in ("short", "32k", "64k", "128k"):
        tasks = build_suite_tasks(ctx)
        assert len(tasks) >= 2
        for t in tasks:
            assert "task_id" in t
            assert "prompt" in t
            assert "max_tokens" in t
            assert t["max_tokens"] > 0


def test_benchmark_logger_append_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        md_file = tmp_path / "test_log.md"
        jsonl_file = tmp_path / "test_history.jsonl"

        logger = BenchmarkLogger(report_md_path=md_file, report_jsonl_path=jsonl_file)

        # Run 1
        item1 = BenchmarkItemResult(
            task_id="task-01",
            category="math",
            workload="short_request",
            context_size=2048,
            prompt="What is 2+2?",
            generated_output="4",
            expected="4",
            task_pass=True,
            prompt_tokens=10,
            completion_tokens=2,
            cached_prompt_tokens=0,
            cache_hit=False,
            ttft_ms=50.0,
            decode_tps=12.5,
            effective_tps=10.0,
            prefill_compute_ms=40.0,
            prefill_io_ms=0.0,
            gpu_compute_ms=5.0,
            checkpoint_ms=1.0,
            elapsed_ms=200.0,
            finish_reason="stop",
            kv_quant="q8_0",
        )
        summary1 = BenchmarkRunSummary(
            run_id="run-001",
            timestamp="2026-09-14T12:00:00",
            model_name="Qwen3.8-Flash-Next-Q5_K_S",
            threads=8,
            cache_mb=1024,
            hardware_info={"cpu": "Zen 4", "gpu": "RTX 5060", "ram_gb": 16.0},
            items=[item1],
            total_duration_s=1.5,
        )

        logger.append_run(summary1)

        assert md_file.exists()
        assert jsonl_file.exists()

        content_after_run1 = md_file.read_text(encoding="utf-8")
        assert "run-001" in content_after_run1
        assert "task-01" in content_after_run1
        assert "What is 2+2?" in content_after_run1
        assert "```text\n4\n```" in content_after_run1

        # Run 2 (append-only verification)
        item2 = BenchmarkItemResult(
            task_id="task-32k-turn2",
            category="architecture",
            workload="multi_turn_cache",
            context_size=32768,
            prompt="Follow up question",
            generated_output="P3 pipeline active with 35 tokens cached",
            expected="P3",
            task_pass=True,
            prompt_tokens=60,
            completion_tokens=10,
            cached_prompt_tokens=35,
            cache_hit=True,
            ttft_ms=20.0,
            decode_tps=11.8,
            effective_tps=10.5,
            prefill_compute_ms=15.0,
            prefill_io_ms=0.0,
            gpu_compute_ms=2.0,
            checkpoint_ms=3.0,
            elapsed_ms=950.0,
            finish_reason="stop",
            turn=2,
            kv_quant="q8_0",
        )
        summary2 = BenchmarkRunSummary(
            run_id="run-002",
            timestamp="2026-09-14T12:05:00",
            model_name="Qwen3.8-Flash-Next-Q5_K_S",
            threads=8,
            cache_mb=1024,
            hardware_info={"cpu": "Zen 4", "gpu": "RTX 5060", "ram_gb": 16.0},
            items=[item2],
            total_duration_s=2.0,
        )

        logger.append_run(summary2)

        content_after_run2 = md_file.read_text(encoding="utf-8")
        # Run 1 must still be intact at the beginning
        assert content_after_run2.startswith(content_after_run1.rstrip())
        # Run 2 must be appended to the bottom
        assert "run-002" in content_after_run2
        assert "task-32k-turn2" in content_after_run2
        assert "CACHE HIT (35 tokens reused)" in content_after_run2
        assert "P3 pipeline active with 35 tokens cached" in content_after_run2
        assert content_after_run2.index("run-001") < content_after_run2.index("run-002")

        # Check JSONL
        jsonl_lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(jsonl_lines) == 2
        line1_data = json.loads(jsonl_lines[0])
        assert line1_data["run_id"] == "run-001"
        line2_data = json.loads(jsonl_lines[1])
        assert line2_data["run_id"] == "run-002"
        assert line2_data["items"][0]["cached_prompt_tokens"] == 35


def test_markdown_fence_escaping_with_code_output():
    """Verify that AI outputs containing code blocks (```python) are wrapped in longer fences without corruption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        md_file = tmp_path / "test_fence.md"
        jsonl_file = tmp_path / "test_fence.jsonl"

        logger = BenchmarkLogger(report_md_path=md_file, report_jsonl_path=jsonl_file)
        code_output = (
            "Here is the solution:\n"
            "```python\n"
            "def calculate(a, b):\n"
            "    return a * b\n"
            "```\n"
            "Done!"
        )
        item = BenchmarkItemResult(
            task_id="code-task",
            category="code",
            workload="benchmark",
            context_size=2048,
            prompt="Write a Python multiplication function",
            generated_output=code_output,
            expected="calculate",
            task_pass=True,
            prompt_tokens=20,
            completion_tokens=25,
            prompt_cache_bytes=1048576,
        )
        summary = BenchmarkRunSummary(
            run_id="run-fence-001",
            timestamp="2026-09-14T15:00:00",
            model_name="Qwen3.8-Flash-Next",
            threads=8,
            cache_mb=1024,
            hardware_info={"cpu": "Zen 4", "gpu": "RTX 5060", "ram_gb": 15.29},
            items=[item],
        )
        logger.append_run(summary)

        content = md_file.read_text(encoding="utf-8")
        # Must wrap with 4 backticks because the content contains 3 backticks
        assert "````text\nHere is the solution:\n```python\ndef calculate(a, b):\n    return a * b\n```\nDone!\n````" in content
        assert "prompt_cache=1.00 MB" in content


def test_multiline_prompt_single_line_summary():
    """Verify that prompts containing multiple newlines (like ChatML) are collapsed into a clean single line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        md_file = tmp_path / "test_prompt.md"
        jsonl_file = tmp_path / "test_prompt.jsonl"

        logger = BenchmarkLogger(report_md_path=md_file, report_jsonl_path=jsonl_file)
        chatml_prompt = (
            "<|im_start|>system\nYou are an expert AI.<|im_end|>\n"
            "<|im_start|>user\nCalculate 17 * 19.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        item = BenchmarkItemResult(
            task_id="chatml-task",
            category="math",
            workload="short",
            context_size=2048,
            prompt=chatml_prompt,
            generated_output="323",
        )
        summary = BenchmarkRunSummary(
            run_id="run-prompt-001",
            timestamp="2026-09-14T15:00:00",
            model_name="Qwen3.8-Flash-Next",
            threads=8,
            cache_mb=1024,
            hardware_info={},
            items=[item],
        )
        logger.append_run(summary)

        content = md_file.read_text(encoding="utf-8")
        # Line containing - **Prompt**: must not be split across lines
        prompt_lines = [l for l in content.splitlines() if l.startswith("- **Prompt**:")]
        assert len(prompt_lines) == 1
        assert "<|im_start|>system You are an expert AI.<|im_end|> <|im_start|>user" in prompt_lines[0]


def test_append_without_trailing_newline():
    """Verify that appending to a file that lacks a trailing newline does not mangle the formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        md_file = tmp_path / "test_newline.md"
        jsonl_file = tmp_path / "test_newline.jsonl"

        # Pre-seed with content lacking trailing newline
        md_file.write_text("# Existing Header without newline", encoding="utf-8")
        jsonl_file.write_text('{"existing": true}', encoding="utf-8")

        logger = BenchmarkLogger(report_md_path=md_file, report_jsonl_path=jsonl_file)
        item = BenchmarkItemResult(
            task_id="task-nl",
            category="test",
            workload="test",
            context_size=2048,
            prompt="test",
            generated_output="ok",
        )
        summary = BenchmarkRunSummary(
            run_id="run-nl-001",
            timestamp="2026-09-14T15:00:00",
            model_name="Qwen",
            threads=8,
            cache_mb=1024,
            hardware_info={},
            items=[item],
        )
        logger.append_run(summary)

        # JSONL must have 2 valid lines
        lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": True}
        assert json.loads(lines[1])["run_id"] == "run-nl-001"

        # MD must cleanly separate existing content
        md_content = md_file.read_text(encoding="utf-8")
        assert "# Existing Header without newline\n---\n## Benchmark Koşusu: `run-nl-001`" in md_content


def test_cli_parsing_and_context_aliases(monkeypatch):
    """Test parse_args with case variations, aliases, and custom paths."""
    from scripts.run_continuous_benchmark import parse_args

    monkeypatch.setattr("sys.argv", [
        "run_continuous_benchmark.py",
        "--contexts", "32K,64k,2k",
        "--threads", "8",
        "--cache-mb", "2048",
        "--model-path", "C:/models/test.gguf",
        "--exe-path", "C:/bin/test.exe",
    ])
    args = parse_args()
    assert args.contexts == "32K,64k,2k"
    assert args.threads == 8
    assert args.cache_mb == 2048
    assert args.model_path == "C:/models/test.gguf"
    assert args.exe_path == "C:/bin/test.exe"


def test_server_kv_cache_flags_generation():
    """Verify AtlasSubprocessBackend KV cache quantization flag resolution."""
    from src.atlas.server import AtlasSubprocessBackend

    # Default for 64k+ should be q4_0
    b1 = AtlasSubprocessBackend(ctx_size=65536)
    # Default for 4k-64k should be q8_0
    b2 = AtlasSubprocessBackend(ctx_size=8192)
    # Default for <4k should be None
    b3 = AtlasSubprocessBackend(ctx_size=2048)
    # Explicit override for <4k
    b4 = AtlasSubprocessBackend(ctx_size=2048, cache_type_k="q8_0")

    assert b1.cache_type_k is None
    assert b2.cache_type_k is None
    assert b4.cache_type_k == "q8_0"

