import json

import pytest

from scripts.p0_benchmark import (
    QUALITY_FLOOR,
    SCHEMA_VERSION,
    build_quality_tasks,
    compare_cache_modes,
    compare_runs,
    sampled_file_identity,
    validate_resumable_run,
)


def _run(token_ids, text, *, cached=0, workload="short_request"):
    return {
        "results": [{
            "task_id": "math-01",
            "category": "math",
            "workload": workload,
            "repetition": 0,
            "turn": 0,
            "prompt": "2+2?",
            "expected": "^4$",
            "token_ids": token_ids,
            "text": text,
            "ttft_ms": 10.0,
            "elapsed_s": 1.0,
            "effective_tps": 2.0,
            "decode_tps": 2.0,
            "done": {
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "cached_prompt_tokens": cached,
            },
        }],
    }


def test_quality_set_is_fixed_balanced_and_unique():
    tasks = build_quality_tasks()
    assert len(tasks) == 100
    assert len({task["task_id"] for task in tasks}) == 100
    assert len({task["prompt"] for task in tasks}) == 100
    assert {category: sum(t["category"] == category for t in tasks) for category in {
        "code", "math", "logic", "turkish", "instruction"
    }} == {
        "code": 20,
        "math": 20,
        "logic": 20,
        "turkish": 20,
        "instruction": 20,
    }
    assert QUALITY_FLOOR == pytest.approx(0.80)


def test_comparison_separates_task_accuracy_token_fidelity_and_cache():
    reference = _run([1, 2], "4", cached=0)
    candidate = _run([1, 3], "4", cached=2)
    report = compare_runs(reference, candidate)
    assert report["quality"]["task_accuracy_pct"] == 100.0
    assert report["fidelity"]["exact_token_match_pct"] == 0.0
    assert report["quality"]["passes_80_pct_floor"] is True
    assert report["cache"]["candidate_cached_prompt_tokens"] == 2
    assert report["general_quality_status"] == "NOT_EVALUATED"
    assert report["coverage"]["unique_prompts"] == 1


def test_cache_gain_is_reported_separately():
    cold = _run([1, 2], "4", cached=0)
    warm = _run([1, 2], "4", cached=3)
    warm["results"][0]["ttft_ms"] = 5.0
    warm["results"][0]["effective_tps"] = 4.0
    report = compare_cache_modes(cold, warm)
    assert report == {
        "cached_prompt_tokens": 3,
        "median_ttft_speedup": 2.0,
        "median_effective_speedup": 2.0,
    }


def test_comparison_rejects_unpaired_inputs():
    reference = _run([1], "4")
    candidate = _run([1], "4")
    candidate["results"][0]["prompt"] = "different"
    with pytest.raises(ValueError, match="prompt"):
        compare_runs(reference, candidate)


def test_sampled_identity_changes_when_file_edges_change(tmp_path):
    artifact = tmp_path / "runtime.bin"
    artifact.write_bytes(b"a" * 4096)
    first = sampled_file_identity(artifact, sample_bytes=64)
    artifact.write_bytes(b"b" + b"a" * 4094 + b"c")
    second = sampled_file_identity(artifact, sample_bytes=64)
    assert first["sampled_sha256"] != second["sampled_sha256"]
    json.dumps(second)


def test_resume_rejects_changed_runtime_identity(tmp_path):
    data = {
        "schema": SCHEMA_VERSION,
        "profile": "reference",
        "cache_mb": 0,
        "task_set_sha256": "tasks",
        "identity": {"exe": "old"},
        "threads": 14,
        "context": 4096,
        "repetitions": 3,
    }
    with pytest.raises(ValueError, match="identity"):
        validate_resumable_run(
            data, path=tmp_path / "run.json", profile="reference", cache_mb=0,
            task_hash="tasks", identities={"exe": "new"}, threads=14,
            context=4096, repetitions=3,
        )
