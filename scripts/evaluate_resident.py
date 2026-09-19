"""Fail-closed task smoke checks and token equivalence, separate from speed."""
import argparse
import json
import re
import statistics
from pathlib import Path

EXPECTED = {"math": r"^323$", "code": r"\[1,\s*2,\s*3\]", "logic": r"^yes\b"}
QUALITY_FLOOR = 80


def evaluate(baseline, candidate):
    before = {(r["case"], r["repetition"]): r for r in baseline["results"]}
    after = {(r["case"], r["repetition"]): r for r in candidate["results"]}
    if not before or before.keys() != after.keys() or {k[0] for k in before} != EXPECTED.keys():
        raise ValueError("Missing, mismatched, or unknown evaluation cases")
    rows = []
    for key, b in before.items():
        c = after[key]
        if b.get("prompt") != c.get("prompt"):
            raise ValueError("Cannot compare different prompts")
        for r in (b, c):
            if not r["token_ids"] or r["done"]["finish_reason"] not in ("stop", "length"):
                raise ValueError("Failed generation cannot pass evaluation")
        rows.append(dict(case=key[0], repetition=key[1],
                         exact_tokens=b["token_ids"] == c["token_ids"],
                         task_pass=bool(re.search(EXPECTED[key[0]], c["text"].strip(), re.I)),
                         ttft_speedup=b["ttft_ms"]/c["ttft_ms"],
                         effective_speedup=c["effective_tps"]/b["effective_tps"]))
    accuracy = 100 * sum(r["task_pass"] for r in rows)/len(rows)
    fidelity = 100 * sum(r["exact_tokens"] for r in rows)/len(rows)
    return dict(scope="Three task smoke checks; not a general coding/reasoning quality certification",
                task_accuracy_pct=accuracy, exact_sequence_match_pct=fidelity,
                smoke_quality_gate_pass=accuracy >= QUALITY_FLOOR and fidelity >= QUALITY_FLOOR,
                general_quality_status="NOT_EVALUATED", comparisons=rows,
                median_effective_speedup=statistics.median(r["effective_speedup"] for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = evaluate(json.loads(Path(a.baseline).read_text()), json.loads(Path(a.candidate).read_text()))
    Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["smoke_quality_gate_pass"] else 1)
