"""Summarize the fixed-input native audit without certifying general task quality."""
import argparse
import json
import statistics
from pathlib import Path


def compare(reference, candidate):
    def index(run):
        result = {}
        for row in run["results"]:
            key = (row["case"], row["repetition"])
            if key in result:
                raise ValueError(f"Duplicate case: {key}")
            if not row["token_ids"] or row["done"]["cached_prompt_tokens"] != 0:
                raise ValueError("This audit requires nonempty outputs and no reused prompt tokens")
            result[key] = row
        if not result:
            raise ValueError("Empty benchmark")
        return result

    a, b = index(reference), index(candidate)
    if a.keys() != b.keys():
        raise ValueError("Case sets differ")
    rows = []
    for key, before in a.items():
        after = b[key]
        if before["prompt"] != after["prompt"]:
            raise ValueError("Prompts differ")
        rows.append({
            "case": key[0], "repetition": key[1],
            "exact_tokens": before["token_ids"] == after["token_ids"],
            "reference_tokens": len(before["token_ids"]),
            "effective_speedup": after["effective_tps"] / before["effective_tps"],
            "decode_speedup": after["decode_tps"] / before["decode_tps"],
            "ttft_speedup": before["ttft_ms"] / after["ttft_ms"],
        })
    return {
        "general_task_quality": "NOT_EVALUATED",
        "scope": "Fixed-input greedy token fidelity; no prompt checkpoint hits",
        "exact_sequence_matches": sum(r["exact_tokens"] for r in rows),
        "cases": len(rows),
        "reference_tokens": sum(r["reference_tokens"] for r in rows),
        "all_sequences_match": all(r["exact_tokens"] for r in rows),
        "median_paired_speedup": {
            metric: statistics.median(r[metric] for r in rows)
            for metric in ("effective_speedup", "decode_speedup", "ttft_speedup")
        },
        "comparisons": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
