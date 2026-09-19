"""P0 paired runtime benchmark with independent quality and token-fidelity gates."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.server import (  # noqa: E402
    AtlasSubprocessBackend,
    ChatMessage,
    DEFAULT_EXE_PATH,
    DEFAULT_MODEL_PATH,
    format_chatml_prompt,
)


QUALITY_FLOOR = 0.80
SCHEMA_VERSION = "tendou-p0/v2"
READABLE_RUN_SCHEMAS = {"tendou-p0/v1", SCHEMA_VERSION}
RUNTIME_DLLS = ("llama.dll", "llama-common.dll", "ggml.dll", "ggml-cpu.dll", "ggml-cuda.dll")


def build_quality_tasks() -> list[dict[str, Any]]:
    """Return the versioned, deterministic 100-task quality set (20 per domain)."""
    tasks: list[dict[str, Any]] = []
    for index in range(20):
        a, b = 13 + index, 7 + (index % 6)
        tasks.append({
            "task_id": f"math-{index + 1:02d}", "category": "math", "workload": "short_request",
            "prompt": f"{a} ile {b} sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.",
            "expected": rf"^{a * b}$", "max_tokens": 16,
        })

    for index in range(20):
        values = (index + 2, index + 4, index + 6)
        expression = f"sum([{values[0]}, {values[1]}, {values[2]}])"
        tasks.append({
            "task_id": f"code-{index + 1:02d}", "category": "code", "workload": "short_request",
            "prompt": f"Python ifadesinin sonucunu yalnizca deger olarak yaz: {expression}",
            "expected": rf"^{sum(values)}$", "max_tokens": 24,
        })

    logic_subjects = (
        "laleler", "serceler", "kediler", "martilar", "cinarlar", "balinalar", "karincalar", "atmacalar",
        "menekseler", "kaplumbagalar", "yunuslar", "kargalar", "orkideler", "penguenler", "aslanlar",
        "leylekler", "papatyalar", "koalalar", "turnalar", "zambaklar",
    )
    for index in range(20):
        positive = index % 2 == 0
        subject = logic_subjects[index]
        premise = (f"Tum {subject} canlidir. Tum canlilar enerji kullanir." if positive
                   else f"Bazi {subject} hizli degildir. Ada bu {subject} grubundadir.")
        question = (f"Tum {subject} enerji kullanir mi?" if positive
                    else "Ada'nin hizli oldugu kesin midir?")
        tasks.append({
            "task_id": f"logic-{index + 1:02d}", "category": "logic", "workload": "short_request",
            "prompt": f"{premise} {question} Yalnizca evet veya hayir yaz.",
            "expected": r"^evet$" if positive else r"^hayir$", "max_tokens": 16,
        })

    names = ("Ayse", "Mehmet", "Deniz", "Ece", "Ali", "Selin", "Mert", "Zeynep", "Can", "Derya",
             "Emre", "Elif", "Bora", "Ceren", "Ozan", "Leyla", "Kerem", "Ipek", "Baris", "Aylin")
    objects = ("kitabi", "anahtari", "kalemi", "bileti", "defteri", "cantayi", "dosyayi", "telefonu",
               "gozlugu", "saati", "mektubu", "haritayi", "kutuyu", "fotografi", "cuzdani", "notu",
               "karti", "paketi", "sapkasini", "semsiyeyi")
    for index in range(20):
        name, obj = names[index], objects[index]
        tasks.append({
            "task_id": f"turkish-{index + 1:02d}", "category": "turkish", "workload": "short_request",
            "prompt": f"'{name} sabah {obj} masaya birakti ve aksam oradan aldi.' {obj.capitalize()} kim aldi? Yalnizca adi yaz.",
            "expected": rf"^{name}$", "max_tokens": 16,
        })

    words = ("atlas", "tendou", "router", "expert", "bellek", "katman", "model", "token", "onbellek",
             "islem", "aktarim", "kuyruk", "tahmin", "guven", "ornek", "matris", "vektor", "cekirdek",
             "sunucu", "istemci")
    for index in range(20):
        word = words[index]
        tasks.append({
            "task_id": f"instruction-{index + 1:02d}", "category": "instruction", "workload": "short_request",
            "prompt": f"Yalnizca BUYUK HARFLERLE '{word}' yaz; aciklama ve noktalama ekleme.",
            "expected": rf"^{word.upper()}$", "max_tokens": 16,
        })
    return tasks


def build_performance_tasks() -> list[dict[str, Any]]:
    long_context = "Tendou yerel bir MoE calistiricisidir. Router expert secimini degistirmez. " * 120
    return [
        {"task_id": "perf-short", "category": "performance", "workload": "short_request",
         "prompt": "17 kere 19 kac eder? Yalnizca sayiyi yaz.", "expected": r"^323$", "max_tokens": 16},
        {"task_id": "perf-long-generation", "category": "performance", "workload": "long_generation",
         "prompt": "Yerel MoE inference sistemlerinde bellek katmanlarini teknik olarak acikla.", "expected": r".+", "max_tokens": 128},
        {"task_id": "perf-long-prompt", "category": "performance", "workload": "long_prompt",
         "prompt": long_context + " Metindeki sistemin adini yalnizca iki kelimeyle yaz.",
         "expected": r"^tendou\s+launcher$", "max_tokens": 24},
        {"task_id": "perf-multi-turn", "category": "performance", "workload": "multi_turn",
         "turns": [
             {"prompt": "Benim tuttugum sayi 7. Yalnizca tamam yaz.", "expected": r"^tamam$"},
             {"prompt": "Benim tuttugum sayi 7. Sen tamam dedin. Sayinin iki katini yalnizca sayi olarak yaz.", "expected": r"^14$"},
             {"prompt": "Benim tuttugum sayi 7. Sen tamam dedin ve iki katinin 14 oldugunu buldun. Simdi bir fazlasini yaz.", "expected": r"^15$"},
         ], "max_tokens": 16},
    ]


def sampled_file_identity(path: Path, sample_bytes: int = 1024 * 1024) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if stat.st_size > sample_bytes:
            stream.seek(max(0, stat.st_size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sample_bytes_per_edge": sample_bytes, "sampled_sha256": digest.hexdigest()}


def full_file_identity(path: Path) -> dict[str, Any]:
    identity = sampled_file_identity(path, sample_bytes=0)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    identity.pop("sample_bytes_per_edge")
    identity.pop("sampled_sha256")
    identity["sha256"] = digest.hexdigest()
    return identity


def runtime_identity(exe: Path, model: Path) -> dict[str, Any]:
    model_pattern = re.sub(r"-00001-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model.name)
    shards = sorted(model.parent.glob(model_pattern)) or [model]
    dlls = {name: full_file_identity(exe.parent / name) for name in RUNTIME_DLLS if (exe.parent / name).is_file()}
    return {"exe": full_file_identity(exe), "runtime_dlls": dlls,
            "model_shards": [sampled_file_identity(shard) for shard in shards]}


def validate_resumable_run(data: dict[str, Any], *, path: Path, profile: str, cache_mb: int,
                           task_hash: str, identities: dict[str, Any], threads: int,
                           context: int, repetitions: int) -> None:
    expected = {
        "schema": SCHEMA_VERSION,
        "profile": profile,
        "cache_mb": cache_mb,
        "task_set_sha256": task_hash,
        "identity": identities,
        "threads": threads,
        "context": context,
        "repetitions": repetitions,
    }
    mismatches = [key for key, value in expected.items() if data.get(key) != value]
    if mismatches:
        raise ValueError(f"incompatible resumable run ({', '.join(mismatches)}): {path}")


def _rows_by_key(run: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = {}
    for row in run.get("results", []):
        key = (row["task_id"], row["repetition"], row.get("turn", 0))
        if key in rows:
            raise ValueError(f"duplicate result key: {key}")
        rows[key] = row
    if not rows:
        raise ValueError("empty benchmark run")
    return rows


def compare_runs(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before, after = _rows_by_key(reference), _rows_by_key(candidate)
    if before.keys() != after.keys():
        raise ValueError("reference and candidate result keys differ")
    comparisons = []
    for key, left in before.items():
        right = after[key]
        if left["prompt"] != right["prompt"]:
            raise ValueError(f"prompt differs for {key}")
        if not left["token_ids"] or not right["token_ids"]:
            raise ValueError(f"empty generation for {key}")
        expected = right.get("expected", r".+")
        comparisons.append({
            "task_id": key[0], "repetition": key[1], "turn": key[2],
            "category": right["category"], "workload": right["workload"],
            "task_pass": bool(re.fullmatch(expected, right["text"].strip(), re.I | re.S)),
            "exact_tokens": left["token_ids"] == right["token_ids"],
            "ttft_speedup": left["ttft_ms"] / right["ttft_ms"],
            "effective_speedup": right["effective_tps"] / left["effective_tps"],
        })
    accuracy = sum(row["task_pass"] for row in comparisons) / len(comparisons)
    exact = sum(row["exact_tokens"] for row in comparisons) / len(comparisons)
    workloads = sorted({row["workload"] for row in comparisons})
    unique_tasks = {row["task_id"] for row in comparisons}
    unique_prompts = {row["prompt"] for row in after.values()}
    quality_categories = {row["category"] for row in comparisons}
    general_evaluated = len(unique_tasks) >= 100 and len(unique_prompts) >= 100 and {
        "code", "math", "logic", "turkish", "instruction"
    }.issubset(quality_categories)
    category_metrics = {}
    for category in sorted(quality_categories):
        category_rows = [row for row in comparisons if row["category"] == category]
        category_metrics[category] = {
            "task_accuracy_pct": round(100 * sum(row["task_pass"] for row in category_rows) / len(category_rows), 3),
            "exact_token_match_pct": round(100 * sum(row["exact_tokens"] for row in category_rows) / len(category_rows), 3),
        }
    return {
        "scope": "Paired greedy runtime checks; general quality requires the 100-task quality suite",
        "general_quality_status": "EVALUATED" if general_evaluated else "NOT_EVALUATED",
        "quality": {"task_accuracy_pct": round(accuracy * 100, 3),
                    "quality_floor_pct": QUALITY_FLOOR * 100,
                    "passes_80_pct_floor": accuracy >= QUALITY_FLOOR},
        "fidelity": {"exact_token_match_pct": round(exact * 100, 3),
                     "all_tokens_match": exact == 1.0},
        "coverage": {"unique_task_ids": len(unique_tasks), "unique_prompts": len(unique_prompts),
                     "categories": category_metrics},
        "performance": {workload: {
            "median_ttft_speedup": statistics.median(r["ttft_speedup"] for r in comparisons if r["workload"] == workload),
            "median_effective_speedup": statistics.median(r["effective_speedup"] for r in comparisons if r["workload"] == workload),
        } for workload in workloads},
        "cache": {
            "reference_cached_prompt_tokens": sum(r["done"].get("cached_prompt_tokens", 0) for r in before.values()),
            "candidate_cached_prompt_tokens": sum(r["done"].get("cached_prompt_tokens", 0) for r in after.values()),
        },
        "comparisons": comparisons,
    }


def compare_cache_modes(cache_off: dict[str, Any], cache_on: dict[str, Any]) -> dict[str, Any]:
    cold, warm = _rows_by_key(cache_off), _rows_by_key(cache_on)
    if cold.keys() != warm.keys():
        raise ValueError("cache result keys differ")
    rows = []
    for key, left in cold.items():
        right = warm[key]
        if left["prompt"] != right["prompt"]:
            raise ValueError(f"cache prompt differs for {key}")
        rows.append({
            "workload": right["workload"],
            "ttft_speedup": left["ttft_ms"] / right["ttft_ms"],
            "effective_speedup": right["effective_tps"] / left["effective_tps"],
            "cached_prompt_tokens": right["done"].get("cached_prompt_tokens", 0),
        })
    return {
        "cached_prompt_tokens": sum(row["cached_prompt_tokens"] for row in rows),
        "median_ttft_speedup": statistics.median(row["ttft_speedup"] for row in rows),
        "median_effective_speedup": statistics.median(row["effective_speedup"] for row in rows),
    }


def comparison_summary(manifest: dict[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "output": str(output_path),
        "task_count": manifest["task_count"],
        "reports": {
            key: {
                "general_quality_status": value["general_quality_status"],
                "quality": value["quality"],
                "fidelity": value["fidelity"],
                "coverage": value["coverage"],
                "performance": value["performance"],
            }
            for key, value in manifest["reports_by_cache_mb"].items()
        },
    }


def _expanded_turns(task: dict[str, Any]) -> Iterable[tuple[int, str, str]]:
    for turn, item in enumerate(task.get("turns") or [{"prompt": task["prompt"], "expected": task["expected"]}]):
        yield turn, item["prompt"], item["expected"]


async def collect_run(args: argparse.Namespace, profile: str, tasks: list[dict[str, Any]], cache_mb: int,
                      task_hash: str, identities: dict[str, Any]) -> dict[str, Any]:
    aggressive = profile == "candidate"
    cand_k = getattr(args, "candidate_k", 5) if aggressive else 0
    extra_args = [
        "--atlas-k", str(cand_k),
        "--atlas-quality-target", str(QUALITY_FLOOR),
        "--atlas-prompt-cache-mb", str(cache_mb),
    ]
    backend = AtlasSubprocessBackend(
        exe_path=str(args.exe), model_path=str(args.model), threads=args.threads, ctx_size=args.context,
        boost=aggressive,
        odmoe_lead=4 if aggressive else 2,
        spice_conf_high=0.50 if aggressive else 0.65,
        spice_conf_mid=0.15 if aggressive else 0.30,
        tutti_async_io=aggressive,
        readback_interval=8,
        prefetch_candidates=16 if aggressive else 10,
        tutti_queue_depth=128 if aggressive else 64,
        p2_gpu_binding=aggressive,
        p2_vram_budget_mib=1600 if aggressive else 0,
        p3_async_transfer=aggressive,
        extra_args=extra_args,
    )
    progress_path = args.output.resolve() / f"{profile}-cache-{cache_mb}.partial.json"
    results = []
    if args.resume and progress_path.is_file():
        checkpoint = json.loads(progress_path.read_text(encoding="utf-8"))
        validate_resumable_run(checkpoint, path=progress_path, profile=profile, cache_mb=cache_mb,
                               task_hash=task_hash, identities=identities, threads=args.threads,
                               context=args.context, repetitions=args.repetitions)
        results = checkpoint.get("results", [])
    completed = {(row["task_id"], row["repetition"], row.get("turn", 0)) for row in results}
    started = time.perf_counter()
    try:
        await backend.start()
        startup_s = time.perf_counter() - started
        for task in tasks:
            for repetition in range(args.repetitions):
                for turn, prompt, expected in _expanded_turns(task):
                    if (task["task_id"], repetition, turn) in completed:
                        continue
                    prompt = format_chatml_prompt([ChatMessage(role="user", content=prompt)])
                    row_start = time.perf_counter()
                    first = last = None
                    token_ids, pieces, done = [], [], None
                    async for event in backend.generate(prompt, n_predict=task["max_tokens"], temp=0):
                        if event["event"] == "error":
                            raise RuntimeError(event)
                        if event["event"] == "token":
                            last = time.perf_counter()
                            first = first or last
                            token_ids.append(event["token_id"])
                            pieces.append(event["token"])
                        elif event["event"] == "done":
                            done = event
                    elapsed = time.perf_counter() - row_start
                    if first is None or last is None or done is None:
                        raise RuntimeError(f"incomplete generation: {task['task_id']}")
                    results.append({
                        "task_id": task["task_id"], "category": task["category"],
                        "workload": task["workload"], "repetition": repetition, "turn": turn,
                        "prompt": prompt, "expected": expected, "token_ids": token_ids,
                        "text": "".join(pieces), "ttft_ms": (first - row_start) * 1000,
                        "elapsed_s": elapsed, "effective_tps": len(token_ids) / elapsed,
                        "decode_tps": ((len(token_ids) - 1) / (last - first)) if len(token_ids) > 1 else None,
                        "done": done,
                    })
                    progress_path.write_text(json.dumps({
                        "schema": SCHEMA_VERSION, "profile": profile, "cache_mb": cache_mb,
                        "task_set_sha256": task_hash, "identity": identities,
                        "threads": args.threads, "context": args.context,
                        "repetitions": args.repetitions, "complete": False, "results": results,
                    }, indent=2), encoding="utf-8")
    finally:
        await backend.stop()
    progress_path.unlink(missing_ok=True)
    return {"schema": SCHEMA_VERSION, "profile": profile, "cache_mb": cache_mb,
            "threads": args.threads, "context": args.context, "startup_s": startup_s,
            "repetitions": args.repetitions,
            "results": results}


async def run(args: argparse.Namespace) -> None:
    if args.recompare_existing:
        recompare_existing(args)
        return
    tasks = build_performance_tasks()
    if args.suite in ("quality", "all"):
        tasks = build_quality_tasks() if args.suite == "quality" else tasks + build_quality_tasks()
    task_hash = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identities = runtime_identity(args.exe, args.model)
    cache_modes = [0, 256] if args.cache == "both" else [0 if args.cache == "off" else 256]
    reports = {}
    all_runs: dict[int, dict[str, dict[str, Any]]] = {}
    for cache_mb in cache_modes:
        runs = {}
        for profile in ("reference", "candidate"):
            path = output / f"{profile}-cache-{cache_mb}.json"
            if args.resume and path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                validate_resumable_run(data, path=path, profile=profile, cache_mb=cache_mb,
                                       task_hash=task_hash, identities=identities, threads=args.threads,
                                       context=args.context, repetitions=args.repetitions)
            else:
                data = await collect_run(args, profile, tasks, cache_mb, task_hash, identities)
                data["identity"] = identities
                data["task_set_sha256"] = task_hash
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            runs[profile] = data
        reports[str(cache_mb)] = compare_runs(runs["reference"], runs["candidate"])
        all_runs[cache_mb] = runs
    cache_effects = None
    if 0 in all_runs and 256 in all_runs:
        cache_effects = {profile: compare_cache_modes(all_runs[0][profile], all_runs[256][profile])
                         for profile in ("reference", "candidate")}
    manifest = {"schema": SCHEMA_VERSION, "task_set_sha256": task_hash,
                "task_count": len(tasks), "repetitions": args.repetitions,
                "serial_execution": True, "reports_by_cache_mb": reports,
                "cache_effects": cache_effects}
    comparison_path = output / "comparison.json"
    comparison_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(comparison_summary(manifest, comparison_path), indent=2))


def recompare_existing(args: argparse.Namespace) -> None:
    """Regenerate a comparison from immutable run artifacts without launching the model."""
    output = args.output.resolve()
    cache_modes = [0, 256] if args.cache == "both" else [0 if args.cache == "off" else 256]
    reports = {}
    all_runs: dict[int, dict[str, dict[str, Any]]] = {}
    task_hash = None
    for cache_mb in cache_modes:
        runs = {}
        for profile in ("reference", "candidate"):
            path = output / f"{profile}-cache-{cache_mb}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") not in READABLE_RUN_SCHEMAS:
                raise ValueError(f"incompatible completed run: {path}")
            current_hash = data.get("task_set_sha256")
            if task_hash is not None and current_hash != task_hash:
                raise ValueError(f"task set differs in completed run: {path}")
            task_hash = current_hash
            runs[profile] = data
        reports[str(cache_mb)] = compare_runs(runs["reference"], runs["candidate"])
        all_runs[cache_mb] = runs
    first_run = all_runs[cache_modes[0]]["reference"]
    unique_tasks = {row["task_id"] for row in first_run["results"]}
    repetitions = max(row["repetition"] for row in first_run["results"]) + 1
    cache_effects = None
    if 0 in all_runs and 256 in all_runs:
        cache_effects = {profile: compare_cache_modes(all_runs[0][profile], all_runs[256][profile])
                         for profile in ("reference", "candidate")}
    manifest = {"schema": SCHEMA_VERSION, "task_set_sha256": task_hash,
                "task_count": len(unique_tasks), "repetitions": repetitions,
                "serial_execution": True, "source": "existing_runs",
                "reports_by_cache_mb": reports, "cache_effects": cache_effects}
    comparison_path = output / "comparison.json"
    comparison_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(comparison_summary(manifest, comparison_path), indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=Path(DEFAULT_EXE_PATH))
    p.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL_PATH))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", choices=("performance", "quality", "all"), default="performance")
    p.add_argument("--cache", choices=("off", "on", "both"), default="both")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--candidate-k", type=int, default=5, help="Active K for candidate profile (default: 5, Clipgfy/MoE pruning)")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--resume", action="store_true", help="Continue compatible partial runs and reuse completed profile JSON")
    p.add_argument("--recompare-existing", action="store_true",
                   help="Regenerate comparison.json from existing profile JSON without running inference")
    return p


if __name__ == "__main__":
    cli_args = parser().parse_args()
    if cli_args.repetitions < 3:
        raise SystemExit("P0 comparisons require at least 3 paired repetitions")
    asyncio.run(asyncio.wait_for(run(cli_args), timeout=24 * 60 * 60))
