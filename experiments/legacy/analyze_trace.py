"""Atlas Engine trace analyzer.

The analyzer deliberately separates:
  * request/session boundaries;
  * prompt vs decode phases when the collector recorded them;
  * ordinary session-local LRU;
  * an oracle next-step prefetch upper bound;
  * an online, causal next-token expert predictor;
  * same-token next-layer predictability;
  * expert co-activation / affinity statistics.

It is backward compatible with the old trace format. Old traces have no explicit
phase field and are treated as decode-only, which is reported in the output.
"""

import argparse
import json
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path


def load_trace(path: str):
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {lineno}: {e}") from e
            if "records" not in obj:
                raise ValueError(f"Trace line {lineno} has no 'records' field")
            sessions.append(obj)
    return sessions


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] * (hi - k) + s[hi] * (k - lo))


def phase_of(rec):
    phase = rec.get("phase")
    if phase in {"prompt", "decode"}:
        return phase
    return "unknown"


def record_layers(rec):
    return rec.get("experts_by_layer", {}) or {}


def normalize_records(records):
    out = []
    for i, rec in enumerate(records):
        x = dict(rec)
        x["_step"] = i
        x["_phase"] = phase_of(rec)
        out.append(x)
    return out


def split_session_records(session):
    records = normalize_records(session.get("records", []))
    prompt = [r for r in records if r["_phase"] == "prompt"]
    decode = [r for r in records if r["_phase"] == "decode"]
    unknown = [r for r in records if r["_phase"] == "unknown"]
    if not prompt and not decode:
        # empty session
        return prompt, decode, unknown
    return prompt, decode, unknown

def effective_records_by_session(sessions, phase="decode"):
    result = []
    for session in sessions:
        records = normalize_records(session.get("records", []))
        if phase == "all":
            selected = records
        else:
            selected = [r for r in records if r["_phase"] == phase]
            if not selected and phase == "decode":
                selected = [r for r in records if r["_phase"] == "unknown"]
        if selected:
            result.append(selected)
    return result


def session_average_working_set(record_groups):
    per_session = []
    for records in record_groups:
        ws = working_set_size(records)
        if ws:
            per_session.append(sum(ws.values()) / len(ws))
    return per_session


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def layer_to_layer_overlap(records):
    vals = []
    for rec in records:
        layers = sorted(record_layers(rec), key=lambda x: int(x))
        for a, b in zip(layers, layers[1:]):
            vals.append(jaccard(record_layers(rec)[a], record_layers(rec)[b]))
    return sum(vals) / len(vals) if vals else 0.0


def token_to_token_overlap_per_layer(records):
    histories = defaultdict(list)
    for rec in records:
        for layer, experts in record_layers(rec).items():
            histories[layer].append(set(experts))
    result = {}
    for layer, hist in histories.items():
        vals = [jaccard(a, b) for a, b in zip(hist, hist[1:])]
        result[layer] = sum(vals) / len(vals) if vals else 0.0
    return result


def working_set_size(records):
    unique = defaultdict(set)
    for rec in records:
        for layer, experts in record_layers(rec).items():
            unique[layer].update(experts)
    return {layer: len(v) for layer, v in unique.items()}


def reuse_distances(records):
    last_seen = defaultdict(dict)
    distances = []
    for step, rec in enumerate(records):
        for layer, experts in record_layers(rec).items():
            seen = last_seen[layer]
            for e in set(experts):
                if e in seen:
                    distances.append(step - seen[e])
                seen[e] = step
    distances.sort()
    return distances


def cold_start_per_step(records):
    seen = defaultdict(set)
    result = []
    for rec in records:
        n = 0
        for layer, experts in record_layers(rec).items():
            s = seen[layer]
            for e in set(experts):
                if e not in s:
                    s.add(e)
                    n += 1
        result.append(n)
    return result


def lru_simulate(records, capacity):
    caches = defaultdict(OrderedDict)
    hits = misses = 0
    miss_per_step = []
    for rec in records:
        step_miss = 0
        for layer, experts in record_layers(rec).items():
            cache = caches[layer]
            for e in experts:
                if e in cache:
                    cache.move_to_end(e)
                    hits += 1
                else:
                    misses += 1
                    step_miss += 1
                    cache[e] = True
                    if len(cache) > capacity:
                        cache.popitem(last=False)
        miss_per_step.append(step_miss)
    return hits, misses, miss_per_step


def lru_simulate_sessions(sessions, capacity, phase="decode"):
    # Important: a new request/session gets a fresh cache. This avoids treating
    # unrelated prompts as one continuous request.
    total_h = total_m = 0
    per_session = []
    for session in sessions:
        records = normalize_records(session.get("records", []))
        if phase != "all":
            phase_records = [r for r in records if r["_phase"] == phase]
            if not phase_records:
                # backward-compatible old traces: all records are decode/unknown
                phase_records = records if phase == "decode" else []
        else:
            phase_records = records
        h, m, miss = lru_simulate(phase_records, capacity)
        total_h += h
        total_m += m
        per_session.append((h, m, miss))
    return total_h, total_m, per_session


def transition_predictor(records):
    """Causal same-layer next-token predictor.

    For each layer, each currently active expert votes for experts observed in the
    next record. The score is normalized only by ranking; top-k is taken from the
    accumulated transition counts. Training is online, so the current target is
    never used to predict itself.
    """
    transitions = defaultdict(lambda: defaultdict(Counter))
    prev_by_layer = None
    predictions = []
    actuals = []

    for idx, rec in enumerate(records):
        current = {l: set(es) for l, es in record_layers(rec).items()}
        if prev_by_layer is not None:
            pred_by_layer = {}
            for layer, actual in current.items():
                top_k = len(actual)
                scores = Counter()
                for prev_e in prev_by_layer.get(layer, set()):
                    scores.update(transitions[layer][prev_e])
                pred_by_layer[layer] = {e for e, _ in scores.most_common(top_k)}
            predictions.append(pred_by_layer)
            actuals.append(current)
        else:
            predictions.append({})
            actuals.append(current)

        if prev_by_layer is not None:
            for layer, prev_set in prev_by_layer.items():
                nxt = current.get(layer, set())
                for pe in prev_set:
                    transitions[layer][pe].update(nxt)
        prev_by_layer = current

    return predictions, actuals


def same_token_next_layer_predictor(records):
    """Online same-token layer L -> L+1 expert predictor."""
    transitions = defaultdict(lambda: defaultdict(Counter))
    observations = []
    precisions = []
    recalls = []
    for rec in records:
        layers = sorted(record_layers(rec), key=lambda x: int(x))
        for a, b in zip(layers, layers[1:]):
            src = set(record_layers(rec)[a])
            actual = set(record_layers(rec)[b])
            scores = Counter()
            for e in src:
                scores.update(transitions[(a, b)][e])
            if scores:
                pred = {e for e, _ in scores.most_common(len(actual))}
                precisions.append(len(pred & actual) / len(pred) if pred else 0.0)
                recalls.append(len(pred & actual) / len(actual) if actual else 0.0)
                observations.append((a, b, pred, actual))
            for e in src:
                transitions[(a, b)][e].update(actual)
    return precisions, recalls, observations


def prompt_profile_predictor(session):
    """Predict decode experts from prompt-phase expert frequencies.

    This is intentionally a weak baseline, not a semantic classifier. It tests
    whether prompt routing itself contains a useful signal for decode routing.
    """
    records = normalize_records(session.get("records", []))
    prompt = [r for r in records if r["_phase"] == "prompt"]
    decode = [r for r in records if r["_phase"] == "decode"]
    if not prompt or not decode:
        return None

    profile = defaultdict(Counter)
    for rec in prompt:
        for layer, experts in record_layers(rec).items():
            profile[layer].update(experts)

    recalls = []
    precisions = []
    for rec in decode:
        for layer, actual_list in record_layers(rec).items():
            actual = set(actual_list)
            k = len(actual)
            pred = {e for e, _ in profile[layer].most_common(k)}
            if pred:
                precisions.append(len(pred & actual) / len(pred))
            if actual:
                recalls.append(len(pred & actual) / len(actual))
    return {
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "n": len(recalls),
    }


def affinity_edges(records, min_count=2):
    """Count same-layer co-activation edges, then return strongest edges."""
    counts = Counter()
    for rec in records:
        for layer, experts in record_layers(rec).items():
            es = sorted(set(experts))
            for i, a in enumerate(es):
                for b in es[i + 1:]:
                    counts[(layer, a, b)] += 1
    return [(c, layer, a, b) for (layer, a, b), c in counts.items() if c >= min_count]


def region_discovery(records, min_edge_count=2, max_region_size=8):
    """Greedy affinity regions per layer.

    This is deliberately a deterministic discovery baseline, not a claim that
    these are semantic "reasoning" or "coding" experts.
    """
    edges = affinity_edges(records, min_edge_count)
    by_layer = defaultdict(list)
    for count, layer, a, b in sorted(edges, reverse=True):
        by_layer[layer].append((count, a, b))

    regions = {}
    rid = 0
    for layer, ledges in by_layer.items():
        adj = defaultdict(set)
        for _, a, b in ledges:
            adj[a].add(b)
            adj[b].add(a)
        unassigned = set(adj)
        while unassigned:
            seed = max(unassigned, key=lambda e: len(adj[e] & unassigned))
            region = [seed]
            unassigned.remove(seed)
            candidates = sorted(unassigned, key=lambda e: len(adj[e] & set(region)), reverse=True)
            for e in candidates:
                if len(region) >= max_region_size:
                    break
                if adj[e] & set(region):
                    region.append(e)
                    unassigned.remove(e)
            regions[rid] = {"layer": layer, "experts": sorted(region)}
            rid += 1
    return regions


def oracle_next_step_prefetch(records, capacity):
    """Oracle next-token prefetch simulation.

    At the end of step t, the simulator is allowed to know the exact expert set
    of step t+1 and load it into the layer cache. Step t itself still pays its
    normal misses. This is an upper bound for causal next-token prediction,
    not a physically implementable strategy.
    """
    caches = defaultdict(OrderedDict)
    hits = misses = 0
    miss_per_step = []

    for idx, rec in enumerate(records):
        step_miss = 0
        for layer, experts in record_layers(rec).items():
            cache = caches[layer]
            for e in experts:
                if e in cache:
                    cache.move_to_end(e)
                    hits += 1
                else:
                    misses += 1
                    step_miss += 1
                    cache[e] = True
                    if len(cache) > capacity:
                        cache.popitem(last=False)
        miss_per_step.append(step_miss)

        if idx + 1 < len(records):
            nxt = records[idx + 1]
            for layer, experts in record_layers(nxt).items():
                cache = caches[layer]
                for e in experts:
                    if e not in cache:
                        cache[e] = True
                        if len(cache) > capacity:
                            cache.popitem(last=False)
                    else:
                        cache.move_to_end(e)
    return hits, misses, miss_per_step


def predicted_next_step_prefetch(records, capacity):
    """Causal next-token prefetch using the online transition predictor.

    Prediction is generated from data available after the current step, then
    applied before the next step. Incorrect prefetch consumes cache space, so
    this measures both performance and pollution rather than granting free hits.
    """
    caches = defaultdict(OrderedDict)
    transitions = defaultdict(lambda: defaultdict(Counter))
    prev = None
    hits = misses = 0
    miss_per_step = []
    prefetched = actual_prefetched = 0
    false_prefetch = 0
    precision_vals = []
    recall_vals = []

    for idx, rec in enumerate(records):
        current = {l: set(es) for l, es in record_layers(rec).items()}
        step_miss = 0
        for layer, experts in current.items():
            cache = caches[layer]
            for e in experts:
                if e in cache:
                    cache.move_to_end(e)
                    hits += 1
                else:
                    misses += 1
                    step_miss += 1
                    cache[e] = True
                    if len(cache) > capacity:
                        cache.popitem(last=False)
        miss_per_step.append(step_miss)

        if prev is not None:
            for layer, prev_set in prev.items():
                nxt = current.get(layer, set())
                for pe in prev_set:
                    transitions[layer][pe].update(nxt)

        if idx + 1 < len(records):
            nxt = {l: set(es) for l, es in record_layers(records[idx + 1]).items()}
            for layer in set(nxt) | set(current):
                actual = nxt.get(layer, set())
                scores = Counter()
                for pe in current.get(layer, set()):
                    scores.update(transitions[layer][pe])
                pred = {e for e, _ in scores.most_common(len(actual))} if scores else set()
                if pred:
                    p = len(pred & actual) / len(pred)
                    r = len(pred & actual) / len(actual) if actual else 0.0
                    precision_vals.append(p)
                    recall_vals.append(r)
                prefetched += len(pred)
                actual_prefetched += len(pred & actual)
                false_prefetch += len(pred - actual)
                cache = caches[layer]
                for e in pred:
                    if e not in cache:
                        cache[e] = True
                        if len(cache) > capacity:
                            cache.popitem(last=False)
                    else:
                        cache.move_to_end(e)
        prev = current

    return {
        "hits": hits,
        "misses": misses,
        "miss_per_step": miss_per_step,
        "prefetched": prefetched,
        "correct_prefetch": actual_prefetched,
        "false_prefetch": false_prefetch,
        "precision": sum(precision_vals) / len(precision_vals) if precision_vals else 0.0,
        "recall": sum(recall_vals) / len(recall_vals) if recall_vals else 0.0,
    }


def distribution(label, values, unit="MB"):
    if not values:
        print(f"{label}: (məlumat yoxdur)")
        return
    s = sorted(values)
    print(
        f"{label}\n"
        f"    orta={sum(s)/len(s):8.1f}{unit}  "
        f"P50={percentile(s,50):8.1f}{unit}  "
        f"P75={percentile(s,75):8.1f}{unit}  "
        f"P90={percentile(s,90):8.1f}{unit}  "
        f"P95={percentile(s,95):8.1f}{unit}  "
        f"P99={percentile(s,99):8.1f}{unit}  "
        f"max={s[-1]:8.1f}{unit}"
    )


def print_cache_row(name, hits, misses, miss_per_step, n_records, expert_mb, target_tps, nvme):
    total = hits + misses
    hit = hits / total if total else 0.0
    mb = misses * expert_mb / n_records if n_records else 0.0
    gbps = mb * target_tps / 1024.0
    fits = "BƏLİ (idealized)" if gbps <= nvme else "XEYR"
    print(f"{name:>22} | {hit*100:>7.1f}% | {mb:>12.1f} | {gbps:>10.2f} | {fits:>18}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--expert-size-mb", type=float, required=True)
    ap.add_argument("--cache-sizes", default="16,32,64,128,256")
    ap.add_argument("--nvme-gbps", type=float, default=7.0)
    ap.add_argument("--target-tps", type=float, default=20.0)
    ap.add_argument("--affinity-top", type=int, default=20)
    ap.add_argument("--min-affinity-count", type=int, default=2)
    ap.add_argument("--region-size", type=int, default=8)
    ap.add_argument("--json-report", default=None, help="Optional machine-readable report path")
    args = ap.parse_args()

    sessions = load_trace(args.trace)
    cache_sizes = [int(x) for x in args.cache_sizes.split(",") if x.strip()]

    all_records = []
    decode_records = []
    prompt_records = []
    unknown_records = []
    for s in sessions:
        records = normalize_records(s.get("records", []))
        all_records.extend(records)
        prompt_records.extend(r for r in records if r["_phase"] == "prompt")
        decode_records.extend(r for r in records if r["_phase"] == "decode")
        unknown_records.extend(r for r in records if r["_phase"] == "unknown")

    # Old trace format fallback: each session remains an independent request.
    decode_groups = effective_records_by_session(sessions, "decode")
    effective_decode = [r for group in decode_groups for r in group]
    phase_mode = "explicit prompt/decode" if decode_records else "legacy decode-only"

    print(f"[+] {len(sessions)} sessiya yükləndi")
    print(f"[+] Trace fazası: {phase_mode}")
    print(f"[+] Records: all={len(all_records)} prompt={len(prompt_records)} decode={len(decode_records)} unknown={len(unknown_records)}\n")

    print("=" * 72)
    print("TEST 0 — TRACE VALIDATION / REQUEST BOUNDARIES")
    print("=" * 72)
    for i, s in enumerate(sessions):
        recs = normalize_records(s.get("records", []))
        p = sum(r["_phase"] == "prompt" for r in recs)
        d = sum(r["_phase"] == "decode" for r in recs)
        u = sum(r["_phase"] == "unknown" for r in recs)
        print(f"  session={i:>3} records={len(recs):>4} prompt={p:>4} decode={d:>4} unknown={u:>4} prompt_idx={s.get('prompt_idx', i)}")
    if not decode_records:
        print("  Qeyd: bu trace köhnə formatdadır; session record-ları decode kimi istifadə olunur.")
    print()

    print("=" * 72)
    print("TEST 1 — LOCALITY ANALYSIS (SESSION-LOCAL)")
    print("=" * 72)
    l2l_vals = [layer_to_layer_overlap(g) for g in decode_groups if g]
    t2t_vals = []
    for g in decode_groups:
        t = token_to_token_overlap_per_layer(g)
        if t:
            t2t_vals.append(sum(t.values()) / len(t))
    ws_per_session = session_average_working_set(decode_groups)
    l2l = sum(l2l_vals) / len(l2l_vals) if l2l_vals else 0.0
    avg_t2t = sum(t2t_vals) / len(t2t_vals) if t2t_vals else 0.0
    print(f"Layer-to-layer overlap (session orta): {l2l*100:.1f}%")
    print(f"Token-to-token overlap (session orta): {avg_t2t*100:.1f}%")
    if t2t_vals:
        print(f"  (session min={min(t2t_vals)*100:.1f}%, max={max(t2t_vals)*100:.1f}%)")
    avg_ws = sum(ws_per_session) / len(ws_per_session) if ws_per_session else 0.0
    print(f"Working-set (session orta): {avg_ws:.1f} expert/layer")
    if ws_per_session:
        print(f"  (session min={min(ws_per_session):.1f}, max={max(ws_per_session):.1f})")

    rdist = []
    for g in decode_groups:
        rdist.extend(reuse_distances(g))
    rdist.sort()
    if rdist:
        print(f"Reuse distance: n={len(rdist)} median={percentile(rdist,50):.1f} P90={percentile(rdist,90):.1f} P95={percentile(rdist,95):.1f} P99={percentile(rdist,99):.1f} max={rdist[-1]}")
    cold = []
    for g in decode_groups:
        cold.extend(cold_start_per_step(g))
    distribution("Cold-start MB/token:", [x * args.expert_size_mb for x in cold])
    print()

    print("=" * 72)
    print("TEST 2 — SESSION-LOCAL LRU vs OLD CROSS-SESSION BASELINE")
    print("=" * 72)
    print(f"{'Cache/layer':>22} | {'Hit %':>7} | {'orta MB/tok':>12} | {'Req. GB/s':>10} | {'Fits?':>18}")
    print("-" * 80)
    for cap in cache_sizes:
        h, m, per = lru_simulate_sessions(sessions, cap, "decode")
        # flatten miss distribution across sessions
        miss_steps = [x for _, _, ms in per for x in ms]
        print_cache_row(str(cap), h, m, miss_steps, len(miss_steps), args.expert_size_mb, args.target_tps, args.nvme_gbps)
    print("  Session sərhədi hər request üçün cache-i sıfırlayır; unrelated prompts artıq bir-birinə cache miras vermir.")
    print()

    print("=" * 72)
    print("TEST 3 — PROMPT → DECODE SIGNAL")
    print("=" * 72)
    if prompt_records:
        results = []
        for s in sessions:
            r = prompt_profile_predictor(s)
            if r:
                results.append(r)
        if results:
            p = sum(x["precision"] for x in results) / len(results)
            r = sum(x["recall"] for x in results) / len(results)
            print(f"Prompt-profile precision: {p*100:.1f}%")
            print(f"Prompt-profile recall:    {r*100:.1f}%")
            print("  Bu semantic classifier deyil; prompt routing frequency-sindən decode expert namizədi çıxaran zəif baseline-dır.")
        else:
            print("  Prompt/decode birlikdə olan uyğun session yoxdur.")
    else:
        print("  Ölçülə bilmədi: trace-də explicit prompt-phase record yoxdur.")
        print("  Yeni GGUF collector bu fazanı ayrıca yazacaq.")
    print()

    print("=" * 72)
    print("TEST 4 — EXPERT AFFINITY / REGION DISCOVERY")
    print("=" * 72)
    edges = affinity_edges(effective_decode, args.min_affinity_count)
    edges.sort(reverse=True)
    print(f"Affinity edges (count >= {args.min_affinity_count}): {len(edges)}")
    for count, layer, a, b in edges[:args.affinity_top]:
        print(f"  layer={layer:>3} E{a:<4} <-> E{b:<4}  co-activation={count}")
    regions = region_discovery(effective_decode, args.min_affinity_count, args.region_size)
    print(f"Discovered logical regions: {len(regions)}")
    for rid, region in list(regions.items())[:10]:
        print(f"  R{rid:04d} layer={region['layer']} experts={region['experts']}")
    print("  Qeyd: bunlar semantic 'reasoning/coding' qrupları deyil; yalnız trace-dən çıxan co-activation region-larıdır.")
    print()

    print("=" * 72)
    print("TEST 5 — SAME-TOKEN NEXT-LAYER PREDICTABILITY")
    print("=" * 72)
    pvals, rvals, _ = same_token_next_layer_predictor(effective_decode)
    if pvals:
        print(f"Next-layer prediction precision: {sum(pvals)/len(pvals)*100:.1f}%")
        print(f"Next-layer prediction recall:    {sum(rvals)/len(rvals)*100:.1f}%")
        print(f"Observations: {len(pvals)}")
    else:
        print("  Yetərli tarixçə yoxdur.")
    print()

    print("=" * 72)
    print("TEST 6 — PREFETCH SIMULATION: LRU vs ORACLE vs CAUSAL PREDICTOR")
    print("=" * 72)
    print(f"{'Cache':>8} | {'LRU hit':>8} | {'Oracle hit':>11} | {'Pred hit':>10} | {'Pred P':>8} | {'Pred R':>8} | {'LRU MB/t':>10} | {'Pred MB/t':>10}")
    print("-" * 100)
    prefetch_report = {}
    for cap in cache_sizes:
        h, m, per = lru_simulate_sessions(sessions, cap, "decode")
        # Oracle/predictor are run session-by-session to preserve request boundaries.
        oh = om = 0
        ph = pm = 0
        pp = rr = 0.0
        pn = 0
        lru_steps = [x for _, _, ms in per for x in ms]
        pred_steps = []
        pred_prefetched = pred_correct = pred_false = 0
        for s in sessions:
            recs = normalize_records(s.get("records", []))
            recs = [r for r in recs if r["_phase"] == "decode"] or [r for r in recs if r["_phase"] == "unknown"]
            if not recs:
                continue
            a,b,ms = oracle_next_step_prefetch(recs, cap)
            oh += a; om += b
            pr = predicted_next_step_prefetch(recs, cap)
            ph += pr["hits"]; pm += pr["misses"]
            pred_prefetched += pr["prefetched"]
            pred_correct += pr["correct_prefetch"]
            pred_false += pr["false_prefetch"]
            if pr["precision"] or pr["recall"]:
                pp += pr["precision"]; rr += pr["recall"]; pn += 1
            pred_steps.extend(pr["miss_per_step"])
        ltotal = h + m
        ototal = oh + om
        ptotal = ph + pm
        lhit = h / ltotal if ltotal else 0
        ohit = oh / ototal if ototal else 0
        phit = ph / ptotal if ptotal else 0
        lmb = sum(lru_steps) * args.expert_size_mb / len(lru_steps) if lru_steps else 0
        pmb = sum(pred_steps) * args.expert_size_mb / len(pred_steps) if pred_steps else 0
        precision = pred_correct / pred_prefetched if pred_prefetched else 0
        recall = pred_correct / (pred_correct + pred_false) if False else (pp / pn if pn else 0)
        # The per-step recall is the requested recall metric; aggregate correct/actual
        # is not directly recoverable because actual K differs only if malformed traces.
        recall = rr / pn if pn else 0
        print(f"{cap:>8} | {lhit*100:>7.1f}% | {ohit*100:>10.1f}% | {phit*100:>9.1f}% | {precision*100:>7.1f}% | {recall*100:>7.1f}% | {lmb:>9.1f} | {pmb:>9.1f}")
        prefetch_report[cap] = {
            "lru_hit": lhit, "oracle_hit": ohit, "pred_hit": phit,
            "pred_precision": precision, "pred_recall": recall,
            "lru_mb_per_token": lmb, "pred_mb_per_token": pmb,
        }
    print("  Oracle = növbəti token-un exact expert-lərini əvvəlcədən bilən nəzəri üst sərhəddir.")
    print("  Predictor = yalnız əvvəlki routing-dən istifadə edir; gələcəyi görmür.")
    print()

    print("=" * 72)
    print("TEST 7 — MB/TOKEN TAILS")
    print("=" * 72)
    for cap in cache_sizes:
        _, _, per = lru_simulate_sessions(sessions, cap, "decode")
        vals = [m * args.expert_size_mb for _, _, ms in per for m in ms]
        distribution(f"  LRU cache={cap}:", vals)
    print()

    print("=" * 72)
    print("XÜLASƏ / ATLAS QƏRAR METRİKLƏRİ")
    print("=" * 72)
    print("  1) LRU artıq session-lar arasında qarışdırılmır.")
    print("  2) Prompt→decode yalnız explicit phase trace olduqda ölçülür.")
    print("  3) Oracle ilə predictor arasında böyük boşluq varsa, prefetch üçün real optimizasiya sahəsi var.")
    print("  4) Predictor precision/recall yüksəkdirsə, yanlış prefetch yalnız bandwidth/cache pollution maliyyətidir; router dəyişdirilmir.")
    print("  5) Region discovery yalnız co-activation əsaslı storage/prefetch grouping-dir; semantic expert etiketi deyil.")
    print("  6) Növbəti real mərhələ: bu statistikaları real NVMe/RAM/VRAM transfer simulator və sonra real runtime ilə təsdiqləməkdir.")

    if args.json_report:
        report = {
            "trace": str(Path(args.trace)),
            "sessions": len(sessions),
            "records": {"all": len(all_records), "prompt": len(prompt_records), "decode": len(decode_records), "unknown": len(unknown_records)},
            "cache_sizes": cache_sizes,
            "prefetch": prefetch_report,
            "working_set_session_mean": avg_ws,
            "working_set_session_values": ws_per_session,
            "reuse_distance": {
                "n": len(rdist), "median": percentile(rdist,50), "p90": percentile(rdist,90),
                "p95": percentile(rdist,95), "p99": percentile(rdist,99), "max": max(rdist) if rdist else 0,
            },
            "regions": regions,
        }
        Path(args.json_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[+] JSON report: {args.json_report}")


if __name__ == "__main__":
    main()
