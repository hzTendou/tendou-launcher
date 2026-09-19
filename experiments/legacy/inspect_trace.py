"""Trace inspection script for Atlas analysis."""
import json
from collections import Counter, defaultdict

with open('trace_35b_a3b_q2k.jsonl') as f:
    sessions = [json.loads(l) for l in f if l.strip()]

print(f"Total sessions: {len(sessions)}")

s0 = sessions[0]
recs = s0['records']
r0 = recs[0]
layers0 = r0.get('experts_by_layer', {})
lkeys = sorted(layers0.keys(), key=lambda x: int(x))

print(f"Session 0: {len(recs)} records")
print(f"Record keys: {list(r0.keys())}")
print(f"Phase: {r0.get('phase', 'NONE')}")
print(f"Total layers: {len(lkeys)}")
print(f"Layer range: {lkeys[0]} to {lkeys[-1]}")
print(f"Experts per layer (sample): { {l: len(layers0[l]) for l in lkeys[:6]} }")
total_per_rec = sum(len(v) for v in layers0.values())
print(f"Total experts per record: {total_per_rec}")

# Examine k (experts per layer) consistency
k_by_layer = [len(v) for v in layers0.values()]
print(f"k distribution: min={min(k_by_layer)}, max={max(k_by_layer)}, mean={sum(k_by_layer)/len(k_by_layer):.1f}")

# Check how many unique experts appear across the whole trace
all_experts_by_layer = defaultdict(set)
expert_freq = defaultdict(Counter)
for s in sessions:
    for rec in s['records']:
        for layer, es in rec.get('experts_by_layer', {}).items():
            all_experts_by_layer[layer].update(es)
            for e in es:
                expert_freq[layer][e] += 1

total_unique_experts = sum(len(v) for v in all_experts_by_layer.values())
print(f"\nTotal unique experts across all layers: {total_unique_experts}")
for l in sorted(all_experts_by_layer.keys(), key=int)[:5]:
    freqs = expert_freq[l]
    vals = sorted(freqs.values(), reverse=True)
    print(f"  Layer {l}: {len(all_experts_by_layer[l])} unique experts, top freq={vals[0]}, median={vals[len(vals)//2]}")

# Token-to-token expert overlap in session 0 (per layer)
print("\nToken-to-token overlap within session 0 (by layer, first 5 tokens):")
layer_sample = lkeys[len(lkeys)//2]  # middle layer
for i in range(min(5, len(recs)-1)):
    ea = set(recs[i].get('experts_by_layer', {}).get(layer_sample, []))
    eb = set(recs[i+1].get('experts_by_layer', {}).get(layer_sample, []))
    overlap = ea & eb
    print(f"  Tok {i}->{i+1} layer={layer_sample}: {len(overlap)}/{len(ea)} experts repeat ({len(overlap)/len(ea)*100:.0f}%)")

# Compute expert repetition distance - how far back is each expert seen?
print("\nReuse distance analysis (across all sessions, decode records):")
all_reuse = []
for s in sessions:
    last_seen = defaultdict(dict)
    for step, rec in enumerate(s['records']):
        for layer, es in rec.get('experts_by_layer', {}).items():
            seen = last_seen[layer]
            for e in es:
                if e in seen:
                    all_reuse.append(step - seen[e])
                seen[e] = step

all_reuse.sort()
n = len(all_reuse)
print(f"  n={n}, median={all_reuse[n//2]}, P90={all_reuse[int(0.9*n)]}, P95={all_reuse[int(0.95*n)]}, P99={all_reuse[int(0.99*n)]}, max={all_reuse[-1]}")

# How many transitions are seen at least once?
print("\nTransition coverage analysis (order-1, within session, per layer sample):")
layer_trans = defaultdict(lambda: defaultdict(Counter))
for s in sessions:
    prev = None
    for rec in s['records']:
        cur = {l: set(es) for l, es in rec.get('experts_by_layer', {}).items()}
        if prev is not None:
            for layer, cur_set in cur.items():
                for pe in prev.get(layer, set()):
                    layer_trans[layer][pe].update(cur_set)
        prev = cur

total_source_experts = sum(len(v) for v in layer_trans.values())
print(f"  Total (layer, prev_expert) source nodes with transitions: {total_source_experts}")
sample_l = lkeys[len(lkeys)//2]
src_count = len(layer_trans[sample_l])
print(f"  Layer {sample_l}: {src_count} source experts have transitions out of {len(all_experts_by_layer[sample_l])} unique")

# Distribution of next-expert predictions
print("\nPrediction 'headroom' - how many next-experts are truly predictable?")
predictable = 0
total_actuals = 0
for s in sessions:
    prev = None
    for rec in s['records']:
        cur = {l: set(es) for l, es in rec.get('experts_by_layer', {}).items()}
        if prev is not None:
            for layer, cur_set in cur.items():
                total_actuals += len(cur_set)
                k = len(cur_set)
                # Can we predict any of these from prev?
                scores = Counter()
                for pe in prev.get(layer, set()):
                    scores.update(layer_trans[layer][pe])
                top_k = {e for e, _ in scores.most_common(k)}
                predictable += len(top_k & cur_set)
        prev = cur

print(f"  Predictable experts with top-k transition: {predictable}/{total_actuals} = {predictable/total_actuals*100:.1f}%")
print("  (This is the IDEAL causal transition recall - upper bound without oracle)")
