"""
CRITICAL VERIFICATION: Is the set-signature result overfitting?

The 35B trace has:
- 12 sessions
- ~756 decode steps (63 per session)
- Each step has 40 layers with 8 experts each = 320 experts
- The current SET of 8 experts at each layer = frozenset signature

Key question: Are the set-signatures REPEATING across sessions?
If yes, the 98.6% set-signature recall is a real signal.
If no (each session has unique set-signatures), it's memorization.

We need to test on HELD-OUT sessions.
"""
import json
from collections import Counter, defaultdict

with open('trace_35b_a3b_q2k.jsonl') as f:
    sessions = [json.loads(l) for l in f if l.strip()]

def get_decode_recs(session):
    recs = session['records']
    decode = [r for r in recs if r.get('phase') == 'decode']
    if decode:
        return decode
    return recs

def layer_experts(rec):
    return {str(l): set(int(e) for e in es) for l, es in rec.get('experts_by_layer', {}).items()}

print(f"Total sessions: {len(sessions)}")
print(f"Decode records per session: {[len(get_decode_recs(s)) for s in sessions]}")

# Count how many times each set-signature appears across ALL sessions
all_sigs = defaultdict(int)
for s in sessions:
    recs = get_decode_recs(s)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                prev_set = prev.get(layer, set())
                sig = frozenset(prev_set)
                all_sigs[(layer, sig)] += 1
        prev = cur

print(f"\nTotal unique set-signatures across all sessions: {len(all_sigs)}")
counts = sorted(all_sigs.values(), reverse=True)
print(f"Signature frequency: max={counts[0]}, median={counts[len(counts)//2]}, singletons={sum(1 for c in counts if c==1)}/{len(counts)}")

# CRITICAL TEST: Leave-one-out cross validation
print("\n" + "="*70)
print("LEAVE-ONE-OUT CROSS VALIDATION")
print("="*70)
print("Train on N-1 sessions, test on held-out session")
print("If set-signature recall drops on held-out data, it's memorization.")

# Build models
def build_models(train_sessions):
    """Build individual transition and set-signature transition models."""
    trans = defaultdict(lambda: defaultdict(Counter))
    set_trans = defaultdict(lambda: defaultdict(Counter))
    for s in train_sessions:
        recs = get_decode_recs(s)
        prev = None
        for rec in recs:
            cur = layer_experts(rec)
            if prev is not None:
                for layer, cur_set in cur.items():
                    prev_set = prev.get(layer, set())
                    sig = frozenset(prev_set)
                    for pe in prev_set:
                        trans[layer][pe].update(cur_set)
                    set_trans[layer][sig].update(cur_set)
            prev = cur
    return trans, set_trans

def evaluate_on_session(test_session, trans, set_trans):
    ind_correct = ind_total = 0
    sig_correct = sig_total = 0
    sig_hit = 0  # times the set-sig was found in training data
    
    recs = get_decode_recs(test_session)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                k = len(cur_set)
                prev_set = prev.get(layer, set())
                sig = frozenset(prev_set)
                
                # Individual transition
                scores = Counter()
                for pe in prev_set:
                    scores.update(trans[layer][pe])
                ind_pred = {e for e, _ in scores.most_common(k)}
                ind_correct += len(ind_pred & cur_set)
                ind_total += k
                
                # Set-signature
                sig_scores = set_trans[layer].get(sig, Counter())
                if sig_scores:
                    sig_pred = {e for e, _ in sig_scores.most_common(k)}
                    sig_hit += 1
                else:
                    sig_pred = ind_pred  # fallback to ind
                sig_correct += len(sig_pred & cur_set)
                sig_total += k
        prev = cur
    
    return ind_correct, ind_total, sig_correct, sig_total, sig_hit

loo_ind_correct = loo_ind_total = 0
loo_sig_correct = loo_sig_total = 0
loo_sig_hits = 0

for test_idx in range(len(sessions)):
    train = [s for i, s in enumerate(sessions) if i != test_idx]
    trans, set_trans = build_models(train)
    ic, it, sc, st, sh = evaluate_on_session(sessions[test_idx], trans, set_trans)
    loo_ind_correct += ic
    loo_ind_total += it
    loo_sig_correct += sc
    loo_sig_total += st
    loo_sig_hits += sh

print(f"\nLOO Individual recall:      {loo_ind_correct/loo_ind_total*100:.1f}%")
print(f"LOO Set-signature recall:   {loo_sig_correct/loo_sig_total*100:.1f}%")
print(f"LOO Sig-hit rate:           {loo_sig_hits}/{loo_sig_total} = {loo_sig_hits/loo_sig_total*100:.1f}% of steps had sig in training data")
print()
print("INTERPRETATION:")
print("  If LOO set-sig recall is MUCH lower than 98.6%, the 98.6% was memorization.")
print("  If LOO set-sig recall remains high, the set-signature is a genuine signal.")

print()
print("="*70)
print("ANALYSIS: How much cross-session sharing do expert SETS have?")
print("="*70)

# Count how many unique 'prev set signatures' appear in multiple sessions
sig_sessions = defaultdict(set)
for si, s in enumerate(sessions):
    recs = get_decode_recs(s)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                prev_set = prev.get(layer, set())
                sig = frozenset(prev_set)
                sig_sessions[(layer, sig)].add(si)
        prev = cur

cross_sess = sum(1 for sess_set in sig_sessions.values() if len(sess_set) > 1)
single_sess = sum(1 for sess_set in sig_sessions.values() if len(sess_set) == 1)
print(f"Set-signatures appearing in >1 sessions: {cross_sess}/{len(sig_sessions)} = {cross_sess/len(sig_sessions)*100:.1f}%")
print(f"Set-signatures unique to one session:    {single_sess}/{len(sig_sessions)} = {single_sess/len(sig_sessions)*100:.1f}%")

print()
print("="*70)
print("ROOT CAUSE DIAGNOSIS: What is the effective causal recall upper bound?")
print("="*70)
# The theoretical upper bound from individual transitions is what matters.
# With MORE training data (more sessions), would individual recall improve?
# Estimate by looking at transition frequency distribution.

all_trans = defaultdict(lambda: defaultdict(Counter))
for s in sessions:
    recs = get_decode_recs(s)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                for pe in prev.get(layer, set()):
                    all_trans[layer][pe].update(cur_set)
        prev = cur

# For each (layer, prev_expert), what fraction of transitions are to
# the top-1 next expert? This measures "determinism" at individual level.
top1_probs = []
for layer, src_dict in all_trans.items():
    for pe, counts in src_dict.items():
        total = sum(counts.values())
        if total < 2:
            continue
        top_count = counts.most_common(1)[0][1]
        top1_probs.append(top_count / total)

top1_probs.sort()
n = len(top1_probs)
print(f"Individual transition top-1 probability distribution:")
print(f"  n={n}, median={top1_probs[n//2]:.3f}, mean={sum(top1_probs)/n:.3f}")
print(f"  P10={top1_probs[int(0.1*n)]:.3f}, P25={top1_probs[int(0.25*n)]:.3f}")
print(f"  P75={top1_probs[int(0.75*n)]:.3f}, P90={top1_probs[int(0.9*n)]:.3f}, max={top1_probs[-1]:.3f}")
print(f"  Fraction with top-1 > 0.5 (majority winner): {sum(1 for p in top1_probs if p > 0.5)/n*100:.1f}%")
print(f"  Fraction with top-1 > 0.9 (near-deterministic): {sum(1 for p in top1_probs if p > 0.9)/n*100:.1f}%")

print()
print("="*70)
print("CONCLUSION: The fundamental nature of the routing distribution")
print("="*70)
# Compute how many experts the top-k transition can recover given UNLIMITED budget
total_experts = sum(
    sum(len(set(rec.get('experts_by_layer', {}).get(l, []))) for l in rec.get('experts_by_layer', {}))
    for s in sessions for rec in get_decode_recs(s)
)
print(f"Total expert activations in decode: {total_experts}")
