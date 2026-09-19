"""
Deep analysis to understand the 43% vs 98.8% oracle gap.

Questions to answer:
1. Is the gap due to SET prediction being treated as independent transitions?
2. Does knowing the current SET help more than knowing just prev experts?
3. What is the upper bound with oracle knowledge of current SET only?
4. How much improvement comes from knowing prev 2 steps vs prev 1?
5. What fraction of tokens have "repeatable" routing (low entropy)?
"""
import json
from collections import Counter, defaultdict
import math

with open('trace_35b_a3b_q2k.jsonl') as f:
    sessions = [json.loads(l) for l in f if l.strip()]

# Only use records that are part of decode phase or all if no phase
def get_decode_recs(session):
    recs = session['records']
    decode = [r for r in recs if r.get('phase') == 'decode']
    if decode:
        return decode
    return recs  # legacy format

def all_keys(rec):
    return {(str(l), int(e)) for l, es in rec.get('experts_by_layer', {}).items() for e in es}

def layer_experts(rec):
    return {str(l): set(int(e) for e in es) for l, es in rec.get('experts_by_layer', {}).items()}

print("="*70)
print("ANALYSIS 1: What fraction of the oracle gap is from SET vs individual prediction?")
print("="*70)
# The oracle knows the NEXT record perfectly.
# The causal predictor only knows prev→cur at expert level.
# Key question: if we knew the current set perfectly and just asked
# "which experts repeat from this set in the next token?",
# how much recall do we get?

repeat_recall_num = 0
repeat_recall_den = 0
for s in sessions:
    recs = get_decode_recs(s)
    for i in range(len(recs)-1):
        cur_set = all_keys(recs[i])
        nxt_set = all_keys(recs[i+1])
        # How many of next token's experts are already in current token?
        overlap = cur_set & nxt_set
        repeat_recall_num += len(overlap)
        repeat_recall_den += len(nxt_set)

print(f"Fraction of next-token experts already in current token: {repeat_recall_num}/{repeat_recall_den} = {repeat_recall_num/repeat_recall_den*100:.1f}%")
print("  → This is the 'pure repetition' upper bound (no prediction needed)")

print()
print("="*70)
print("ANALYSIS 2: Co-activation within a token: how predictable is the SET?")
print("="*70)
# Given that experts {a, b, c, ...} are active at layer L token T,
# how well can we predict the FULL set vs top-k from transitions?

# Let's measure: given prev expert e at layer l, what fraction of
# the NEXT expert set is predictable via top-k transitions?

# Build full transition model on all sessions
trans = defaultdict(lambda: defaultdict(Counter))
for s in sessions:
    recs = get_decode_recs(s)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                for pe in prev.get(layer, set()):
                    trans[layer][pe].update(cur_set)
        prev = cur

print("\nAnalysis 2A: Token-level recall vs k-budget")
for budget_mult in [1.0, 1.5, 2.0, 3.0, 4.0, 8.0]:
    total_correct = 0
    total_actual = 0
    total_predicted = 0
    for s in sessions:
        recs = get_decode_recs(s)
        prev = None
        for rec in recs:
            cur = layer_experts(rec)
            if prev is not None:
                for layer, cur_set in cur.items():
                    k = len(cur_set)
                    budget = int(k * budget_mult)
                    scores = Counter()
                    for pe in prev.get(layer, set()):
                        scores.update(trans[layer][pe])
                    top_pred = {e for e, _ in scores.most_common(budget)}
                    total_correct += len(top_pred & cur_set)
                    total_actual += len(cur_set)
                    total_predicted += len(top_pred)
            prev = cur
    prec = total_correct/total_predicted if total_predicted else 0
    rec = total_correct/total_actual if total_actual else 0
    print(f"  Budget {budget_mult:.0f}x k: recall={rec*100:.1f}%, precision={prec*100:.1f}%, predicted={total_predicted}")

print()
print("="*70)
print("ANALYSIS 3: SET-CONTEXT (current routing SET) as predictor feature")
print("="*70)
# Key hypothesis: "treating each expert transition independently" loses
# information about the current expert SET context.
# Test: does knowing the FULL current set help predict next set?

# Build a simple set-signature → next-expert model
# To avoid explosion, hash the current set per layer into a signature
set_trans = defaultdict(lambda: defaultdict(Counter))  # (layer, frozenset) -> next expert counts
for s in sessions:
    recs = get_decode_recs(s)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                prev_set = prev.get(layer, set())
                sig = frozenset(prev_set)
                set_trans[layer][sig].update(cur_set)
        prev = cur

# Now evaluate: using the set-signature model vs individual transition model
set_recall_total = 0
ind_recall_total = 0
total_actual = 0
seen_sig = 0
for s in sessions:
    recs = get_decode_recs(s)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                k = len(cur_set)
                prev_set = prev.get(layer, set())
                sig = frozenset(prev_set)
                
                # Set-signature model
                sig_scores = set_trans[layer].get(sig, Counter())
                set_pred = {e for e, _ in sig_scores.most_common(k)}
                set_recall_total += len(set_pred & cur_set)
                
                # Individual transition model
                ind_scores = Counter()
                for pe in prev_set:
                    ind_scores.update(trans[layer][pe])
                ind_pred = {e for e, _ in ind_scores.most_common(k)}
                ind_recall_total += len(ind_pred & cur_set)
                
                total_actual += len(cur_set)
                if sig_scores:
                    seen_sig += 1
        prev = cur

print(f"Individual transition recall: {ind_recall_total/total_actual*100:.1f}%")
print(f"Set-signature recall:         {set_recall_total/total_actual*100:.1f}%")
print(f"Set signatures with data: {seen_sig} prediction events")
print("  → If set-signature >> individual, the current set context matters significantly")

print()
print("="*70)
print("ANALYSIS 4: How much is the gap from TEMPORAL context (prev 2 tokens)?")
print("="*70)

# Build order-2 model (prev_prev + prev → next)
trans2 = defaultdict(lambda: defaultdict(Counter))  # (layer, prev_prev_sig, prev_e) -> next
for s in sessions:
    recs = get_decode_recs(s)
    prev_prev = None
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None and prev_prev is not None:
            for layer, cur_set in cur.items():
                prev_set = prev.get(layer, set())
                pp_set = prev_prev.get(layer, set())
                for pe in prev_set:
                    pp_sig = frozenset(pp_set)
                    trans2[layer][(pp_sig, pe)].update(cur_set)
        prev_prev = prev
        prev = cur

ord2_recall = 0
ord1_recall = 0
total_act2 = 0
for s in sessions:
    recs = get_decode_recs(s)
    prev_prev = None
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None and prev_prev is not None:
            for layer, cur_set in cur.items():
                k = len(cur_set)
                prev_set = prev.get(layer, set())
                pp_set = prev_prev.get(layer, set())
                
                # Order-1
                scores1 = Counter()
                for pe in prev_set:
                    scores1.update(trans[layer][pe])
                pred1 = {e for e, _ in scores1.most_common(k)}
                ord1_recall += len(pred1 & cur_set)
                
                # Order-2
                scores2 = Counter()
                pp_sig = frozenset(pp_set)
                for pe in prev_set:
                    scores2.update(trans2[layer].get((pp_sig, pe), Counter()))
                if scores2:
                    pred2 = {e for e, _ in scores2.most_common(k)}
                else:
                    pred2 = pred1  # fall back
                ord2_recall += len(pred2 & cur_set)
                
                total_act2 += len(cur_set)
        prev_prev = prev
        prev = cur

print(f"Order-1 recall (prev→next):       {ord1_recall/total_act2*100:.1f}%")
print(f"Order-2 recall (prev_prev+prev→next): {ord2_recall/total_act2*100:.1f}%")

print()
print("="*70)
print("ANALYSIS 5: Per-layer entropy of routing (which layers are most predictable?)")
print("="*70)
layer_recalls = defaultdict(lambda: [0, 0])
for s in sessions:
    recs = get_decode_recs(s)
    prev = None
    for rec in recs:
        cur = layer_experts(rec)
        if prev is not None:
            for layer, cur_set in cur.items():
                k = len(cur_set)
                scores = Counter()
                for pe in prev.get(layer, set()):
                    scores.update(trans[layer][pe])
                pred = {e for e, _ in scores.most_common(k)}
                layer_recalls[layer][0] += len(pred & cur_set)
                layer_recalls[layer][1] += len(cur_set)
        prev = cur

print("Layer-by-layer recall (transition model, top-k):")
for layer in sorted(layer_recalls.keys(), key=int):
    correct, total = layer_recalls[layer]
    r = correct/total if total else 0
    print(f"  Layer {int(layer):>2}: recall={r*100:.1f}% ({correct}/{total})")

print()
print("="*70)
print("ANALYSIS 6: Token-to-token routing STABILITY (how often does the same set repeat?)")
print("="*70)
exact_repeat = partial_repeat = total_pairs = 0
exact_recall_from_repeat = total_from_repeat = 0
for s in sessions:
    recs = get_decode_recs(s)
    for i in range(len(recs)-1):
        cur = all_keys(recs[i])
        nxt = all_keys(recs[i+1])
        total_pairs += 1
        overlap = len(cur & nxt)
        if overlap == len(nxt):
            exact_repeat += 1
        if overlap > 0:
            partial_repeat += 1
            exact_recall_from_repeat += overlap
        total_from_repeat += len(nxt)

print(f"Exact set repeat (cur == nxt): {exact_repeat}/{total_pairs} = {exact_repeat/total_pairs*100:.1f}%")
print(f"Any overlap: {partial_repeat}/{total_pairs} = {partial_repeat/total_pairs*100:.1f}%")
print(f"Average overlap recall: {exact_recall_from_repeat/total_from_repeat*100:.1f}%")

print()
print("="*70)
print("ANALYSIS 7: Is expert routing DETERMINISTIC for given input patterns?")
print("="*70)
# If the same 'previous expert set' at a layer always leads to the same
# 'next expert set', routing is deterministic and fully predictable from history.
# If not, there's fundamental uncertainty.

entropy_by_layer = {}
for layer, sigdict in set_trans.items():
    total_entropy = 0
    total_events = 0
    for sig, next_counts in sigdict.items():
        total = sum(next_counts.values())
        if total < 3:
            continue
        # Entropy of next-expert distribution given this set-signature
        probs = [c/total for c in next_counts.values()]
        h = -sum(p * math.log(p) for p in probs if p > 0)
        hmax = math.log(len(probs)) if len(probs) > 1 else 1.0
        normalized_h = h / hmax if hmax > 0 else 0
        total_entropy += normalized_h * total
        total_events += total
    if total_events > 0:
        entropy_by_layer[layer] = total_entropy / total_events

if entropy_by_layer:
    mean_entropy = sum(entropy_by_layer.values()) / len(entropy_by_layer)
    print(f"Mean normalized routing entropy across layers: {mean_entropy:.3f}")
    print("  (0 = fully deterministic, 1 = maximum randomness)")
    worst = sorted(entropy_by_layer.items(), key=lambda x: -x[1])[:5]
    best = sorted(entropy_by_layer.items(), key=lambda x: x[1])[:5]
    print(f"Most deterministic layers: {[(l, f'{v:.3f}') for l,v in best]}")
    print(f"Most stochastic layers:    {[(l, f'{v:.3f}') for l,v in worst]}")
