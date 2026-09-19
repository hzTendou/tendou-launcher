"""Fast, leakage-safe Atlas predictor benchmark.

The expensive part (causal prediction) is evaluated once per predictor history
configuration. Confidence floors, budget scales and candidate caps are then
applied to the cached causal candidate lists. This makes large sweeps practical
without changing the predictor itself.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from atlas_simulator import load_sessions, layers, all_keys
from atlas_predictor import AtlasPredictor


def floats(s): return [float(x) for x in s.split(',') if x.strip()]
def ints(s): return [int(x) for x in s.split(',') if x.strip()]


def collect_predictions(sessions, history_window, max_candidates):
    """Run the causal predictor once and cache candidates/confidence/targets."""
    predictor = AtlasPredictor(history_window=history_window,
                               max_candidates_per_layer=max_candidates,
                               min_count=2, confidence_floor=0.0,
                               budget_scale=1.0)
    steps = []
    for si, sess in enumerate(sessions):
        st = predictor.begin_session()
        recs = sess['records']
        for i, rec in enumerate(recs):
            phase = str(rec.get('phase', 'decode'))
            cur = layers(rec)
            if i + 1 < len(recs) and recs[i + 1].get('phase', 'decode') == phase:
                p = st.predict(cur, phase=phase)
                target = all_keys(recs[i + 1])
                
                ranked = p.ranked_keys_by_layer or {}
                steps.append((si, phase, ranked, p.confidence, p.abstained_layers, target))
            st.observe(cur, phase=phase)
        st.finish()
    return steps


def score_steps(steps, confidence_floor, budget_scale, max_candidates):
    pred = correct = actual = 0
    conf_sum = 0.0
    decode_steps = 0
    abstained = 0
    for _, phase, candidates_by_layer, conf, _, target in steps:
        if phase != 'decode':
            continue
        decode_steps += 1
        conf_sum += conf
        if conf < confidence_floor:
            abstained += 1
            continue
        # Apply the budget independently per layer.
        chosen = set()
        target_k = {}
        for layer, _ in target:
            target_k[layer] = target_k.get(layer, 0) + 1
        for layer, candidates in candidates_by_layer.items():
            k = target_k.get(layer, 1)
            budget_k = max(k, k + int((max(0.0, budget_scale - 1.0) * k * conf)))
            chosen.update(candidates[:min(max_candidates, budget_k)])
        pred += len(chosen)
        correct += len(chosen & target)
        actual += len(target)
    precision = correct / pred if pred else 0.0
    recall = correct / actual if actual else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    f2 = 5*precision*recall/(4*precision+recall) if 4*precision+recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2,
            'predicted': pred, 'correct': correct, 'actual': actual,
            'steps': decode_steps, 'mean_confidence': conf_sum/decode_steps if decode_steps else 0.0,
            'abstained_steps': abstained}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--confidence-floors', default='0.00,0.10,0.15,0.20,0.25,0.30,0.35,0.40')
    ap.add_argument('--budget-scales', default='1.00,1.02,1.05,1.10,1.15')
    ap.add_argument('--history-windows', default='2,4,6')
    ap.add_argument('--max-candidates', default='8,12,16,24,32')
    ap.add_argument('--out', default='predictor_sweep.json')
    args = ap.parse_args()
    sessions = load_sessions(args.trace)
    floors, budgets, histories, maxcands = (floats(args.confidence_floors),
                                             floats(args.budget_scales),
                                             ints(args.history_windows),
                                             ints(args.max_candidates))
    rows = []
    for hw in histories:
        cached = collect_predictions(sessions, hw, max(maxcands))
        for floor in floors:
            for budget in budgets:
                for mc in maxcands:
                    r = score_steps(cached, floor, budget, mc)
                    rows.append({'confidence_floor': floor, 'budget_scale': budget,
                                 'history_window': hw, 'max_candidates': mc,
                                 **r})
    rows.sort(key=lambda x: (-x['f2'], -x['precision'], x['predicted']))
    report = {'config': vars(args), 'best': rows[:30], 'all': rows}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[+] predictor sweep:', args.out)
    print(' rank floor budget history maxcand precision recall F1 F2 predicted')
    for i, r in enumerate(rows[:15], 1):
        print(f" {i:>2} {r['confidence_floor']:.2f} {r['budget_scale']:.2f} "
              f"{r['history_window']:>7} {r['max_candidates']:>7} "
              f"{r['precision']*100:>8.2f}% {r['recall']*100:>7.2f}% "
              f"{r['f1']*100:>6.2f}% {r['f2']*100:>6.2f}% {r['predicted']:>8}")

if __name__ == '__main__': main()
