"""Atlas v5 routing predictor.

Leakage-safe, phase-aware next-token expert predictor for MoE prefetching.
The predictor is advisory only: it never changes the model router's decision.

Key properties:
- prompt and decode routing histories are isolated;
- predictions use only observations available before the target token;
- local (current-request) evidence is preferred over persistent history;
- transition probabilities are normalized per source expert instead of raw-count
  normalization, which makes confidence and ranking much more meaningful;
- candidate generation is sparse and bounded, avoiding the old all-history
  candidate explosion;
- budget is derived from confidence and remains close to the router top-k;
- completed sessions are committed only after the session ends.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math


Key = tuple[str, int]


def _entropy_confidence(scores: Counter, chosen: int) -> float:
    """Confidence from concentration among the candidate distribution."""
    if not scores or chosen <= 0:
        return 0.0
    total = float(sum(scores.values()))
    if total <= 0:
        return 0.0
    top_mass = sum(v for _, v in scores.most_common(chosen)) / total
    probs = [v / total for v in scores.values() if v > 0]
    if len(probs) <= 1:
        concentration = 1.0
    else:
        h = -sum(p * math.log(p) for p in probs)
        hmax = math.log(len(probs))
        concentration = max(0.0, 1.0 - h / hmax) if hmax else 1.0
    return max(0.0, min(1.0, 0.70 * top_mass + 0.30 * concentration))


@dataclass
class Prediction:
    keys: set[Key]
    confidence: float
    per_layer_confidence: dict[str, float]
    candidate_count: int
    budget_count: int
    abstained_layers: int = 0
    ranked_keys_by_layer: dict[str, tuple[Key, ...]] | None = None


class _Model:
    def __init__(self):
        self.t1 = defaultdict(Counter)          # (layer, prev_expert) -> next counts
        self.global_counts = defaultdict(Counter)  # layer -> expert occurrence counts


class AtlasPredictor:
    """Phase-aware adaptive online predictor.

    ``decode`` is the primary production phase.  Prompt and decode state are
    deliberately separate because prompt routing has a different statistical
    regime and should not poison next-token decode transitions.
    """

    def __init__(self, history_window=4, max_candidates_per_layer=32,
                 min_count=2, order2_weight=0.0, history_weight=0.0,
                 prior_weight=0.0, confidence_floor=0.22, budget_scale=1.05,
                 local_weight=1.0, persistent_weight=0.0,
                 source_top_n=8, prior_smoothing=0.15,
                 min_sources=1, decode_only=True, persistent_fallback=False):
        self.history_window = max(2, int(history_window))
        self.max_candidates = max(1, int(max_candidates_per_layer))
        self.min_count = max(1, int(min_count))
        # Kept for API compatibility with v4; v5 intentionally does not use
        # exact set-context order-2 because those contexts are too sparse.
        self.order2_weight = float(order2_weight)
        self.history_weight = float(history_weight)
        self.prior_weight = float(prior_weight)
        self.confidence_floor = float(confidence_floor)
        self.budget_scale = max(1.0, float(budget_scale))
        self.local_weight = max(0.0, float(local_weight))
        self.persistent_weight = max(0.0, float(persistent_weight))
        norm = self.local_weight + self.persistent_weight
        if norm <= 0:
            self.local_weight, self.persistent_weight = 0.7, 0.3
        else:
            self.local_weight /= norm
            self.persistent_weight /= norm
        self.source_top_n = max(1, int(source_top_n))
        self.prior_smoothing = max(0.0, float(prior_smoothing))
        self.min_sources = max(1, int(min_sources))
        self.decode_only = bool(decode_only)
        self.persistent_fallback = bool(persistent_fallback)

        self._models = defaultdict(_Model)
        # Cross-session prompt-last -> first-decode transition model. It is
        # intentionally persistent-only: the current session's boundary is
        # never available before the first decode token is predicted.
        self.boundary_t1 = defaultdict(Counter)
        self.sessions_seen = 0

        # v4 compatibility: expose decode model under the old attributes.
        self.t1 = self._models["decode"].t1
        self.global_counts = self._models["decode"].global_counts

    def begin_session(self):
        return _SessionState(self)

    @staticmethod
    def _top_conditional(counter: Counter, top_n: int):
        """Return only top-N conditional probabilities. Ranking is identical
        to raw counts, so we avoid scanning the entire sparse counter."""
        total = sum(counter.values())
        if total <= 0:
            return ()
        return tuple((e, n / total) for e, n in counter.most_common(top_n))

    def _score_layer(self, phase, layer, current, local_t1, k):
        """Sparse conditional transition scorer."""
        model = self._models[phase]
        persistent = model.t1

        scores = Counter()
        evidence_sources = defaultdict(int)
        local_sources = 0
        for pe in current:
            lc = local_t1.get((layer, int(pe)), Counter())
            local_top = self._top_conditional(lc, self.source_top_n)
            if local_top:
                local_sources += 1
                for e, prob in local_top:
                    scores[int(e)] += prob
                    evidence_sources[int(e)] += 1

        if local_sources:
            for e in list(scores):
                scores[e] /= local_sources

        # With persistent_fallback enabled, use prior-session evidence even
        # when local evidence exists.  Local routing remains dominant, while a
        # bounded persistent prior restores candidate coverage for transitions
        # that are sparse in the current request.
        if self.persistent_fallback:
            persistent_scores = Counter()
            persistent_sources = 0
            for pe in current:
                pc = persistent.get((layer, int(pe)), Counter())
                persistent_top = self._top_conditional(pc, self.source_top_n)
                if persistent_top:
                    persistent_sources += 1
                    for e, prob in persistent_top:
                        persistent_scores[int(e)] += prob
            if persistent_sources:
                for e in list(persistent_scores):
                    persistent_scores[e] /= persistent_sources
                if local_sources:
                    lw, pw = 0.75, 0.25
                    merged = Counter()
                    for e, v in scores.items():
                        merged[e] += lw * v
                    for e, v in persistent_scores.items():
                        merged[e] += pw * v
                    scores = merged
                else:
                    scores = persistent_scores

        if not scores:
            return Counter(), 0.0

        scores = Counter(dict(scores.most_common(self.max_candidates)))
        confidence = _entropy_confidence(scores, k)
        return scores, confidence

    def predict_boundary(self, prompt_last_layers, max_candidates_per_layer=None):
        """Predict first decode experts from the last prompt routing state.

        This uses only completed *previous* sessions. ``prompt_last_layers`` is
        observed before generation starts, so it is a legitimate cold-start
        signal and is especially valuable because decode history does not yet
        exist.
        """
        cap = self.max_candidates if max_candidates_per_layer is None else max(1, int(max_candidates_per_layer))
        out = set(); ranked = {}
        for layer, current in prompt_last_layers.items():
            scores = Counter()
            for pe in current:
                c = self.boundary_t1.get((str(layer), int(pe)), Counter())
                total = sum(c.values())
                if total:
                    for e, n in c.most_common(cap):
                        scores[int(e)] += n / total
            if scores:
                ranked[str(layer)] = tuple((str(layer), int(e)) for e, _ in scores.most_common(cap))
                out.update((str(layer), int(e)) for e, _ in scores.most_common(cap))
        return Prediction(out, 1.0 if out else 0.0, {}, sum(len(v) for v in ranked.values()), len(out), 0, ranked)

    def predict(self, current_layers, prev_prev_layers=None, local_trans=None,
                prompt_prior=None, confidence_floor=None, phase="decode",
                exclude_keys=None):
        phase = str(phase or "decode")
        if self.decode_only and phase != "decode":
            return Prediction(set(), 0.0, {}, 0, 0, 0)
        local_t1 = local_trans if local_trans is not None else defaultdict(Counter)
        floor = self.confidence_floor if confidence_floor is None else float(confidence_floor)

        out = set()
        ranked = {}
        per_layer = {}
        candidates = 0
        budget = 0
        abstained = 0

        for layer, current in current_layers.items():
            cur = set(int(e) for e in current)
            k = max(1, len(cur))
            scores, conf = self._score_layer(phase, str(layer), cur, local_t1, k)
            if exclude_keys:
                scores = Counter({e:v for e,v in scores.items() if (str(layer), int(e)) not in exclude_keys})
            per_layer[str(layer)] = conf
            candidates += len(scores)
            if not scores or conf < floor:
                abstained += 1
                continue

            # v5 uses only a small confidence-scaled expansion.  At the default
            # 1.05 this is normally k or k+1, avoiding the v4 over-prefetch.
            budget_k = int(math.ceil(k * (1.0 + max(0.0, self.budget_scale - 1.0) * conf)))
            budget_k = min(len(scores), budget_k)
            budget_k = max(1, budget_k)
            ranked[str(layer)] = tuple((str(layer), int(e)) for e, _ in scores.most_common(min(len(scores), self.max_candidates)))
            for e, _ in scores.most_common(budget_k):
                out.add((str(layer), int(e)))
            budget += budget_k

        confidence = sum(per_layer.values()) / len(per_layer) if per_layer else 0.0
        return Prediction(out, confidence, per_layer, candidates, budget, abstained, ranked)

    def commit_session(self, records):
        """Commit only a completed session; never train on its future first."""
        previous = None
        previous_phase = None
        pending_prompt_last = None
        for item in records:
            if isinstance(item, tuple) and len(item) == 2:
                phase, layers = item
            else:
                phase, layers = "decode", item
            phase = str(phase or "decode")
            if self.decode_only and phase != "decode":
                previous = None
                previous_phase = phase
                pending_prompt_last = {str(l): set(int(e) for e in layers.get(l, [])) for l in layers}
                continue
            cur = {str(l): set(int(e) for e in es) for l, es in layers.items()}
            model = self._models[phase]
            if phase == "decode" and previous_phase == "prompt" and pending_prompt_last is not None:
                for l, ps in pending_prompt_last.items():
                    nxt = cur.get(l, set())
                    for pe in ps:
                        self.boundary_t1[(l, int(pe))].update(int(ne) for ne in nxt)
                pending_prompt_last = None
            if previous is not None and previous_phase == phase:
                for l, ps in previous.items():
                    nxt = cur.get(l, set())
                    if not nxt:
                        continue
                    for pe in ps:
                        model.t1[(l, int(pe))].update(int(ne) for ne in nxt)
            for l, es in cur.items():
                model.global_counts[l].update(int(e) for e in es)
            previous = cur
            previous_phase = phase
        self.sessions_seen += 1


class _SessionState:
    """Transient per-request state; persistent state is committed on finish."""
    def __init__(self, predictor: AtlasPredictor):
        self.predictor = predictor
        self.recent = deque(maxlen=predictor.history_window)
        self.local_t1 = defaultdict(Counter)
        self.records = []
        self.current_phase = None

    def _reset_phase(self, phase):
        if self.current_phase != phase:
            self.recent.clear()
            self.local_t1.clear()
            self.current_phase = phase

    def predict_boundary(self, prompt_last_layers, max_candidates_per_layer=None):
        return self.predictor.predict_boundary(prompt_last_layers, max_candidates_per_layer)

    def predict(self, current_layers, prompt_prior=None, confidence_floor=None,
                phase="decode", exclude_keys=None):
        phase = str(phase or "decode")
        self._reset_phase(phase)
        return self.predictor.predict(
            current_layers,
            self.recent[-2] if len(self.recent) >= 2 else None,
            self.local_t1,
            prompt_prior,
            confidence_floor,
            phase=phase,
            exclude_keys=exclude_keys,
        )

    def observe(self, layers, phase="decode"):
        phase = str(phase or "decode")
        self._reset_phase(phase)
        cur = {str(l): set(int(e) for e in es) for l, es in layers.items()}
        if self.recent:
            prev = self.recent[-1]
            for l, ps in prev.items():
                nxt = cur.get(l, set())
                for pe in ps:
                    self.local_t1[(l, int(pe))].update(int(ne) for ne in nxt)
        self.recent.append(cur)
        self.records.append((phase, cur))

    def finish(self):
        self.predictor.commit_session(self.records)
