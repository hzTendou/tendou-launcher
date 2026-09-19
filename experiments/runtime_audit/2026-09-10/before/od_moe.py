"""OD-MoE: Predictive Expert Prefetch & Quick Eviction Policy.

Key capabilities:
- Advance multi-layer expert prediction (lookahead across L+1..L+lead layers)
- Overlapped compute and PCIe/NVMe expert transfers
- Quick post-layer eviction of non-reused experts under VRAM memory pressure
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .spice_scheduler import ConfidenceTier, SpiceScheduler


class ODMoEPrefetcher:
    """Multi-layer advance predictor and quick eviction coordinator."""

    def __init__(
        self,
        lead_layers: int = 2,
        quick_evict: bool = True,
        gpu_expert_layers: int = 2,
        scheduler: Optional[SpiceScheduler] = None,
    ):
        self.lead_layers = max(1, min(8, int(lead_layers)))
        self.quick_evict = bool(quick_evict)
        self.gpu_expert_layers = max(0, int(gpu_expert_layers))
        self.scheduler = scheduler or SpiceScheduler()

        # Cross-layer routing transitions: (src_layer, tgt_layer) -> src_expert -> tgt_expert -> count
        self.cross_layer_transitions: Dict[Tuple[int, int], Dict[int, Dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.observed_token_layers: Dict[int, List[int]] = {}
        self.last_observed_layer = -1
        self.last_observed_experts: List[int] = []
        self.lead_predictions: Dict[int, Set[int]] = {}

        # Metrics
        self.prefetch_dispatches = 0
        self.quick_evictions = 0
        self.lookahead_hits = 0

    def observe_layer(self, layer: int, active_experts: List[int]) -> None:
        """Observes active experts at layer and updates cross-layer transitions across lead horizon."""
        # Check hits against previous lookahead predictions
        if layer in self.lead_predictions:
            for exp in active_experts:
                if exp in self.lead_predictions[layer]:
                    self.lookahead_hits += 1

        # Record transitions from all preceding layers within lookahead window in this token
        for src_l, src_experts in self.observed_token_layers.items():
            if 1 <= (layer - src_l) <= self.lead_layers:
                pair = (src_l, layer)
                for src_e in src_experts:
                    for tgt_e in active_experts:
                        self.cross_layer_transitions[pair][src_e][tgt_e] += 1

        self.observed_token_layers[layer] = list(active_experts)
        self.last_observed_layer = layer
        self.last_observed_experts = list(active_experts)

    def end_token(self) -> None:
        """Resets per-token observation state."""
        self.observed_token_layers.clear()
        self.lead_predictions.clear()
        self.last_observed_layer = -1
        self.last_observed_experts = []

    def predict_lookahead_layers(
        self,
        current_layer: int,
        current_experts: List[int],
        k: int = 10,
        total_layers: int = 48,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Predicts active experts and confidence for layers current_layer + 1 .. current_layer + lead."""
        predictions: Dict[int, List[Tuple[int, float]]] = {}

        for h in range(1, self.lead_layers + 1):
            target_layer = (current_layer + h) % total_layers
            scores: Dict[int, float] = defaultdict(float)

            # 1. Cross-layer correlation from current active experts to specific target layer
            pair = (current_layer, target_layer)
            if pair in self.cross_layer_transitions and current_experts:
                matched_sources = 0
                for cur_e in current_experts:
                    if cur_e in self.cross_layer_transitions[pair]:
                        row = self.cross_layer_transitions[pair][cur_e]
                        total = sum(row.values())
                        if total > 0:
                            matched_sources += 1
                            for tgt_e, count in row.items():
                                scores[tgt_e] += 0.65 * (float(count) / float(total))
                if matched_sources > 0:
                    for e in scores:
                        scores[e] /= float(matched_sources)

            # 2. If insufficient cross-layer evidence, fallback to seed distribution
            if not scores:
                for i in range(max(1, k)):
                    scores[i] = 0.30

            # Sort and bound to top-k
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            # Convert raw score to bounded confidence [0.05, 1.0]
            conf_list = [
                (exp, max(0.05, min(1.0, score))) for exp, score in ranked
            ]
            predictions[target_layer] = conf_list
            self.prefetch_dispatches += len(conf_list)
            self.lead_predictions[target_layer] = {exp for exp, _ in conf_list}

        return predictions

    def plan_prefetches(
        self,
        current_layer: int,
        current_experts: List[int],
        k: int = 10,
    ) -> Dict[int, Dict[ConfidenceTier, List[Tuple[int, float]]]]:
        """Generates SPICE-partitioned prefetch plan for upcoming layers."""
        lookahead = self.predict_lookahead_layers(current_layer, current_experts, k=k)
        plan: Dict[int, Dict[ConfidenceTier, List[Tuple[int, float]]]] = {}

        for target_layer, candidates in lookahead.items():
            plan[target_layer] = self.scheduler.schedule_candidates(target_layer, candidates)

        return plan

    def identify_quick_evictions(
        self,
        completed_layer: int,
        vram_resident_experts: List[Tuple[int, int]],  # list of (layer, expert)
        upcoming_plan: Dict[int, Dict[ConfidenceTier, List[Tuple[int, float]]]],
        vram_used_bytes: int,
        vram_capacity_bytes: int,
        high_watermark_ratio: float = 0.80,
    ) -> List[Tuple[int, int]]:
        """Identifies experts in completed_layer that can be immediately evicted from VRAM."""
        if not self.quick_evict:
            return []

        # Anchor layers (layers 0..gpu_expert_layers-1) must NEVER be evicted from VRAM
        if completed_layer < self.gpu_expert_layers:
            return []

        # Only evict if memory pressure exceeds high watermark
        if vram_used_bytes < vram_capacity_bytes * high_watermark_ratio:
            return []

        # Collect upcoming planned VRAM experts to avoid thrashing
        keep_experts: Set[int] = set()
        for tgt_layer, tiered in upcoming_plan.items():
            for exp, _ in tiered.get(ConfidenceTier.HIGH, []):
                keep_experts.add(exp)

        evict_list: List[Tuple[int, int]] = []
        for l, e in vram_resident_experts:
            if l == completed_layer and e not in keep_experts:
                evict_list.append((l, e))
                self.quick_evictions += 1

        return evict_list

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "lead_layers": self.lead_layers,
            "quick_evict_enabled": self.quick_evict,
            "gpu_expert_layers": self.gpu_expert_layers,
            "prefetch_dispatches": self.prefetch_dispatches,
            "quick_evictions": self.quick_evictions,
            "lookahead_hits": self.lookahead_hits,
        }
