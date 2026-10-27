"""SPICE: Confidence-Aware Scheduler with Lossless Fallback Guarantee.

Partitioning policy:
- High Confidence (>= conf_high, default 0.65): Prefetch to GPU VRAM
- Medium Confidence (>= conf_mid, default 0.30): Warm in Host RAM / Pinned Staging Pool
- Low Confidence (< conf_mid): Leave on NVMe disk

Lossless Guarantee:
Under NO circumstances is an approximate or surrogate expert substituted.
On prediction miss, the fallback scheduler triggers lossless on-demand paging,
preserving 100% router fidelity and mathematical precision.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ConfidenceTier(str, Enum):
    LOW = "low"        # Stays on NVMe (< conf_mid)
    MEDIUM = "medium"  # Warmed in Host RAM / Pinned Staging (>= conf_mid && < conf_high)
    HIGH = "high"      # Prefetched to GPU VRAM (>= conf_high)


class SpiceScheduler:
    """Confidence-aware multi-tier expert scheduler."""

    def __init__(self, conf_high: float = 0.65, conf_mid: float = 0.30):
        if not (0.0 <= conf_mid <= conf_high <= 1.0):
            raise ValueError(f"Invalid thresholds: conf_mid={conf_mid}, conf_high={conf_high}")
        self.conf_high = float(conf_high)
        self.conf_mid = float(conf_mid)

        # Metrics
        self.vram_scheduled = 0
        self.ram_scheduled = 0
        self.nvme_left = 0
        self.vram_hits = 0
        self.lossless_fallbacks = 0
        self.total_evaluations = 0

    def classify_confidence(self, confidence: float) -> Tuple[ConfidenceTier, str]:
        """Maps confidence score to (ConfidenceTier, MemoryTier)."""
        c = max(0.0, min(1.0, float(confidence)))
        if c >= self.conf_high:
            return ConfidenceTier.HIGH, "vram"
        elif c >= self.conf_mid:
            return ConfidenceTier.MEDIUM, "ram"
        else:
            return ConfidenceTier.LOW, "nvme"

    def compute_expert_confidence(
        self,
        markov_prob: float = 0.0,
        stickiness: bool = False,
        frequency: float = 0.0,
        cross_layer_prob: float = 0.0,
    ) -> float:
        """Computes composite confidence score bounded in [0.0, 1.0]."""
        score = 0.15
        if cross_layer_prob > 0.0:
            score += 0.45 * min(1.0, cross_layer_prob)
        if markov_prob > 0.0:
            score += 0.30 * min(1.0, markov_prob)
        if stickiness:
            score += 0.25
        if frequency > 0.02:
            score += 0.15 * min(1.0, frequency * 5.0)
        return max(0.05, min(1.0, score))

    def schedule_candidates(
        self,
        layer: int,
        candidates: List[Tuple[int, float]],
    ) -> Dict[ConfidenceTier, List[Tuple[int, float]]]:
        """Partitions candidate experts into HIGH (VRAM), MEDIUM (RAM), and LOW (NVME)."""
        result: Dict[ConfidenceTier, List[Tuple[int, float]]] = {
            ConfidenceTier.HIGH: [],
            ConfidenceTier.MEDIUM: [],
            ConfidenceTier.LOW: [],
        }

        for exp_id, conf in candidates:
            tier, target_mem = self.classify_confidence(conf)
            result[tier].append((exp_id, conf))
            self.total_evaluations += 1
            if tier == ConfidenceTier.HIGH:
                self.vram_scheduled += 1
            elif tier == ConfidenceTier.MEDIUM:
                self.ram_scheduled += 1
            else:
                self.nvme_left += 1

        return result

    def fallback_schedule(
        self,
        layer: int,
        requested_expert: int,
        current_residency: str,
    ) -> Dict[str, Any]:
        """Handles prediction miss with a guaranteed lossless on-demand promotion.

        Approximate / surrogate expert substitution is strictly forbidden.
        """
        is_hit = (current_residency == "vram")
        action = "cache_hit"
        source_tier = current_residency

        if is_hit:
            self.vram_hits += 1
        else:
            self.lossless_fallbacks += 1
            if current_residency in ("ram", "pinned"):
                action = "h2d_transfer"
            else:
                action = "nvme_demand_load"

        return {
            "layer": layer,
            "expert": requested_expert,
            "hit": is_hit,
            "action": action,
            "source_tier": source_tier,
            "target_tier": "vram",
            "lossless": True,
        }

    def verify_lossless(
        self,
        actual_router_experts: List[int],
        executed_experts: List[int],
    ) -> bool:
        """Verifies that no approximate or surrogate substitution occurred."""
        if actual_router_experts != executed_experts:
            raise ValueError(
                f"Lossless violation: router expected {actual_router_experts} "
                f"but got {executed_experts}"
            )
        return True

    def get_metrics(self) -> Dict[str, Any]:
        total_accesses = self.vram_hits + self.lossless_fallbacks
        hit_rate = (self.vram_hits / max(1, total_accesses)) * 100.0 if total_accesses > 0 else 0.0
        return {
            "vram_scheduled": self.vram_scheduled,
            "ram_scheduled": self.ram_scheduled,
            "nvme_left": self.nvme_left,
            "vram_hits": self.vram_hits,
            "lossless_fallbacks": self.lossless_fallbacks,
            "hit_rate_pct": hit_rate,
            "total_evaluations": self.total_evaluations,
            "conf_high": self.conf_high,
            "conf_mid": self.conf_mid,
        }
