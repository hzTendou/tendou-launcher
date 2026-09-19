"""P4 Predictive Placement Simulator — Tendou Launcher.

OD-MoE lookahead + SPICE confidence + FreeToken q* policy entegrasyonu.
Gerçek inference kanıtı değil; politika simülatörü.
"""

from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass

class PlacementTier(Enum):
    GPU = "gpu"
    RAM = "ram"
    DISK = "disk"
    CPU_FALLBACK = "cpu_fallback"

@dataclass
class ExpertPrediction:
    layer: int
    expert: int
    confidence: float
    predicted_by: str

class PlacementPolicy:
    def __init__(
        self,
        gpu_confidence_threshold: float = 0.70,
        ram_confidence_threshold: float = 0.30,
        vram_budget_bytes: int = 0,
        ram_budget_bytes: int = 0
    ):
        self.gpu_confidence_threshold = gpu_confidence_threshold
        self.ram_confidence_threshold = ram_confidence_threshold
        self.vram_budget_bytes = vram_budget_bytes
        self.ram_budget_bytes = ram_budget_bytes
        
        self.vram_used = 0
        self.ram_used = 0
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.tier_counts = {
            PlacementTier.GPU: 0,
            PlacementTier.RAM: 0,
            PlacementTier.DISK: 0,
            PlacementTier.CPU_FALLBACK: 0
        }
        self.miss_records = []
        
        self.EXPERT_SIZE = 1 

    def decide(self, prediction: ExpertPrediction) -> PlacementTier:
        tier = PlacementTier.DISK
        
        if prediction.confidence >= self.gpu_confidence_threshold:
            if self.vram_used + self.EXPERT_SIZE <= self.vram_budget_bytes:
                tier = PlacementTier.GPU
                self.vram_used += self.EXPERT_SIZE
            elif self.ram_used + self.EXPERT_SIZE <= self.ram_budget_bytes:
                tier = PlacementTier.RAM
                self.ram_used += self.EXPERT_SIZE
        elif prediction.confidence >= self.ram_confidence_threshold:
            if self.ram_used + self.EXPERT_SIZE <= self.ram_budget_bytes:
                tier = PlacementTier.RAM
                self.ram_used += self.EXPERT_SIZE
                
        self.tier_counts[tier] += 1
        return tier

    def free(self, tier: PlacementTier):
        if tier == PlacementTier.GPU:
            self.vram_used = max(0, self.vram_used - self.EXPERT_SIZE)
        elif tier == PlacementTier.RAM:
            self.ram_used = max(0, self.ram_used - self.EXPERT_SIZE)

    def register_miss(self, layer: int, expert: int):
        self.misses += 1
        self.miss_records.append((layer, expert))
        self.tier_counts[PlacementTier.CPU_FALLBACK] += 1

    def get_stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "tier_counts": {k.value: v for k, v in self.tier_counts.items()},
            "vram_used": self.vram_used,
            "ram_used": self.ram_used
        }

class LookaheadPlacementScheduler:
    def __init__(self, policy: PlacementPolicy, lookahead_layers: int = 4):
        self.policy = policy
        self.lookahead_layers = lookahead_layers
        self.queue: Dict[Tuple[int, int], ExpertPrediction] = {}
        self.plan: Dict[Tuple[int, int], PlacementTier] = {}

    def submit_predictions(self, predictions: List[ExpertPrediction]):
        for p in predictions:
            self.queue[(p.layer, p.expert)] = p

    def get_placement_plan(self) -> Dict[Tuple[int, int], PlacementTier]:
        for key, prediction in self.queue.items():
            if key not in self.plan:
                tier = self.policy.decide(prediction)
                self.plan[key] = tier
                self.policy.hits += 1 # A prediction was placed
        return self.plan

    def on_layer_complete(self, layer: int):
        # Evict stale predictions for this layer
        keys_to_remove = [k for k in self.queue.keys() if k[0] == layer]
        for k in keys_to_remove:
            del self.queue[k]
            if k in self.plan:
                self.policy.free(self.plan[k])
                del self.plan[k]

    def on_miss(self, layer: int, expert: int):
        self.policy.register_miss(layer, expert)
