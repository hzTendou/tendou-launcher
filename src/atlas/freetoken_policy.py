"""FreeToken Edge-Native MoE Serving Policy."""
from typing import Dict, List, Optional, Set, Tuple

class LRUExpertCache:
    """Global LRU Expert Caching: dynamic VRAM cache for experts."""
    def __init__(self, capacity_slots: int, vram_budget_bytes: int = 0):
        self.capacity_slots = capacity_slots
        self.vram_budget_bytes = vram_budget_bytes
        self.cache: List[Tuple[int, int]] = []
        self.expert_sizes: Dict[Tuple[int, int], int] = {}
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0

    def access(self, layer: int, expert: int, size_bytes: int = 0) -> bool:
        key = (layer, expert)
        if key in self.cache:
            self.cache.remove(key)
            self.cache.append(key)
            self.hits += 1
            return True
        else:
            self.misses += 1
            # Evict if adding this exceeds slots or VRAM
            while self.cache and (
                (self.capacity_slots > 0 and len(self.cache) >= self.capacity_slots) or 
                (self.vram_budget_bytes > 0 and self.used_bytes + size_bytes > self.vram_budget_bytes)
            ):
                evicted_key = self.cache.pop(0)
                evicted_size = self.expert_sizes.pop(evicted_key, 0)
                self.used_bytes -= evicted_size
            
            # Add new expert if it fits
            if self.vram_budget_bytes == 0 or self.used_bytes + size_bytes <= self.vram_budget_bytes:
                self.cache.append(key)
                self.expert_sizes[key] = size_bytes
                self.used_bytes += size_bytes
            return False

    def get_stats(self) -> Dict[str, int]:
        return {
            "hits": self.hits, 
            "misses": self.misses, 
            "capacity_slots": self.capacity_slots, 
            "vram_budget_bytes": self.vram_budget_bytes,
            "used_slots": len(self.cache),
            "used_bytes": self.used_bytes
        }


class BandwidthCeilingPolicy:
    """Bandwidth-aware q* policy to decide between hybrid/offload based on thresholds."""
    def __init__(self, cpu_bw_gbs: float, pcie_bw_gbs: float, threshold: float = 2.0):
        # Bound against zero to avoid div by zero or negative bandwidth weirdness
        self.cpu_bw_gbs = max(0.001, cpu_bw_gbs)
        self.pcie_bw_gbs = max(0.0, pcie_bw_gbs)
        self.threshold = threshold

    def should_offload_to_gpu(self) -> bool:
        ratio = self.pcie_bw_gbs / self.cpu_bw_gbs
        return ratio >= self.threshold


class DoubleBufferPrefillPlanner:
    """Full-Layer Double-Buffered Prefill Streaming to overlap I/O with compute."""
    def __init__(self, slot_count: int = 2):
        self.slot_count = slot_count
        self.buffers: Dict[int, Optional[int]] = {i: None for i in range(slot_count)}
        self.current_slot = 0

    def plan_prefetch(self, target_layer: int) -> int:
        # If already buffered, return that slot
        for slot, layer in self.buffers.items():
            if layer == target_layer:
                return slot
                
        slot = self.current_slot
        self.buffers[slot] = target_layer
        self.current_slot = (self.current_slot + 1) % self.slot_count
        return slot

    def get_buffered_layers(self) -> List[int]:
        return [v for v in self.buffers.values() if v is not None]


class SemanticAnchorCache:
    """Semantic-Aware Caching for lossless KV-cache snapshot logic."""
    def __init__(self):
        self.snapshots: Dict[str, Dict[str, any]] = {}

    def save_snapshot(self, anchor_key: str, state: Dict[str, any]):
        self.snapshots[anchor_key] = state

    def load_snapshot(self, anchor_key: str) -> Optional[Dict[str, any]]:
        return self.snapshots.get(anchor_key)

    def has_snapshot(self, anchor_key: str) -> bool:
        return anchor_key in self.snapshots
