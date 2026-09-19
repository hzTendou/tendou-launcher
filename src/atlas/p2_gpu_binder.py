"""P2 GPU Binder Simulator."""
from typing import Dict, List, Set, Tuple

class ExpertIDMapper:
    """Compact expert ID mapping (layer_id × expert_id -> buffer index)."""
    def __init__(self, num_layers: int = 48, num_experts: int = 512):
        self.num_layers = num_layers
        self.num_experts = num_experts

    def get_compact_id(self, layer: int, expert: int) -> int:
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"Invalid layer {layer}")
        if expert < 0 or expert >= self.num_experts:
            raise ValueError(f"Invalid expert {expert}")
        return layer * self.num_experts + expert

class GPUExpertBuffer:
    """Capacity-limited VRAM buffer model, tracks binding status."""
    def __init__(self, capacity_bytes: int, expert_size_bytes: int = 3481600):
        self.capacity_bytes = capacity_bytes
        self.expert_size_bytes = expert_size_bytes
        self.used_bytes = 0
        self.resident_experts: Set[Tuple[int, int]] = set()
        self.lru_order: List[Tuple[int, int]] = []
        
    def can_fit(self) -> bool:
        return self.used_bytes + self.expert_size_bytes <= self.capacity_bytes

    def bind_expert(self, layer: int, expert: int) -> bool:
        if layer < 0 or layer >= 48 or expert < 0 or expert >= 512:
            return False
        key = (layer, expert)
        if key in self.resident_experts:
            # Update LRU
            self.lru_order.remove(key)
            self.lru_order.append(key)
            return True
            
        # Needs to be bound, check capacity
        if not self.can_fit():
            return False
            
        self.resident_experts.add(key)
        self.lru_order.append(key)
        self.used_bytes += self.expert_size_bytes
        return True

    def evict_lru(self, keep_experts: Set[Tuple[int, int]] = frozenset()) -> Tuple[int, int]:
        """Evicts the least recently used expert that is not in keep_experts."""
        for i, key in enumerate(self.lru_order):
            if key not in keep_experts:
                evicted = self.lru_order.pop(i)
                self.resident_experts.remove(evicted)
                self.used_bytes -= self.expert_size_bytes
                return evicted
        raise RuntimeError("Buffer is empty or all experts are pinned")

class ExpertBindingPlan:
    """Plans which experts go to GPU, which fallback to CPU."""
    def __init__(self, buffer: GPUExpertBuffer):
        self.buffer = buffer

    def plan_binding(self, layer: int, router_experts: List[int]) -> Dict[str, List[int]]:
        """
        router_experts: list of expert IDs selected by the router (K=10).
        Returns a dictionary with 'gpu' and 'cpu' lists.
        """
        gpu_bound = []
        cpu_bound = []
        
        if layer < 0 or layer >= 48:
            return {'gpu': [], 'cpu': list(router_experts)}

        # We shouldn't evict experts we just bound for this layer
        currently_binding = set((layer, e) for e in router_experts if 0 <= e < 512)
        
        for expert in router_experts:
            if expert < 0 or expert >= 512:
                cpu_bound.append(expert)
                continue

            if self.buffer.bind_expert(layer, expert):
                gpu_bound.append(expert)
            else:
                # Try to evict if capacity is constrained
                evicted = False
                if self.buffer.capacity_bytes > 0:
                    try:
                        while not self.buffer.can_fit():
                            self.buffer.evict_lru(keep_experts=currently_binding)
                        if self.buffer.bind_expert(layer, expert):
                            gpu_bound.append(expert)
                            evicted = True
                    except RuntimeError:
                        pass
                
                if not evicted:
                    # Lossless fallback
                    cpu_bound.append(expert)
                    
        return {'gpu': gpu_bound, 'cpu': cpu_bound}
