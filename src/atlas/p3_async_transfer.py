"""P3 Async Transfer Simulator — Tendou Launcher."""
from typing import List, Tuple, Dict, Optional

class PinnedStagingBuffer:
    def __init__(self, max_slots: int, slot_size_bytes: int):
        self.max_slots = max_slots
        self.slot_size_bytes = slot_size_bytes
        self.available_slots = set(range(max_slots))
        self.used_slots = set()
    
    def acquire_slot(self) -> Optional[int]:
        if not self.available_slots:
            return None
        slot = self.available_slots.pop()
        self.used_slots.add(slot)
        return slot
        
    def release_slot(self, slot: int):
        if slot in self.used_slots:
            self.used_slots.remove(slot)
            self.available_slots.add(slot)

    def clear(self):
        self.used_slots.clear()
        self.available_slots = set(range(self.max_slots))
            
    def is_full(self) -> bool:
        return len(self.available_slots) == 0

class H2DTransferQueue:
    def __init__(self, max_depth: int):
        self.max_depth = max(0, max_depth)
        self.queue = []
        self.pending_count = 0
        self.completed_count = 0
        
    def enqueue(self, layer: int, expert: int, staging_slot: int) -> bool:
        if self.max_depth == 0 or len(self.queue) >= self.max_depth:
            return False
        self.queue.append((layer, expert, staging_slot))
        self.pending_count += 1
        return True
        
    def dequeue(self) -> Optional[Tuple[int, int, int]]:
        if not self.queue:
            return None
        item = self.queue.pop(0)
        self.pending_count -= 1
        self.completed_count += 1
        return item
        
    def remove(self, layer: int, expert: int) -> bool:
        for i, item in enumerate(self.queue):
            if item[0] == layer and item[1] == expert:
                self.queue.pop(i)
                self.pending_count -= 1
                return True
        return False

    def clear(self):
        self.queue.clear()
        self.pending_count = 0

    def is_empty(self) -> bool:
        return len(self.queue) == 0

class CUDAEventFence:
    def __init__(self):
        self.transfers_started = set()
        self.computes_started = set()
        self.transfers_completed = set()
        self.compute_hazards = set()
        
    def record_transfer_start(self, expert_key: Tuple[int, int]):
        self.transfers_started.add(expert_key)
        
    def record_transfer_complete(self, expert_key: Tuple[int, int]):
        self.transfers_completed.add(expert_key)

    def record_transfer_cancelled(self, expert_key: Tuple[int, int]):
        self.transfers_started.discard(expert_key)
        self.transfers_completed.discard(expert_key)
        self.compute_hazards.discard(expert_key)

    def record_compute_start(self, expert_key: Tuple[int, int]):
        self.computes_started.add(expert_key)
        if expert_key in self.transfers_started and expert_key not in self.transfers_completed:
            self.compute_hazards.add(expert_key)
        
    def transfer_complete_before_compute(self, expert_key: Tuple[int, int]) -> bool:
        return (expert_key in self.transfers_completed) and (expert_key not in self.compute_hazards)

    def has_hazard(self, expert_key: Tuple[int, int]) -> bool:
        return expert_key in self.compute_hazards

    def clear(self):
        self.transfers_started.clear()
        self.computes_started.clear()
        self.transfers_completed.clear()
        self.compute_hazards.clear()

class AsyncTransferPipeline:
    def __init__(self, max_slots: int, slot_size_bytes: int, max_queue_depth: int):
        self.staging_buffer = PinnedStagingBuffer(max_slots, slot_size_bytes)
        self.transfer_queue = H2DTransferQueue(max_queue_depth)
        self.event_fence = CUDAEventFence()
        self.active_transfers = {} # key -> slot
        
    def submit_expert(self, layer: int, expert: int, size_bytes: int) -> bool:
        if layer < 0 or layer >= 48 or expert < 0 or expert >= 512:
            return False
        if size_bytes > self.staging_buffer.slot_size_bytes:
            return False
            
        key = (layer, expert)
        if key in self.active_transfers:
            # Already in flight; avoid redundant slot allocation and staging leak
            return False

        slot = self.staging_buffer.acquire_slot()
        if slot is None:
            return False
            
        if not self.transfer_queue.enqueue(layer, expert, slot):
            self.staging_buffer.release_slot(slot)
            return False
            
        self.active_transfers[key] = slot
        self.event_fence.record_transfer_start(key)
        return True
        
    def drain_completed(self) -> List[Tuple[int, int]]:
        completed = []
        while True:
            item = self.transfer_queue.dequeue()
            if item is None:
                break
            layer, expert, slot = item
            key = (layer, expert)
            
            # release buffer only after H2D transfer completed
            self.staging_buffer.release_slot(slot)
            if key in self.active_transfers:
                del self.active_transfers[key]
            
            self.event_fence.record_transfer_complete(key)
            completed.append((layer, expert))
            
        return completed

    def cancel_transfer(self, layer: int, expert: int) -> bool:
        key = (layer, expert)
        if key not in self.active_transfers:
            return False
            
        slot = self.active_transfers.pop(key)
        self.staging_buffer.release_slot(slot)
        self.event_fence.record_transfer_cancelled(key)
        
        # We must also remove it from the queue if it hasn't been processed
        return self.transfer_queue.remove(layer, expert)

    def clear(self):
        self.transfer_queue.clear()
        self.staging_buffer.clear()
        self.active_transfers.clear()
        self.event_fence.clear()
