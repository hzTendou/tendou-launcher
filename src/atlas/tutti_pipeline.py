"""Offline queue/capacity simulator. No disk reads, pinned allocation, H2D or GPU work.

Timing fields below are estimates, not measured inference speed or overlap.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple


@dataclass
class TuttiRequest:
    layer: int
    expert: int
    size_bytes: int = 3481600
    staged_offset: int = 0
    enqueued_at: float = field(default_factory=time.perf_counter)
    completed_at: float = 0.0
    is_ready: bool = False


class TuttiPipeline:
    """Capacity and queue policy model for a future staging consumer."""

    def __init__(
        self,
        pinned_capacity_bytes: int = 668 * 1024 * 1024,
        max_queue_depth: int = 64,
        h2d_bandwidth_gbps: float = 12.8,
        nvme_bandwidth_gbps: float = 4.5,
    ):
        if pinned_capacity_bytes <= 0 or max_queue_depth <= 0 or nvme_bandwidth_gbps <= 0:
            raise ValueError("Capacity, queue depth and bandwidth must be positive")
        self.next_offset = 0
        self.pinned_capacity = pinned_capacity_bytes
        self.max_queue_depth = max_queue_depth
        self.h2d_bandwidth = h2d_bandwidth_gbps
        self.nvme_bandwidth = nvme_bandwidth_gbps

        # Staging state
        self.queue: Deque[TuttiRequest] = deque()
        self.in_flight: Dict[Tuple[int, int], TuttiRequest] = {}
        self.staged_experts: Dict[Tuple[int, int], TuttiRequest] = {}
        self.staged_offsets: Dict[Tuple[int, int], int] = {}
        self.bytes_staged = 0

        # Metrics
        self.total_reads = 0
        self.completed_reads = 0
        self.io_stall_seconds = 0.0
        self.overlap_seconds = 0.0

    def enqueue_read(self, layer: int, expert: int, size_bytes: int = 3481600) -> bool:
        """Enqueues an asynchronous NVMe read into the pinned staging pipeline."""
        key = (layer, expert)
        if key in self.in_flight or key in self.staged_experts:
            return True  # Already enqueued or staged

        if len(self.queue) >= self.max_queue_depth:
            return False  # Queue full

        if size_bytes <= 0 or size_bytes > self.pinned_capacity - self.next_offset:
            return False
        offset = self.next_offset
        self.next_offset += size_bytes
        req = TuttiRequest(layer=layer, expert=expert, size_bytes=size_bytes, staged_offset=offset)
        self.queue.append(req)
        self.in_flight[key] = req
        self.total_reads += 1
        return True

    def step_io_worker(self, elapsed_delta: float = 0.001) -> int:
        """Processes pending NVMe reads in background without blocking compute."""
        completed_this_step = 0
        now = time.perf_counter()

        while self.queue:
            req = self.queue.popleft()
            # Estimate I/O read time from NVMe to pinned host RAM: size / bandwidth
            io_latency = req.size_bytes / (self.nvme_bandwidth * 1e9)
            req.completed_at = req.enqueued_at + io_latency
            req.is_ready = True
            key = (req.layer, req.expert)
            self.staged_experts[key] = req
            self.staged_offsets[key] = req.staged_offset
            self.bytes_staged += req.size_bytes
            self.completed_reads += 1
            completed_this_step += 1
            if key in self.in_flight:
                del self.in_flight[key]

        return completed_this_step

    def is_staged(self, layer: int, expert: int) -> bool:
        """Checks if an expert is already staged in pinned host RAM."""
        return (layer, expert) in self.staged_experts

    def get_staged_offset(self, layer: int, expert: int) -> Optional[int]:
        """Returns offset within pinned staging buffer if expert is staged."""
        return self.staged_offsets.get((layer, expert))

    def wait_staged(self, layer: int, expert: int) -> float:
        """Awaits an expert to finish staging; returns any stall latency."""
        key = (layer, expert)
        if key in self.staged_experts:
            return 0.0  # Zero stall! Overlap succeeded

        t0 = time.perf_counter()
        if key in self.in_flight:
            req = self.in_flight.pop(key)
            self.queue.remove(req)
            self.bytes_staged += req.size_bytes
            req.is_ready = True
            self.staged_experts[key] = req
            self.staged_offsets[key] = req.staged_offset
            self.completed_reads += 1

        stall = time.perf_counter() - t0
        self.io_stall_seconds += stall
        return stall

    def clear(self) -> None:
        """Clears pipeline state."""
        self.next_offset = 0
        self.queue.clear()
        self.in_flight.clear()
        self.staged_experts.clear()
        self.staged_offsets.clear()
        self.bytes_staged = 0

    def get_metrics(self) -> Dict[str, Any]:
        overlap_eff = 100.0 if (self.completed_reads > 0 and self.io_stall_seconds == 0.0) else (
            max(0.0, 100.0 - (self.io_stall_seconds / max(1e-6, self.completed_reads * 0.001)) * 100.0)
            if self.completed_reads > 0 else 0.0
        )
        return {
            "simulation_only": True,
            "measured_overlap_efficiency_pct": None,
            "total_reads": self.total_reads,
            "completed_reads": self.completed_reads,
            "staged_count": len(self.staged_experts),
            "bytes_staged_mb": self.bytes_staged / (1024 * 1024),
            "io_stall_ms": self.io_stall_seconds * 1000.0,
            "overlap_efficiency_pct": overlap_eff,
            "pinned_capacity_mb": self.pinned_capacity / (1024 * 1024),
        }
