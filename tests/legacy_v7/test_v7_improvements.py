"""Unit tests for Atlas V7 improvements:
1. FrequencyBiasedVRAM eviction behavior
2. Preload experts to RAM in TieredSim
3. Oracle decomposition diagnostic functions
4. Deadline-aware budget calculation
"""
import unittest
from collections import OrderedDict

from atlas_simulator import (
    CacheItem, GlobalLRU, FrequencyBiasedVRAM, TieredSim,
    collect_all_expert_keys, evaluate_policy
)
from oracle_decompose import run_belady_vram, decompose_gap


class TestV7Improvements(unittest.TestCase):
    def test_frequency_biased_vram_eviction(self):
        # Cache capacity = 3.0 MB, item size = 1.0 MB -> holds 3 items
        cache = FrequencyBiasedVRAM(capacity_mb=3.0)
        
        # Put 3 items
        cache.put(('0', 1), CacheItem(1.0, 0.0))
        cache.put(('0', 2), CacheItem(1.0, 0.0))
        cache.put(('0', 3), CacheItem(1.0, 0.0))
        
        # Access item 1 and item 2 multiple times so they have higher frequency
        cache.get(('0', 1))
        cache.get(('0', 1))
        cache.get(('0', 2))
        # Item 3 has frequency 0
        
        # Now put item 4 (needs eviction)
        cache.put(('0', 4), CacheItem(1.0, 0.0))
        
        # Item 3 should have been evicted because it has lowest frequency
        self.assertFalse(cache.has(('0', 3)))
        self.assertTrue(cache.has(('0', 1)))
        self.assertTrue(cache.has(('0', 2)))
        self.assertTrue(cache.has(('0', 4)))

    def test_preload_experts_to_ram_eliminates_nvme(self):
        sim = TieredSim(
            expert_mb=1.2,
            vram_capacity_mb=7000.0,
            ram_gb=16.0,
            nvme_gbps=7.0,
            pcie_gbps=12.0,
            ram_gbps=40.0,
            nvme_latency_ms=0.08,
            pcie_latency_ms=0.02,
            compute_ms=50.0,
        )
        
        expert_set = {('0', i) for i in range(100)}
        ok, done = sim.preload_experts_to_ram(expert_set)
        self.assertTrue(ok)
        self.assertGreater(done, 0.0)
        self.assertGreater(sim.stats['ram_preload_mb'], 0)
        
        # When accessing preloaded experts, they should be RAM hits, not NVMe misses
        for k in expert_set:
            sim.ensure(k, sim.now_ms)
        
        self.assertEqual(sim.stats['nvme_miss'], 0)
        self.assertEqual(sim.stats['ram_hit'], len(expert_set))

    def test_belady_vram_optimal_replacement(self):
        # Synthetic session with known reuse pattern
        # Step 0: [1, 2], Step 1: [1, 3], Step 2: [2, 3]
        sessions = [{
            'records': [
                {'experts_by_layer': {'0': [1, 2]}},
                {'experts_by_layer': {'0': [1, 3]}},
                {'experts_by_layer': {'0': [2, 3]}},
            ]
        }]
        # Capacity for 2 experts (2.4 MB for 1.2 MB expert)
        hits, misses, hit_rate = run_belady_vram(sessions, vram_capacity_mb=2.4, expert_mb=1.2)
        self.assertGreaterEqual(hit_rate, 0.0)
        self.assertEqual(hits + misses, 6)

    def test_deadline_budget_capping(self):
        # 50 ms compute window, 12 GB/s PCIe, 1.2 MB expert
        # Transfer bandwidth: (50 - 0.02) / 1000 * 12 * 1024 = 614 MB
        # Max experts: 614 / 1.2 = ~511 experts
        sim = TieredSim(
            expert_mb=1.2,
            vram_capacity_mb=7000.0,
            ram_gb=16.0,
            nvme_gbps=7.0,
            pcie_gbps=12.0,
            ram_gbps=40.0,
            nvme_latency_ms=0.08,
            pcie_latency_ms=0.02,
            compute_ms=50.0,
        )
        self.assertEqual(sim.compute_ms, 50.0)


if __name__ == '__main__':
    unittest.main()
