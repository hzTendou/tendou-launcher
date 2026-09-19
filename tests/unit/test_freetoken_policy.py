import pytest
from src.atlas.freetoken_policy import (
    LRUExpertCache,
    BandwidthCeilingPolicy,
    DoubleBufferPrefillPlanner,
    SemanticAnchorCache,
)

class TestLRUExpertCache:
    def test_initial_state(self):
        cache = LRUExpertCache(capacity_slots=2)
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache.cache) == 0

    def test_access_miss(self):
        cache = LRUExpertCache(capacity_slots=2)
        assert cache.access(0, 1) is False
        assert cache.misses == 1

    def test_access_hit(self):
        cache = LRUExpertCache(capacity_slots=2)
        cache.access(0, 1)
        assert cache.access(0, 1) is True
        assert cache.hits == 1

    def test_eviction_slots(self):
        cache = LRUExpertCache(capacity_slots=2)
        cache.access(0, 1)
        cache.access(0, 2)
        cache.access(0, 3)
        assert (0, 1) not in cache.cache
        assert (0, 2) in cache.cache
        assert (0, 3) in cache.cache

    def test_eviction_bytes(self):
        cache = LRUExpertCache(capacity_slots=0, vram_budget_bytes=100)
        cache.access(0, 1, size_bytes=60)
        cache.access(0, 2, size_bytes=50) # Should evict (0, 1)
        assert (0, 1) not in cache.cache
        assert (0, 2) in cache.cache
        assert cache.used_bytes == 50
        
    def test_get_stats(self):
        cache = LRUExpertCache(capacity_slots=10)
        cache.access(1, 1)
        cache.access(1, 1)
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["used_slots"] == 1
        assert stats["capacity_slots"] == 10


class TestBandwidthCeilingPolicy:
    def test_should_offload_true(self):
        policy = BandwidthCeilingPolicy(cpu_bw_gbs=10.0, pcie_bw_gbs=25.0, threshold=2.0)
        assert policy.should_offload_to_gpu() is True

    def test_should_offload_false(self):
        policy = BandwidthCeilingPolicy(cpu_bw_gbs=15.0, pcie_bw_gbs=25.0, threshold=2.0)
        assert policy.should_offload_to_gpu() is False

    def test_should_offload_zero_cpu_bw(self):
        policy = BandwidthCeilingPolicy(cpu_bw_gbs=0.0, pcie_bw_gbs=25.0, threshold=2.0)
        assert policy.should_offload_to_gpu() is True

    def test_threshold_exact_match(self):
        policy = BandwidthCeilingPolicy(cpu_bw_gbs=10.0, pcie_bw_gbs=20.0, threshold=2.0)
        assert policy.should_offload_to_gpu() is True


class TestDoubleBufferPrefillPlanner:
    def test_initial_state(self):
        planner = DoubleBufferPrefillPlanner(slot_count=2)
        assert planner.get_buffered_layers() == []

    def test_plan_prefetch(self):
        planner = DoubleBufferPrefillPlanner(slot_count=2)
        slot = planner.plan_prefetch(5)
        assert slot == 0
        assert 5 in planner.get_buffered_layers()

    def test_buffer_cycling(self):
        planner = DoubleBufferPrefillPlanner(slot_count=2)
        assert planner.plan_prefetch(1) == 0
        assert planner.plan_prefetch(2) == 1
        assert planner.plan_prefetch(3) == 0 # Overwrites slot 0
        layers = planner.get_buffered_layers()
        assert 1 not in layers
        assert 3 in layers
        assert 2 in layers

    def test_no_duplicate_allocation(self):
        planner = DoubleBufferPrefillPlanner(slot_count=2)
        assert planner.plan_prefetch(1) == 0
        assert planner.plan_prefetch(1) == 0 # Should return same slot
        assert planner.current_slot == 1


class TestSemanticAnchorCache:
    def test_initial_state(self):
        cache = SemanticAnchorCache()
        assert cache.has_snapshot("test") is False

    def test_save_and_load(self):
        cache = SemanticAnchorCache()
        cache.save_snapshot("anchor1", {"kv": "data"})
        assert cache.has_snapshot("anchor1") is True
        assert cache.load_snapshot("anchor1") == {"kv": "data"}

    def test_load_nonexistent(self):
        cache = SemanticAnchorCache()
        assert cache.load_snapshot("missing") is None
