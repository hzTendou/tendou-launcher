"""Tests for P2 GPU Binder Simulator."""
import pytest
from src.atlas.p2_gpu_binder import ExpertIDMapper, GPUExpertBuffer, ExpertBindingPlan

# ExpertIDMapper Tests
def test_expert_id_mapper_valid():
    mapper = ExpertIDMapper(num_layers=48, num_experts=512)
    assert mapper.get_compact_id(0, 0) == 0
    assert mapper.get_compact_id(0, 511) == 511
    assert mapper.get_compact_id(1, 0) == 512
    assert mapper.get_compact_id(47, 511) == 48 * 512 - 1

def test_expert_id_mapper_invalid_layer():
    mapper = ExpertIDMapper()
    with pytest.raises(ValueError):
        mapper.get_compact_id(-1, 0)
    with pytest.raises(ValueError):
        mapper.get_compact_id(48, 0)

def test_expert_id_mapper_invalid_expert():
    mapper = ExpertIDMapper()
    with pytest.raises(ValueError):
        mapper.get_compact_id(0, -1)
    with pytest.raises(ValueError):
        mapper.get_compact_id(0, 512)

def test_expert_id_mapper_collision():
    mapper = ExpertIDMapper(num_layers=48, num_experts=512)
    ids = set()
    for l in range(2):
        for e in range(512):
            ids.add(mapper.get_compact_id(l, e))
    assert len(ids) == 2 * 512

# GPUExpertBuffer Tests
def test_gpu_expert_buffer_zero_capacity():
    buffer = GPUExpertBuffer(capacity_bytes=0)
    assert buffer.bind_expert(0, 0) == False
    assert buffer.bind_expert(0, 1) == False
    assert len(buffer.resident_experts) == 0

def test_gpu_expert_buffer_capacity():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 2) # fits 2 experts
    assert buffer.bind_expert(0, 0) == True
    assert buffer.bind_expert(0, 1) == True
    assert buffer.bind_expert(0, 2) == False
    assert len(buffer.resident_experts) == 2

def test_gpu_expert_buffer_eviction():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 2)
    buffer.bind_expert(0, 0)
    buffer.bind_expert(0, 1)
    # Evict 0,0
    evicted = buffer.evict_lru()
    assert evicted == (0, 0)
    assert (0, 0) not in buffer.resident_experts
    assert buffer.bind_expert(0, 2) == True

def test_gpu_expert_buffer_eviction_keep():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 2)
    buffer.bind_expert(0, 0)
    buffer.bind_expert(0, 1)
    # Evict, but keep 0,0
    evicted = buffer.evict_lru(keep_experts={(0, 0)})
    assert evicted == (0, 1)

# ExpertBindingPlan Tests
def test_expert_binding_plan_all_gpu():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 10)
    plan = ExpertBindingPlan(buffer)
    router_experts = list(range(10))
    result = plan.plan_binding(0, router_experts)
    assert len(result['gpu']) == 10
    assert len(result['cpu']) == 0

def test_expert_binding_plan_mixed():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 5)
    plan = ExpertBindingPlan(buffer)
    router_experts = list(range(10))
    result = plan.plan_binding(0, router_experts)
    assert len(result['gpu']) == 5
    assert len(result['cpu']) == 5
    assert set(result['gpu']) == set(range(5))
    assert set(result['cpu']) == set(range(5, 10))

def test_expert_binding_plan_lossless_fallback():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 1) # capacity 1
    plan = ExpertBindingPlan(buffer)
    result = plan.plan_binding(0, [0, 1])
    assert result['gpu'] == [0]
    assert result['cpu'] == [1]

def test_expert_binding_plan_with_eviction():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 2)
    plan = ExpertBindingPlan(buffer)
    # Layer 0 uses 0, 1
    plan.plan_binding(0, [0, 1])
    # Layer 1 uses 2, 3 -> should evict 0, 1
    result = plan.plan_binding(1, [2, 3])
    assert result['gpu'] == [2, 3]
    assert len(result['cpu']) == 0
    assert (0, 0) not in buffer.resident_experts
    assert (1, 2) in buffer.resident_experts

def test_expert_binding_plan_invalid_expert_no_eviction():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 2)
    buffer.bind_expert(0, 0)
    buffer.bind_expert(0, 1)
    plan = ExpertBindingPlan(buffer)
    result = plan.plan_binding(0, [999])
    assert result['gpu'] == []
    assert result['cpu'] == [999]
    assert (0, 0) in buffer.resident_experts
    assert (0, 1) in buffer.resident_experts

def test_gpu_expert_buffer_invalid_bounds():
    buffer = GPUExpertBuffer(capacity_bytes=3481600 * 2)
    assert not buffer.bind_expert(-1, 0)
    assert not buffer.bind_expert(48, 0)
    assert not buffer.bind_expert(0, -1)
    assert not buffer.bind_expert(0, 512)
