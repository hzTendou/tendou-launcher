import pytest
from src.atlas.p4_placement import (
    PlacementTier,
    ExpertPrediction,
    PlacementPolicy,
    LookaheadPlacementScheduler
)

def test_placement_policy_high_confidence_gpu():
    policy = PlacementPolicy(gpu_confidence_threshold=0.7, ram_confidence_threshold=0.3, vram_budget_bytes=10)
    pred = ExpertPrediction(layer=0, expert=1, confidence=0.8, predicted_by="od_moe")
    tier = policy.decide(pred)
    assert tier == PlacementTier.GPU

def test_placement_policy_medium_confidence_ram():
    policy = PlacementPolicy(gpu_confidence_threshold=0.7, ram_confidence_threshold=0.3, ram_budget_bytes=10)
    pred = ExpertPrediction(layer=0, expert=1, confidence=0.5, predicted_by="spice")
    tier = policy.decide(pred)
    assert tier == PlacementTier.RAM

def test_placement_policy_low_confidence_disk():
    policy = PlacementPolicy(gpu_confidence_threshold=0.7, ram_confidence_threshold=0.3, vram_budget_bytes=10, ram_budget_bytes=10)
    pred = ExpertPrediction(layer=0, expert=1, confidence=0.2, predicted_by="od_moe")
    tier = policy.decide(pred)
    assert tier == PlacementTier.DISK

def test_placement_policy_gpu_fallback_to_ram():
    policy = PlacementPolicy(gpu_confidence_threshold=0.7, vram_budget_bytes=1, ram_budget_bytes=5)
    pred1 = ExpertPrediction(layer=0, expert=1, confidence=0.9, predicted_by="od_moe")
    pred2 = ExpertPrediction(layer=0, expert=2, confidence=0.9, predicted_by="od_moe")
    
    tier1 = policy.decide(pred1)
    tier2 = policy.decide(pred2)
    
    assert tier1 == PlacementTier.GPU
    assert tier2 == PlacementTier.RAM

def test_placement_policy_gpu_fallback_to_disk_when_ram_full():
    policy = PlacementPolicy(gpu_confidence_threshold=0.7, vram_budget_bytes=1, ram_budget_bytes=1)
    pred1 = ExpertPrediction(layer=0, expert=1, confidence=0.9, predicted_by="od_moe")
    pred2 = ExpertPrediction(layer=0, expert=2, confidence=0.9, predicted_by="od_moe")
    pred3 = ExpertPrediction(layer=0, expert=3, confidence=0.9, predicted_by="od_moe")
    
    policy.decide(pred1)
    policy.decide(pred2)
    tier3 = policy.decide(pred3)
    
    assert tier3 == PlacementTier.DISK

def test_placement_policy_ram_fallback_to_disk():
    policy = PlacementPolicy(ram_confidence_threshold=0.3, vram_budget_bytes=0, ram_budget_bytes=1)
    pred1 = ExpertPrediction(layer=0, expert=1, confidence=0.5, predicted_by="spice")
    pred2 = ExpertPrediction(layer=0, expert=2, confidence=0.5, predicted_by="spice")
    
    tier1 = policy.decide(pred1)
    tier2 = policy.decide(pred2)
    
    assert tier1 == PlacementTier.RAM
    assert tier2 == PlacementTier.DISK

def test_placement_policy_zero_budget():
    policy = PlacementPolicy(vram_budget_bytes=0, ram_budget_bytes=0)
    pred = ExpertPrediction(layer=0, expert=1, confidence=0.9, predicted_by="od_moe")
    tier = policy.decide(pred)
    assert tier == PlacementTier.DISK

def test_placement_policy_lossless_miss_record():
    policy = PlacementPolicy()
    policy.register_miss(layer=2, expert=5)
    
    stats = policy.get_stats()
    assert stats["misses"] == 1
    assert stats["tier_counts"][PlacementTier.CPU_FALLBACK.value] == 1
    assert policy.miss_records == [(2, 5)]

def test_scheduler_submit_predictions():
    policy = PlacementPolicy(vram_budget_bytes=10, ram_budget_bytes=10)
    scheduler = LookaheadPlacementScheduler(policy)
    
    scheduler.submit_predictions([
        ExpertPrediction(0, 1, 0.8, "od_moe"),
        ExpertPrediction(0, 2, 0.4, "spice")
    ])
    
    assert len(scheduler.queue) == 2

def test_scheduler_get_placement_plan():
    policy = PlacementPolicy(vram_budget_bytes=10, ram_budget_bytes=10)
    scheduler = LookaheadPlacementScheduler(policy)
    
    scheduler.submit_predictions([
        ExpertPrediction(0, 1, 0.8, "od_moe"),
        ExpertPrediction(0, 2, 0.4, "spice")
    ])
    
    plan = scheduler.get_placement_plan()
    assert (0, 1) in plan
    assert (0, 2) in plan
    assert plan[(0, 1)] == PlacementTier.GPU
    assert plan[(0, 2)] == PlacementTier.RAM

def test_scheduler_on_layer_complete_eviction():
    policy = PlacementPolicy(vram_budget_bytes=10, ram_budget_bytes=10)
    scheduler = LookaheadPlacementScheduler(policy)
    
    scheduler.submit_predictions([
        ExpertPrediction(0, 1, 0.8, "od_moe"),
        ExpertPrediction(1, 1, 0.8, "od_moe")
    ])
    scheduler.get_placement_plan()
    
    assert policy.vram_used == 2
    scheduler.on_layer_complete(0)
    assert policy.vram_used == 1
    
    assert (0, 1) not in scheduler.queue
    assert (0, 1) not in scheduler.plan
    assert (1, 1) in scheduler.queue
    assert (1, 1) in scheduler.plan

def test_scheduler_on_miss():
    policy = PlacementPolicy()
    scheduler = LookaheadPlacementScheduler(policy)
    
    scheduler.on_miss(3, 4)
    stats = policy.get_stats()
    assert stats["misses"] == 1
    assert stats["tier_counts"][PlacementTier.CPU_FALLBACK.value] == 1

def test_scheduler_stats_hits_increment():
    policy = PlacementPolicy(vram_budget_bytes=10, ram_budget_bytes=10)
    scheduler = LookaheadPlacementScheduler(policy)
    
    scheduler.submit_predictions([
        ExpertPrediction(0, 1, 0.8, "od_moe")
    ])
    scheduler.get_placement_plan()
    
    stats = policy.get_stats()
    assert stats["hits"] == 1

def test_all_tiers_disk_when_capacity_full():
    policy = PlacementPolicy(gpu_confidence_threshold=0.9, ram_confidence_threshold=0.5, vram_budget_bytes=1, ram_budget_bytes=1)
    
    policy.decide(ExpertPrediction(0, 1, 0.95, "od_moe")) # GPU
    policy.decide(ExpertPrediction(0, 2, 0.6, "spice"))   # RAM
    
    t3 = policy.decide(ExpertPrediction(0, 3, 0.95, "od_moe")) # VRAM and RAM full -> DISK
    t4 = policy.decide(ExpertPrediction(0, 4, 0.6, "spice"))   # RAM full -> DISK
    
    assert t3 == PlacementTier.DISK
    assert t4 == PlacementTier.DISK

def test_placement_policy_default_thresholds():
    policy = PlacementPolicy()
    assert policy.gpu_confidence_threshold == 0.70
    assert policy.ram_confidence_threshold == 0.30

def test_lookahead_layers_config():
    policy = PlacementPolicy()
    scheduler = LookaheadPlacementScheduler(policy, lookahead_layers=5)
    assert scheduler.lookahead_layers == 5
