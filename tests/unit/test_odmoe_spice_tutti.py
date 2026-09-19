"""Unit tests for OD-MoE, SPICE, and Tutti pipeline implementations.

Validates:
1. SPICE: Confidence-aware 3-tier partitioning (VRAM, RAM, NVMe)
2. SPICE: Strict lossless fallback guarantee (zero surrogate tolerance)
3. OD-MoE: Multi-layer advance lookahead prediction (cross-layer routing)
4. OD-MoE: Fast post-layer eviction with anti-thrashing protection
5. Tutti: Async NVMe read queue, pinned staging pool, and compute/IO overlap
6. End-to-end multi-tier pipeline orchestration
"""
import pytest
from src.atlas.spice_scheduler import ConfidenceTier, SpiceScheduler
from src.atlas.od_moe import ODMoEPrefetcher
from src.atlas.tutti_pipeline import TuttiPipeline, TuttiRequest


class TestSpiceScheduler:
    def test_confidence_classification(self):
        scheduler = SpiceScheduler(conf_high=0.70, conf_mid=0.35)

        # High confidence -> VRAM
        tier, mem = scheduler.classify_confidence(0.85)
        assert tier == ConfidenceTier.HIGH
        assert mem == "vram"

        tier, mem = scheduler.classify_confidence(0.70)
        assert tier == ConfidenceTier.HIGH
        assert mem == "vram"

        # Medium confidence -> RAM
        tier, mem = scheduler.classify_confidence(0.50)
        assert tier == ConfidenceTier.MEDIUM
        assert mem == "ram"

        tier, mem = scheduler.classify_confidence(0.35)
        assert tier == ConfidenceTier.MEDIUM
        assert mem == "ram"

        # Low confidence -> NVMe
        tier, mem = scheduler.classify_confidence(0.20)
        assert tier == ConfidenceTier.LOW
        assert mem == "nvme"

        tier, mem = scheduler.classify_confidence(0.0)
        assert tier == ConfidenceTier.LOW
        assert mem == "nvme"

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="Invalid thresholds"):
            SpiceScheduler(conf_high=0.30, conf_mid=0.70)

        with pytest.raises(ValueError, match="Invalid thresholds"):
            SpiceScheduler(conf_high=1.5, conf_mid=0.5)

    def test_candidate_scheduling_partitioning(self):
        scheduler = SpiceScheduler(conf_high=0.65, conf_mid=0.30)
        candidates = [(10, 0.90), (12, 0.50), (15, 0.15), (20, 0.75), (25, 0.28)]

        partitioned = scheduler.schedule_candidates(layer=5, candidates=candidates)

        high_ids = [c[0] for c in partitioned[ConfidenceTier.HIGH]]
        mid_ids = [c[0] for c in partitioned[ConfidenceTier.MEDIUM]]
        low_ids = [c[0] for c in partitioned[ConfidenceTier.LOW]]

        assert high_ids == [10, 20]
        assert mid_ids == [12]
        assert low_ids == [15, 25]

        metrics = scheduler.get_metrics()
        assert metrics["vram_scheduled"] == 2
        assert metrics["ram_scheduled"] == 1
        assert metrics["nvme_left"] == 2

    def test_lossless_fallback_and_no_surrogates(self):
        scheduler = SpiceScheduler()

        # Cache hit in VRAM
        hit = scheduler.fallback_schedule(layer=4, requested_expert=42, current_residency="vram")
        assert hit["hit"] is True
        assert hit["action"] == "cache_hit"
        assert hit["lossless"] is True

        # Cache miss from RAM -> H2D transfer
        miss_ram = scheduler.fallback_schedule(layer=4, requested_expert=43, current_residency="ram")
        assert miss_ram["hit"] is False
        assert miss_ram["action"] == "h2d_transfer"
        assert miss_ram["lossless"] is True

        # Cache miss from NVMe -> demand load
        miss_nvme = scheduler.fallback_schedule(layer=4, requested_expert=44, current_residency="nvme")
        assert miss_nvme["hit"] is False
        assert miss_nvme["action"] == "nvme_demand_load"
        assert miss_nvme["lossless"] is True

        # Verify lossless check
        actual = [1, 5, 8]
        executed = [1, 5, 8]
        assert scheduler.verify_lossless(actual, executed) is True

        # Strictly reject surrogate / approximate substitutions
        with pytest.raises(ValueError, match="Lossless violation"):
            surrogate_executed = [1, 5, 9]  # Expert 8 replaced by surrogate 9
            scheduler.verify_lossless(actual, surrogate_executed)

    def test_composite_confidence_calculation(self):
        scheduler = SpiceScheduler()
        c_baseline = scheduler.compute_expert_confidence()
        assert 0.05 <= c_baseline <= 1.0

        c_boosted = scheduler.compute_expert_confidence(
            markov_prob=0.8, stickiness=True, frequency=0.08, cross_layer_prob=0.9
        )
        assert c_boosted > c_baseline
        assert c_boosted <= 1.0


class TestODMoEPrefetcher:
    def test_multistep_lookahead_prediction(self):
        scheduler = SpiceScheduler(conf_high=0.65, conf_mid=0.30)
        odmoe = ODMoEPrefetcher(lead_layers=3, scheduler=scheduler)

        # Train transitions across layers: layer 2 -> layer 3 -> layer 4
        odmoe.observe_layer(2, [10, 11])
        odmoe.observe_layer(3, [20, 21])
        odmoe.observe_layer(4, [30, 31])
        odmoe.end_token()

        # Another token
        odmoe.observe_layer(2, [10, 11])
        odmoe.observe_layer(3, [20, 21])
        odmoe.observe_layer(4, [30, 31])
        odmoe.end_token()

        # Predict from layer 2: lookahead should cover layer 3, 4, 5
        predictions = odmoe.predict_lookahead_layers(current_layer=2, current_experts=[10, 11], k=4)
        assert 3 in predictions
        assert 4 in predictions
        assert 5 in predictions
        assert len(predictions[3]) > 0
        assert len(predictions[4]) > 0

        # Layer 3 should have highest probability for 20, 21
        top_l3 = [exp for exp, _ in predictions[3][:2]]
        assert 20 in top_l3 and 21 in top_l3

        # Multi-layer lookahead check: Layer 4 should predict 30, 31 (NOT repeat Layer 3)
        top_l4 = [exp for exp, _ in predictions[4][:2]]
        assert 30 in top_l4 and 31 in top_l4

        # Hit tracking: when layer 3 runs with [20, 21], hits must be recorded
        hits_before = odmoe.lookahead_hits
        odmoe.observe_layer(3, [20, 21])
        assert odmoe.lookahead_hits > hits_before

    def test_plan_prefetches_partitioned(self):
        scheduler = SpiceScheduler(conf_high=0.60, conf_mid=0.25)
        odmoe = ODMoEPrefetcher(lead_layers=2, scheduler=scheduler)

        # Train
        odmoe.observe_layer(0, [1, 2])
        odmoe.observe_layer(1, [10, 20])
        odmoe.end_token()

        plan = odmoe.plan_prefetches(current_layer=0, current_experts=[1, 2], k=4)
        assert 1 in plan
        assert 2 in plan
        assert ConfidenceTier.HIGH in plan[1]
        assert ConfidenceTier.MEDIUM in plan[1]
        assert ConfidenceTier.LOW in plan[1]

    def test_quick_eviction_under_memory_pressure(self):
        odmoe = ODMoEPrefetcher(lead_layers=2, quick_evict=True)

        vram_resident = [(5, 10), (5, 11), (6, 20), (7, 30)]
        capacity = 1000
        # High pressure (85% > 80% watermark)
        high_pressure_used = 850

        upcoming_plan = {
            6: {ConfidenceTier.HIGH: [(20, 0.9)]},
            7: {ConfidenceTier.HIGH: [(30, 0.8)]},
        }

        # Layer 5 just finished, not in upcoming plan -> should be quick evicted
        evictions = odmoe.identify_quick_evictions(
            completed_layer=5,
            vram_resident_experts=vram_resident,
            upcoming_plan=upcoming_plan,
            vram_used_bytes=high_pressure_used,
            vram_capacity_bytes=capacity,
        )
        assert (5, 10) in evictions
        assert (5, 11) in evictions
        assert (6, 20) not in evictions  # layer 6 not completed

    def test_quick_eviction_anti_thrashing_and_low_pressure(self):
        odmoe = ODMoEPrefetcher(lead_layers=2, quick_evict=True)

        vram_resident = [(5, 10), (5, 11)]
        capacity = 1000

        # Case 1: Low pressure (50% < 80% watermark) -> no evictions needed
        evictions_low = odmoe.identify_quick_evictions(
            completed_layer=5,
            vram_resident_experts=vram_resident,
            upcoming_plan={},
            vram_used_bytes=500,
            vram_capacity_bytes=capacity,
        )
        assert len(evictions_low) == 0

        # Case 2: Expert 10 is predicted to be reused in upcoming layer -> protect from eviction!
        upcoming_with_reuse = {
            5: {ConfidenceTier.HIGH: [(10, 0.9)]}  # Same layer and expert reused.
        }
        evictions_protect = odmoe.identify_quick_evictions(
            completed_layer=5,
            vram_resident_experts=[(5, 10), (5, 11)],
            upcoming_plan=upcoming_with_reuse,
            vram_used_bytes=850,
            vram_capacity_bytes=capacity,
        )
        # 11 evicted, 10 protected
        assert (5, 11) in evictions_protect
        assert (5, 10) not in evictions_protect

    def test_anchor_layers_protected_from_eviction(self):
        odmoe = ODMoEPrefetcher(lead_layers=2, quick_evict=True, gpu_expert_layers=2)
        # Layers 0 and 1 are GPU anchor layers
        vram_resident = [(0, 1), (0, 2), (1, 5), (2, 10)]
        capacity = 1000
        high_pressure_used = 900

        # Attempt to quick-evict completed layer 0 (anchor)
        evictions_l0 = odmoe.identify_quick_evictions(
            completed_layer=0,
            vram_resident_experts=vram_resident,
            upcoming_plan={},
            vram_used_bytes=high_pressure_used,
            vram_capacity_bytes=capacity,
        )
        assert len(evictions_l0) == 0  # Completely protected!

        # Attempt to quick-evict completed layer 1 (anchor)
        evictions_l1 = odmoe.identify_quick_evictions(
            completed_layer=1,
            vram_resident_experts=vram_resident,
            upcoming_plan={},
            vram_used_bytes=high_pressure_used,
            vram_capacity_bytes=capacity,
        )
        assert len(evictions_l1) == 0  # Completely protected!

        # Layer 2 is NOT an anchor -> can be evicted
        evictions_l2 = odmoe.identify_quick_evictions(
            completed_layer=2,
            vram_resident_experts=vram_resident,
            upcoming_plan={},
            vram_used_bytes=high_pressure_used,
            vram_capacity_bytes=capacity,
        )
        assert (2, 10) in evictions_l2

    def test_lead_boundary_clamping(self):
        # Extremely large lead_layers should be clamped safely to <= 8
        odmoe_large = ODMoEPrefetcher(lead_layers=50)
        assert odmoe_large.lead_layers <= 8

        odmoe_zero = ODMoEPrefetcher(lead_layers=0)
        assert odmoe_zero.lead_layers >= 1


class TestTuttiPipeline:
    def test_queue_and_staging(self):
        pipeline = TuttiPipeline(pinned_capacity_bytes=100 * 1024 * 1024, max_queue_depth=8)

        # Enqueue reads
        assert pipeline.enqueue_read(layer=1, expert=5) is True
        assert pipeline.enqueue_read(layer=1, expert=6) is True
        assert pipeline.enqueue_read(layer=1, expert=5) is True  # Duplicate, returns True

        assert len(pipeline.queue) == 2
        assert not pipeline.is_staged(layer=1, expert=5)

        # Step background I/O worker
        processed = pipeline.step_io_worker()
        assert processed == 2
        assert len(pipeline.queue) == 0
        assert pipeline.is_staged(layer=1, expert=5)
        assert pipeline.is_staged(layer=1, expert=6)

        # Zero stall when already staged
        stall = pipeline.wait_staged(layer=1, expert=5)
        assert stall == 0.0

        metrics = pipeline.get_metrics()
        assert metrics["completed_reads"] == 2
        assert metrics["staged_count"] == 2
        assert metrics["bytes_staged_mb"] > 0
        assert metrics["overlap_efficiency_pct"] == 100.0

        # Verify staged offset retrieval
        offset5 = pipeline.get_staged_offset(layer=1, expert=5)
        offset6 = pipeline.get_staged_offset(layer=1, expert=6)
        assert offset5 is not None
        assert offset6 is not None
        assert offset5 != offset6

    def test_queue_depth_limit(self):
        pipeline = TuttiPipeline(max_queue_depth=2)
        assert pipeline.enqueue_read(1, 1) is True
        assert pipeline.enqueue_read(1, 2) is True
        assert pipeline.enqueue_read(1, 3) is False  # Queue full!

    def test_clear_pipeline(self):
        pipeline = TuttiPipeline()
        pipeline.enqueue_read(2, 4)
        pipeline.step_io_worker()
        assert pipeline.is_staged(2, 4)

        pipeline.clear()
        assert not pipeline.is_staged(2, 4)
        assert len(pipeline.queue) == 0


class TestOrchestratedPipeline:
    def test_full_pipeline_flow_lossless(self):
        """Simulates end-to-end OD-MoE prediction -> SPICE scheduling -> Tutti staging -> Lossless compute."""
        scheduler = SpiceScheduler(conf_high=0.65, conf_mid=0.30)
        odmoe = ODMoEPrefetcher(lead_layers=2, scheduler=scheduler)
        tutti = TuttiPipeline(pinned_capacity_bytes=50 * 1024 * 1024)

        # Train real correlations; cold start must not invent experts to prefetch.
        odmoe.observe_layer(0, [2, 3])
        odmoe.observe_layer(1, [4, 5])
        odmoe.end_token()

        # Simulate execution of Layer 0
        active_l0 = [2, 3]
        odmoe.observe_layer(0, active_l0)

        # OD-MoE plans lookahead for Layer 1 and 2
        plan = odmoe.plan_prefetches(current_layer=0, current_experts=active_l0, k=4)

        # Tutti queues reads for HIGH (VRAM) and MEDIUM (RAM) experts
        for tgt_l, tiers in plan.items():
            for exp, _ in tiers[ConfidenceTier.HIGH] + tiers[ConfidenceTier.MEDIUM]:
                tutti.enqueue_read(layer=tgt_l, expert=exp)

        # I/O worker stages experts asynchronously in background
        completed = tutti.step_io_worker()
        assert completed > 0

        # Layer 1 execution: actual router chooses experts
        actual_l1 = [1, 2]
        odmoe.observe_layer(1, actual_l1)

        # Lossless fallback for each expert:
        executed_l1 = []
        for exp in actual_l1:
            residency = "ram" if tutti.is_staged(1, exp) else "nvme"
            fallback_res = scheduler.fallback_schedule(1, exp, residency)
            assert fallback_res["lossless"] is True
            executed_l1.append(exp)

        # Verify lossless fidelity guarantee: exact router outputs executed
        assert scheduler.verify_lossless(actual_l1, executed_l1) is True

        # Quick eviction after layer 2 (non-anchor layer)
        vram_res = [(2, 2), (2, 3)]
        evicted = odmoe.identify_quick_evictions(
            completed_layer=2,
            vram_resident_experts=vram_res,
            upcoming_plan=plan,
            vram_used_bytes=900,
            vram_capacity_bytes=1000,
        )
        assert len(evicted) > 0


def test_multilabel_confidence_and_token_boundary():
    predictor = ODMoEPrefetcher()
    targets = list(range(20, 30))
    for _ in range(10):
        predictor.observe_layer(0, [1, 2])
        predictor.observe_layer(1, targets)
        # Boundary is detected even without an explicit end_token call.
    predicted = predictor.predict_lookahead_layers(0, [1, 2])[1]
    assert {e for e, c in predicted} == set(targets)
    assert all(c > 0.8 for e, c in predicted)
    assert predictor.predict_lookahead_layers(47, [1]) == {}


def test_expert_identity_includes_layer():
    predictor = ODMoEPrefetcher()
    evicted = predictor.identify_quick_evictions(5, [(5, 10)],
        {6: {ConfidenceTier.HIGH: [(10, 0.9)]}}, 900, 1000)
    assert evicted == [(5, 10)]


def test_staging_model_capacity_and_single_completion():
    pipeline = TuttiPipeline(pinned_capacity_bytes=10)
    assert pipeline.enqueue_read(0, 1, 6)
    assert pipeline.enqueue_read(0, 2, 4)
    assert not pipeline.enqueue_read(0, 3, 1)
    pipeline.wait_staged(0, 1)
    assert pipeline.step_io_worker() == 1
    assert pipeline.completed_reads == 2
    assert pipeline.bytes_staged == 10
    assert pipeline.get_staged_offset(0, 2) == 6
    assert pipeline.get_metrics()["simulation_only"]


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_confidence_rejected(confidence):
    with pytest.raises(ValueError, match="finite"):
        SpiceScheduler().classify_confidence(confidence)
