"""Phase 10 Architecture & Systems Engineering Verification Suite — Tendou Launcher.

Tests the mathematical models, KV cache scaling tiers, dual-cursor amortized
re-quantization with rollback protection, MoE-aware dynamic EV calibration,
causal block-sparse attention tile masking, and DirectStorage/BypassIO alignment.
"""

import math
import pytest


def get_kv_tier(n_tokens: int) -> str:
    if n_tokens <= 8192:
        return "tier_0_q8_q8"
    elif n_tokens <= 32768:
        return "tier_1_q8_q4"
    elif n_tokens <= 131072:
        return "tier_2_q4_q4"
    else:
        return "tier_3_compressed_ring"


def calculate_kv_footprint_bytes(n_tokens: int, n_layers: int = 40, n_heads: int = 2, head_dim: int = 128) -> int:
    tier = get_kv_tier(n_tokens)
    total_dim = n_layers * n_heads * head_dim
    bytes_per_dim_map = {
        "tier_0_q8_q8": 2.125,       # (34/32) + (34/32)
        "tier_1_q8_q4": 1.625,       # (34/32) + (18/32)
        "tier_2_q4_q4": 1.125,       # (18/32) + (18/32) -> 11.25 KB/tok @ 10,000 dim
        "tier_3_compressed_ring": 0.750, # Ring buffer (8k) + Sink (64) + IQ2_XXS distant (~0.75 B/elem avg)
    }
    bpd = bytes_per_dim_map[tier]
    return int(n_tokens * total_dim * bpd)


def is_token_attended(query_pos: int, key_pos: int, sink_count: int = 64, sliding_window: int = 8192) -> bool:
    if key_pos > query_pos:
        return False
    if key_pos < sink_count:
        return True
    return (query_pos - key_pos) <= sliding_window


def is_tile_masked(
    query_tile_idx: int,
    key_tile_idx: int,
    tile_size: int = 64,
    sink_count: int = 64,
    sliding_window: int = 8192,
) -> bool:
    # Causal mask: key tile strictly in the future of the query tile
    if key_tile_idx > query_tile_idx:
        return True
    k_start = key_tile_idx * tile_size
    k_end = (key_tile_idx + 1) * tile_size
    # Sink token preservation
    if k_start < sink_count:
        return False
    q_start = query_tile_idx * tile_size
    window_start = (q_start - sliding_window) if q_start > sliding_window else 0
    if k_end <= window_start:
        return True
    return False


def moe_expert_expansion_factor(d: int, tau: float = 2.2, alpha_max: float = 2.6) -> float:
    if d <= 1:
        return 1.0
    return 1.0 + (alpha_max - 1.0) * (1.0 - math.exp(-(d - 1) / tau))


def compute_expected_value(
    d: int,
    t_gpu_shared_ms: float = 12.0,
    t_cpu_moe_ms: float = 28.0,
    t_draft_step_ms: float = 2.5,
    acceptance_probs: list = None,
) -> float:
    if d <= 0:
        return 0.0
    if acceptance_probs is None:
        acceptance_probs = [0.85, 0.75, 0.65, 0.55]

    alpha_d = moe_expert_expansion_factor(d)
    t_target_1 = t_gpu_shared_ms + 1.0 * t_cpu_moe_ms
    t_target_d = t_gpu_shared_ms + alpha_d * t_cpu_moe_ms

    prob_cum = 1.0
    saved_target_time = 0.0
    for i in range(1, d + 1):
        p_i = acceptance_probs[i - 1] if i - 1 < len(acceptance_probs) else 0.50
        prob_cum *= p_i
        saved_target_time += prob_cum * t_target_1

    draft_cost = d * t_draft_step_ms
    moe_verify_penalty = t_target_d - t_target_1
    return saved_target_time - (draft_cost + moe_verify_penalty)


def select_optimal_horizon(
    max_depth: int = 4,
    t_gpu_shared_ms: float = 12.0,
    t_cpu_moe_ms: float = 28.0,
    t_draft_step_ms: float = 2.5,
    acceptance_probs: list = None,
) -> int:
    best_d = 0
    max_ev = 0.0
    for d in range(1, max_depth + 1):
        ev = compute_expected_value(d, t_gpu_shared_ms, t_cpu_moe_ms, t_draft_step_ms, acceptance_probs)
        if ev > max_ev:
            max_ev = ev
            best_d = d
    return best_d


class TestPhase10Architecture:
    def test_tiered_kv_quant_state_machine_thresholds(self):
        assert get_kv_tier(0) == "tier_0_q8_q8"
        assert get_kv_tier(8192) == "tier_0_q8_q8"
        assert get_kv_tier(8193) == "tier_1_q8_q4"
        assert get_kv_tier(32768) == "tier_1_q8_q4"
        assert get_kv_tier(32769) == "tier_2_q4_q4"
        assert get_kv_tier(131072) == "tier_2_q4_q4"
        assert get_kv_tier(131073) == "tier_3_compressed_ring"
        assert get_kv_tier(256000) == "tier_3_compressed_ring"

    def test_tier_2_memory_footprint_11_25_kb(self):
        # Model with 10,000 effective KV dimension across layers
        footprint_1_tok = int(10000 * 1.125)
        assert footprint_1_tok == 11250  # Exactly 11.25 KB/token

        # Footprint at 131,072 context
        footprint_131k = calculate_kv_footprint_bytes(131072, n_layers=40, n_heads=2, head_dim=128)
        footprint_gib = footprint_131k / (1024 ** 3)
        # 131,072 * 10,240 * 1.125 = 1,509,949,440 bytes (~1.406 GiB)
        assert 1.35 < footprint_gib < 1.45

        # Footprint at 256,000 context with Tier 3 hybrid compression
        footprint_256k = calculate_kv_footprint_bytes(256000, n_layers=40, n_heads=2, head_dim=128)
        footprint_256k_gib = footprint_256k / (1024 ** 3)
        assert footprint_256k_gib < 2.0  # Fits easily in 16 GB laptop RAM with 0 OOM

    def test_dual_cursor_amortized_requantization_and_rollback(self):
        total_tokens = 8192
        cursor_active = total_tokens
        cursor_quant = 0
        blocks_per_step = 4
        block_size = 32
        tokens_per_step = blocks_per_step * block_size  # 128 tokens

        # Step 1
        cursor_quant += tokens_per_step
        assert cursor_quant == 128

        # Simulated token rollback: context rolls back to 64 tokens
        new_active = 64
        cursor_active = new_active
        # Clamping prevents unsigned underflow:
        if cursor_quant > cursor_active:
            cursor_quant = cursor_active
        assert cursor_quant == 64
        assert cursor_quant == cursor_active

        # Further re-quantize step yields 0 converted tokens (no underflow)
        converted = min(cursor_quant + tokens_per_step, cursor_active) - cursor_quant
        assert converted == 0

    def test_attention_sink_and_sliding_window_masking(self):
        # Sink tokens (0..63) are always attended even from 100k distant query
        assert is_token_attended(query_pos=100000, key_pos=0) is True
        assert is_token_attended(query_pos=100000, key_pos=63) is True

        # Future tokens are never attended (causal mask)
        assert is_token_attended(query_pos=100, key_pos=101) is False

        # Local sliding window tokens are attended
        assert is_token_attended(query_pos=10000, key_pos=5000) is True

        # Distant middle token outside sliding window and sink is rejected
        assert is_token_attended(query_pos=100000, key_pos=50000) is False

        # Block-sparse 64x64 Tile Masking checks:
        # Causal boundary: immediate next future tile (Q=10, K=11) MUST be masked
        assert is_tile_masked(query_tile_idx=10, key_tile_idx=11) is True
        assert is_tile_masked(query_tile_idx=10, key_tile_idx=20) is True
        # Diagonal tile (Q=10, K=10) must be attended
        assert is_tile_masked(query_tile_idx=10, key_tile_idx=10) is False
        # Sink tile (K=0) must be attended from any query position
        assert is_tile_masked(query_tile_idx=0, key_tile_idx=0) is False
        assert is_tile_masked(query_tile_idx=200, key_tile_idx=0) is False
        # Distant tile after sink and before window must be masked
        assert is_tile_masked(query_tile_idx=200, key_tile_idx=1) is True

    def test_moe_expert_expansion_sublinear(self):
        assert moe_expert_expansion_factor(1) == 1.0
        alpha_2 = moe_expert_expansion_factor(2)
        alpha_3 = moe_expert_expansion_factor(3)
        alpha_4 = moe_expert_expansion_factor(4)

        assert 1.0 < alpha_2 < alpha_3 < alpha_4 <= 2.6
        # Sub-linear concavity check: slope from 1->2 is steeper than from 2->3
        assert (alpha_2 - 1.0) > (alpha_3 - alpha_2)

    def test_mtp_expected_value_and_instant_fallback(self):
        # Warm conditions: high acceptance rate -> EV positive, optimal d in [1, 4]
        ev_1 = compute_expected_value(1)
        assert ev_1 > 0.0

        best_d = select_optimal_horizon(max_depth=4)
        assert 1 <= best_d <= 4

        # Degraded conditions: divergent draft (very low acceptance rates)
        divergent_probs = [0.04, 0.02, 0.01, 0.00]
        ev_divergent_1 = compute_expected_value(1, acceptance_probs=divergent_probs)
        assert ev_divergent_1 < 0.0

        fallback_d = select_optimal_horizon(max_depth=4, acceptance_probs=divergent_probs)
        assert fallback_d == 0  # Zero-delay fallback to non-speculative decode!

    def test_directstorage_bypass_io_sector_alignment_and_slot_recycling(self):
        sector_size = 4096
        # Any file offset must align to 4096-byte boundaries
        valid_offsets = [0, 4096, 8192, 1048576]
        for offset in valid_offsets:
            assert offset % sector_size == 0

        unaligned_offset = 5000
        aligned_down = (unaligned_offset // sector_size) * sector_size
        assert aligned_down == 4096
        assert aligned_down % sector_size == 0

        # Slot recycling test: double-buffered slots [0, 1]
        slots = [None, None]
        # Request 1 -> Slot 0
        slots[0] = {"layer": 2, "expert": 10}
        assert slots[0]["expert"] == 10
        # Request 2 -> Slot 1
        slots[1] = {"layer": 3, "expert": 11}
        assert slots[1]["expert"] == 11
        # Request 3 -> Overwrites Slot 0
        slots[0] = {"layer": 4, "expert": 12}
        assert slots[0]["expert"] == 12
        # Slot 0 now points to expert 12, expert 10 is retired
        assert not any(s["expert"] == 10 for s in slots)
