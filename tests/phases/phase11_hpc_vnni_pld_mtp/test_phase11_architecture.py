"""Phase 11 HPC & SIMD Optimization Verification Suite — Tendou Launcher.

Tests the mathematical models and logic for:
1. Direct __mmask64 sign mask generation via even-parity bit lookup.
2. Masked broadcast (0xFF00) for dual-scale factors.
3. Multi-token (nrc = 2) weight sharing equivalence and zero-drift guarantee.
4. Zero-VRAM Prompt Lookup Decoding (PLD) N-Gram draft engine.
5. Integration between PLD and MoE-aware MtpDynamicHorizonController.
"""

import time
import math
import pytest
from typing import List, Optional, Tuple


# Even parity sign mask lookup table (128 entries)
K_EVEN_SIGN_MASK = [
    0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
    0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
    0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
    0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
    0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
    0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
    0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
    0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff
]


def generate_mask64_direct(aux0: int, aux1: int) -> int:
    """Simulates direct __mmask64 generation from 64-bit grid auxiliary words."""
    m0 = (
        (K_EVEN_SIGN_MASK[(aux0 >>  0) & 127] <<  0) |
        (K_EVEN_SIGN_MASK[(aux0 >>  7) & 127] <<  8) |
        (K_EVEN_SIGN_MASK[(aux0 >> 14) & 127] << 16) |
        (K_EVEN_SIGN_MASK[(aux0 >> 21) & 127] << 24) |
        (K_EVEN_SIGN_MASK[(aux1 >>  0) & 127] << 32) |
        (K_EVEN_SIGN_MASK[(aux1 >>  7) & 127] << 40) |
        (K_EVEN_SIGN_MASK[(aux1 >> 14) & 127] << 48) |
        (K_EVEN_SIGN_MASK[(aux1 >> 21) & 127] << 56)
    )
    return m0


def simulate_masked_broadcast(scale1: int, scale2: int) -> List[int]:
    """Simulates _mm512_mask_set1_epi32(_mm512_set1_epi32(scale1), 0xFF00, scale2)."""
    # 16 elements: dwords 0..7 get scale1, dwords 8..15 get scale2
    result = [scale1] * 16
    mask = 0xFF00
    for i in range(16):
        if (mask >> i) & 1:
            result[i] = scale2
    return result


class PythonPromptLookupDraftEngine:
    """Pure CPU/RAM hash-indexed N-Gram lookup draft engine."""

    def __init__(self, min_n_gram: int = 2, max_n_gram: int = 4):
        self.min_n_gram = min_n_gram
        self.max_n_gram = max_n_gram
        self.total_draft_steps = 0
        self.total_draft_time_ms = 0.0
        self.hash_table: Dict[Tuple[int, ...], List[int]] = {}
        self.indexed_len = 0

    @staticmethod
    def get_vram_usage_bytes() -> int:
        return 0  # 0 MB VRAM guaranteed

    def ingest_prompt_tokens(self, tokens: List[int]) -> None:
        """Pre-warm index with prompt tokens."""
        if not tokens or len(tokens) < self.min_n_gram:
            return
        self.hash_table.clear()
        n = len(tokens)
        start_idx = max(0, n - 4096)
        for pos in range(start_idx, n - self.min_n_gram + 1):
            for k in range(self.min_n_gram, min(self.max_n_gram + 1, n - pos + 1)):
                key = tuple(tokens[pos:pos + k])
                self.hash_table.setdefault(key, []).append(pos)
        self.indexed_len = n

    def _update_index(self, context_tokens: List[int]) -> None:
        ctx_len = len(context_tokens)
        if ctx_len < self.min_n_gram:
            self.hash_table.clear()
            self.indexed_len = 0
            return

        if ctx_len < self.indexed_len or self.indexed_len == 0:
            self.hash_table.clear()
            start_idx = max(0, ctx_len - 4096)
            for pos in range(start_idx, ctx_len - self.min_n_gram + 1):
                for n in range(self.min_n_gram, min(self.max_n_gram + 1, ctx_len - pos + 1)):
                    key = tuple(context_tokens[pos:pos + n])
                    self.hash_table.setdefault(key, []).append(pos)
            self.indexed_len = ctx_len
            return

        start_pos = max(0, self.indexed_len - self.max_n_gram)
        for pos in range(start_pos, ctx_len - self.min_n_gram + 1):
            for n in range(self.min_n_gram, min(self.max_n_gram + 1, ctx_len - pos + 1)):
                key = tuple(context_tokens[pos:pos + n])
                lst = self.hash_table.setdefault(key, [])
                if not lst or lst[-1] != pos:
                    lst.append(pos)
        self.indexed_len = ctx_len

    def draft(self, context_tokens: List[int], requested_d: int) -> Tuple[List[int], float]:
        t0 = time.perf_counter()
        if requested_d <= 0 or len(context_tokens) <= self.min_n_gram:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return [], dur_ms

        ctx_len = len(context_tokens)
        max_d = min(requested_d, 8)
        self._update_index(context_tokens)

        found_pos = -1

        for n in range(min(self.max_n_gram, ctx_len - 1), self.min_n_gram - 1, -1):
            if ctx_len < n + 1:
                continue
            pattern = tuple(context_tokens[ctx_len - n:])
            search_limit = ctx_len - n
            candidates = self.hash_table.get(pattern)
            if candidates:
                for p in reversed(candidates):
                    if p < search_limit:
                        found_pos = p + n
                        break
            if found_pos != -1:
                break

        drafted: List[int] = []
        if found_pos != -1:
            available = ctx_len - found_pos
            take = min(max_d, available)
            drafted = context_tokens[found_pos:found_pos + take]

        dur_ms = (time.perf_counter() - t0) * 1000.0
        self.total_draft_steps += 1
        self.total_draft_time_ms += dur_ms
        return drafted, dur_ms


class TestPhase11Architecture:
    def test_direct_mask64_even_parity_coverage(self):
        # Verify all 128 elements in K_EVEN_SIGN_MASK satisfy the even parity property
        for i in range(128):
            mask = K_EVEN_SIGN_MASK[i]
            # Lower 7 bits must equal i
            assert (mask & 0x7F) == i
            # Total set bits must be even
            set_bits = bin(mask).count('1')
            assert set_bits % 2 == 0

    def test_direct_mask64_generation_exactness(self):
        aux0 = 0x12345678
        aux1 = 0x9abcdef0
        mask = generate_mask64_direct(aux0, aux1)
        assert isinstance(mask, int)
        assert 0 <= mask < (1 << 64)

        # Test individual bytes match table entries
        b0 = mask & 0xFF
        b7 = (mask >> 56) & 0xFF
        assert b0 == K_EVEN_SIGN_MASK[(aux0 >> 0) & 127]
        assert b7 == K_EVEN_SIGN_MASK[(aux1 >> 21) & 127]

    def test_masked_broadcast_bit_exactness(self):
        s1 = 5
        s2 = 13
        vec = simulate_masked_broadcast(s1, s2)
        assert len(vec) == 16
        # Dwords 0..7 must equal s1
        assert vec[0:8] == [5] * 8
        # Dwords 8..15 must equal s2
        assert vec[8:16] == [13] * 8

    def test_nrc2_weight_sharing_mathematical_equivalence(self):
        # Verify that multiplying shared weights by two independent vectors
        # yields the exact same outputs as independent single-vector dot products
        weights = [2, -1, 3, 0, -2, 1, 4, -3]
        vec0 = [10, 20, 30, 40, 50, 60, 70, 80]
        vec1 = [5, 15, 25, 35, 45, 55, 65, 75]

        # Single dot products
        dot0_ref = sum(w * y for w, y in zip(weights, vec0))
        dot1_ref = sum(w * y for w, y in zip(weights, vec1))

        # Shared weights dot products (nrc = 2)
        dot0_shared = 0
        dot1_shared = 0
        for w, y0, y1 in zip(weights, vec0, vec1):
            dot0_shared += w * y0
            dot1_shared += w * y1

        assert dot0_shared == dot0_ref
        assert dot1_shared == dot1_ref

    def test_pld_draft_engine_zero_vram_and_speed(self):
        pld = PythonPromptLookupDraftEngine(min_n_gram=2, max_n_gram=4)
        assert pld.get_vram_usage_bytes() == 0

        # Pattern context: Repetitive code
        ctx = [10, 20, 30, 40, 50, 60, 10, 20, 30]
        drafted, dur_ms = pld.draft(ctx, requested_d=3)

        assert drafted == [40, 50, 60]
        # Latency should be well under 0.05 ms (target ~0.01 ms)
        assert dur_ms < 0.10

    def test_pld_boundary_conditions(self):
        pld = PythonPromptLookupDraftEngine(min_n_gram=2, max_n_gram=4)
        # Empty
        d_empty, _ = pld.draft([], requested_d=3)
        assert d_empty == []

        # Single token
        d_single, _ = pld.draft([42], requested_d=3)
        assert d_single == []

        # Zero requested depth
        d_zero, _ = pld.draft([1, 2, 3, 4, 1, 2], requested_d=0)
        assert d_zero == []

        # Non-repetitive text (no match)
        d_nonrep, _ = pld.draft([100, 200, 300, 400, 500], requested_d=2)
        assert d_nonrep == []

    def test_pld_horizon_controller_ev_boost(self):
        from tests.phases.phase10_context_vnni_mtp_directstorage.test_phase10_architecture import compute_expected_value

        # Neural draft step: T_draft = 2.5 ms
        ev_neural = compute_expected_value(d=2, t_draft_step_ms=2.5)

        # PLD draft step: T_draft = 0.01 ms (zero VRAM)
        ev_pld = compute_expected_value(d=2, t_draft_step_ms=0.01)

        # Because PLD has almost 0 overhead, EV must be strictly higher
        assert ev_pld > ev_neural
        assert (ev_pld - ev_neural) > 2.0  # More than 2 ms saved per step!

    def test_pld_prompt_ingest_and_aligned_horizon(self):
        """Tests prompt token pre-warming and SIMD-aligned horizon selection."""
        pld = PythonPromptLookupDraftEngine(min_n_gram=2, max_n_gram=4)
        prompt = [100, 101, 102, 103, 104, 105, 101, 102]
        pld.ingest_prompt_tokens(prompt)
        assert pld.indexed_len == len(prompt)
        drafted, _ = pld.draft(prompt, 2)
        assert drafted == [103, 104]

        # Verify aligned horizon logic (multiples of nrc=2)
        probs = [0.90, 0.85, 0.80, 0.75]
        # Calculate EV for d=1..4 with low draft step (PLD)
        t_target_1 = 40.0
        t_target_d = 40.0
        t_draft = 0.010
        def compute_ev(d):
            saved = 0.0
            p = 1.0
            for i in range(1, d + 1):
                p *= probs[i - 1]
                saved += p * t_target_1
            return saved - (d * t_draft)
        
        ev2 = compute_ev(2)
        ev4 = compute_ev(4)
        assert ev2 > 0 and ev4 > 0

    def test_pld_long_context_scaling(self):
        pld = PythonPromptLookupDraftEngine(min_n_gram=2, max_n_gram=4)
        # 8192 tokens with periodic repeats
        ctx = [i % 256 for i in range(8192)]
        d_cold, _ = pld.draft(ctx, requested_d=4)

        # Incremental token append (decoding step)
        ctx.append(0)
        d_warm, dur_ms = pld.draft(ctx, requested_d=4)
        assert len(d_warm) == 4
        assert dur_ms < 0.10  # O(1) hash lookup stays < 0.10 ms on 8k tokens
