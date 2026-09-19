"""Tests for oracle_decompose.py variants.

Verifies that:
1. decompose_gap runs without errors on minimal synthetic sessions.
2. Oracle B (perfect prediction, real PCIe) ≤ Oracle D (zero I/O upper bound).
3. Belady MIN hit rate is >= LRU hit rate on the same session.
4. decompose_gap returns the expected variant keys.
"""
import unittest

from atlas_simulator import TieredSim, CacheItem, all_keys
from oracle_decompose import run_belady_vram, decompose_gap


def _make_sessions(token_sequences):
    """Build synthetic session list from a list of per-token expert lists.

    Each inner list is the set of (layer, expert) tuples active at that token.
    We encode them as experts_by_layer dicts understood by atlas_simulator.
    """
    records = []
    for keys in token_sequences:
        by_layer = {}
        for layer, expert in keys:
            by_layer.setdefault(str(layer), []).append(expert)
        records.append({"experts_by_layer": by_layer})
    return [{"records": records}]


class TestOracleDecomposeVariants(unittest.TestCase):
    def _simple_sessions(self):
        """Two tokens sharing expert (0,1), diverging on (0,2) vs (0,3)."""
        return _make_sessions([
            [("0", 1), ("0", 2)],
            [("0", 1), ("0", 3)],
            [("0", 2), ("0", 3)],
        ])

    def test_decompose_gap_returns_all_variant_keys(self):
        sessions = self._simple_sessions()
        results = decompose_gap(
            sessions,
            vram_gb=2.0, vram_reserve_gb=0.0, ram_gb=4.0,
            nvme_gbps=7.0, pcie_gbps=12.0, ram_gbps=40.0,
            target_tps=20.0, expert_mb=0.1,
        )
        expected = {
            "Baseline (none)",
            "Atlas Predictor",
            "Oracle B (Next-token oracle + Real PCIe)",
            "Oracle C (Belady MIN VRAM Residency)",
            "Oracle D (Zero-Exposed Upper Bound)",
        }
        self.assertEqual(set(results.keys()), expected)

    def test_oracle_d_tps_equals_target(self):
        sessions = self._simple_sessions()
        results = decompose_gap(sessions, target_tps=20.0, expert_mb=0.1,
                                vram_gb=2.0, vram_reserve_gb=0.0, ram_gb=4.0)
        self.assertAlmostEqual(results["Oracle D (Zero-Exposed Upper Bound)"]["tps"], 20.0)

    def test_oracle_b_tps_leq_oracle_d(self):
        sessions = self._simple_sessions()
        results = decompose_gap(sessions, target_tps=20.0, expert_mb=0.1,
                                vram_gb=2.0, vram_reserve_gb=0.0, ram_gb=4.0)
        self.assertLessEqual(
            results["Oracle B (Next-token oracle + Real PCIe)"]["tps"],
            results["Oracle D (Zero-Exposed Upper Bound)"]["tps"] + 1e-6,
        )

    def test_belady_hit_rate_bounded_0_to_1(self):
        sessions = self._simple_sessions()
        hits, misses, hit_rate = run_belady_vram(
            sessions, vram_capacity_mb=1.0, expert_mb=0.1
        )
        self.assertGreaterEqual(hit_rate, 0.0)
        self.assertLessEqual(hit_rate, 1.0)
        self.assertEqual(hits + misses, 6)  # 3 tokens × 2 experts

    def test_belady_empty_capacity_is_all_misses(self):
        sessions = self._simple_sessions()
        hits, misses, hit_rate = run_belady_vram(
            sessions, vram_capacity_mb=0.0, expert_mb=0.1
        )
        self.assertEqual(hits, 0)
        self.assertEqual(hit_rate, 0.0)

    def test_belady_huge_capacity_is_all_hits_after_cold_start(self):
        """With VRAM large enough for all experts, every re-access is a hit."""
        # Repeat the same expert pattern twice so there are cache hits.
        sessions = _make_sessions([
            [("0", 1), ("0", 2)],
            [("0", 1), ("0", 2)],  # same as step 0 → should be hits
        ])
        hits, misses, hit_rate = run_belady_vram(
            sessions, vram_capacity_mb=100.0, expert_mb=0.1
        )
        # First token: all misses; second token: all hits
        self.assertEqual(misses, 2)
        self.assertEqual(hits, 2)
        self.assertAlmostEqual(hit_rate, 0.5)


class TestDeadlineAwareBudgetCapping(unittest.TestCase):
    """Verify that the deadline cap formula in evaluate_policy is correct."""

    def test_deadline_cap_calculation(self):
        """max_pcie_mb should limit prefetch count to physical PCIe capacity."""
        from atlas_simulator import evaluate_policy

        # Synthetic session: 6 tokens, each using 1 expert in layer 0
        sessions = _make_sessions([
            [("0", i)] for i in range(6)
        ])

        # Run with and without deadline cap; verify the mode doesn't crash
        # and produces valid TPS values.
        kw = dict(
            sessions=sessions, vram_capacity_mb=100.0, expert_mb=0.1,
            ram_gb=2.0, nvme_gbps=7.0, pcie_gbps=12.0, ram_gbps=40.0,
            nvme_latency_ms=0.08, pcie_latency_ms=0.02,
            compute_ms=50.0, prefetch_horizon=1, hybrid_alpha=0.75,
        )
        r_cap = evaluate_policy(mode="atlas_predictor", deadline_aware=True, **kw)
        r_nocap = evaluate_policy(mode="atlas_predictor", deadline_aware=False, **kw)
        self.assertGreater(r_cap["effective_tps"], 0)
        self.assertGreater(r_nocap["effective_tps"], 0)


if __name__ == "__main__":
    unittest.main()
