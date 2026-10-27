import unittest
from atlas_simulator import TieredSim, evaluate_policy, predict_from_transition
from collections import defaultdict, Counter


class SimulatorUnitTests(unittest.TestCase):
    def test_pcie_bandwidth_controls_ram_to_vram(self):
        # A 1024 MB transfer should take ~1000 ms at 1 GB/s before latency.
        fast = TieredSim(1024, 2048, 2, 10, 10, 100, 0, 0, 1)
        slow = TieredSim(1024, 2048, 2, 10, 1, 100, 0, 0, 1)
        fast_done = fast._transfer_ram_to_vram(('0', 0), 0)
        slow_done = slow._transfer_ram_to_vram(('0', 0), 0)
        self.assertLess(fast_done, slow_done)
        self.assertAlmostEqual(slow_done, 1000.0, places=5)

    def test_vram_is_global_byte_budget(self):
        sim = TieredSim(600, 1000, 2, 10, 10, 100, 0, 0, 1)
        sim._vram_put(('0', 0), 0)
        sim._vram_put(('1', 0), 0)
        self.assertLessEqual(sim.vram_used_mb, 1000)
        self.assertEqual(len(sim.vram), 1)

    def test_causal_predictor_does_not_need_future_expert_ids(self):
        transitions = defaultdict(lambda: defaultdict(Counter))
        transitions['0'][1].update({2: 9, 3: 1})
        pred = predict_from_transition(transitions, {'0': {1}}, {'0': 1})
        self.assertEqual(pred, {('0', 2)})

    def test_prefetch_can_be_hidden_by_compute(self):
        sim = TieredSim(1, 100, 1, 100, 100, 100, 0, 0, 100)
        exposed, _ = sim.run_token({('0', 0)}, {('0', 1)})
        # First token pays for its actual expert; the prefetched next expert is
        # scheduled during compute and should already be ready by token 2.
        self.assertGreater(exposed, 0)
        self.assertTrue(sim._vram_has(('0', 1), sim.now_ms))


if __name__ == '__main__':
    unittest.main()
