import unittest

from atlas_physical_simulator import PhysicalSim


class PhysicalSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.pm = {
            "source": "tiny",
            "experts": {
                "0:0": {"exact": True, "chunks": [{"offset": 0, "size_bytes": 1024 * 1024}]},
                "0:1": {"exact": True, "chunks": [{"offset": 1024 * 1024, "size_bytes": 1024 * 1024}]},
            },
        }

    def test_cold_vs_eviction_classification(self):
        sim = PhysicalSim(self.pm, 2, 4, 7, 12, 40, 0, 0, 50)
        sim.ensure_batch({("0", 0)}, 0)
        self.assertEqual(sim.stats['nvme_miss'], 1)
        # With enough RAM, the second access is not an NVMe miss.
        sim.ensure_batch({("0", 0)}, 50)
        self.assertEqual(sim.stats['nvme_miss'], 1)

    def test_preload_fits_and_avoids_decode_nvme_misses(self):
        sim = PhysicalSim(self.pm, 2, 4, 7, 12, 40, 0, 0, 50)
        ok, done, mb = sim._preload_experts_to_ram(0)
        self.assertTrue(ok)
        self.assertAlmostEqual(mb, 2.0, places=6)
        self.assertGreater(done, 0)
        sim.ensure_batch({("0", 0), ("0", 1)}, done)
        self.assertEqual(sim.stats['nvme_miss'], 0)


if __name__ == '__main__':
    unittest.main()
