import unittest

import numpy as np
import warp as wp

from warp_trap.simulation import METHODS, Particles, initial_state
from warp_trap.validation import validate


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_one_step_matches_batched_update_equations(self):
        x, v, omega = initial_state(257)
        h = 0.125
        expected_v = v - h * omega[:, None]**2 * x
        for method in METHODS:
            with self.subTest(method=method):
                particles = Particles(x, v, omega, "cpu")
                particles.step(method, h)
                actual_x, actual_v = particles.snapshot()
                expected_x = x + h * (v if method == "explicit" else expected_v)
                np.testing.assert_allclose(actual_x, expected_x, atol=2e-7, rtol=2e-7)
                np.testing.assert_allclose(actual_v, expected_v, atol=2e-7, rtol=2e-7)

    def test_cpu_validation(self):
        summary, _, _ = validate("cpu")
        self.assertTrue(all(summary["checks"].values()), summary["checks"])

    def test_cuda_trajectory_matches_cpu(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Warp detected no CUDA device")
        x, v, omega = initial_state(1025)
        for method in METHODS:
            with self.subTest(method=method):
                cpu = Particles(x, v, omega, "cpu")
                gpu = Particles(x, v, omega, devices[0])
                for _ in range(256):
                    cpu.step(method, 0.01)
                    gpu.step(method, 0.01)
                for actual, expected in zip(gpu.snapshot(), cpu.snapshot(), strict=True):
                    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-6)

    def test_autodiff_gradients_match_finite_differences(self):
        from warp_trap.sysid import gradient_check

        self.assertLess(gradient_check("cpu"), 2e-2)

    def test_system_identification_recovers_parameters(self):
        from warp_trap.sysid import run_sysid

        report, *_ = run_sysid("cpu", count=128, steps=150, iterations=150)
        self.assertLess(report["omega_median_rel_error"], 1e-2)
        self.assertLess(report["zeta_median_rel_error"], 5e-2)


if __name__ == "__main__":
    unittest.main()
