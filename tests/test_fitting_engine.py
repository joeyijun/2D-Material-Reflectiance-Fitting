import unittest

import numpy as np

from fitting_engine import (
    composite_derivative_residual,
    fit_spectrum,
    guess_resonances,
    resonance_balanced_sigma,
    smoothed_spectral_derivative,
)


class FittingEngineTests(unittest.TestCase):
    def test_smoothed_derivatives_suppress_noise(self):
        rng = np.random.default_rng(3)
        energy = np.linspace(1.6, 2.0, 1001)
        clean = np.exp(-0.5 * ((energy - 1.8) / 0.02) ** 2)
        noisy = clean + rng.normal(0.0, 0.002, energy.size)
        truth = -(energy - 1.8) / 0.02**2 * clean
        raw = np.gradient(noisy, energy)
        smooth = smoothed_spectral_derivative(noisy, energy, order=1)
        self.assertLess(np.sqrt(np.mean((smooth - truth) ** 2)), np.sqrt(np.mean((raw - truth) ** 2)))

    def test_composite_derivative_residual_keeps_original_spectrum(self):
        energy = np.linspace(1.6, 2.0, 501)
        measured = np.exp(-0.5 * ((energy - 1.8) / 0.02) ** 2)
        shifted = np.exp(-0.5 * ((energy - 1.805) / 0.02) ** 2)
        residual = composite_derivative_residual(measured, shifted, energy, maximum_order=2)
        self.assertEqual(residual.size, measured.size * 3)
        self.assertGreater(np.linalg.norm(residual), 0.0)
        exact = composite_derivative_residual(measured, measured, energy, maximum_order=2)
        np.testing.assert_allclose(exact, 0.0, atol=1e-12)

    def test_auto_guess_ignores_drift_and_finds_peak_and_dip(self):
        energy = np.linspace(1.5, 2.7, 1201)
        drift = 0.1 + 0.08 * (energy - 2.1)
        peak = 0.25 * np.exp(-0.5 * ((energy - 1.85) / 0.018) ** 2)
        dip = -0.4 * np.exp(-0.5 * ((energy - 2.35) / 0.025) ** 2)
        guesses = guess_resonances(energy, drift + peak + dip)
        self.assertEqual(len(guesses), 2, msg=str(guesses))
        np.testing.assert_allclose(guesses[:, 0], [1.85, 2.35], atol=0.005)

    def test_auto_guess_merges_dispersive_peak_dip_pair(self):
        energy = np.linspace(1.9, 2.3, 801)
        peak = 0.25 * np.exp(-0.5 * ((energy - 2.085) / 0.009) ** 2)
        dip = -0.8 * np.exp(-0.5 * ((energy - 2.106) / 0.010) ** 2)
        guesses = guess_resonances(energy, peak + dip)
        self.assertEqual(len(guesses), 1, msg=str(guesses))
        self.assertAlmostEqual(guesses[0, 0], 2.106, delta=0.01)

    def test_auto_guess_finds_weak_peak_on_curved_background(self):
        energy = np.linspace(1.55, 2.15, 1201)
        interference = 0.08 * np.sin(2 * np.pi * (energy - energy[0]) / 0.55)
        drift = 0.03 * (energy - np.mean(energy))
        strong = 0.42 * np.exp(-0.5 * ((energy - 1.72) / 0.014) ** 2)
        weak = 0.045 * np.exp(-0.5 * ((energy - 1.82) / 0.012) ** 2)
        guesses = guess_resonances(
            energy, interference + drift + strong + weak, max_peaks=4
        )
        self.assertGreaterEqual(len(guesses), 2, msg=str(guesses))
        self.assertTrue(np.any(np.abs(guesses[:, 0] - 1.72) < 0.008), msg=str(guesses))
        self.assertTrue(np.any(np.abs(guesses[:, 0] - 1.82) < 0.010), msg=str(guesses))

    def test_resonance_balancing_upweights_a_weak_local_feature(self):
        energy = np.linspace(1.6, 2.0, 801)
        strong = np.exp(-0.5 * ((energy - 1.72) / 0.008) ** 2)
        weak = 0.04 * np.exp(-0.5 * ((energy - 1.82) / 0.012) ** 2)
        measured = strong + weak
        sigma = resonance_balanced_sigma(
            energy, measured, [(1.72, 0.016), (1.82, 0.024)]
        )
        weak_window = np.abs(energy - 1.82) < 0.02
        outside = (energy > 1.90) & (energy < 1.98)
        self.assertLess(np.median(sigma[weak_window]), np.median(sigma[outside]) / 5)

    def test_recovers_parameters_with_drift_and_outliers(self):
        rng = np.random.default_rng(7)
        energy = np.linspace(1.5, 2.5, 301)

        def model(params):
            amplitude, center, width = params
            return amplitude / (1.0 + ((energy - center) / width) ** 2)

        truth = np.array([0.35, 2.03, 0.045])
        drift = 0.025 - 0.018 * (energy - 2.0)
        measured = model(truth) + drift + rng.normal(0.0, 0.002, energy.size)
        measured[[40, 172, 250]] += [0.12, -0.15, 0.10]
        result = fit_spectrum(
            energy,
            measured,
            model,
            [0.2, 1.98, 0.08],
            ([0.01, 1.8, 0.005], [1.0, 2.2, 0.2]),
            baseline_order=1,
            robust=True,
        )

        np.testing.assert_allclose(result.params, truth, rtol=0.025, atol=0.002)
        self.assertIn("LM", result.method)
        self.assertTrue(np.all(np.isfinite(result.standard_errors)))
        self.assertLess(result.rmse, 0.02)

    def test_locked_parameter_is_preserved(self):
        energy = np.linspace(1.0, 2.0, 100)
        model = lambda params: params[0] + params[1] * energy
        result = fit_spectrum(
            energy,
            model([0.4, 1.7]),
            model,
            [0.4, 1.0],
            ([-1.0, 0.0], [1.0, 3.0]),
            locked_mask=[True, False],
            baseline_order=-1,
            robust=False,
        )
        self.assertEqual(result.params[0], 0.4)
        self.assertAlmostEqual(result.params[1], 1.7, places=8)
        self.assertEqual(result.standard_errors[0], 0.0)

    def test_rejects_initial_value_outside_bounds(self):
        with self.assertRaises(ValueError):
            fit_spectrum(
                np.arange(10.0),
                np.arange(10.0),
                lambda params: np.zeros(10),
                [2.0],
                ([0.0], [1.0]),
            )


if __name__ == "__main__":
    unittest.main()
