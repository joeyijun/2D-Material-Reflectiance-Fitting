import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fitting_engine import guess_resonances
from wse2_benchmark import run_case


ROOT = Path(__file__).resolve().parents[1]


def load_data(filename):
    data = pd.read_csv(ROOT / "example" / filename, header=None).apply(
        pd.to_numeric, errors="coerce"
    ).dropna().to_numpy()
    return data[:, 0], data[:, 1]


def load_contrast(sample_filename, substrate_filename=None):
    axis, sample = load_data(sample_filename)
    if substrate_filename is None:
        energy, contrast = axis, sample
    else:
        substrate_axis, substrate = load_data(substrate_filename)
        sample_on_substrate = np.interp(substrate_axis, axis, sample)
        energy = 1239.8419843320026 / substrate_axis
        contrast = (sample_on_substrate - substrate) / substrate
    order = np.argsort(energy)
    return energy[order], contrast[order]


class WSe2ExampleTests(unittest.TestCase):
    def assert_guesses_near(self, filename, expected, substrate_filename=None):
        energy, contrast = load_contrast(filename, substrate_filename)
        guesses = guess_resonances(energy, contrast)
        self.assertEqual(len(guesses), len(expected), guesses)
        np.testing.assert_allclose(guesses[:, 0], expected, atol=0.012)

    def test_hbn_auto_guess_finds_two_main_resonances(self):
        self.assert_guesses_near(
            "hBN-1L WSe2-hBN-285nm SiO2 Si-sample.csv", [1.713, 1.814]
        )

    def test_quartz_auto_guess_finds_two_main_resonances(self):
        self.assert_guesses_near(
            "1L-WSe2-quartz-sample.csv",
            [1.737, 1.900],
            "1L-WSe2-quartz-substrate.csv",
        )

    def test_hbn_voigt_fit_keeps_the_181_ev_resonance(self):
        result = run_case("hbn", "voigt", derivative_order=0)
        second = result.params[5:9]
        self.assertGreater(second[0], 0.005)
        self.assertAlmostEqual(second[1], 1.814, delta=0.008)
        self.assertLess(second[2], 0.03)
        self.assertLess(second[3], 0.03)
        self.assertGreater(result.r_squared, 0.98)
        second_local = result.resonance_diagnostics[1]
        self.assertGreater(second_local["local_r_squared"], 0.9)
        self.assertGreater(second_local["amplitude_ratio"], 0.7)
        self.assertLess(second_local["amplitude_ratio"], 1.3)


if __name__ == "__main__":
    unittest.main()
