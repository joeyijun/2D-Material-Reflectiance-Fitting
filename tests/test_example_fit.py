import unittest
from pathlib import Path

import numpy as np

from fitting_engine import fit_spectrum, guess_resonances
from materials import MaterialLoader
from optical_model import HC_EV_NM, calculate_contrast_dynamic, dielectric_func_lorentz


ROOT = Path(__file__).resolve().parents[1]


class ExampleSpectrumFitTests(unittest.TestCase):
    def test_full_spectrum_auto_model_exceeds_point_99_gof(self):
        substrate = np.loadtxt(
            ROOT / "example" / "1L-WS2-300 nm SiO2_Si-substrate.csv", delimiter=","
        )
        sample = np.loadtxt(
            ROOT / "example" / "1L-WS2-300 nm SiO2_Si-sample.csv", delimiter=","
        )
        order = np.argsort(substrate[:, 0])
        energy = substrate[order, 0]
        measured = sample[order, 1] / substrate[order, 1] - 1.0
        wavelength = HC_EV_NM / energy
        guesses = guess_resonances(energy, measured)
        self.assertEqual(len(guesses), 3, msg=str(guesses))

        initial = [12.0]
        lower = [1.0]
        upper = [50.0]
        for center, linewidth in guesses:
            initial.extend([0.1, center, linewidth])
            lower.extend([0.0, center - 0.15, 0.0001])
            upper.extend([20.0, center + 0.15, 0.5])
        physical_count = len(initial)
        initial.append(300.0)
        lower.append(280.0)
        upper.append(320.0)
        config = {
            "substrate_type": "Si/SiO2",
            "temp": 298.0,
            "sio2_thick": 300.0,
            "sample_thick": 0.65,
            "has_top_hbn": False,
            "has_bot_hbn": False,
            "numerical_aperture": 0.0,
            "contrast_definition": "relative",
        }

        for si_filename in ("Si_data.csv", "Schinke.csv", "Green-2008.csv"):
            materials = MaterialLoader(ROOT / si_filename)

            def model(params):
                fit_config = {**config, "sio2_thick": params[-1]}
                epsilon = dielectric_func_lorentz(energy, params[:physical_count])
                return calculate_contrast_dynamic(wavelength, epsilon, materials, fit_config)

            result = fit_spectrum(
                energy,
                measured,
                model,
                initial,
                (lower, upper),
                baseline_order=3,
                robust=True,
                max_nfev=10000,
            )
            fitted_centers = result.params[2:physical_count:3]
            self.assertGreaterEqual(result.r_squared, 0.99, msg=si_filename)
            self.assertLess(np.min(np.abs(fitted_centers - 2.1)), 0.02)
            self.assertLess(np.min(np.abs(fitted_centers - 2.5)), 0.03)

    def test_ws2_a_exciton_fit_tracks_measured_resonance(self):
        substrate = np.loadtxt(
            ROOT / "example" / "1L-WS2-300 nm SiO2_Si-substrate.csv", delimiter=","
        )
        sample = np.loadtxt(
            ROOT / "example" / "1L-WS2-300 nm SiO2_Si-sample.csv", delimiter=","
        )
        energy = substrate[:, 0]
        measured = (sample[:, 1] - substrate[:, 1]) / substrate[:, 1]
        mask = (energy >= 1.91) & (energy <= 2.25)
        energy = energy[mask]
        measured = measured[mask]
        wavelengths = HC_EV_NM / energy
        materials = MaterialLoader(ROOT / "Si_data.csv")
        config = {
            "substrate_type": "Si/SiO2",
            "temp": 298.0,
            "sio2_thick": 300.0,
            "sample_thick": 0.65,
            "has_top_hbn": False,
            "has_bot_hbn": False,
            "contrast_definition": "relative",
        }

        def model(params):
            epsilon = dielectric_func_lorentz(energy, params)
            return calculate_contrast_dynamic(wavelengths, epsilon, materials, config)

        initial = np.array([12.0, 1.0, 2.08, 0.04])
        initial_rmse = np.sqrt(np.mean((model(initial) - measured) ** 2))
        result = fit_spectrum(
            energy,
            measured,
            model,
            initial,
            ([1.0, 0.0, 1.95, 0.001], [50.0, 50.0, 2.20, 0.30]),
            baseline_order=1,
            robust=True,
        )

        measured_minimum = energy[np.argmin(measured)]
        self.assertTrue(result.success)
        self.assertLess(abs(result.params[2] - measured_minimum), 0.01)
        self.assertLess(result.rmse, initial_rmse * 0.5)


if __name__ == "__main__":
    unittest.main()
