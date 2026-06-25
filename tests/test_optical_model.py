import unittest

import numpy as np

from fitting_engine import fit_spectrum

from optical_model import (
    HC_EV_NM,
    calculate_contrast_dynamic,
    config_with_layer_thicknesses,
    dielectric_func_lorentz,
    dielectric_func_voigt,
    layer_fit_parameters,
    normalized_layer_stack,
    spectral_derivative,
)


class ConstantMaterials:
    def get_si_n_with_temp(self, wavelength_nm, temperature_k):
        return np.full_like(wavelength_nm, 3.8 + 0.02j, dtype=complex)

    def get_sio2_n(self, wavelength_nm):
        return np.full_like(wavelength_nm, 1.46, dtype=complex)

    def get_hbn_n(self, wavelength_nm):
        return np.full_like(wavelength_nm, 2.1, dtype=complex)

    def get_graphene_n(self, wavelength_nm):
        return np.full_like(wavelength_nm, 2.5 + 1.3j, dtype=complex)

    def get_quartz_n(self, wavelength_nm):
        return np.full_like(wavelength_nm, 1.46, dtype=complex)

    def get_sapphire_n(self, wavelength_nm):
        return np.full_like(wavelength_nm, 1.77, dtype=complex)

    def get_tio2_n(self, wavelength_nm):
        return np.full_like(wavelength_nm, 2.5, dtype=complex)


class OpticalModelTests(unittest.TestCase):
    def setUp(self):
        self.materials = ConstantMaterials()
        self.wavelengths = np.linspace(450.0, 750.0, 101)
        self.config = {
            "substrate_type": "Si/SiO2",
            "sio2_thick": 285.0,
            "sample_thick": 0.65,
            "has_top_hbn": False,
            "has_bot_hbn": False,
        }

    def test_default_contrast_is_relative(self):
        epsilon = dielectric_func_lorentz(
            HC_EV_NM / self.wavelengths, [12.0, 0.5, 2.0, 0.04]
        )
        relative = calculate_contrast_dynamic(
            self.wavelengths, epsilon, self.materials, self.config
        )
        symmetric = calculate_contrast_dynamic(
            self.wavelengths,
            epsilon,
            self.materials,
            {**self.config, "contrast_definition": "symmetric"},
        )
        np.testing.assert_allclose(symmetric, relative / (relative + 2.0), rtol=1e-12)

    def test_zero_thickness_sample_has_zero_contrast(self):
        contrast = calculate_contrast_dynamic(
            self.wavelengths,
            np.full(self.wavelengths.shape, 10.0 + 1.0j),
            self.materials,
            {**self.config, "sample_thick": 0.0},
        )
        np.testing.assert_allclose(contrast, 0.0, atol=1e-14)

    def test_finite_numerical_aperture_is_angle_averaged(self):
        epsilon = dielectric_func_lorentz(
            HC_EV_NM / self.wavelengths, [12.0, 0.5, 2.0, 0.04]
        )
        normal = calculate_contrast_dynamic(
            self.wavelengths, epsilon, self.materials, self.config
        )
        finite_na = calculate_contrast_dynamic(
            self.wavelengths,
            epsilon,
            self.materials,
            {**self.config, "numerical_aperture": 0.65},
        )
        self.assertTrue(np.all(np.isfinite(finite_na)))
        self.assertGreater(np.max(np.abs(finite_na - normal)), 1e-4)

    def test_hbn_reference_stack_changes_encapsulated_contrast(self):
        epsilon = dielectric_func_lorentz(
            HC_EV_NM / self.wavelengths, [12.0, 0.5, 2.0, 0.04]
        )
        encapsulated = {
            **self.config,
            "has_top_hbn": True,
            "top_hbn_thick": 12.0,
            "has_bot_hbn": True,
            "bot_hbn_thick": 18.0,
        }
        bare_reference = calculate_contrast_dynamic(
            self.wavelengths, epsilon, self.materials, encapsulated
        )
        hbn_reference = calculate_contrast_dynamic(
            self.wavelengths,
            epsilon,
            self.materials,
            {**encapsulated, "reference_includes_hbn": True},
        )
        self.assertGreater(np.max(np.abs(hbn_reference - bare_reference)), 1e-4)

    def test_arbitrary_graphene_encapsulated_layer_table_is_supported(self):
        epsilon = dielectric_func_lorentz(
            HC_EV_NM / self.wavelengths, [12.0, 0.1, 1.9, 0.04]
        )
        config = {
            "substrate_type": "Si",
            "layers": [
                {"material": "hBN", "thickness_nm": 8.0, "in_reference": True,
                 "fit": True, "min_nm": 0.0, "max_nm": 30.0},
                {"material": "Graphene", "thickness_nm": 0.335, "in_reference": False,
                 "fit": False, "min_nm": 0.1, "max_nm": 2.0},
                {"material": "Sample", "thickness_nm": 0.65, "in_reference": False,
                 "fit": False, "min_nm": 0.1, "max_nm": 2.0},
                {"material": "Graphene", "thickness_nm": 0.335, "in_reference": False,
                 "fit": False, "min_nm": 0.1, "max_nm": 2.0},
                {"material": "hBN", "thickness_nm": 15.0, "in_reference": True,
                 "fit": True, "min_nm": 0.0, "max_nm": 30.0},
                {"material": "SiO2", "thickness_nm": 285.0, "in_reference": True,
                 "fit": True, "min_nm": 260.0, "max_nm": 310.0},
            ],
        }
        contrast = calculate_contrast_dynamic(
            self.wavelengths, epsilon, self.materials, config
        )
        self.assertTrue(np.all(np.isfinite(contrast)))
        self.assertEqual(len(layer_fit_parameters(config)), 3)
        updated = config_with_layer_thicknesses(config, [9.0, 16.0, 286.0])
        self.assertEqual(
            [row["thickness_nm"] for row in normalized_layer_stack(updated) if row["fit"]],
            [9.0, 16.0, 286.0],
        )

    def test_layer_table_requires_exactly_one_sample(self):
        with self.assertRaisesRegex(ValueError, "exactly one Sample"):
            normalized_layer_stack({"layers": [
                {"material": "hBN", "thickness_nm": 10.0,
                 "min_nm": 0.0, "max_nm": 20.0}
            ]})

    def test_fitted_layer_thickness_is_recovered_from_synthetic_spectrum(self):
        energy = HC_EV_NM / self.wavelengths
        epsilon = dielectric_func_lorentz(energy, [12.0, 0.25, 1.9, 0.035])
        config = {
            "substrate_type": "Si",
            "layers": [
                {"material": "hBN", "thickness_nm": 12.0, "in_reference": True,
                 "fit": True, "min_nm": 2.0, "max_nm": 25.0},
                {"material": "Graphene", "thickness_nm": 0.335, "in_reference": False,
                 "fit": False, "min_nm": 0.1, "max_nm": 2.0},
                {"material": "Sample", "thickness_nm": 0.65, "in_reference": False,
                 "fit": False, "min_nm": 0.1, "max_nm": 2.0},
                {"material": "SiO2", "thickness_nm": 285.0, "in_reference": True,
                 "fit": False, "min_nm": 260.0, "max_nm": 310.0},
            ],
        }
        measured = calculate_contrast_dynamic(
            self.wavelengths, epsilon, self.materials, config
        )
        initial_config = config_with_layer_thicknesses(config, [7.0])

        def model(params):
            candidate = config_with_layer_thicknesses(initial_config, params)
            return calculate_contrast_dynamic(
                self.wavelengths, epsilon, self.materials, candidate
            )

        result = fit_spectrum(
            energy, measured, model, [7.0], ([2.0], [25.0]),
            baseline_order=-1, robust=False, max_nfev=1000,
        )
        self.assertAlmostEqual(result.params[0], 12.0, delta=0.02)
        self.assertGreater(result.r_squared, 0.999999)

    def test_lorentz_model_is_absorbing_for_positive_linewidth(self):
        epsilon = dielectric_func_lorentz(np.array([1.9, 2.0, 2.1]), [4.0, 1.0, 2.0, 0.05])
        self.assertTrue(np.all(np.imag(epsilon) > 0.0))

    def test_lorentz_rejects_nonphysical_oscillator_parameters(self):
        with self.assertRaises(ValueError):
            dielectric_func_lorentz(np.array([2.0]), [4.0, -1.0, 2.0, 0.05])
        with self.assertRaises(ValueError):
            dielectric_func_lorentz(np.array([2.0]), [4.0, 1.0, 2.0, 0.0])

    def test_voigt_model_is_passive_and_peaks_at_resonance(self):
        energy = np.linspace(1.6, 2.0, 801)
        epsilon = dielectric_func_voigt(energy, [4.0, 0.02, 1.8, 0.015, 0.025])
        self.assertTrue(np.all(np.imag(epsilon) > 0.0))
        peak_energy = energy[np.argmax(np.imag(epsilon))]
        self.assertAlmostEqual(peak_energy, 1.8, delta=0.001)

    def test_voigt_rejects_nonpositive_width(self):
        with self.assertRaises(ValueError):
            dielectric_func_voigt(np.array([1.8]), [4.0, 0.02, 1.8, 0.01, 0.0])

    def test_voigt_rejects_nonphysical_oscillator_parameters(self):
        with self.assertRaises(ValueError):
            dielectric_func_voigt(np.array([1.8]), [4.0, -0.02, 1.8, 0.01, 0.02])
        with self.assertRaises(ValueError):
            dielectric_func_voigt(np.array([1.8]), [4.0, 0.02, 0.0, 0.01, 0.02])

    def test_derivative_is_with_respect_to_energy(self):
        wavelengths = np.linspace(800.0, 400.0, 201)
        energy = HC_EV_NM / wavelengths
        values = energy**2
        derivative = spectral_derivative(values, wavelengths)
        np.testing.assert_allclose(derivative, 2.0 * energy, rtol=2e-4)

    def test_invalid_parameter_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            dielectric_func_lorentz(np.array([2.0]), [4.0, 1.0])


if __name__ == "__main__":
    unittest.main()
