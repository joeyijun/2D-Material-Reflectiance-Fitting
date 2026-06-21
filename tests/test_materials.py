import unittest
import io
from pathlib import Path

import numpy as np

from materials import MaterialLoader, read_si_optical_constants


class MaterialLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = MaterialLoader(Path(__file__).resolve().parents[1] / "Si_data.csv")

    def test_pchip_interpolation_remains_passive(self):
        wavelengths = np.linspace(400.0, 1300.0, 1000)
        index = self.loader.get_si_n(wavelengths)
        self.assertTrue(np.all(np.isfinite(index)))
        self.assertTrue(np.all(np.imag(index) >= 0.0))

    def test_endpoint_extrapolation_remains_finite_and_passive(self):
        index = self.loader.get_si_n(np.array([389.0, 1320.0]))
        self.assertTrue(np.all(np.isfinite(index)))
        self.assertTrue(np.all(np.real(index) > 0.0))
        self.assertTrue(np.all(np.imag(index) >= 0.0))

    def test_rejects_temperature_outside_calibration(self):
        with self.assertRaises(ValueError):
            self.loader.get_si_n_with_temp(np.array([600.0]), 5.0)

    def test_reads_segmented_schinke_and_green_tables(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("Schinke.csv", "Green-2008.csv"):
            wavelength, n_values, k_values = read_si_optical_constants(root / filename)
            self.assertEqual(wavelength.shape, n_values.shape)
            self.assertEqual(wavelength.shape, k_values.shape)
            self.assertLessEqual(wavelength.min(), 250.0)
            self.assertGreaterEqual(wavelength.max(), 1450.0)
            self.assertTrue(np.all(n_values > 0.0))
            self.assertTrue(np.all(k_values >= 0.0))
            loader = MaterialLoader(root / filename)
            index = loader.get_si_n(np.array([389.0, 500.0, 650.0]))
            self.assertTrue(np.all(np.isfinite(index)))
            self.assertTrue(np.all(np.imag(index) >= 0.0))

    def test_reads_standard_three_column_header(self):
        source = io.StringIO("wavelength,n,k\n400,5.5,0.3\n500,4.3,0.1\n")
        wavelength, n_values, k_values = read_si_optical_constants(source)
        np.testing.assert_allclose(wavelength, [400.0, 500.0])
        np.testing.assert_allclose(n_values, [5.5, 4.3])
        np.testing.assert_allclose(k_values, [0.3, 0.1])


if __name__ == "__main__":
    unittest.main()
