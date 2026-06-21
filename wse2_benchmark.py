"""Benchmark Lorentz and Faddeeva-Voigt models on the WSe2 examples."""

from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import interp1d

from fitting_engine import (
    composite_derivative_residual,
    finalize_physical_fit,
    fit_spectrum,
    resonance_balanced_sigma,
    resonance_diagnostics,
    resonance_windows_from_parameters,
)
from materials import MaterialLoader
from optical_model import (
    HC_EV_NM,
    calculate_contrast_dynamic,
    dielectric_func_lorentz,
    dielectric_func_voigt,
)


ROOT = Path(__file__).resolve().parent
CASES = {
    "hbn": {
        "substrate": "hBN-1L WSe2-hBN-285nm SiO2 Si-substrate.csv",
        "sample": "hBN-1L WSe2-hBN-285nm SiO2 Si-sample.csv",
        "centers": [1.713, 1.814],
        "direct_contrast": True,
        "fit_hbn_thickness": True,
        "config": {
            "substrate_type": "Si/SiO2",
            "temp": 298.0,
            "sio2_thick": 285.0,
            "sample_thick": 0.65,
            "has_top_hbn": True,
            "top_hbn_thick": 10.0,
            "has_bot_hbn": True,
            "bot_hbn_thick": 10.0,
            "reference_includes_hbn": False,
            "numerical_aperture": 0.0,
            "contrast_definition": "relative",
        },
    },
    "quartz": {
        "substrate": "1L-WSe2-quartz-substrate.csv",
        "sample": "1L-WSe2-quartz-sample.csv",
        "centers": [1.737, 1.900],
        "config": {
            "substrate_type": "Quartz",
            "sample_thick": 0.65,
            "has_top_hbn": False,
            "has_bot_hbn": False,
            "numerical_aperture": 0.0,
            "contrast_definition": "relative",
        },
    },
}


def load_pair(case):
    substrate = np.loadtxt(ROOT / "example" / case["substrate"], delimiter=",")
    sample = np.loadtxt(ROOT / "example" / case["sample"], delimiter=",")

    def wavelength_data(data):
        x, values = data[:, 0], data[:, 1]
        wavelength = HC_EV_NM / x if np.mean(x) < 100.0 else x
        order = np.argsort(wavelength)
        return wavelength[order], values[order]

    substrate_wavelength, substrate_values = wavelength_data(substrate)
    sample_wavelength, sample_values = wavelength_data(sample)
    interpolated = interp1d(
        sample_wavelength, sample_values, kind="cubic", bounds_error=False, fill_value=np.nan
    )(substrate_wavelength)
    valid = np.isfinite(interpolated) & (substrate_values != 0)
    wavelength = substrate_wavelength[valid]
    if case.get("direct_contrast", False):
        measured = interpolated[valid]
    else:
        measured = interpolated[valid] / substrate_values[valid] - 1.0
    energy = HC_EV_NM / wavelength
    order = np.argsort(energy)
    return energy[order], wavelength[order], measured[order]


def run_case(case_name, line_shape, derivative_order=0):
    case = CASES[case_name]
    energy, wavelength, measured = load_pair(case)
    materials = MaterialLoader(ROOT / "Si_data.csv")
    config = case["config"].copy()
    initial = [12.0]
    lower = [1.0]
    upper = [50.0]
    if line_shape == "lorentz":
        strength_initial = 0.1
        strength_upper = 50.0
        for center in case["centers"]:
            center_margin = 0.00875
            initial.extend([strength_initial, center, 0.025])
            lower.extend([0.0, center - center_margin, 0.001])
            upper.extend([strength_upper, center + center_margin, 0.10])
        dielectric = dielectric_func_lorentz
    else:
        strength_initial = 0.01
        strength_upper = 10.0
        for center in case["centers"]:
            approximate_fwhm = 0.5346 * 0.015 + np.sqrt(0.2166 * 0.015**2 + 0.020**2)
            center_margin = max(0.004, 0.35 * approximate_fwhm)
            initial.extend([strength_initial, center, 0.015, 0.020])
            lower.extend([0.0, center - center_margin, 0.001, 0.001])
            upper.extend([
                strength_upper, center + center_margin,
                4.0 * approximate_fwhm, 4.0 * approximate_fwhm,
            ])
        dielectric = dielectric_func_voigt
    physical_count = len(initial)
    resonances = resonance_windows_from_parameters(initial, line_shape)
    balanced_sigma = resonance_balanced_sigma(energy, measured, resonances)
    fit_oxide = config["substrate_type"] == "Si/SiO2"
    if fit_oxide:
        initial.append(config["sio2_thick"])
        lower.append(config["sio2_thick"] - 20.0)
        upper.append(config["sio2_thick"] + 20.0)
    fit_hbn = case.get("fit_hbn_thickness", False)
    if fit_hbn:
        initial.extend([config["top_hbn_thick"], config["bot_hbn_thick"]])
        lower.extend([0.0, 0.0])
        upper.extend([100.0, 100.0])

    def physical_model(params):
        model_config = config.copy()
        if fit_oxide:
            model_config["sio2_thick"] = params[physical_count]
        if fit_hbn:
            model_config["top_hbn_thick"] = params[-2]
            model_config["bot_hbn_thick"] = params[-1]
        epsilon = dielectric(energy, params[:physical_count])
        return calculate_contrast_dynamic(wavelength, epsilon, materials, model_config)

    def model(params):
        physical = physical_model(params)
        if derivative_order:
            return composite_derivative_residual(
                measured, physical, energy,
                maximum_order=derivative_order,
                resonances=resonances,
            )
        return physical

    if derivative_order:
        objective_energy = np.tile(energy, derivative_order + 1)
        target = np.zeros(objective_energy.size)
        warm_start = fit_spectrum(
            energy,
            measured,
            physical_model,
            initial,
            (lower, upper),
            baseline_order=3,
            robust=True,
            sigma=balanced_sigma,
            max_nfev=1500,
        )
        initial = warm_start.params
    else:
        objective_energy = energy
        target = measured
    result = fit_spectrum(
        objective_energy,
        target,
        model,
        initial,
        (lower, upper),
        baseline_order=-1 if derivative_order else 3,
        robust=True,
        sigma=np.ones_like(target) if derivative_order else balanced_sigma,
        max_nfev=1500,
    )
    result = finalize_physical_fit(
        result, energy, measured, physical_model(result.params),
        baseline_order=3, sigma=balanced_sigma,
    )
    result.resonance_diagnostics = resonance_diagnostics(
        energy, measured, result.fitted, resonances
    )
    return result


def main():
    requested = [item for item in sys.argv[1:] if item in CASES] or list(CASES)
    requested_shapes = [item for item in sys.argv[1:] if item in {"lorentz", "voigt"}] or [
        "lorentz", "voigt"
    ]
    requested_modes = [item for item in sys.argv[1:] if item in {"standard", "d1", "d2"}] or [
        "standard"
    ]
    for case_name in requested:
        for line_shape in requested_shapes:
            for mode in requested_modes:
                derivative_order = {"standard": 0, "d1": 1, "d2": 2}[mode]
                result = run_case(case_name, line_shape, derivative_order)
                stride = 3 if line_shape == "lorentz" else 4
                physical_end = 1 + stride * len(CASES[case_name]["centers"])
                centers = result.params[2:physical_end:stride]
                print(
                    f"{case_name:6s} {line_shape:7s} {mode:8s} R2={result.r_squared:.8f} "
                    f"RMSE={result.rmse:.6g} E0={np.round(centers, 5)} params={np.round(result.params, 5)}"
                )


if __name__ == "__main__":
    main()
