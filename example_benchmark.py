"""Benchmark full-spectrum example fits across Si optical-constant datasets."""

from pathlib import Path

import numpy as np

from fitting_engine import fit_spectrum
from materials import MaterialLoader
from optical_model import HC_EV_NM, calculate_contrast_dynamic, dielectric_func_lorentz


ROOT = Path(__file__).resolve().parent
CENTERS = {
    "3osc": [2.106, 2.515, 3.059],
    "4osc": [2.106, 2.515, 2.94, 3.059],
    "5osc": [1.995, 2.106, 2.515, 2.94, 3.059],
}


def load_example():
    substrate = np.loadtxt(
        ROOT / "example" / "1L-WS2-300 nm SiO2_Si-substrate.csv", delimiter=","
    )
    sample = np.loadtxt(
        ROOT / "example" / "1L-WS2-300 nm SiO2_Si-sample.csv", delimiter=","
    )
    energy = substrate[:, 0]
    measured = sample[:, 1] / substrate[:, 1] - 1.0
    order = np.argsort(energy)
    return energy[order], measured[order]


def run_trial(si_filename, centers, baseline_order):
    energy, measured = load_example()
    wavelength = HC_EV_NM / energy
    materials = MaterialLoader(ROOT / si_filename)
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

    lorentz_initial = [12.0]
    lorentz_lower = [1.0]
    lorentz_upper = [50.0]
    for center in centers:
        lorentz_initial.extend([0.5 if abs(center - 2.106) < 0.02 else 0.1, center, 0.04])
        lorentz_lower.extend([0.0, center - 0.10, 0.002])
        lorentz_upper.extend([50.0, center + 0.10, 0.50])
    oscillator_count = len(centers)
    initial = np.asarray(lorentz_initial + [300.0])
    lower = np.asarray(lorentz_lower + [280.0])
    upper = np.asarray(lorentz_upper + [320.0])

    def model(params):
        config["sio2_thick"] = params[-1]
        epsilon = dielectric_func_lorentz(energy, params[: 1 + 3 * oscillator_count])
        return calculate_contrast_dynamic(wavelength, epsilon, materials, config)

    result = fit_spectrum(
        energy,
        measured,
        model,
        initial,
        (lower, upper),
        baseline_order=baseline_order,
        robust=True,
        global_search=False,
        max_nfev=4000,
    )
    return result


def main():
    for si_filename in ("Si_data.csv", "Schinke.csv", "Green-2008.csv"):
        for label, centers in CENTERS.items():
            for baseline_order in (1, 3):
                result = run_trial(si_filename, centers, baseline_order)
                resonances = result.params[2:-1:3]
                print(
                    f"{si_filename:14s} {label} baseline={baseline_order} "
                    f"R2={result.r_squared:.8f} RMSE={result.rmse:.6f} "
                    f"oxide={result.params[-1]:.3f} E0={np.round(resonances, 4)}"
                )


if __name__ == "__main__":
    main()
