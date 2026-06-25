"""Core normal-incidence optical model for 2D-material contrast fitting."""

import numpy as np
from scipy.special import wofz


HC_EV_NM = 1239.8419843320026
LAYER_MATERIALS = ("Sample", "hBN", "Graphene", "SiO2", "Quartz", "Sapphire", "TiO2")


def default_layer_stack(substrate_type="Si/SiO2"):
    """Return a table-friendly top-to-bottom layer stack."""
    layers = [{"material": "Sample", "thickness_nm": 0.65, "in_reference": False,
               "fit": False, "min_nm": 0.1, "max_nm": 5.0}]
    if substrate_type == "Si/SiO2":
        layers.append({"material": "SiO2", "thickness_nm": 285.0, "in_reference": True,
                       "fit": True, "min_nm": 265.0, "max_nm": 305.0})
    return layers


def normalized_layer_stack(config):
    """Validate a layer table, or translate the legacy fixed-layer config."""
    if "layers" in config:
        layers = []
        for row in config["layers"]:
            material = str(row.get("material", "")).strip()
            if material not in LAYER_MATERIALS:
                raise ValueError(f"Unsupported layer material: {material}")
            thickness = float(row.get("thickness_nm", 0.0))
            lower = float(row.get("min_nm", 0.0))
            upper = float(row.get("max_nm", max(thickness, 1.0)))
            if not np.isfinite(thickness) or thickness < 0:
                raise ValueError(f"Invalid {material} thickness")
            if lower < 0 or upper <= lower or not lower <= thickness <= upper:
                raise ValueError(f"Invalid fit bounds for {material}: {lower} <= {thickness} <= {upper}")
            layers.append({
                "material": material,
                "thickness_nm": thickness,
                "in_reference": False if material == "Sample" else bool(
                    row.get("in_reference", True)
                ),
                "fit": bool(row.get("fit", False)),
                "min_nm": lower,
                "max_nm": upper,
            })
        if sum(layer["material"] == "Sample" for layer in layers) != 1:
            raise ValueError("The layer stack must contain exactly one Sample row")
        return layers

    layers = []
    reference_hbn = bool(config.get("reference_includes_hbn", False))
    if config.get("has_top_hbn", False):
        value = float(config.get("top_hbn_thick", 10.0))
        layers.append({"material": "hBN", "thickness_nm": value,
                       "in_reference": reference_hbn, "fit": False,
                       "min_nm": 0.0, "max_nm": 100.0})
    layers.append({"material": "Sample", "thickness_nm": float(config.get("sample_thick", 0.65)),
                   "in_reference": False, "fit": False, "min_nm": 0.1, "max_nm": 5.0})
    if config.get("has_bot_hbn", False):
        value = float(config.get("bot_hbn_thick", 10.0))
        layers.append({"material": "hBN", "thickness_nm": value,
                       "in_reference": reference_hbn, "fit": False,
                       "min_nm": 0.0, "max_nm": 100.0})
    if config.get("substrate_type", "Si/SiO2") == "Si/SiO2":
        value = float(config.get("sio2_thick", 285.0))
        layers.append({"material": "SiO2", "thickness_nm": value,
                       "in_reference": True, "fit": False,
                       "min_nm": max(0.0, value - 20.0), "max_nm": value + 20.0})
    return layers


def layer_fit_parameters(config):
    """Return (row index, value, lower, upper) for fitted layer thicknesses."""
    return [
        (index, row["thickness_nm"], row["min_nm"], row["max_nm"])
        for index, row in enumerate(normalized_layer_stack(config)) if row["fit"]
    ]


def config_with_layer_thicknesses(config, fitted_values):
    """Copy config and apply fitted thicknesses in table order."""
    updated = dict(config)
    layers = [dict(row) for row in normalized_layer_stack(config)]
    fitted_rows = [index for index, row in enumerate(layers) if row["fit"]]
    if len(fitted_values) != len(fitted_rows):
        raise ValueError("Fitted layer-thickness count does not match the layer table")
    for index, value in zip(fitted_rows, fitted_values):
        layers[index]["thickness_nm"] = float(value)
    updated["layers"] = layers
    return updated


def dielectric_func_lorentz(energy_ev, params):
    """Return epsilon(E) for [eps_inf, f1, E01, gamma1, ...]."""
    energy_ev = np.asarray(energy_ev, dtype=float)
    params = np.asarray(params, dtype=float)
    if params.size < 1 or (params.size - 1) % 3:
        raise ValueError("Lorentz parameters must be [eps_inf, f, E0, gamma, ...]")

    epsilon = np.full(energy_ev.shape, params[0], dtype=complex)
    for strength, resonance, linewidth in params[1:].reshape(-1, 3):
        if strength < 0 or resonance <= 0 or linewidth <= 0:
            raise ValueError("Lorentz oscillators require f >= 0, E0 > 0, and gamma > 0")
        denominator = resonance**2 - energy_ev**2 - 1j * energy_ev * linewidth
        epsilon += strength / denominator
    return epsilon


def dielectric_func_voigt(energy_ev, params):
    """Complex Voigt dielectric response using Faddeeva oscillators.

    Parameters are ``[eps_b, f1, E1, wL1, wG1, ...]`` where ``wL`` and
    ``wG`` are Lorentzian and Gaussian FWHM values in eV.
    """
    energy_ev = np.asarray(energy_ev, dtype=float)
    params = np.asarray(params, dtype=float)
    if params.size < 1 or (params.size - 1) % 4:
        raise ValueError("Voigt parameters must be [eps_b, f, E0, wL, wG, ...]")

    epsilon = np.full(energy_ev.shape, params[0], dtype=complex)
    root_ln2 = np.sqrt(np.log(2.0))
    prefactor_constant = 2j * np.sqrt(np.pi * np.log(2.0))
    for strength, resonance, lorentz_fwhm, gaussian_fwhm in params[1:].reshape(-1, 4):
        if strength < 0 or resonance <= 0:
            raise ValueError("Voigt oscillators require f >= 0 and E0 > 0")
        if lorentz_fwhm <= 0 or gaussian_fwhm <= 0:
            raise ValueError("Voigt linewidths must be positive")
        argument = (
            2.0 * root_ln2 * (energy_ev - resonance) / gaussian_fwhm
            + 1j * root_ln2 * lorentz_fwhm / gaussian_fwhm
        )
        epsilon += strength * prefactor_constant / gaussian_fwhm * wofz(argument)
    return epsilon


def spectral_derivative(values, wavelengths_nm, order=1):
    """Differentiate spectral values with respect to photon energy (eV)."""
    result = np.asarray(values, dtype=float)
    energy_ev = HC_EV_NM / np.asarray(wavelengths_nm, dtype=float)
    for _ in range(order):
        result = np.gradient(result, energy_ev, edge_order=2)
    return result


def calculate_contrast_dynamic(wavelengths_nm, eps_sample, mat_loader, config):
    """Calculate normal-incidence multilayer reflectance contrast.

    ``contrast_definition`` may be ``relative`` (the default,
    (R_sample-R_ref)/R_ref) or ``symmetric``
    ((R_sample-R_ref)/(R_sample+R_ref)).
    """
    wavelengths_nm = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    eps_sample = np.broadcast_to(np.asarray(eps_sample, dtype=complex), wavelengths_nm.shape)
    if np.any(~np.isfinite(wavelengths_nm)) or np.any(wavelengths_nm <= 0):
        raise ValueError("Wavelengths must be finite and positive")

    n_2d = np.sqrt(eps_sample)
    # Select the passive square-root branch for the exp(-i*omega*t) convention.
    n_2d = np.where(np.imag(n_2d) < 0, -n_2d, n_2d)
    substrate_type = config.get("substrate_type", "Si/SiO2")

    if substrate_type in ("Si/SiO2", "Si"):
        n_substrate = mat_loader.get_si_n_with_temp(
            wavelengths_nm, config.get("temp", 298.0)
        )
        n_oxide = mat_loader.get_sio2_n(wavelengths_nm)
    elif substrate_type == "TiO2":
        n_substrate = mat_loader.get_tio2_n(wavelengths_nm)
        n_oxide = None
    elif substrate_type == "Quartz":
        n_substrate = mat_loader.get_quartz_n(wavelengths_nm)
        n_oxide = None
    elif substrate_type == "Sapphire":
        n_substrate = mat_loader.get_sapphire_n(wavelengths_nm)
        n_oxide = None
    else:
        raise ValueError(f"Unsupported substrate type: {substrate_type}")

    n_substrate = np.broadcast_to(np.asarray(n_substrate, dtype=complex), wavelengths_nm.shape)
    layer_rows = normalized_layer_stack(config)

    def material_index(material):
        if material == "Sample":
            return n_2d
        method_name = {
            "hBN": "get_hbn_n", "Graphene": "get_graphene_n",
            "SiO2": "get_sio2_n", "Quartz": "get_quartz_n",
            "Sapphire": "get_sapphire_n", "TiO2": "get_tio2_n",
        }[material]
        return np.broadcast_to(
            np.asarray(getattr(mat_loader, method_name)(wavelengths_nm), dtype=complex),
            wavelengths_nm.shape,
        )

    def layer_stack(reference=False):
        return [
            (material_index(row["material"]), row["thickness_nm"])
            for row in layer_rows
            if not reference or row["in_reference"]
        ]

    def reflectance_at_angle(reference, sine_squared, polarization):
        layers = layer_stack(reference)
        matrix = np.broadcast_to(np.eye(2, dtype=complex), (wavelengths_nm.size, 2, 2)).copy()
        for index, thickness_nm in layers:
            normal_wavevector = np.sqrt(index**2 - sine_squared + 0j)
            normal_wavevector = np.where(
                np.imag(normal_wavevector) < 0, -normal_wavevector, normal_wavevector
            )
            admittance = (
                normal_wavevector
                if polarization == "s"
                else index**2 / normal_wavevector
            )
            phase = 2 * np.pi * normal_wavevector * thickness_nm / wavelengths_nm
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            layer_matrix = np.empty_like(matrix)
            layer_matrix[:, 0, 0] = cos_phase
            layer_matrix[:, 0, 1] = -1j * sin_phase / admittance
            layer_matrix[:, 1, 0] = -1j * admittance * sin_phase
            layer_matrix[:, 1, 1] = cos_phase
            matrix = matrix @ layer_matrix

        substrate_wavevector = np.sqrt(n_substrate**2 - sine_squared + 0j)
        substrate_wavevector = np.where(
            np.imag(substrate_wavevector) < 0, -substrate_wavevector, substrate_wavevector
        )
        substrate_admittance = (
            substrate_wavevector
            if polarization == "s"
            else n_substrate**2 / substrate_wavevector
        )
        incident_wavevector = np.sqrt(1.0 - sine_squared + 0j)
        incident_admittance = (
            incident_wavevector
            if polarization == "s"
            else 1.0 / incident_wavevector
        )
        denominator = matrix[:, 0, 0] + matrix[:, 0, 1] * substrate_admittance
        admittance = (
            matrix[:, 1, 0] + matrix[:, 1, 1] * substrate_admittance
        ) / denominator
        amplitude = (incident_admittance - admittance) / (incident_admittance + admittance)
        return np.abs(amplitude) ** 2

    def reflectance(reference):
        numerical_aperture = float(config.get("numerical_aperture", 0.0))
        if not 0.0 <= numerical_aperture < 1.0:
            raise ValueError("numerical_aperture must be in [0, 1)")
        if numerical_aperture == 0.0:
            return reflectance_at_angle(reference, 0.0, "s")

        # Uniform illumination of the objective pupil is uniform in sin(theta)^2.
        angular_points = max(2, int(config.get("angular_points", 7)))
        nodes, weights = np.polynomial.legendre.leggauss(angular_points)
        sine_squared = 0.5 * numerical_aperture**2 * (nodes + 1.0)
        normalized_weights = 0.5 * weights
        averaged = np.zeros(wavelengths_nm.shape, dtype=float)
        for value, weight in zip(sine_squared, normalized_weights):
            reflectance_s = reflectance_at_angle(reference, value, "s")
            reflectance_p = reflectance_at_angle(reference, value, "p")
            averaged += weight * 0.5 * (reflectance_s + reflectance_p)
        return averaged

    sample_reflectance = reflectance(reference=False)
    reference_reflectance = reflectance(reference=True)
    definition = config.get("contrast_definition", "relative")
    if definition == "symmetric":
        denominator = sample_reflectance + reference_reflectance
    elif definition == "relative":
        denominator = reference_reflectance
    else:
        raise ValueError(f"Unsupported contrast definition: {definition}")

    return np.divide(
        sample_reflectance - reference_reflectance,
        denominator,
        out=np.zeros_like(sample_reflectance),
        where=np.abs(denominator) > np.finfo(float).eps,
    )
