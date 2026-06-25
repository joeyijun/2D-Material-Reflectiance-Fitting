"""Robust nonlinear least-squares engine for optical spectrum metrology."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.special import expit


class FitCancelled(RuntimeError):
    """Raised when a caller requests cancellation during optimization."""


@dataclass
class FitResult:
    params: np.ndarray
    fitted: np.ndarray
    physical_model: np.ndarray
    baseline_coefficients: np.ndarray
    baseline_center: float
    baseline_scale: float
    standard_errors: np.ndarray
    covariance: np.ndarray
    success: bool
    message: str
    method: str
    nfev: int
    rmse: float
    mae: float
    r_squared: float
    reduced_chi_squared: float
    jacobian_condition: float
    durbin_watson: float

    def add_baseline(self, physical_model, energy_ev):
        """Apply the fitted additive baseline to model values."""
        physical_model = np.asarray(physical_model, dtype=float)
        if self.baseline_coefficients.size == 0:
            return physical_model
        x = (np.asarray(energy_ev, dtype=float) - self.baseline_center) / self.baseline_scale
        design = np.vander(x, N=self.baseline_coefficients.size, increasing=True)
        return physical_model + design @ self.baseline_coefficients


def smoothed_spectral_derivative(values, energy_ev, order=1, smoothing_ev=None):
    """Savitzky-Golay derivative on a uniform energy grid with noise suppression."""
    values = np.asarray(values, dtype=float)
    energy_ev = np.asarray(energy_ev, dtype=float)
    if values.shape != energy_ev.shape or values.ndim != 1 or values.size < 7:
        raise ValueError("derivative data must be equal one-dimensional arrays with >=7 points")
    if order not in (1, 2):
        raise ValueError("derivative order must be 1 or 2")
    sort_order = np.argsort(energy_ev)
    energy_sorted = energy_ev[sort_order]
    values_sorted = values[sort_order]
    uniform_energy = np.linspace(energy_sorted[0], energy_sorted[-1], energy_sorted.size)
    uniform_values = np.interp(uniform_energy, energy_sorted, values_sorted)
    spacing = float(uniform_energy[1] - uniform_energy[0])
    smoothing_ev = (0.004 if order == 1 else 0.008) if smoothing_ev is None else smoothing_ev
    window = max(order + 4, int(np.ceil(smoothing_ev / spacing)))
    window += 1 - window % 2
    maximum_window = values.size if values.size % 2 else values.size - 1
    window = min(window, maximum_window)
    polynomial_order = min(max(3, order + 2), window - 2)
    derivative_uniform = savgol_filter(
        uniform_values,
        window,
        polynomial_order,
        deriv=order,
        delta=spacing,
        mode="interp",
    )
    derivative_sorted = np.interp(energy_sorted, uniform_energy, derivative_uniform)
    derivative = np.empty_like(derivative_sorted)
    derivative[sort_order] = derivative_sorted
    return derivative


def finalize_physical_fit(
    result, energy_ev, measured, physical_model, baseline_order=3, sigma=None
):
    """Re-evaluate a fit in the original spectrum domain after derivative fitting."""
    energy_ev = np.asarray(energy_ev, dtype=float)
    measured = np.asarray(measured, dtype=float)
    physical_model = np.asarray(physical_model, dtype=float)
    design, center, scale = _baseline_design(energy_ev, baseline_order)
    weights = None
    if sigma is not None:
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), measured.shape)
        weights = 1.0 / sigma**2
    coefficients, baseline = _profile_baseline(
        measured - physical_model, design, weights
    )
    fitted = physical_model + baseline
    residual = fitted - measured
    rss = float(np.sum(residual**2))
    total = float(np.sum((measured - np.mean(measured)) ** 2))
    result.objective_r_squared = result.r_squared
    result.baseline_coefficients = coefficients
    result.baseline_center = center
    result.baseline_scale = scale
    result.physical_model = physical_model
    result.fitted = fitted
    result.rmse = float(np.sqrt(np.mean(residual**2)))
    result.mae = float(np.mean(np.abs(residual)))
    result.r_squared = 1.0 - rss / total if total > 0 else np.nan
    result.durbin_watson = float(np.sum(np.diff(residual) ** 2) / rss) if rss > 0 else np.nan
    return result


def resonance_balanced_sigma(energy_ev, measured, resonances):
    """Per-point scales that give each requested resonance local influence.

    ``resonances`` contains ``(center_eV, approximate_fwhm_eV)`` pairs. The
    full-spectrum scale is replaced near each resonance by its locally
    detrended amplitude, with a robust noise floor to avoid fitting noise.
    """
    energy_ev = np.asarray(energy_ev, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if energy_ev.shape != measured.shape or energy_ev.ndim != 1:
        raise ValueError("energy and measured arrays must have equal one-dimensional shapes")
    global_scale = max(float(np.std(measured)), np.finfo(float).eps)
    noise_floor = 5.0 * _noise_scale(measured)
    sigma = np.full(measured.shape, global_scale)
    for center, fwhm in resonances:
        radius = float(np.clip(2.0 * max(float(fwhm), 0.005), 0.02, 0.08))
        mask = np.abs(energy_ev - float(center)) <= radius
        if np.count_nonzero(mask) < 7:
            continue
        local_energy = energy_ev[mask]
        normalized = (local_energy - np.mean(local_energy)) / max(np.ptp(local_energy), 1e-12)
        trend = np.polyval(np.polyfit(normalized, measured[mask], 1), normalized)
        local_scale = max(float(np.std(measured[mask] - trend)), noise_floor)
        sigma[mask] = np.minimum(sigma[mask], local_scale)
    return sigma


def resonance_windows_from_parameters(params, line_shape):
    """Extract (center, approximate FWHM) pairs from dielectric parameters."""
    params = np.asarray(params, dtype=float)
    if str(line_shape).lower() == "voigt":
        rows = params[1:].reshape(-1, 4)
        widths = 0.5346 * rows[:, 2] + np.sqrt(
            0.2166 * rows[:, 2] ** 2 + rows[:, 3] ** 2
        )
    else:
        rows = params[1:].reshape(-1, 3)
        widths = rows[:, 2]
    return [(float(row[1]), float(width)) for row, width in zip(rows, widths)]


def resonance_diagnostics(energy_ev, measured, fitted, resonances):
    """Return detrended local fit metrics for every requested resonance."""
    energy_ev = np.asarray(energy_ev, dtype=float)
    measured = np.asarray(measured, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    diagnostics = []
    for center, fwhm in resonances:
        radius = float(np.clip(2.0 * max(float(fwhm), 0.005), 0.02, 0.08))
        mask = np.abs(energy_ev - float(center)) <= radius
        if np.count_nonzero(mask) < 7:
            continue
        x = energy_ev[mask]
        x = (x - np.mean(x)) / max(np.ptp(x), 1e-12)
        measured_local = measured[mask]
        fitted_local = fitted[mask]
        measured_feature = measured_local - np.polyval(
            np.polyfit(x, measured_local, 1), x
        )
        fitted_feature = fitted_local - np.polyval(
            np.polyfit(x, fitted_local, 1), x
        )
        rss = float(np.sum((fitted_feature - measured_feature) ** 2))
        total = float(np.sum(measured_feature**2))
        measured_amplitude = float(np.ptp(measured_feature))
        diagnostics.append({
            "center_ev": float(center),
            "local_r_squared": 1.0 - rss / total if total > 0 else np.nan,
            "amplitude_ratio": (
                float(np.ptp(fitted_feature)) / measured_amplitude
                if measured_amplitude > 0 else np.nan
            ),
            "points": int(np.count_nonzero(mask)),
        })
    return diagnostics


def composite_derivative_residual(
    measured, predicted, energy_ev, maximum_order, baseline_order=3, resonances=None
):
    """Joint spectrum/derivative residual used for stable derivative fitting."""
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    energy_ev = np.asarray(energy_ev, dtype=float)
    design, _, _ = _baseline_design(energy_ev, baseline_order)
    _, baseline = _profile_baseline(measured - predicted, design)
    fitted = predicted + baseline
    residual_blocks = []
    derivative_weights = (1.0, 1.0, 0.5)
    measured_component = measured
    fitted_component = fitted
    feature_weight = np.ones_like(measured)
    if resonances:
        local_sigma = resonance_balanced_sigma(energy_ev, measured, resonances)
        feature_weight = np.median(local_sigma) / local_sigma
    for order in range(maximum_order + 1):
        if order:
            measured_component = smoothed_spectral_derivative(
                measured, energy_ev, order=order
            )
            fitted_component = smoothed_spectral_derivative(
                fitted, energy_ev, order=order
            )
        scale = max(float(np.std(measured_component)), np.finfo(float).eps)
        residual_blocks.append(
            derivative_weights[order] * feature_weight
            * (fitted_component - measured_component) / scale
        )
    return np.concatenate(residual_blocks)


def guess_resonances(
    energy_ev,
    contrast,
    max_peaks=6,
    min_separation_ev=0.04,
    max_linewidth_ev=0.12,
    peak_dip_merge_ev=0.035,
):
    """Return robust initial (energy, FWHM) guesses after removing slow drift."""
    energy_ev = np.asarray(energy_ev, dtype=float)
    contrast = np.asarray(contrast, dtype=float)
    if energy_ev.ndim != 1 or energy_ev.shape != contrast.shape or energy_ev.size < 11:
        return np.empty((0, 2))
    order = np.argsort(energy_ev)
    energy = energy_ev[order]
    values = contrast[order]
    spacing = float(np.median(np.diff(energy)))
    if spacing <= 0:
        return np.empty((0, 2))

    def odd_window(fraction, minimum):
        window = max(minimum, int(round(values.size * fraction)))
        window += 1 - window % 2
        return min(window, values.size - 1 + values.size % 2)

    smooth_window = odd_window(0.015, 7)
    smooth = savgol_filter(values, smooth_window, min(3, smooth_window - 2))
    normalized_energy = (energy - np.mean(energy)) / max(np.ptp(energy), np.finfo(float).eps)
    background = np.polyval(np.polyfit(normalized_energy, smooth, deg=min(3, values.size - 1)), normalized_energy)
    feature = smooth - background
    prominence = max(5.0 * _noise_scale(values), 0.008 * np.ptp(feature))
    distance = max(1, int(np.ceil(min_separation_ev / spacing)))

    def add_feature_candidates(feature_values, feature_prominence):
        found = []
        positive, positive_properties = find_peaks(
            feature_values, prominence=feature_prominence, distance=distance
        )
        negative, negative_properties = find_peaks(
            -feature_values, prominence=feature_prominence, distance=distance
        )
        if positive.size:
            widths = peak_widths(feature_values, positive, rel_height=0.5)[0] * spacing
            found.extend(zip(positive, positive_properties["prominences"], widths))
        if negative.size:
            widths = peak_widths(-feature_values, negative, rel_height=0.5)[0] * spacing
            found.extend(zip(negative, negative_properties["prominences"], widths))
        return found

    candidates = add_feature_candidates(feature, prominence)
    broad_window = odd_window(0.12, 31)
    if broad_window > smooth_window:
        local_background = savgol_filter(smooth, broad_window, min(3, broad_window - 2))
        local_feature = smooth - local_background
        local_prominence = max(4.0 * _noise_scale(local_feature), 0.006 * np.ptp(local_feature))
        local_candidates = add_feature_candidates(local_feature, local_prominence)
        for local in local_candidates:
            local_energy = energy[local[0]]
            local_width = float(local[2])
            if all(
                abs(local_energy - energy[global_candidate[0]])
                >= max(peak_dip_merge_ev, 2.0 * max(local_width, float(global_candidate[2])))
                for global_candidate in candidates
            ):
                candidates.append(local)
    if not candidates:
        return np.empty((0, 2))
    candidates = [
        item
        for item in candidates
        if item[2] <= max_linewidth_ev
        and min(energy[item[0]] - energy[0], energy[-1] - energy[item[0]]) >= item[2]
    ]
    if not candidates:
        return np.empty((0, 2))
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = []
    for candidate in candidates:
        candidate_energy = energy[candidate[0]]
        candidate_width = float(candidate[2])
        if all(
            abs(candidate_energy - energy[item[0]])
            >= max(peak_dip_merge_ev, 1.5 * max(candidate_width, float(item[2])))
            for item in selected
        ):
            selected.append(candidate)
        if len(selected) == max_peaks:
            break
    guesses = np.array([
        (energy[index], np.clip(width, 0.005, 0.5))
        for index, _, width in selected
    ])
    return guesses[np.argsort(guesses[:, 0])]


def _noise_scale(values):
    """Estimate point noise from second differences without fitting spectral level."""
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        scale = np.std(values)
    else:
        second_difference = np.diff(values, n=2)
        median = np.median(second_difference)
        scale = 1.4826 * np.median(np.abs(second_difference - median)) / np.sqrt(6.0)
    dynamic_range = np.ptp(values)
    floor = max(np.finfo(float).eps, dynamic_range * 1e-6)
    return max(float(scale), floor)


def _baseline_design(energy_ev, order, center=None, scale=None):
    if order < 0:
        return np.empty((len(energy_ev), 0)), 0.0, 1.0
    energy_ev = np.asarray(energy_ev, dtype=float)
    center = float(np.mean(energy_ev)) if center is None else float(center)
    scale = float(np.ptp(energy_ev) / 2.0) if scale is None else float(scale)
    if scale <= 0:
        scale = 1.0
    normalized = (energy_ev - center) / scale
    return np.vander(normalized, N=order + 1, increasing=True), center, scale


def _profile_baseline(target_minus_model, design, weights=None):
    if design.shape[1] == 0:
        return np.empty(0), np.zeros_like(target_minus_model)
    if weights is None:
        coefficients, *_ = np.linalg.lstsq(design, target_minus_model, rcond=None)
    else:
        root_weight = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * root_weight[:, None], target_minus_model * root_weight, rcond=None
        )
    return coefficients, design @ coefficients


def _huber_weights(residual, tuning=1.345):
    residual = np.asarray(residual, dtype=float)
    center = np.median(residual)
    scale = 1.4826 * np.median(np.abs(residual - center))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.ones_like(residual)
    normalized = np.abs(residual - center) / (tuning * scale)
    weights = np.ones_like(normalized)
    np.divide(1.0, normalized, out=weights, where=normalized > 1.0)
    return weights


def _to_unconstrained(params, lower, upper):
    fraction = (params - lower) / (upper - lower)
    fraction = np.clip(fraction, 1e-9, 1.0 - 1e-9)
    return np.log(fraction / (1.0 - fraction))


def _from_unconstrained(values, lower, upper):
    return lower + (upper - lower) * expit(values)


def fit_spectrum(
    energy_ev,
    measured,
    model,
    initial_params,
    bounds,
    locked_mask=None,
    *,
    baseline_order=1,
    robust=True,
    global_search=False,
    sigma=None,
    max_nfev=5000,
    cancel_check=None,
    random_seed=0,
):
    """Fit a spectrum using robust TRF followed by transformed-variable LM.

    ``model`` receives the full physical parameter vector. Finite bounds are
    enforced during LM through a logistic parameter transform. An additive
    polynomial baseline is solved analytically at every nonlinear iteration
    (variable projection), so nuisance drift does not enlarge the nonlinear
    parameter space.
    """
    energy_ev = np.asarray(energy_ev, dtype=float)
    measured = np.asarray(measured, dtype=float)
    initial_params = np.asarray(initial_params, dtype=float)
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    if energy_ev.shape != measured.shape or energy_ev.ndim != 1:
        raise ValueError("energy_ev and measured must be one-dimensional arrays of equal length")
    if initial_params.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError("initial_params and bounds must have identical shapes")
    if np.any(~np.isfinite(energy_ev)) or np.any(~np.isfinite(measured)):
        raise ValueError("fit data must be finite")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("bounded LM requires finite lower and upper bounds")
    if np.any(lower >= upper) or np.any(initial_params < lower) or np.any(initial_params > upper):
        raise ValueError("initial parameters must lie inside valid bounds")

    locked = np.zeros(initial_params.size, dtype=bool) if locked_mask is None else np.asarray(locked_mask, dtype=bool)
    if locked.shape != initial_params.shape:
        raise ValueError("locked_mask must match initial_params")
    free = ~locked
    if measured.size <= np.count_nonzero(free) + max(baseline_order + 1, 0):
        raise ValueError("not enough spectral points for the requested parameters")

    design, baseline_center, baseline_scale = _baseline_design(energy_ev, baseline_order)
    noise = np.full(measured.shape, _noise_scale(measured)) if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=float), measured.shape)
    if np.any(~np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("sigma must contain finite positive values")

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise FitCancelled("Fitting stopped")

    def full_params(free_params):
        params = initial_params.copy()
        params[free] = free_params
        return params

    def raw_residual(free_params, weights=None):
        check_cancelled()
        prediction = np.asarray(model(full_params(free_params)), dtype=float)
        if prediction.shape != measured.shape or np.any(~np.isfinite(prediction)):
            raise ValueError("model returned non-finite values or an invalid shape")
        profile_weights = 1.0 / noise**2
        if weights is not None:
            profile_weights = profile_weights * weights
        coefficients, baseline = _profile_baseline(
            measured - prediction, design, profile_weights
        )
        return prediction + baseline - measured, coefficients, prediction + baseline

    free_initial = initial_params[free]
    free_lower = lower[free]
    free_upper = upper[free]
    if not np.any(free):
        residual, coefficients, fitted = raw_residual(free_initial)
        final_free = free_initial
        jacobian = np.empty((measured.size, 0))
        method = "fixed"
        nfev = 1
        success = True
        message = "All physical parameters were locked"
        final_weights = np.ones_like(measured)
    else:
        if global_search:
            def global_objective(candidate):
                residual, _, _ = raw_residual(candidate)
                normalized = residual / noise
                # A clipped quadratic prevents a few detector spikes dominating initialization.
                return float(np.sum(np.minimum(normalized**2, 25.0)))

            global_result = differential_evolution(
                global_objective,
                list(zip(free_lower, free_upper)),
                seed=random_seed,
                strategy="best1bin",
                popsize=12,
                maxiter=120,
                tol=1e-7,
                polish=False,
                workers=1,
                updating="immediate",
            )
            free_initial = global_result.x

        def trf_residual(candidate):
            residual, _, _ = raw_residual(candidate)
            return residual / noise

        trf = least_squares(
            trf_residual,
            free_initial,
            bounds=(free_lower, free_upper),
            method="trf",
            loss="soft_l1" if robust else "linear",
            f_scale=1.0,
            x_scale="jac",
            jac="3-point",
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        trf_raw, _, _ = raw_residual(trf.x)
        robust_weights = _huber_weights(trf_raw / noise) if robust else np.ones_like(measured)
        final_weights = robust_weights
        root_weight = np.sqrt(robust_weights) / noise
        unconstrained_initial = _to_unconstrained(trf.x, free_lower, free_upper)

        def lm_residual(unconstrained):
            candidate = _from_unconstrained(unconstrained, free_lower, free_upper)
            residual, _, _ = raw_residual(candidate, robust_weights)
            return residual * root_weight

        # LM is efficient near the solution; the transform retains physical bounds.
        lm = least_squares(
            lm_residual,
            unconstrained_initial,
            method="lm",
            jac="3-point",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        final_free = _from_unconstrained(lm.x, free_lower, free_upper)
        residual, coefficients, fitted = raw_residual(final_free, robust_weights)
        jacobian = lm.jac
        method = ("DE + " if global_search else "") + "robust TRF + bounded LM"
        nfev = trf.nfev + lm.nfev
        success = bool(trf.success and lm.success)
        message = f"TRF: {trf.message}; LM: {lm.message}"

    params = full_params(final_free)
    physical_model = np.asarray(model(params), dtype=float)
    if design.shape[1]:
        fitted = physical_model + design @ coefficients
    residual = fitted - measured
    degrees_of_freedom = measured.size - np.count_nonzero(free) - design.shape[1]
    weighted_rss = float(np.sum(final_weights * (residual / noise) ** 2))
    covariance_free = np.full((np.count_nonzero(free), np.count_nonzero(free)), np.nan)
    condition = np.inf
    if np.any(free) and jacobian.size:
        information = jacobian.T @ jacobian
        condition = float(np.linalg.cond(information))
        if np.linalg.matrix_rank(information) == information.shape[0]:
            covariance_z = np.linalg.inv(information) * weighted_rss / max(degrees_of_freedom, 1)
            fraction = (final_free - free_lower) / (free_upper - free_lower)
            derivative = (free_upper - free_lower) * fraction * (1.0 - fraction)
            covariance_free = covariance_z * np.outer(derivative, derivative)

    covariance = np.full((params.size, params.size), np.nan)
    covariance[np.ix_(free, free)] = covariance_free
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    standard_errors[locked] = 0.0
    total_variance = float(np.sum((measured - np.mean(measured)) ** 2))
    rss = float(np.sum(residual**2))
    r_squared = 1.0 - rss / total_variance if total_variance > 0 else np.nan
    differences = np.diff(residual)
    durbin_watson = float(np.sum(differences**2) / rss) if rss > 0 else np.nan

    return FitResult(
        params=params,
        fitted=fitted,
        physical_model=physical_model,
        baseline_coefficients=coefficients,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        standard_errors=standard_errors,
        covariance=covariance,
        success=success,
        message=message,
        method=method,
        nfev=nfev,
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(np.abs(residual))),
        r_squared=float(r_squared),
        reduced_chi_squared=weighted_rss / max(degrees_of_freedom, 1),
        jacobian_condition=condition,
        durbin_watson=durbin_watson,
    )
