
import streamlit as st
import numpy as np
import pandas as pd
from fitting_engine import (
    composite_derivative_residual,
    finalize_physical_fit,
    fit_spectrum,
    guess_resonances,
    resonance_balanced_sigma,
    resonance_diagnostics,
    resonance_windows_from_parameters,
)
from materials import read_si_optical_constants
from optical_model import (
    HC_EV_NM,
    LAYER_MATERIALS,
    calculate_contrast_dynamic as calculate_contrast_core,
    config_with_layer_thicknesses,
    dielectric_func_lorentz as dielectric_func_lorentz_core,
    dielectric_func_voigt,
    layer_fit_parameters,
    normalized_layer_stack,
    spectral_derivative,
)
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import re
import sys

# --- Material Loader Class (Adapted) ---
class MaterialLoader:
    def __init__(self, si_filename=None, uploaded_si_file=None):
        self.si_n_interp = None
        self.si_k_interp = None
        
        # Priority: Uploaded > Local Default
        if uploaded_si_file is not None:
             self._init_si_interpolation_from_file(uploaded_si_file)
        elif si_filename and os.path.exists(si_filename):
             self._init_si_interpolation_from_path(si_filename)
        else:
             # Try to load default from current directory if nothing else
             if os.path.exists("Si_data.csv"):
                 self._init_si_interpolation_from_path("Si_data.csv")

    def _init_si_interpolation_from_path(self, path):
        try:
            self._process_data(path)
        except Exception as e:
            st.error(f"Error loading substrate file: {e}")

    def _init_si_interpolation_from_file(self, uploaded_file):
        try:
            self._process_data(uploaded_file)
        except Exception as e:
             st.error(f"Error loading uploaded substrate file: {e}")

    def _process_data(self, source):
        lam_nm, n_vals, k_vals = read_si_optical_constants(source)
        order = np.argsort(lam_nm)
        lam_nm = lam_nm[order]
        self.si_wavelength_range = (float(lam_nm[0]), float(lam_nm[-1]))
        self.si_n_interp = PchipInterpolator(lam_nm, n_vals[order], extrapolate=False)
        self.si_k_interp = PchipInterpolator(lam_nm, k_vals[order], extrapolate=False)
        self.si_n_edge_slopes = tuple(
            float(self.si_n_interp.derivative()(edge)) for edge in self.si_wavelength_range
        )
        self.si_k_edge_slopes = tuple(
            float(self.si_k_interp.derivative()(edge)) for edge in self.si_wavelength_range
        )

    def _evaluate_si_component(self, wavelengths_nm, interpolation, edge_slopes):
        lower, upper = self.si_wavelength_range
        clipped = np.clip(wavelengths_nm, lower, upper)
        values = interpolation(clipped)
        lower_value = float(interpolation(lower))
        upper_value = float(interpolation(upper))
        values = np.where(
            wavelengths_nm < lower,
            lower_value + edge_slopes[0] * (wavelengths_nm - lower),
            values,
        )
        return np.where(
            wavelengths_nm > upper,
            upper_value + edge_slopes[1] * (wavelengths_nm - upper),
            values,
        )

    def get_si_n(self, lam_nm):
        if self.si_n_interp is None: return 1.0 + 0j
        lam_nm = np.asarray(lam_nm, dtype=float)
        if np.any(~np.isfinite(lam_nm)) or np.any(lam_nm <= 0):
            raise ValueError("Si wavelengths must be finite and positive")
        n = self._evaluate_si_component(lam_nm, self.si_n_interp, self.si_n_edge_slopes)
        k = self._evaluate_si_component(lam_nm, self.si_k_interp, self.si_k_edge_slopes)
        n = np.maximum(n, np.finfo(float).eps)
        return n + 1j * np.maximum(k, 0.0)

    def get_si_n_with_temp(self, lam_nm, temp_k):
        if not 10.0 <= temp_k <= 300.0:
            raise ValueError("Si temperature correction is calibrated only for 10-300 K")
        n_base = self.get_si_n(lam_nm)
        lam_um = lam_nm / 1000.0
        delta_n_abs = 0.02514 + 0.00850 / (lam_um**2 - 0.10165)
        delta_k_abs = 270.16 * np.exp(-18.91 * lam_um) + 0.0029
        ratio = (298.0 - temp_k) / (298.0 - 10.0)
        correction_n = ratio * delta_n_abs
        correction_k = ratio * delta_k_abs
        n_real = np.real(n_base) - correction_n
        n_imag = np.maximum(np.imag(n_base) - correction_k, 0.0)
        return n_real + 1j * n_imag

    def get_sio2_n(self, lam_nm):
        x = lam_nm / 1000.0
        term1 = 0.9310 / (1 - (0.079 / x)**2)
        term2 = 0.1735 / (1 - (0.130 / x)**2)
        term3 = 2.1121 / (1 - (14.918 / x)**2)
        n_squared = 1 + term1 + term2 + term3
        return np.sqrt(n_squared) + 0j

    def get_hbn_n(self, lam_nm):
        lam_um = lam_nm / 1000.0
        denom = lam_um**2 - 0.1644**2
        n_squared = 1 + (3.263 * lam_um**2) / denom
        return np.sqrt(n_squared) + 0j

    def get_quartz_n(self, lam_nm):
        return self.get_sio2_n(lam_nm)

    def get_sapphire_n(self, lam_nm):
        x = lam_nm / 1000.0
        term1 = 1.4313493 / (1 - (0.0726631 / x)**2)
        term2 = 0.65054713/ (1 - (0.1193242 / x)**2)
        term3 = 5.3414021 / (1 - (18.028251 / x)**2)
        n_squared = 1 + term1 + term2 + term3
        return np.sqrt(n_squared) + 0j

    def get_tio2_n(self, lam_nm):
        x = lam_nm / 1000.0
        n_squared = 5.913 + 0.2441 / (x**2 - 0.0803)
        return np.sqrt(n_squared) + 0j

# --- Core Physics Functions ---

def dielectric_func_lorentz(E, params):
    eps_inf = params[0]
    eps = eps_inf + 0j
    num_oscillators = (len(params) - 1) // 3
    for i in range(num_oscillators):
        f = params[1 + i*3]
        E0 = params[2 + i*3]
        g = params[3 + i*3]
        numerator = f
        denominator = E0**2 - E**2 - 1j * E * g
        eps += numerator / denominator
    return eps

def calculate_contrast_dynamic(wavelengths_nm, eps_sample, mat_loader, structure_config):
    lam_arr = np.array(wavelengths_nm)
    n_2d_arr = np.sqrt(eps_sample)
    
    sub_type = structure_config.get('substrate_type', 'Si/SiO2')
    temp_k = structure_config.get('temp', 298.0)
    
    if sub_type == 'Si/SiO2':
        n_sub_arr = mat_loader.get_si_n_with_temp(lam_arr, temp_k)
        n_oxide_arr = mat_loader.get_sio2_n(lam_arr)
    elif sub_type == 'TiO2':
        n_sub_arr = mat_loader.get_tio2_n(lam_arr)
        n_oxide_arr = None
    elif sub_type == 'Quartz':
        n_sub_arr = mat_loader.get_quartz_n(lam_arr)
        n_oxide_arr = None
    elif sub_type == 'Sapphire':
        n_sub_arr = mat_loader.get_sapphire_n(lam_arr)
        n_oxide_arr = None
    
    n_hbn_val = mat_loader.get_hbn_n(lam_arr)
    if np.isscalar(n_hbn_val):
        n_hbn_arr = np.full_like(lam_arr, n_hbn_val, dtype=complex)
    else:
        n_hbn_arr = n_hbn_val
        
    def solve_stack(include_sample, include_hbn):
        stack_n = []
        stack_d = []
        
        if include_hbn and structure_config['has_top_hbn']:
            stack_n.append(n_hbn_arr)
            stack_d.append(structure_config['top_hbn_thick'])
            
        if include_sample:
            stack_n.append(n_2d_arr)
            stack_d.append(structure_config['sample_thick'])
            
        if include_hbn and structure_config['has_bot_hbn']:
            stack_n.append(n_hbn_arr)
            stack_d.append(structure_config['bot_hbn_thick'])
            
        if n_oxide_arr is not None:
             stack_n.append(n_oxide_arr)
             stack_d.append(structure_config['sio2_thick'])
             
        M = np.broadcast_to(np.eye(2, dtype=complex), (len(lam_arr), 2, 2)).copy()
        
        for n_layer, d_layer in zip(stack_n, stack_d):
            delta = 2 * np.pi * n_layer * d_layer / lam_arr
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            m11 = cos_d
            m12 = -1j / n_layer * sin_d
            m21 = -1j * n_layer * sin_d
            m22 = cos_d
            M_layer = np.array([[m11, m12], [m21, m22]]) 
            M_layer = np.moveaxis(M_layer, 2, 0) 
            M = np.matmul(M, M_layer)
            
        n_s = n_sub_arr
        n_0 = 1.0 
        M11 = M[:, 0, 0]
        M12 = M[:, 0, 1]
        M21 = M[:, 1, 0]
        M22 = M[:, 1, 1]
        denom = (M11 + M12 * n_s)
        denom[denom == 0] = 1e-10
        Y = (M21 + M22 * n_s) / denom
        r = (n_0 - Y) / (n_0 + Y)
        return np.abs(r)**2
    
    try:
        R_sample = solve_stack(include_sample=True, include_hbn=True)
        R_ref = solve_stack(include_sample=False, include_hbn=False)
        R_ref[R_ref == 0] = 1e-10
        contrast = (R_sample - R_ref) / R_ref
        return contrast
    except Exception as e:
        raise RuntimeError("Optical model calculation failed") from e


dielectric_func_lorentz = dielectric_func_lorentz_core
calculate_contrast_dynamic = calculate_contrast_core

# --- Streamlit App ---

st.set_page_config(page_title="2D Material Reflection Fitting", layout="wide")
st.title("2D Material Reflectance Fitting")

# Sidebar - Files
st.sidebar.header("1. Material Data (Target)")
# --- Experimental Data Loading ---
st.sidebar.subheader("Experimental Spectra")
uploaded_sub_file = st.sidebar.file_uploader("Upload Substrate Spectrum (Ref)", type=["txt", "csv"])
uploaded_samp_file = st.sidebar.file_uploader("Upload Sample Spectrum", type=["txt", "csv"])

# Advanced optical settings remain code-level defaults for future use.
si_source = "Si_data.csv"
loader = MaterialLoader(si_filename=si_source) if os.path.exists(si_source) else None
if loader is None or loader.si_n_interp is None:
    st.sidebar.error("Default Si optical data (Si_data.csv) is missing.")
    st.stop()

hbn_filename_hint = any(
    uploaded is not None and len(re.findall(r"hBN", uploaded.name, re.IGNORECASE)) >= 2
    for uploaded in (uploaded_sub_file, uploaded_samp_file)
)
layer_presets = {
    "Sample / SiO2 / Si": [
        ["Sample", 0.65, False, False, 0.1, 2.0],
        ["SiO2", 285.0, True, True, 265.0, 305.0],
    ],
    "hBN / Sample / hBN / SiO2 / Si": [
        ["hBN", 10.0, False, True, 0.0, 100.0],
        ["Sample", 0.65, False, False, 0.1, 2.0],
        ["hBN", 10.0, False, True, 0.0, 100.0],
        ["SiO2", 285.0, True, True, 265.0, 305.0],
    ],
    "hBN / Graphene / Sample / Graphene / hBN / SiO2 / Si": [
        ["hBN", 10.0, False, True, 0.0, 100.0],
        ["Graphene", 0.335, False, False, 0.1, 2.0],
        ["Sample", 0.65, False, False, 0.1, 2.0],
        ["Graphene", 0.335, False, False, 0.1, 2.0],
        ["hBN", 10.0, False, True, 0.0, 100.0],
        ["SiO2", 285.0, True, True, 265.0, 305.0],
    ],
}
layer_columns = ["Order", "Material", "Thickness (nm)", "In reference", "Fit", "Min (nm)", "Max (nm)"]


def preset_layer_table(rows):
    table = pd.DataFrame(rows, columns=layer_columns[1:])
    table.insert(0, "Order", np.arange(1, len(table) + 1))
    return table


if "structure_layers" not in st.session_state:
    initial_preset = list(layer_presets)[1 if hbn_filename_hint else 0]
    st.session_state.structure_layers = preset_layer_table(layer_presets[initial_preset])

st.header("2. Structure & Optical Model")
structure_col, optics_col = st.columns([2, 1])
with structure_col:
    preset_col, apply_col = st.columns([4, 1])
    selected_preset = preset_col.selectbox("Structure preset", list(layer_presets))
    if apply_col.button("Apply", use_container_width=True):
        st.session_state.structure_layers = preset_layer_table(layer_presets[selected_preset])
    layer_df = st.data_editor(
        st.session_state.structure_layers,
        column_config={
            "Order": st.column_config.NumberColumn("Order", min_value=1, step=1, required=True),
            "Material": st.column_config.SelectboxColumn("Material", options=list(LAYER_MATERIALS), required=True),
            "Thickness (nm)": st.column_config.NumberColumn(min_value=0.0, format="%.4f"),
            "In reference": st.column_config.CheckboxColumn(help="Keep this layer in the reference-region stack"),
            "Fit": st.column_config.CheckboxColumn(help="Fit this layer thickness"),
            "Min (nm)": st.column_config.NumberColumn(min_value=0.0, format="%.4f"),
            "Max (nm)": st.column_config.NumberColumn(min_value=0.0001, format="%.4f"),
        },
        num_rows="dynamic", use_container_width=True, hide_index=True,
        key="layer_editor",
    )
    st.session_state.structure_layers = layer_df

with optics_col:
    sub_type = st.selectbox("Semi-infinite substrate", ["Si", "Quartz", "Sapphire", "TiO2"])
    line_shape = st.selectbox(
        "Exciton line shape", ["Voigt / Faddeeva (Recommended)", "Lorentz"]
    )

temperature = 298.0
numerical_aperture = 0.0
eps_inf = 12.0

layers = [
    {"material": row["Material"], "thickness_nm": row["Thickness (nm)"],
     "in_reference": False if row["Material"] == "Sample" else row["In reference"],
     "fit": row["Fit"],
     "min_nm": row["Min (nm)"], "max_nm": row["Max (nm)"]}
    for _, row in layer_df.sort_values("Order").iterrows()
]
config = {
    'substrate_type': sub_type, 'temp': temperature, 'layers': layers,
    'numerical_aperture': numerical_aperture, 'contrast_definition': 'relative'
}
config['line_shape'] = 'Voigt' if line_shape.startswith('Voigt') else 'Lorentz'

def make_model_signature(model_config):
    return (
        si_source,
        model_config['substrate_type'],
        round(float(model_config.get('temp', 298.0)), 8),
        round(float(model_config['numerical_aperture']), 8),
        tuple(tuple(sorted(row.items())) for row in model_config['layers']),
        model_config['line_shape'],
    )


model_signature = make_model_signature(config)
if 'fit_results' in st.session_state:
    stored = st.session_state.fit_results
    stored_signature = stored[5] if len(stored) >= 6 else None
    if stored_signature != model_signature:
        del st.session_state.fit_results
        st.sidebar.info("Optical-model settings changed. Run the fit again.")

# Helper to load and process experimental data
def process_experiments(sub_file, samp_file):
    if not sub_file or not samp_file:
        return None, None, "Files missing"
    
    try:
        sub_file.seek(0)
        df_sub = pd.read_csv(sub_file, sep=None, engine='python', header=None)
        data_sub = df_sub.iloc[:, 0:2].apply(pd.to_numeric, errors='coerce').dropna().values
        wl_sub = data_sub[:, 0]
        y_sub = data_sub[:, 1]

        samp_file.seek(0)
        df_samp = pd.read_csv(samp_file, sep=None, engine='python', header=None)
        data_samp = df_samp.iloc[:, 0:2].apply(pd.to_numeric, errors='coerce').dropna().values
        wl_samp = data_samp[:, 0]
        y_samp = data_samp[:, 1]

        # Unit Correction (eV -> nm)
        # Assuming if mean < 100, it's eV.
        if np.mean(wl_sub) < 100:
             wl_sub = HC_EV_NM / wl_sub
             # Sort if needed (inverse flips order)
             idx = np.argsort(wl_sub)
             wl_sub = wl_sub[idx]
             y_sub = y_sub[idx]

        if np.mean(wl_samp) < 100:
             wl_samp = HC_EV_NM / wl_samp
             idx = np.argsort(wl_samp)
             wl_samp = wl_samp[idx]
             y_samp = y_samp[idx]

        # Interpolate Sample onto Substrate Grid
        f_samp = interp1d(wl_samp, y_samp, kind='cubic', bounds_error=False, fill_value=np.nan)
        y_samp_interp = f_samp(wl_sub)
        
        # Mask valid
        mask = (~np.isnan(y_samp_interp)) & (y_sub != 0)
        wl_final = wl_sub[mask]
        sub_final = y_sub[mask]
        samp_final = y_samp_interp[mask]
        
        # Calculate Contrast
        substrate_is_unit_placeholder = (
            np.ptp(sub_final) <= 1e-8 and np.allclose(sub_final, 1.0, atol=1e-8)
            and np.any(samp_final < 0)
        )
        contrast = (
            samp_final if substrate_is_unit_placeholder
            else (samp_final - sub_final) / sub_final
        )
        
        # Return x in eV, y_contrast
        x_ev = HC_EV_NM / wl_final
        
        # Sort by eV (low to high)
        idx_ev = np.argsort(x_ev)
        return x_ev[idx_ev], contrast[idx_ev], None

    except Exception as e:
        return None, None, str(e)

# Calculate Experimental Contrast if files available (Process data EARLIER for dynamic defaults)
x_exp_ev, y_exp_contrast = None, None
if uploaded_sub_file and uploaded_samp_file:
    x_exp_ev, y_exp_contrast, err = process_experiments(uploaded_sub_file, uploaded_samp_file)
    if err:
        st.error(f"Error processing spectra: {err}")

# Sidebar - Fitting
st.sidebar.header("3. Fitting Setup")

# Dynamic ROI Defaults
def_min, def_max = 1.5, 3.0
if x_exp_ev is not None:
    def_min = float(np.min(x_exp_ev))
    def_max = float(np.max(x_exp_ev))

roi_min = st.sidebar.number_input("ROI Min (eV)", value=def_min, format="%.2f")
roi_max = st.sidebar.number_input("ROI Max (eV)", value=def_max, format="%.2f")
fit_method = st.sidebar.selectbox("Method", [
    "Robust LM (Recommended)",
    "Global + Robust LM",
    "1st Derivative + LM",
    "2nd Derivative + LM",
])
e0_margin = 0.02
baseline_order = 3
max_nfev = st.sidebar.select_slider(
    "Optimization budget", options=[3000, 8000, 15000], value=8000
)
config['fit_method'] = fit_method
config['high_precision'] = (fit_method == "Global + Robust LM")



# Main Content
col1, col2 = st.columns([3, 2])

# Data processed earlier. Warning if errors are present not needed here if handled above/inline.
# But process_experiments logic was moved up. The original block down here is now redundant/needs removal.

with col1:
    st.subheader("Excitons Control")
    if 'excitons' not in st.session_state:
        # Use simple column names with emojis for UI
        st.session_state.excitons = pd.DataFrame([
            {"f": 0.5, "🔒f": False, "E0": 1.96, "🔒E0": False, "wL": 0.03, "🔒wL": False, "wG": 0.03, "🔒wG": False}
        ])
    elif 'Gamma' in st.session_state.excitons.columns:
        old = st.session_state.excitons
        old['wL'] = old['Gamma']
        old['🔒wL'] = old.get('🔒Gamma', False)
        old['wG'] = np.maximum(old['Gamma'] / 2, 0.001)
        old['🔒wG'] = False
        st.session_state.excitons = old.drop(columns=['Gamma', '🔒Gamma'], errors='ignore')

    def add_exciton():
        new_row = {"f": 0.1, "🔒f": False, "E0": 2.0, "🔒E0": False, "wL": 0.03, "🔒wL": False, "wG": 0.03, "🔒wG": False}
        st.session_state.excitons = pd.concat([st.session_state.excitons, pd.DataFrame([new_row])], ignore_index=True)

    def clear_excitons():
        st.session_state.excitons = pd.DataFrame(columns=["f", "🔒f", "E0", "🔒E0", "wL", "🔒wL", "wG", "🔒wG"])
        
    def auto_guess_excitons():
        if x_exp_ev is None:
            st.error("Upload both spectra first.")
            return

        try:
            # Use ROI
            mask = (x_exp_ev >= roi_min) & (x_exp_ev <= roi_max)
            y_roi = y_exp_contrast[mask]
            x_roi = x_exp_ev[mask]
            
            if len(y_roi) < 5:
                st.error("Not enough points in ROI for auto-guess.")
                return

            guesses = guess_resonances(x_roi, y_roi)
            
            if len(guesses) > 0:
                new_data = []
                for p_ev, linewidth in guesses:
                     new_data.append({
                        "f": 0.1, "🔒f": False, 
                        "E0": float(p_ev), "🔒E0": False, 
                        "wL": float(linewidth) / 2, "🔒wL": False,
                        "wG": float(linewidth) / 2, "🔒wG": False
                     })
                st.session_state.excitons = pd.DataFrame(new_data)
                st.success(f"Found {len(guesses)} peaks.")
            else:
                st.warning("No distinct peaks found.")
                
        except Exception as e:
            st.error(f"Auto-guess error: {e}")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    col_btn1.button("Add Exciton", on_click=add_exciton)
    col_btn2.button("Clear", on_click=clear_excitons)
    col_btn3.button("Auto Guess", on_click=auto_guess_excitons)
    
    # Configure columns for better UI
    column_config = {
        "f": st.column_config.NumberColumn("f (eV²)", format="%.4f"),
        "E0": st.column_config.NumberColumn("E0 (eV)", format="%.4f"),
        "wL": st.column_config.NumberColumn("wL (eV)", format="%.4f"),
        "wG": st.column_config.NumberColumn("wG (eV)", format="%.4f"),
        "🔒f": st.column_config.CheckboxColumn("🔒", width="small"),
        "🔒E0": st.column_config.CheckboxColumn("🔒", width="small"),
        "🔒wL": st.column_config.CheckboxColumn("🔒", width="small"),
        "🔒wG": st.column_config.CheckboxColumn("🔒", width="small"),
    }

    edited_df = st.data_editor(
        st.session_state.excitons, 
        column_config=column_config,
        column_order=(
            ["f", "🔒f", "E0", "🔒E0", "wL", "🔒wL", "wG", "🔒wG"]
            if config['line_shape'] == 'Voigt'
            else ["f", "🔒f", "E0", "🔒E0", "wL", "🔒wL"]
        ),
        num_rows="dynamic", 
        use_container_width=True
    )
    st.session_state.excitons = edited_df

    if st.button("Start Fitting", type="primary"):
        if x_exp_ev is None:
            st.error("Please upload both Substrate and Sample spectra first!")
        else:
            # Filter range
            mask = (x_exp_ev >= roi_min) & (x_exp_ev <= roi_max)
            # x for fitting (model expects nm for TMM but lorentz takes eV)
            # Convert the energy axis back to nm for the optical model.
            x_fit_nm = HC_EV_NM / x_exp_ev[mask]
            # x_exp_ev is low-to-high, so the corresponding wavelength axis is descending.
            
            # Let's keep data consistent pairs
            x_fit_ev_roi = x_exp_ev[mask]
            y_fit_exp = y_exp_contrast[mask]
            
            # The optical model receives wavelength while the dielectric model uses energy.
            x_fit_input = HC_EV_NM / x_fit_ev_roi
            
            if len(x_fit_input) < 5:
                st.error(f"Not enough data points in ROI ({roi_min}-{roi_max} eV). Loaded {len(x_exp_ev)} points.")
            else:
                try:
                    normalized_layer_stack(config)
                except ValueError as exc:
                    st.error(f"Invalid layer stack: {exc}")
                    st.stop()
                # Construct Initial Params
                p0 = [eps_inf]
                bounds_min = [0]
                bounds_max = [50]
                locked_mask = [False] # eps_inf not locked in UI for now
                
                stride = 4 if config['line_shape'] == 'Voigt' else 3
                for idx, row in edited_df.iterrows():
                    if stride == 4:
                        approximate_fwhm = (
                            0.5346 * row['wL']
                            + np.sqrt(0.2166 * row['wL'] ** 2 + row['wG'] ** 2)
                        )
                        width_upper = min(
                            0.5,
                            max(0.015, 4.0 * approximate_fwhm,
                                1.25 * row['wL'], 1.25 * row['wG']),
                        )
                    else:
                        approximate_fwhm = row['wL']
                        width_upper = min(0.5, max(0.015, 4.0 * approximate_fwhm))
                    local_e0_margin = min(e0_margin, max(0.004, 0.35 * approximate_fwhm))
                    values = [row['f'], row['E0'], row['wL']]
                    lower = [0, max(0.1, row['E0'] - local_e0_margin), 0.0001]
                    upper = [50, row['E0'] + local_e0_margin, width_upper]
                    locks = [row['🔒f'], row['🔒E0'], row['🔒wL']]
                    if stride == 4:
                        values.append(row['wG'])
                        lower.append(0.0001)
                        upper.append(width_upper)
                        locks.append(row['🔒wG'])
                    p0.extend(values)
                    bounds_min.extend(lower)
                    bounds_max.extend(upper)
                    locked_mask.extend(locks)
                
                p0 = np.array(p0)
                bounds = (bounds_min, bounds_max)
                locked_mask = np.array(locked_mask)
                
                # --- FITTING LOGIC ---
                progress_bar = st.progress(0, text="Fitting...")
                
                derivative_order = {
                    "1st Derivative + LM": 1,
                    "2nd Derivative + LM": 2,
                }.get(fit_method, 0)
                if derivative_order:
                    objective_energy = np.tile(x_fit_ev_roi, derivative_order + 1)
                    y_target = np.zeros(objective_energy.size)
                else:
                    objective_energy = x_fit_ev_roi
                    y_target = y_fit_exp

                physical_count = len(p0)
                resonances = resonance_windows_from_parameters(
                    p0, config['line_shape']
                )
                balanced_sigma = resonance_balanced_sigma(
                    x_fit_ev_roi, y_fit_exp, resonances
                )
                structure_parameters = layer_fit_parameters(config)
                for _, value, lower, upper in structure_parameters:
                    p0 = np.append(p0, value)
                    bounds = (
                        np.append(np.asarray(bounds[0], dtype=float), lower),
                        np.append(np.asarray(bounds[1], dtype=float), upper),
                    )
                    locked_mask = np.append(locked_mask, False)

                dielectric_model = (
                    dielectric_func_voigt
                    if config['line_shape'] == 'Voigt'
                    else dielectric_func_lorentz
                )

                def physical_model(params):
                    model_config = config_with_layer_thicknesses(
                        config,
                        params[physical_count:physical_count + len(structure_parameters)],
                    )
                    eps_2d = dielectric_model(x_fit_ev_roi, params[:physical_count])
                    return calculate_contrast_dynamic(x_fit_input, eps_2d, loader, model_config)

                def model(params):
                    contrast = physical_model(params)
                    if derivative_order:
                        return composite_derivative_residual(
                            y_fit_exp,
                            contrast,
                            x_fit_ev_roi,
                            maximum_order=derivative_order,
                            resonances=resonances,
                        )
                    return contrast
                
                try:
                    progress_bar.progress(10, text="Validating layer stack and parameters...")
                    normalized_layer_stack(config)
                    if config['high_precision']:
                        st.info("Running global initialization, robust TRF, and LM refinement...")
                    if derivative_order:
                        progress_bar.progress(25, text="Warm-starting on the original spectrum...")
                        warm_result = fit_spectrum(
                            x_fit_ev_roi, y_fit_exp, physical_model, p0, bounds,
                            locked_mask, baseline_order=baseline_order, robust=True,
                            global_search=config['high_precision'], max_nfev=max_nfev,
                            sigma=balanced_sigma,
                        )
                        p0 = warm_result.params
                    progress_bar.progress(45, text="Optimizing optical and layer parameters...")
                    fit_result = fit_spectrum(
                        objective_energy,
                        y_target,
                        model,
                        p0,
                        bounds,
                        locked_mask,
                        baseline_order=-1 if derivative_order else baseline_order,
                        robust=True,
                        global_search=config['high_precision'] and not derivative_order,
                        sigma=np.ones_like(y_target) if derivative_order else balanced_sigma,
                        max_nfev=max_nfev,
                    )
                    if not fit_result.success:
                        raise RuntimeError(fit_result.message)
                    
                    progress_bar.progress(90, text="Calculating statistics and uncertainty...")
                    st.success("Fitting Complete!")
                    progress_bar.empty()
                    p_full_final = fit_result.params[:physical_count]
                    fitted_config = config_with_layer_thicknesses(
                        config,
                        fit_result.params[physical_count:physical_count + len(structure_parameters)],
                    )
                    st.session_state.structure_layers = pd.DataFrame([
                        [index + 1, layer['material'], layer['thickness_nm'], layer['in_reference'],
                         layer['fit'], layer['min_nm'], layer['max_nm']]
                        for index, layer in enumerate(fitted_config['layers'])
                    ], columns=layer_columns)
                    st.session_state.pop("layer_editor", None)
                    epsilon_roi = dielectric_model(x_fit_ev_roi, p_full_final)
                    physical_roi = calculate_contrast_dynamic(
                        x_fit_input, epsilon_roi, loader, fitted_config
                    )
                    finalize_physical_fit(
                        fit_result,
                        x_fit_ev_roi,
                        y_fit_exp,
                        physical_roi,
                        baseline_order=baseline_order,
                        sigma=balanced_sigma,
                    )
                    fit_result.resonance_diagnostics = resonance_diagnostics(
                        x_fit_ev_roi, y_fit_exp, fit_result.fitted, resonances
                    )
                    
                    # Update Excitons
                    num_exc = (len(p_full_final) - 1) // stride
                    new_data = []
                    for i in range(num_exc):
                        base = 1 + i*stride
                        new_data.append({
                            "f": p_full_final[base],
                            "🔒f": edited_df.iloc[i]['🔒f'],
                            "E0": p_full_final[base+1],
                            "🔒E0": edited_df.iloc[i]['🔒E0'],
                            "wL": p_full_final[base+2],
                            "🔒wL": edited_df.iloc[i]['🔒wL'],
                            "wG": p_full_final[base+3] if stride == 4 else edited_df.iloc[i]['wG'],
                            "🔒wG": edited_df.iloc[i]['🔒wG']
                        })
                    st.session_state.excitons = pd.DataFrame(new_data)
                    st.session_state.fit_results = (
                        x_fit_input,
                        y_fit_exp,
                        p_full_final,
                        fit_result,
                        fitted_config,
                        make_model_signature(fitted_config),
                    )
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Fitting failed: {e}")

    # --- Export Section (Moved to col1) ---
    if 'fit_results' in st.session_state:
        st.divider()
        st.subheader("Export Results")
        x_fit_nm_res, y_fit_exp_res, p_final, fit_result, fitted_config, _ = st.session_state.fit_results

        # Recalculate y_model for consistency
        # Warning: scope. Let's re-calculate basics.
        x_ev_export = HC_EV_NM / x_fit_nm_res
        dielectric_model = dielectric_func_voigt if fitted_config.get('line_shape') == 'Voigt' else dielectric_func_lorentz
        eps_model_e = dielectric_model(x_ev_export, p_final)
        physical_model_e = calculate_contrast_dynamic(
            x_fit_nm_res, eps_model_e, loader, fitted_config
        )
        y_model_e = fit_result.add_baseline(physical_model_e, x_ev_export)

        st.caption(
            f"{fit_result.method} | RMSE={fit_result.rmse:.4g} | "
            f"R2={fit_result.r_squared:.5f} | condition={fit_result.jacobian_condition:.3g} | "
            f"Durbin-Watson={fit_result.durbin_watson:.3f}"
        )
        for index, diagnostic in enumerate(
            getattr(fit_result, "resonance_diagnostics", []), start=1
        ):
            st.caption(
                f"Local peak {index} ({diagnostic['center_ev']:.4f} eV): "
                f"R2={diagnostic['local_r_squared']:.4f}, "
                f"amplitude={diagnostic['amplitude_ratio']:.2f}x"
            )
        fitted_layer_rows = [
            (index, layer) for index, layer in enumerate(fitted_config['layers']) if layer['fit']
        ]
        for index, layer in fitted_layer_rows:
            st.caption(
                f"Fitted layer {index + 1} ({layer['material']}): "
                f"{layer['thickness_nm']:.3f} nm"
            )
        if fit_result.jacobian_condition > 1e10:
            st.warning("The Jacobian is ill-conditioned; some fitted parameters are not independently identifiable.")
        
        # 1. Export Spectrum Data
        export_df = pd.DataFrame({
            "Energy_eV": x_ev_export,
            "Wavelength_nm": x_fit_nm_res,
            "Contrast_Exp": y_fit_exp_res,
            "Contrast_Fit": y_model_e
        })
        csv_spec = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Fitted Spectrum (CSV)",
            csv_spec,
            "fit_spectrum.csv",
            "text/csv",
            key='download-spectrum'
        )
        
        # 2. Export Parameters
        params_export = []
        params_export.append({"Parameter": "Eps_Inf", "Value": p_final[0], "Std_Error": fit_result.standard_errors[0]})
        stride = 4 if fitted_config.get('line_shape') == 'Voigt' else 3
        num_osc = (len(p_final) - 1) // stride
        for i in range(num_osc):
            base = 1 + i*stride
            params_export.append({"Parameter": f"Oscillator {i+1} f", "Value": p_final[base], "Std_Error": fit_result.standard_errors[base]})
            params_export.append({"Parameter": f"Oscillator {i+1} E0", "Value": p_final[base+1], "Std_Error": fit_result.standard_errors[base+1]})
            params_export.append({"Parameter": f"Oscillator {i+1} wL", "Value": p_final[base+2], "Std_Error": fit_result.standard_errors[base+2]})
            if stride == 4:
                params_export.append({"Parameter": f"Oscillator {i+1} wG", "Value": p_final[base+3], "Std_Error": fit_result.standard_errors[base+3]})
        for offset, (index, layer) in enumerate(fitted_layer_rows):
            params_export.append({
                "Parameter": f"Layer {index + 1} {layer['material']} Thickness (nm)",
                "Value": layer['thickness_nm'],
                "Std_Error": fit_result.standard_errors[len(p_final) + offset],
            })
        
        df_params = pd.DataFrame(params_export)
        csv_params = df_params.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Fit Parameters (CSV)",
            csv_params,
            "fit_parameters.csv",
            "text/csv",
            key='download-params'
        )

with col2:
    st.subheader("Plot")
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Contrast")
    
    # Plot experimental data
    if x_exp_ev is not None:
        ax.plot(x_exp_ev, y_exp_contrast, 'k.', label='Experiment', markersize=2)
        
        # Plot Fit
        if 'fit_results' in st.session_state:
            x_fit_nm_res, y_fit_exp_res, p_final, fit_result, fitted_config, _ = st.session_state.fit_results
            # Show Fit
            wl_full = HC_EV_NM / x_exp_ev
            dielectric_model = dielectric_func_voigt if fitted_config.get('line_shape') == 'Voigt' else dielectric_func_lorentz
            eps_model = dielectric_model(x_exp_ev, p_final)
            physical_model = calculate_contrast_dynamic(
                wl_full, eps_model, loader, fitted_config
            )
            y_model = fit_result.add_baseline(physical_model, x_exp_ev)
            ax.plot(x_exp_ev, y_model, 'r-', label='Fit', linewidth=2)

        ax.set_xlim(roi_min, roi_max)
        
        # Auto ylim
        mask = (x_exp_ev >= roi_min) & (x_exp_ev <= roi_max)
        if np.any(mask):
            y_roi = y_exp_contrast[mask]
            ymin, ymax = np.min(y_roi), np.max(y_roi)
            margin = (ymax - ymin) * 0.1 if (ymax!=ymin) else 0.1
            ax.set_ylim(ymin - margin, ymax + margin)
            
    ax.legend()
    st.pyplot(fig)
