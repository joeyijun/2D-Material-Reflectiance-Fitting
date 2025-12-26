
import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution, minimize, basinhopping
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
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
            df = pd.read_csv(path, header=None, names=['lam_um', 'n', 'k'])
            self._process_data(df)
        except Exception as e:
            st.error(f"Error loading substrate file: {e}")

    def _init_si_interpolation_from_file(self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file, header=None, names=['lam_um', 'n', 'k'])
            self._process_data(df)
        except Exception as e:
             st.error(f"Error loading uploaded substrate file: {e}")

    def _process_data(self, df):
        lam_nm = df['lam_um'].values * 1000.0
        n_vals = df['n'].values
        k_vals = df['k'].values
        self.si_n_interp = interp1d(lam_nm, n_vals, kind='cubic', fill_value="extrapolate")
        self.si_k_interp = interp1d(lam_nm, k_vals, kind='cubic', fill_value="extrapolate")

    def get_si_n(self, lam_nm):
        if self.si_n_interp is None: return 1.0 + 0j
        n = self.si_n_interp(lam_nm)
        k = self.si_k_interp(lam_nm)
        return n + 1j * k

    def get_si_n_with_temp(self, lam_nm, temp_k):
        n_base = self.get_si_n(lam_nm)
        lam_um = lam_nm / 1000.0
        delta_n_abs = 0.02514 + 0.00850 / (lam_um**2 - 0.10165)
        delta_k_abs = 270.16 * np.exp(-18.91 * lam_um) + 0.0029
        ratio = (298.0 - temp_k) / (298.0 - 10.0)
        correction_n = ratio * delta_n_abs
        correction_k = ratio * delta_k_abs
        n_real = np.real(n_base) - correction_n
        n_imag = np.imag(n_base) - correction_k
        if np.isscalar(n_imag):
            if n_imag < 0: n_imag = 0
        else:
            n_imag[n_imag < 0] = 0
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
        return np.zeros_like(lam_arr)

# --- Streamlit App ---

st.set_page_config(page_title="2D Material Reflection Fitting", layout="wide")
st.title("2D Material Reflectance Fitting")

# Sidebar - Files
st.sidebar.header("1. Material Data (Target)")
# --- Experimental Data Loading ---
st.sidebar.subheader("Experimental Spectra")
uploaded_sub_file = st.sidebar.file_uploader("Upload Substrate Spectrum (Ref)", type=["txt", "csv"])
uploaded_samp_file = st.sidebar.file_uploader("Upload Sample Spectrum", type=["txt", "csv"])

st.sidebar.subheader("Theoretical Model Data")
# Configurable theoretical substrate file with default
use_custom_si = st.sidebar.checkbox("Upload Custom Si (n,k) File", value=False)
si_file = None
loader = None

if use_custom_si:
    si_file = st.sidebar.file_uploader("Upload Substrate n,k Data (CSV)", type="csv")
    if si_file:
         loader = MaterialLoader(uploaded_si_file=si_file)
else:
    # Use embedded or local default
    local_si_path = "Si_data.csv"
    if os.path.exists(local_si_path):
        st.sidebar.caption(f"Using default Si (n,k): {local_si_path}")
        loader = MaterialLoader(si_filename=local_si_path)
    else:
        st.sidebar.warning("Default substrate n,k file (Si_data.csv) not found.")

if loader is None or loader.si_n_interp is None:
    if use_custom_si and not si_file:
        st.sidebar.warning("Please upload a substrate n,k file.")
    elif not use_custom_si:
         st.sidebar.error("Default data missing. Please upload custom file.")
    st.stop()

# Sidebar - Structure
st.sidebar.header("2. Structure Config")
sub_type = st.sidebar.selectbox("Substrate", ["Si/SiO2", "Quartz", "Sapphire", "TiO2"])

config = {
    'substrate_type': sub_type,
    'temp': 298.0,
    'sio2_thick': 285.0,
    'has_top_hbn': False,
    'top_hbn_thick': 10.0,
    'has_bot_hbn': False,
    'bot_hbn_thick': 10.0,
    'sample_thick': 0.65
}

if sub_type == "Si/SiO2":
    config['sio2_thick'] = st.sidebar.number_input("SiO2 Thickness (nm)", value=285.0)
    config['temp'] = st.sidebar.number_input("Temperature (K)", value=298.0)

config['has_top_hbn'] = st.sidebar.checkbox("Top hBN")
if config['has_top_hbn']:
    config['top_hbn_thick'] = st.sidebar.number_input("Top hBN Thickness (nm)", value=10.0)

config['sample_thick'] = st.sidebar.number_input("Sample Thickness (nm)", value=0.65, format="%.2f")

config['has_bot_hbn'] = st.sidebar.checkbox("Bottom hBN")
if config['has_bot_hbn']:
    config['bot_hbn_thick'] = st.sidebar.number_input("Bottom hBN Thickness (nm)", value=10.0)

eps_inf = st.sidebar.number_input("Background Eps (Inf)", value=12.0)

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
             wl_sub = 1240.0 / wl_sub
             # Sort if needed (inverse flips order)
             idx = np.argsort(wl_sub)
             wl_sub = wl_sub[idx]
             y_sub = y_sub[idx]

        if np.mean(wl_samp) < 100:
             wl_samp = 1240.0 / wl_samp
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
        contrast = (samp_final - sub_final) / (samp_final + sub_final)
        
        # Return x in eV, y_contrast
        x_ev = 1240.0 / wl_final
        
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
fit_method = st.sidebar.selectbox("Method", ["Standard", "High Precision", "Multi-Stage", "MCMC", "Derivative", "2nd Derivative"])
config['fit_method'] = fit_method
config['high_precision'] = (fit_method == "High Precision")



# Main Content
col1, col2 = st.columns([1, 1])

# Data processed earlier. Warning if errors are present not needed here if handled above/inline.
# But process_experiments logic was moved up. The original block down here is now redundant/needs removal.

with col1:
    st.subheader("Excitons Control")
    if 'excitons' not in st.session_state:
        # Use simple column names with emojis for UI
        st.session_state.excitons = pd.DataFrame([
            {"f": 0.5, "🔒f": False, "E0": 1.96, "🔒E0": False, "Gamma": 0.05, "🔒Gamma": False}
        ])

    def add_exciton():
        new_row = {"f": 0.1, "🔒f": False, "E0": 2.0, "🔒E0": False, "Gamma": 0.05, "🔒Gamma": False}
        st.session_state.excitons = pd.concat([st.session_state.excitons, pd.DataFrame([new_row])], ignore_index=True)

    def clear_excitons():
        st.session_state.excitons = pd.DataFrame(columns=["f", "🔒f", "E0", "🔒E0", "Gamma", "🔒Gamma"])
        
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

            peaks, _ = find_peaks(np.abs(y_roi), prominence=0.01, distance=5)
            
            if len(peaks) > 0:
                new_data = []
                for p_idx in peaks:
                     p_ev = x_roi[p_idx]
                     new_data.append({
                        "f": 0.1, "🔒f": False, 
                        "E0": float(p_ev), "🔒E0": False, 
                        "Gamma": 0.05, "🔒Gamma": False
                     })
                st.session_state.excitons = pd.DataFrame(new_data)
                st.success(f"Found {len(peaks)} peaks.")
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
        "Gamma": st.column_config.NumberColumn("Γ (eV)", format="%.4f"),
        "🔒f": st.column_config.CheckboxColumn("🔒", width="small"),
        "🔒E0": st.column_config.CheckboxColumn("🔒", width="small"),
        "🔒Gamma": st.column_config.CheckboxColumn("🔒", width="small"),
    }

    edited_df = st.data_editor(
        st.session_state.excitons, 
        column_config=column_config,
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
            # Wait, our fit wrapper uses 1240/x. So we better pass x in nm to fit_func if that's what we designed.
            # calculate_contrast_dynamic takes nm.
            # Let's convert x_exp_ev back to nm for fitting x-axis
            x_fit_nm = 1240.0 / x_exp_ev[mask]
            # Since x_exp_ev is low->high, x_fit_nm will be high->low. 
            # It's better to sort x_fit_nm for curve_fit stability although not strictly required for point-wise func
            
            # Let's keep data consistent pairs
            x_fit_ev_roi = x_exp_ev[mask]
            y_fit_exp = y_exp_contrast[mask]
            
            # For fitting function:
            # It expects x which is then passed to dielectric_func(1240/x) [so x is nm]
            # and calculate_contrast(x) [so x is nm]
            # So we should pass nm to curve_fit.
            x_fit_input = 1240.0 / x_fit_ev_roi
            
            if len(x_fit_input) < 5:
                st.error(f"Not enough data points in ROI ({roi_min}-{roi_max} eV). Loaded {len(x_exp_ev)} points.")
            else:
                # Construct Initial Params
                p0 = [eps_inf]
                bounds_min = [0]
                bounds_max = [50]
                locked_mask = [False] # eps_inf not locked in UI for now
                
                for idx, row in edited_df.iterrows():
                    p0.extend([row['f'], row['E0'], row['Gamma']])
                    bounds_min.extend([0, 1.0, 0.001])
                    bounds_max.extend([50, 4.0, 1.0])
                    # Update column references to matches new names
                    locked_mask.extend([row['🔒f'], row['🔒E0'], row['🔒Gamma']])
                
                p0 = np.array(p0)
                bounds = (bounds_min, bounds_max)
                locked_mask = np.array(locked_mask)
                
                # --- FITTING LOGIC ---
                progress_bar = st.progress(0, text="Fitting...")
                
                def fit_func_wrapper(x_nm, *p_unlocked):
                    p_full = p0.copy()
                    p_full[~locked_mask] = p_unlocked
                    # dielectric takes eV = 1240/x_nm
                    eps_2d = dielectric_func_lorentz(1240.0/x_nm, p_full)
                    y_model = calculate_contrast_dynamic(x_nm, eps_2d, loader, config)
                    
                    if fit_method == 'Derivative':
                        # Gradient wrt x_nm? Or eV?
                        # Using raw gradient on current axis.
                        return np.gradient(y_model, x_nm) 
                    return y_model

                # Prepare Data Target
                if fit_method == 'Derivative':
                    y_target = np.gradient(y_fit_exp, x_fit_input)
                else:
                    y_target = y_fit_exp

                p_unlocked0 = p0[~locked_mask]
                b_min_u = np.array(bounds[0])[~locked_mask]
                b_max_u = np.array(bounds[1])[~locked_mask]
                
                try:
                    if config['high_precision']:
                        st.info("Running Differential Evolution...")
                        de_bounds = list(zip(b_min_u, b_max_u))
                        def obj(p):
                            return np.sum((y_target - fit_func_wrapper(x_fit_input, *p))**2)
                            
                        res = differential_evolution(obj, de_bounds, maxiter=50, popsize=10, strategy='best1bin')
                        p_unlocked_final = res.x
                    else:
                        popt, pcov = curve_fit(fit_func_wrapper, x_fit_input, y_target, 
                                               p0=p_unlocked0, bounds=(b_min_u, b_max_u), maxfev=5000)
                        p_unlocked_final = popt
                    
                    st.success("Fitting Complete!")
                    progress_bar.empty()
                    p_full_final = p0.copy()
                    p_full_final[~locked_mask] = p_unlocked_final
                    
                    # Update Excitons
                    num_exc = (len(p_full_final) - 1) // 3
                    new_data = []
                    for i in range(num_exc):
                        base = 1 + i*3
                        new_data.append({
                            "f": p_full_final[base],
                            "🔒f": edited_df.iloc[i]['🔒f'],
                            "E0": p_full_final[base+1],
                            "🔒E0": edited_df.iloc[i]['🔒E0'],
                            "Gamma": p_full_final[base+2],
                            "🔒Gamma": edited_df.iloc[i]['🔒Gamma']
                        })
                    st.session_state.excitons = pd.DataFrame(new_data)
                    st.session_state.fit_results = (x_fit_input, y_fit_exp, p_full_final)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Fitting failed: {e}")

    # --- Export Section (Moved to col1) ---
    if 'fit_results' in st.session_state:
        st.divider()
        st.subheader("Export Results")
        x_fit_nm_res, y_fit_exp_res, p_final = st.session_state.fit_results

        # Recalculate y_model for consistency
        # Warning: scope. Let's re-calculate basics.
        x_ev_export = 1240.0 / x_fit_nm_res
        eps_model_e = dielectric_func_lorentz(x_ev_export, p_final)
        y_model_e = calculate_contrast_dynamic(x_fit_nm_res, eps_model_e, loader, config)
        
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
        params_export.append({"Parameter": "Eps_Inf", "Value": p_final[0]})
        num_osc = (len(p_final) - 1) // 3
        for i in range(num_osc):
            base = 1 + i*3
            params_export.append({"Parameter": f"Oscillator {i+1} f", "Value": p_final[base]})
            params_export.append({"Parameter": f"Oscillator {i+1} E0", "Value": p_final[base+1]})
            params_export.append({"Parameter": f"Oscillator {i+1} Gamma", "Value": p_final[base+2]})
        
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
            x_fit_nm_res, y_fit_exp_res, p_final = st.session_state.fit_results
            # Show Fit
            wl_full = 1240.0 / x_exp_ev
            eps_model = dielectric_func_lorentz(x_exp_ev, p_final) # eV input
            y_model = calculate_contrast_dynamic(wl_full, eps_model, loader, config)
            ax.plot(x_exp_ev, y_model, 'r-', label='Fit', linewidth=2)
            
            # --- Export Section REMOVED (Moved to col1) ---
            pass
            
            
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
