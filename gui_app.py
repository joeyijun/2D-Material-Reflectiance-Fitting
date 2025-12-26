import sys
import os
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QDoubleSpinBox, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox)
from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import interp1d
from tmm import coh_tmm

# Import existing logic
# Ensure the directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --- Helper Functions (Re-implemented to avoid dependency issues) ---
from materials import MaterialLoader
def load_spectrum_file(filepath):
    """
    Load spectrum data from CSV/TXT.
    Assumes two columns: Wavelength (nm), Intensity.
    Returns: wavelength_nm (np.array), intensity (np.array)
    """
    try:
        # Try converting to float logic manually or with pandas
        # Pandas is robust for various delimiters
        df = pd.read_csv(filepath, sep=None, engine='python', header=None)
        # Assuming first two columns are what we want
        data = df.iloc[:, 0:2].apply(pd.to_numeric, errors='coerce').dropna().values
        # Sort by wavelength if needed? Typically data is sorted.
        # Ensure numpy array
        wl = data[:, 0]
        inte = data[:, 1]
        return wl, inte
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def dielectric_func_lorentz(E, params):
    """
    Calculate dielectric function using Lorentz oscillators.
    params: [eps_inf, f1, E01, gamma1, f2, E02, gamma2, ...]
    """
    eps_inf = params[0]
    eps = eps_inf + 0j
    
    num_oscillators = (len(params) - 1) // 3
    for i in range(num_oscillators):
        f = params[1 + i*3]
        E0 = params[2 + i*3]
        g = params[3 + i*3]
        
        # Standard Lorentz Model
        # eps = eps_inf + sum( f / (E0^2 - E^2 - i*E*gamma) )
        # Note: If f is oscillator strength (dimensionless), usually multiplied by something.
        # But commonly in fitting tools f is treated as an amplitude parameter (eV^2).
        numerator = f
        denominator = E0**2 - E**2 - 1j * E * g
        eps += numerator / denominator
        
    return eps

# --- Dynamic Contrast Logic ---

def calculate_contrast_dynamic(wavelengths_nm, eps_sample, mat_loader, structure_config):
    """
    Calculate contrast using Vectorized TMM (Abeles Matrix).
    Returns (R_sample - R_ref) / R_ref.
    R_ref assumed to be substrate + oxide (no hBN, no sample).
    """
    lam_arr = np.array(wavelengths_nm)
    n_2d_arr = np.sqrt(eps_sample)
    
    sub_type = structure_config.get('substrate_type', 'Si/SiO2')
    temp_k = structure_config.get('temp', 298.0)
    
    # --- 1. Material Indices (Arrays) ---
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
    
    # hBN
    n_hbn_val = mat_loader.get_hbn_n(lam_arr)
    if np.isscalar(n_hbn_val):
        n_hbn_arr = np.full_like(lam_arr, n_hbn_val, dtype=complex)
    else:
        n_hbn_arr = n_hbn_val
        
    # --- 2. Helper for Stack Reflectance (Vectorized) ---
    def solve_stack(include_sample, include_hbn):
        # Build layers: Top -> Bottom. (Air is Top, Substrate is Bottom)
        # Abeles Matrix convention
        
        # Lists to store (n_array, d_scalar) for Finite Layers only
        stack_n = []
        stack_d = []
        
        # 1. Top hBN
        if include_hbn and structure_config['has_top_hbn']:
            stack_n.append(n_hbn_arr)
            stack_d.append(structure_config['top_hbn_thick'])
            
        # 2. Sample
        if include_sample:
            stack_n.append(n_2d_arr)
            stack_d.append(structure_config['sample_thick'])
            
        # 3. Bottom hBN
        if include_hbn and structure_config['has_bot_hbn']:
            stack_n.append(n_hbn_arr)
            stack_d.append(structure_config['bot_hbn_thick'])
            
        # 4. Oxide (if exists)
        if n_oxide_arr is not None:
             stack_n.append(n_oxide_arr)
             stack_d.append(structure_config['sio2_thick'])
             
        # Calculation
        # Initialize M = Identity (N_wl, 2, 2)
        M = np.broadcast_to(np.eye(2, dtype=complex), (len(lam_arr), 2, 2)).copy()
        
        for n_layer, d_layer in zip(stack_n, stack_d):
            # Phase shift delta = 2pi * n * d / lam
            delta = 2 * np.pi * n_layer * d_layer / lam_arr
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Construct M_layer (N_wl, 2, 2)
            # [[cos, -i/n * sin], [-i*n*sin, cos]]
            m11 = cos_d
            m12 = -1j / n_layer * sin_d
            m21 = -1j * n_layer * sin_d
            m22 = cos_d
            
            # Stack into (N_wl, 2, 2) and Transpose: (2, 2, N_wl) -> (N_wl, 2, 2)
            M_layer = np.array([[m11, m12], [m21, m22]]) 
            M_layer = np.moveaxis(M_layer, 2, 0) 
            
            # Multiply P = P * M_layer
            M = np.matmul(M, M_layer)
            
        # --- Apply Boundary Conditions ---
        # Substrate (Exit medium)
        n_s = n_sub_arr
        n_0 = 1.0 # Air (Entrance)
        
        # Y = H0/E0 = (M21 + M22*ns) / (M11 + M12*ns)
        M11 = M[:, 0, 0]
        M12 = M[:, 0, 1]
        M21 = M[:, 1, 0]
        M22 = M[:, 1, 1]
        
        denom = (M11 + M12 * n_s)
        # Avoid division by zero (unlikely but safe)
        denom[denom == 0] = 1e-10
        
        Y = (M21 + M22 * n_s) / denom
        
        # r = (n0 - Y) / (n0 + Y)
        r = (n_0 - Y) / (n_0 + Y)
        
        return np.abs(r)**2

    # --- 3. Calculate Contrasts ---
    # Case A: Full Structure (Sample + hBNs + Oxide + Sub)
    R_sample = solve_stack(include_sample=True, include_hbn=True)
    
    # Case B: Reference Structure (Substrate + Oxide only)
    # Assumes Reference is measured on bare substrate spot (no hBN, no Sample)
    R_ref = solve_stack(include_sample=False, include_hbn=False)
    
    # Calculate Contrast
    # Avoid div by zero in Ref
    R_ref[R_ref == 0] = 1e-10
    
    contrast = (R_sample - R_ref) / R_ref
    return contrast


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class CopyableTableWidget(QTableWidget):
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy) or (event.key() == Qt.Key.Key_C and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            selection = self.selectedRanges()
            if selection:
                rows = sorted(list(set(range(r.top(), r.bottom() + 1) for r in selection for r in range(r.top(), r.bottom() + 1))))
                columns = sorted(list(set(range(c.left(), c.right() + 1) for c in selection for c in range(c.left(), c.right() + 1))))
                
                # Check for single rectangular selection for simplicity or handle multiple
                # Better: simple copy of visible processed text
                
                # If multiple ranges, just copy everything in the bounding box? 
                # Or just iterate rows in selection order.
                # Simplest robust way:
                
                text_to_copy = ""
                # Get all selected items
                selected_model_indexes = self.selectedIndexes()
                
                if not selected_model_indexes:
                    return

                # Sort specific to row then col
                rows = sorted(list(set(index.row() for index in selected_model_indexes)))
                cols = sorted(list(set(index.column() for index in selected_model_indexes)))
                
                for r in rows:
                    row_data = []
                    for c in cols:
                        item = self.item(r, c)
                        if item:
                            row_data.append(item.text())
                        else:
                            # Try cell widget?
                            widget = self.cellWidget(r, c)
                            if widget and isinstance(widget, QWidget):
                                # Check for checkbox
                                chk = widget.findChild(QCheckBox)
                                if chk:
                                    row_data.append("Checked" if chk.isChecked() else "Unchecked")
                                else:
                                    row_data.append("")
                            else:
                                row_data.append("")
                    text_to_copy += "\t".join(row_data) + "\n"
                
                QApplication.clipboard().setText(text_to_copy)
                return

        super().keyPressEvent(event)

class FittingWorker(QThread):
    finished = pyqtSignal(object, object, object) # results, popt, pcov
    error = pyqtSignal(str)
    aborted = pyqtSignal()

    def __init__(self, x_data, y_data, mat_loader, p0, bounds, structure_config, fit_range, locked_mask):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.mat_loader = mat_loader
        self.p0 = np.array(p0)
        self.bounds = bounds # (min_arr, max_arr)
        self.structure_config = structure_config
        self.fit_range = fit_range # (min_e, max_e)
        self.fit_range = fit_range # (min_e, max_e)
        self.locked_mask = np.array(locked_mask, dtype=bool)
        self.fit_method = structure_config.get('fit_method', 'Standard')
        self.high_precision = structure_config.get('high_precision', False)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            # 1. Filter Data by Range (x_data is nm)
            # fit_range is in eV (min_e, max_e)
            # x_ev = 1240/x_nm
            x_ev_all = 1240.0 / self.x_data
            mask = (x_ev_all >= self.fit_range[0]) & (x_ev_all <= self.fit_range[1])
            
            x_fit = self.x_data[mask]
            x_fit = self.x_data[mask]
            y_fit_exp = self.y_data[mask]
            
            # Derivative Mode Preparation
            if self.fit_method == 'Derivative':
                # Calculate dy/dx
                y_target = np.gradient(y_fit_exp, x_fit)
            elif self.fit_method == '2nd Derivative':
                 # Calculate d2y/dx2
                 y_target = np.gradient(np.gradient(y_fit_exp, x_fit), x_fit)
            else:
                y_target = y_fit_exp
            
            if len(x_fit) < 10:
                raise ValueError("Too few points in range to fit.")
            
            # 2.5 Calculate Weights for Weighted Least Squares
            # Give higher weight to regions with larger gradients (near peaks)
            # This helps the fit focus on peak shapes rather than flat background
            gradient_mag = np.abs(np.gradient(y_fit_exp, x_fit))
            # Normalize and add baseline (1.0) to avoid zero weights
            weights = 1.0 + 3.0 * (gradient_mag / (np.max(gradient_mag) + 1e-10))
            # sigma = 1/weight for curve_fit (lower sigma = higher weight)
            sigma_fit = 1.0 / weights
            
            # 2. Setup Params
            p_unlocked0 = self.p0[~self.locked_mask]
            b_min = np.array(self.bounds[0])[~self.locked_mask]
            b_max = np.array(self.bounds[1])[~self.locked_mask]
            
            # 3. Fit Function Wrapper (supports abort)
            def fit_func_wrapper(x, *p_unlocked):
                if self._abort:
                    raise RuntimeError("Fitting Stopped.")
                
                # Reconstruct full params
                p_full = self.p0.copy()
                p_full[~self.locked_mask] = p_unlocked
                
                # Calculate Dielectric Function (Lorentz model)
                # Note: dielectric_func_lorentz expects eV input
                eps_2d = dielectric_func_lorentz(1240.0/x, p_full)
                
                # Calculate Contrast (Vectorized)
                y_model = calculate_contrast_dynamic(x, eps_2d, self.mat_loader, self.structure_config)
                
                if self.fit_method == 'Derivative':
                    return np.gradient(y_model, x)
                elif self.fit_method == '2nd Derivative':
                    return np.gradient(np.gradient(y_model, x), x)
                else:
                    return y_model

            # 4. Run Optimization
            # Wrapper for Least Squares (residuals) or Curve Fit
            
            popt_unlocked = p_unlocked0
            pcov_unlocked = None

            if len(p_unlocked0) > 0:
                # A. High Precision Global Search (Differential Evolution)
                if self.high_precision:
                    # DE requires bounds for all parameters. curve_fit bounds are (min_arr, max_arr).
                    # DE requires list of (min, max) tuples.
                    de_bounds = list(zip(b_min, b_max))
                    
                    # Objective function for DE: Sum of Squared Residuals
                    def objective_de(p_current):
                        if self._abort:
                            return 1e10 # Penalize to stop
                        
                        y_pred = fit_func_wrapper(x_fit, *p_current)
                        # Weighted residuals
                        resid = np.sum(weights * (y_target - y_pred)**2)
                        return resid

                    # Run DE
                    # strategy='best1bin' is standard. popsize=15 (default).
                    res_de = differential_evolution(objective_de, de_bounds, strategy='best1bin', 
                                                    maxiter=2000, popsize=20, tol=1e-10, atol=1e-12,
                                                    callback=lambda xk, convergence: True if self._abort else None)
                    
                    if self._abort:
                        return
                        
                    # Use DE result as new initial guess
                    p_unlocked0 = res_de.x
                
                # B. Multi-Stage Fitting (3-stage sequential optimization)
                elif self.fit_method == 'MultiStage':
                    from scipy.optimize import minimize
                    
                    # Objective function for minimize
                    def objective_min(p_current):
                        if self._abort:
                            return 1e10
                        y_pred = fit_func_wrapper(x_fit, *p_current)
                        return np.sum(weights * (y_target - y_pred)**2)
                    
                    current_p = p_unlocked0.copy()
                    n_unlocked = len(current_p)
                    
                    # Determine parameter types for staging
                    # params order after eps_inf: [f1, E01, g1, f2, E02, g2, ...]
                    # For staging, we need to know which indices are E0, f, gamma
                    # eps_inf is index 0 in full params, so unlocked indices start from there
                    
                    # Create masks for different param types (relative to unlocked)
                    # This is complex because locked params shift indices
                    # Simplified: fit all together but in stages with tighter bounds
                    
                    bounds_list = list(zip(b_min, b_max))
                    
                    # Stage 1: Coarse fit with relaxed tolerance
                    res1 = minimize(objective_min, current_p, method='L-BFGS-B',
                                   bounds=bounds_list, options={'maxiter': 500, 'ftol': 1e-6})
                    if self._abort: return
                    current_p = res1.x
                    
                    # Stage 2: Medium precision
                    res2 = minimize(objective_min, current_p, method='L-BFGS-B',
                                   bounds=bounds_list, options={'maxiter': 1000, 'ftol': 1e-9})
                    if self._abort: return
                    current_p = res2.x
                    
                    # Stage 3: High precision final fit
                    res3 = minimize(objective_min, current_p, method='L-BFGS-B',
                                   bounds=bounds_list, options={'maxiter': 2000, 'ftol': 1e-12})
                    if self._abort: return
                    
                    p_unlocked0 = res3.x
                
                # C. MCMC / Basin Hopping for robust global search
                elif self.fit_method == 'MCMC':
                    from scipy.optimize import basinhopping, minimize
                    
                    def objective_min(p_current):
                        if self._abort:
                            return 1e10
                        # Clip to bounds
                        p_clipped = np.clip(p_current, b_min, b_max)
                        y_pred = fit_func_wrapper(x_fit, *p_clipped)
                        return np.sum(weights * (y_target - y_pred)**2)
                    
                    bounds_list = list(zip(b_min, b_max))
                    
                    # Custom step function to respect bounds
                    class BoundedStep:
                        def __init__(self, stepsize=0.1):
                            self.stepsize = stepsize
                        def __call__(self, x):
                            s = self.stepsize
                            x_new = x + np.random.uniform(-s, s, x.shape)
                            # Clip to bounds
                            x_new = np.clip(x_new, b_min, b_max)
                            return x_new
                    
                    # Run Basin Hopping (MCMC-like global optimizer)
                    minimizer_kwargs = {
                        'method': 'L-BFGS-B',
                        'bounds': bounds_list,
                        'options': {'ftol': 1e-10, 'maxiter': 500}
                    }
                    
                    # Callback to check abort
                    def bh_callback(x, f, accept):
                        return self._abort  # Return True to stop
                    
                    res_bh = basinhopping(objective_min, p_unlocked0, 
                                         minimizer_kwargs=minimizer_kwargs,
                                         niter=50, T=1.0,
                                         take_step=BoundedStep(0.05),
                                         callback=bh_callback)
                    
                    if self._abort: return
                    p_unlocked0 = res_bh.x
                
                # D. Local Refinement (Curve Fit / Levenberg-Marquardt / TRF)
                # Even after global search, run curve_fit to get covariance and fine-tune
                # Use weighted fitting: sigma = 1/weight (lower sigma = higher importance)
                popt_unlocked, pcov_unlocked = curve_fit(fit_func_wrapper, x_fit, y_target, 
                                                         p0=p_unlocked0, bounds=(b_min, b_max),
                                                         sigma=sigma_fit, absolute_sigma=True,
                                                         ftol=1e-12, xtol=1e-12, gtol=1e-12,
                                                         maxfev=10000)
                # Reconstruct
                popt_full = self.p0.copy()
                popt_full[~self.locked_mask] = popt_unlocked
            else:
                popt_full = self.p0 # Nothing to fit
            
            if self._abort:
                 return

            # 5. Result Calculation
            # Calculate full range result for plotting
            eps_full = dielectric_func_lorentz(1240.0/self.x_data, popt_full)
            y_fit_full = calculate_contrast_dynamic(self.x_data, eps_full, 
                                                    self.mat_loader, self.structure_config)
            
            # 6. Calculate Goodness of Fit (R-squared)
            ss_res = np.sum((y_fit_exp - calculate_contrast_dynamic(x_fit, dielectric_func_lorentz(1240.0/x_fit, popt_full), self.mat_loader, self.structure_config))**2)
            ss_tot = np.sum((y_fit_exp - np.mean(y_fit_exp))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            self.finished.emit(y_fit_full, popt_full, r_squared)
            
        except RuntimeError as re:
            if "Stopped" in str(re):
                self.aborted.emit()
                return
            self.error.emit(str(re))
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Material Reflectance Fitting")
        self.resize(1300, 900)

        # Data placeholders
        self.si_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Si_data.csv")
        self.sub_path = None
        self.samp_path = None
        self.mat_loader = None
        
        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        self.setup_control_panel()
        self.setup_plot_area()
        
        # Initialize Material Loader
        self.init_materials()
        
        # Post-initialization UI updates
        self.on_substrate_changed(0) 

    def init_materials(self):
        if os.path.exists(self.si_data_path):
            try:
                self.mat_loader = MaterialLoader(self.si_data_path)
                self.status_label.setText(f"Substrate Data: Loaded ({os.path.basename(self.si_data_path)})")
                self.status_label.setStyleSheet("color: green")
            except Exception as e:
                self.status_label.setText(f"Substrate Data: Error loading ({str(e)})")
                self.status_label.setStyleSheet("color: red")
        else:
            self.status_label.setText("Substrate Data: Si_data.csv NOT FOUND")
            self.status_label.setStyleSheet("color: red")

    def setup_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # --- File Selection ---
        group_files = QGroupBox("File Selection")
        form_files = QFormLayout()
        
        self.btn_sub = QPushButton("Select Substrate File")
        self.btn_sub.clicked.connect(self.load_substrate)
        # self.lbl_sub = QLabel("None") # Removed
        
        self.btn_samp = QPushButton("Select Sample File")
        self.btn_samp.clicked.connect(self.load_sample)
        # self.lbl_samp = QLabel("None") # Removed
        
        self.status_label = QLabel("Initializing...")
        
        form_files.addRow("Substrate:", self.btn_sub)
        # form_files.addRow("", self.lbl_sub) # Removed
        form_files.addRow("Sample:", self.btn_samp)
        # form_files.addRow("", self.lbl_samp) # Removed
        form_files.addRow("System:", self.status_label)
        
        group_files.setLayout(form_files)
        layout.addWidget(group_files)
        
        # --- Structure & Constants ---
        group_struct = QGroupBox("Structure Configuration")
        form_struct = QFormLayout()
        
        self.combo_sub = QComboBox()
        self.combo_sub.addItems(["Si/SiO2", "Quartz", "Sapphire", "TiO2"])
        self.combo_sub.currentIndexChanged.connect(self.on_substrate_changed)
        
        # --- Helper for Row with Unit ---
        def create_unit_spinbox(spin: QDoubleSpinBox, unit: str):
            container = QWidget()
            lay = QHBoxLayout(container)
            lay.setContentsMargins(0,0,0,0)
            lay.addWidget(spin)
            if unit:
                lay.addWidget(QLabel(unit))
            return container

        # SiO2
        self.lbl_sio2 = QLabel("SiO2 Thickness:")
        self.spin_sio2 = QDoubleSpinBox()
        self.spin_sio2.setRange(0, 1000)
        self.spin_sio2.setValue(285.0)
        self.container_sio2 = create_unit_spinbox(self.spin_sio2, "nm")
        
        # Temp
        self.lbl_temp = QLabel("Temperature:")
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0, 1000)
        self.spin_temp.setValue(298.0)
        self.container_temp = create_unit_spinbox(self.spin_temp, "K")
        
        # Top hBN
        self.chk_top_hbn = QCheckBox("Top hBN")
        self.spin_top_hbn = QDoubleSpinBox()
        self.spin_top_hbn.setRange(0, 200)
        self.spin_top_hbn.setValue(10.0)
        container_top = create_unit_spinbox(self.spin_top_hbn, "nm")
        
        # Bottom hBN
        self.chk_bot_hbn = QCheckBox("Bottom hBN")
        self.spin_bot_hbn = QDoubleSpinBox()
        self.spin_bot_hbn.setRange(0, 200)
        self.spin_bot_hbn.setValue(10.0)
        container_bot = create_unit_spinbox(self.spin_bot_hbn, "nm")
        
        # Sample Thickness
        self.spin_2d = QDoubleSpinBox()
        self.spin_2d.setRange(0, 100)
        self.spin_2d.setValue(0.65)
        self.spin_2d.setSingleStep(0.01)
        container_2d = create_unit_spinbox(self.spin_2d, "nm")

        # Eps inf
        self.spin_eps_inf = QDoubleSpinBox()
        self.spin_eps_inf.setRange(0, 50)
        self.spin_eps_inf.setValue(12.0)
        
        form_struct.addRow("Substrate:", self.combo_sub)
        form_struct.addRow(self.lbl_sio2, self.container_sio2)
        form_struct.addRow(self.lbl_temp, self.container_temp)
        form_struct.addRow(self.chk_top_hbn, container_top)
        form_struct.addRow("Sample Thickness:", container_2d)
        form_struct.addRow(self.chk_bot_hbn, container_bot)
        form_struct.addRow("Background Eps:", self.spin_eps_inf)
        
        group_struct.setLayout(form_struct)
        layout.addWidget(group_struct)

        # --- Fitting Optimization ---
        group_opt = QGroupBox("Fitting Optimization")
        layout_opt = QHBoxLayout()
        
        self.spin_range_min = QDoubleSpinBox()
        self.spin_range_min.setRange(0, 10)
        self.spin_range_min.setValue(1.5)
        
        self.spin_range_max = QDoubleSpinBox()
        self.spin_range_max.setRange(0, 10)
        self.spin_range_max.setValue(3.0)
        
        layout_opt.addWidget(QLabel("ROI (eV):"))
        layout_opt.addWidget(QLabel("Min"))
        layout_opt.addWidget(self.spin_range_min)
        layout_opt.addWidget(QLabel("Max"))
        layout_opt.addWidget(self.spin_range_max)
        
        self.combo_fit_method = QComboBox()
        self.combo_fit_method.addItems(["Standard", "High Precision", "Multi-Stage", "MCMC", "1st Derivative", "2nd Derivative"])
        layout_opt.addWidget(QLabel("Method:"))
        layout_opt.addWidget(self.combo_fit_method)
        
        group_opt.setLayout(layout_opt)
        layout.addWidget(group_opt)
        
        # --- Excitons ---
        group_excitons = QGroupBox("Excitons (Lorentz Model)")
        vbox_exc = QVBoxLayout()
        
        # Update Table Columns: f, Lock, E0, Lock, G, Lock (Interleaved)
        self.table_exc = CopyableTableWidget(0, 6)
        self.table_exc.setHorizontalHeaderLabels(["f", "🔒", "E0", "🔒", "Gamma", "🔒"])
        self.table_exc.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # Resize lock columns to be small
        for c in [1,3,5]:
            self.table_exc.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            self.table_exc.setColumnWidth(c, 30)
        
        hbox_exc_btns = QHBoxLayout()
        btn_add = QPushButton("Add Exciton")
        btn_add.clicked.connect(self.add_exciton)
        btn_auto = QPushButton("Auto-Guess")
        btn_auto.clicked.connect(self.auto_guess_excitons)
        btn_rem = QPushButton("Remove Selected")
        btn_rem.clicked.connect(self.remove_exciton)
        
        hbox_exc_btns.addWidget(btn_add)
        hbox_exc_btns.addWidget(btn_auto)
        hbox_exc_btns.addWidget(btn_rem)
        
        vbox_exc.addWidget(self.table_exc)
        vbox_exc.addLayout(hbox_exc_btns)
        group_excitons.setLayout(vbox_exc)
        layout.addWidget(group_excitons)

        
        # Pre-populate one exciton
        self.add_exciton_values(0.5, 1.96, 0.05)

        # --- Actions ---
        self.btn_plot = QPushButton("Plot Data Preview")
        self.btn_plot.clicked.connect(self.preview_data)

        self.btn_fit = QPushButton("Start Fitting")
        self.btn_fit.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_fit.clicked.connect(self.start_fitting)
        
        self.btn_save_img = QPushButton("Save Plot")
        self.btn_save_img.clicked.connect(self.save_plot_image)
        
        self.btn_export_data = QPushButton("Export Data")
        self.btn_export_data.clicked.connect(self.export_data)
        
        hbox_actions_top = QHBoxLayout()
        hbox_actions_top.addWidget(self.btn_plot)
        hbox_actions_top.addWidget(self.btn_save_img)
        hbox_actions_top.addWidget(self.btn_export_data)
        
        layout.addLayout(hbox_actions_top)
        layout.addWidget(self.btn_fit)
        
        layout.addStretch()
        self.main_layout.addWidget(panel, 1)

    def setup_plot_area(self):
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.main_layout.addWidget(self.canvas, 2)
        # Connect mouse event for cursor
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        
        # Annotation for cursor
        self.cursor_annot = None
        self.cursor_line_v = None
        self.cursor_line_h = None
        
        # Data storage for export
        self.last_x_ev = None
        self.last_y_exp = None
        self.last_y_fit = None

    def on_substrate_changed(self, index):
        sub_type = self.combo_sub.currentText()
        is_si = (sub_type == "Si/SiO2")
        
        self.lbl_sio2.setVisible(is_si)
        self.container_sio2.setVisible(is_si)
        
        self.lbl_temp.setVisible(is_si)
        self.container_temp.setVisible(is_si)

    def on_mouse_move(self, event):
        if not event.inaxes: 
            if self.cursor_line_v: self.cursor_line_v.set_visible(False)
            if self.cursor_line_h: self.cursor_line_h.set_visible(False)
            if self.cursor_annot: self.cursor_annot.set_visible(False)
            self.canvas.draw_idle()
            return
            
        x, y = event.xdata, event.ydata
        
        # Update or create vertical/horizontal lines
        if self.cursor_line_v is None:
            self.cursor_line_v = self.canvas.axes.axvline(x, color='gray', linestyle=':', alpha=0.5)
            self.cursor_line_h = self.canvas.axes.axhline(y, color='gray', linestyle=':', alpha=0.5)
        else:
            self.cursor_line_v.set_xdata([x, x])
            self.cursor_line_h.set_ydata([y, y])
            self.cursor_line_v.set_visible(True)
            self.cursor_line_h.set_visible(True)
            
        # Using annotation is better
        if self.cursor_annot is None:
            self.cursor_annot = self.canvas.axes.annotate(
                f"E={x:.3f} eV\nC={y:.3f}", 
                xy=(x, y), xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                arrowprops=dict(arrowstyle="->")
            )
        else:
            self.cursor_annot.xy = (x, y)
            self.cursor_annot.set_text(f"E={x:.3f} eV\nC={y:.3f}")
            self.cursor_annot.set_visible(True)
            
        self.canvas.draw_idle()

    def add_exciton(self):
        self.add_exciton_values(0.1, 2.0, 0.05)

    def auto_guess_excitons(self):
        """Automatically find peaks in experimental contrast"""
        x_ev, y_contrast, err = self.get_processed_data()
        if err:
            QMessageBox.warning(self, "Error", err)
            return
            
        if x_ev is None or len(x_ev) == 0:
            return

        from scipy.signal import find_peaks
        # Use abs(contrast) to capture dips or peaks
        # Smooth slightly?
        # Simple detection first
        peaks, _ = find_peaks(np.abs(y_contrast), prominence=0.01, distance=5)
        
        if len(peaks) == 0:
            QMessageBox.information(self, "Info", "No significant peaks found.")
            return

        # Ask user confirmation? No, just populate (easy to remove)
        reply = QMessageBox.question(self, "Auto-Guess", f"Found {len(peaks)} peaks. Replace existing excitons?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.table_exc.setRowCount(0)
            for p in peaks:
                val_E = x_ev[p]
                # Default guess: f=0.1, gamma=0.05. Smart guess for gamma (FWHM) is harder without peak analysis.
                self.add_exciton_values(0.1, round(val_E, 4), 0.05)

    def add_exciton_values(self, f, E0, gamma, locks=(False, False, False)):
        row = self.table_exc.rowCount()
        self.table_exc.insertRow(row)
        # Values in 0, 2, 4
        self.table_exc.setItem(row, 0, QTableWidgetItem(str(f)))
        self.table_exc.setItem(row, 2, QTableWidgetItem(str(E0)))
        self.table_exc.setItem(row, 4, QTableWidgetItem(str(gamma)))
        
        # Add Checkboxes for locks in 1, 3, 5
        for i, col in enumerate([1, 3, 5]):
           # Create a widget to center the checkbox
           cell_widget = QWidget()
           chk = QCheckBox()
           chk.setChecked(locks[i])
           # Remove margin hack, let layout handle centering
           layout = QHBoxLayout(cell_widget)
           layout.addWidget(chk)
           layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
           layout.setContentsMargins(0,0,0,0)
           self.table_exc.setCellWidget(row, col, cell_widget)

    def _get_lock_state(self, row, col_index):
        # col_index passed directly (1, 3, 5)
        widget = self.table_exc.cellWidget(row, col_index)
        if widget:
            # Find checkbox child
            chk = widget.findChild(QCheckBox)
            if chk:
                return chk.isChecked()
        return False

    def remove_exciton(self):
        current_row = self.table_exc.currentRow()
        if current_row >= 0:
            self.table_exc.removeRow(current_row)

    def load_substrate(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Substrate File", "", "Data Files (*.csv *.txt *.dat);;All Files (*)")
        if path:
            self.sub_path = path
            self.btn_sub.setText(f"Substrate: {os.path.basename(path)}")
    
    def load_sample(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Sample File", "", "Data Files (*.csv *.txt *.dat);;All Files (*)")
        if path:
            self.samp_path = path
            self.btn_samp.setText(f"Sample: {os.path.basename(path)}")
            
    def get_processed_data(self):
        """Helper to load and process data returning (x_ev, y_contrast)"""
        if not self.sub_path or not self.samp_path:
            return None, None, "Please select both Substrate and Sample files."
        
        # 1. Load Data
        x_sub_raw, y_sub_raw = load_spectrum_file(self.sub_path)
        x_samp_raw, y_samp_raw = load_spectrum_file(self.samp_path)
        
        if x_sub_raw is None or x_samp_raw is None:
             return None, None, "Failed to load spectrum files. Check format."

        # Helper to standardize to nm
        def to_nm(x, y):
             if np.mean(x) < 100: # Assume eV input
                 wl = 1240.0 / x
                 # Sort because inverse relationship flips order
                 idx = np.argsort(wl)
                 return wl[idx], y[idx]
             else:
                 return x, y

        wl_sub, int_sub = to_nm(x_sub_raw, y_sub_raw)
        wl_samp, int_samp = to_nm(x_samp_raw, y_samp_raw)

        # 2. Process Data (Interpolation & Contrast)
        try:
            # Interpolate Sample onto Substrate Grid
            f_samp = interp1d(wl_samp, int_samp, kind='cubic', bounds_error=False, fill_value=np.nan)
            int_samp_interp = f_samp(wl_sub)
            
            # Mask valid data (valid interpolation and non-zero denominator)
            mask = (~np.isnan(int_samp_interp)) & (int_sub != 0)
            
            wl_final = wl_sub[mask]
            sub_final = int_sub[mask]
            samp_final = int_samp_interp[mask]
            
            # Calculate Contrast
            y_contrast_exp = (samp_final - sub_final) / (samp_final + sub_final)
            
            # Convert back to eV for output
            x_ev = 1240.0 / wl_final
            
            # Sort by Energy (Low -> High) for plotting convenience
            idx_ev = np.argsort(x_ev)
            x_ev = x_ev[idx_ev]
            y_contrast_exp = y_contrast_exp[idx_ev]
            
            # Auto-Range Update (Only if new files are loaded)
            current_paths = (self.sub_path, self.samp_path)
            if x_ev is not None and len(x_ev) > 0:
                # Use getattr to avoid init issues if called early, though unlikely
                last = getattr(self, 'last_loaded_paths', None)
                if current_paths != last:
                    self.spin_range_min.setValue(np.min(x_ev))
                    self.spin_range_max.setValue(np.max(x_ev))
                    self.last_loaded_paths = current_paths
                
            return x_ev, y_contrast_exp, None
            
        except Exception as e:
            return None, None, str(e)


    def preview_data(self):
        x_ev, y_contrast, err = self.get_processed_data()
        if err:
            QMessageBox.warning(self, "Error", err)
            return

        self.canvas.axes.cla()
        self.cursor_line_v = None
        self.cursor_line_h = None
        self.cursor_annot = None
        
        self.canvas.axes.plot(x_ev, y_contrast, 'o', markersize=3, label='Experiment', alpha=0.5)
        
        # Draw ROI lines (Vertical lines showing fit range)
        min_e = self.spin_range_min.value()
        max_e = self.spin_range_max.value()
        self.canvas.axes.axvline(min_e, color='g', linestyle='--', alpha=0.5, label='Min E')
        self.canvas.axes.axvline(max_e, color='g', linestyle='--', alpha=0.5, label='Max E')
        
        self.canvas.axes.set_title(f"Contrast Spectrum (Preview)\nSubstrate={self.combo_sub.currentText()}")
        self.canvas.axes.set_xlabel("Energy (eV)")
        self.canvas.axes.set_ylabel("Contrast")
        self.canvas.axes.legend()
        self.canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()
        
        # Store for export
        self.last_x_ev = x_ev
        self.last_y_exp = y_contrast
        self.last_y_fit = None

    def save_plot_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "", "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf)")
        if path:
            try:
                # Hide cursor elements before saving
                if self.cursor_line_v: self.cursor_line_v.set_visible(False)
                if self.cursor_line_h: self.cursor_line_h.set_visible(False)
                if self.cursor_annot: self.cursor_annot.set_visible(False)
                
                self.canvas.fig.savefig(path, dpi=600)
                
                QMessageBox.information(self, "Success", f"Plot saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot: {e}")

    def export_data(self):
        if self.last_x_ev is None or self.last_y_exp is None:
            QMessageBox.warning(self, "Warning", "No data to export. Please plot or fit first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV File (*.csv);;Text File (*.txt)")
        if path:
            try:
                data = {'Energy_eV': self.last_x_ev, 'Contrast_Exp': self.last_y_exp}
                if self.last_y_fit is not None:
                    data['Contrast_Fit'] = self.last_y_fit
                
                df = pd.DataFrame(data)
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Success", f"Data exported to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")

    def start_fitting(self):
        # Toggle Start/Stop
        if self.btn_fit.text() == "Stop Fitting":
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.abort()
                self.btn_fit.setText("Stopping...")
                self.btn_fit.setEnabled(False) # Disable while stopping
            return # Exit if we initiated a stop

        if not self.sub_path or not self.samp_path:
            QMessageBox.warning(self, "Error", "Please select both Substrate and Sample files.")
            return
        
        if not self.mat_loader:
            QMessageBox.critical(self, "Error", "Material Loader not initialized (Si_data.csv missing).")
            return
            
        x_ev, y_contrast_exp, err = self.get_processed_data()
        if err:
            QMessageBox.warning(self, "Error", err)
            return
            
        # Recover wavelength for fitting because TMM needs nm
        x_exp_nm = 1240.0 / x_ev
            
        # UI State -> Fitting
        self.btn_fit.setText("Stop Fitting")
        self.btn_fit.setStyleSheet("background-color: #ffaaaa; color: black;") # Red tint
        self.btn_fit.setEnabled(True) # Keep enabled so user can click to stop
    
        try:
            # 3. Prepare Config
            
            # Map Dropdown to Config
            method_str = self.combo_fit_method.currentText()
            fit_method = 'Standard'
            high_prec = False
            
            if method_str == "High Precision":
                fit_method = 'Standard'
                high_prec = True
            elif method_str == "Multi-Stage":
                fit_method = 'MultiStage'
            elif method_str == "MCMC":
                fit_method = 'MCMC'
            elif method_str == "1st Derivative":
                fit_method = 'Derivative'
            elif method_str == "2nd Derivative":
                fit_method = '2nd Derivative'
            
            struct_config = {
                'substrate_type': self.combo_sub.currentText(),
                'sio2_thick': self.spin_sio2.value(),
                'temp': self.spin_temp.value(),
                'sample_thick': self.spin_2d.value(),
                'has_top_hbn': self.chk_top_hbn.isChecked(),
                'top_hbn_thick': self.spin_top_hbn.value(),
                'has_bot_hbn': self.chk_bot_hbn.isChecked(),
                'bot_hbn_thick': self.spin_bot_hbn.value(),
                'fit_method': fit_method,
                'high_precision': high_prec
            }

            # 4. Prepare Params & Locks
            p0 = []
            locked_mask = []
            
            # Background Eps
            p0.append(self.spin_eps_inf.value())
            locked_mask.append(False) # Never lock Eps Inf for now (user didn't ask)
            
            # Oscillators
            # Oscillators
            rows = self.table_exc.rowCount()
            for r in range(rows):
                try:
                    # Values at 0, 2, 4
                    f = float(self.table_exc.item(r, 0).text())
                    E0 = float(self.table_exc.item(r, 2).text())
                    g = float(self.table_exc.item(r, 4).text())
                    p0.extend([f, E0, g])
                    
                    # Check Locks at 1, 3, 5
                    lock_f = self._get_lock_state(r, 1)
                    lock_E0 = self._get_lock_state(r, 3)
                    lock_g = self._get_lock_state(r, 5)
                    
                    locked_mask.extend([lock_f, lock_E0, lock_g])
                    
                except ValueError:
                     QMessageBox.warning(self, "Warning", f"Invalid number in Exciton Row {r+1}. Skipping.")
            
            if len(p0) < 4:
                QMessageBox.warning(self, "Warning", "Need at least one exciton to fit.")
                self.btn_fit.setEnabled(True)
                self.btn_fit.setText("Start Fitting")
                return

            # 4b. Construct Bounds
            # p0 structure: [eps_inf, f1, E01, g1, f2, E02, g2, ...]
            # Bounds need to match unlocked parameters only? 
            # No, scipy.optimize.curve_fit expects bounds for ALL parameters if p0 is full, or subset if p0 is subset.
            # Here we pass full bounds (min, max) to worker, and worker slices them.
            
            bounds_min = []
            bounds_max = []
            
            # Eps Inf
            bounds_min.append(1.0)
            bounds_max.append(50.0)
            
            # Oscillators
            num_osc = (len(p0) - 1) // 3
            
            # Global Bounds Logic:
            # Prioritize user-specified exciton peak positions by using TIGHT bounds around E0.
            # This prevents fitting from "wandering" to unintended peak positions.
            # f: [0, 20]
            # E0: [User_E0 - 0.15, User_E0 + 0.15] -> Stay close to user's explicit setting
            # Gamma: [0.0001, 0.5]
            
            # Extract user E0 values for per-oscillator bounds
            user_E0_values = []
            for r in range(rows):
                try:
                    E0 = float(self.table_exc.item(r, 2).text())
                    user_E0_values.append(E0)
                except:
                    user_E0_values.append(2.0)  # Fallback
            
            # Tight E0 margin (±0.15 eV) to prioritize user settings
            E0_margin = 0.15
            
            for i in range(num_osc):
                # f: 0 to 20
                bounds_min.append(0.0)
                bounds_max.append(20.0)
                
                # E0: Tight bounds around user-specified value
                user_E0 = user_E0_values[i] if i < len(user_E0_values) else 2.0
                bounds_min.append(max(0.1, user_E0 - E0_margin))
                bounds_max.append(user_E0 + E0_margin)
                
                # Gamma: 0.0001 to 0.5 eV (Prevent singular or too broad)
                bounds_min.append(0.0001)
                bounds_max.append(0.5)
            
            # 5. Fit Range
            fit_range = (self.spin_range_min.value(), self.spin_range_max.value())
            
            # 6. Start Thread
            self.worker = FittingWorker(x_exp_nm, y_contrast_exp, self.mat_loader, p0, (bounds_min, bounds_max),
                                        struct_config, fit_range, locked_mask)
            self.worker.finished.connect(lambda y, p, r2: self.on_fit_finished(x_exp_nm, y_contrast_exp, y, p, r2, locked_mask))
            self.worker.error.connect(self.on_fit_error)
            self.worker.aborted.connect(self.on_fit_aborted)
            self.worker.start()

        except Exception as e:
            self.on_fit_error(str(e))
            
    def on_fit_aborted(self):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("Start Fitting")
        self.btn_fit.setStyleSheet("")
        self.status_label.setText("Fitting stopped by user.")

    def on_fit_finished(self, x_data, y_data, y_fit, popt, r_squared, locked_mask):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("Start Fitting")
        self.btn_fit.setStyleSheet("") # Reset Color
        
        # Convert Wavelength (nm) to Energy (eV)
        x_ev = 1240.0 / x_data
        
        # Update Plot
        self.canvas.axes.cla()
        # Reset cursor handles
        self.cursor_line_v = None
        self.cursor_line_h = None
        self.cursor_annot = None
        
        # Plot vs Energy
        self.canvas.axes.plot(x_ev, y_data, 'o', markersize=3, label='Experiment', alpha=0.5)
        self.canvas.axes.plot(x_ev, y_fit, 'r-', linewidth=2, label='Fit Model')
        self.canvas.axes.set_title(f"Contrast Fit ({self.combo_sub.currentText()})")
        self.canvas.axes.set_xlabel("Energy (eV)")
        self.canvas.axes.set_ylabel("Contrast")
        self.canvas.axes.legend()
        self.canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()
        # Invert x-axis? Typically high energy is on the right, which is small wavelength.
        # 1240/x is monotonic decreasing. matplotlib will plot it correctly (e.g. 2.5 -> 1.5).
        # Usually spectra are plotted from low eV to high eV (left to right).
        # If x_data was sorted (450 -> 750), then x_ev is (2.75 -> 1.65).
        # Matplotlib will plot 2.75 on right if values are decreasing? No, it plots index order if just plot(x,y).
        # Store for export
        self.last_x_ev = x_ev
        self.last_y_exp = y_data
        self.last_y_fit = y_fit
        
        # Update UI with results
        # popt[0] is eps_inf
        self.spin_eps_inf.setValue(popt[0])
        
        # Update Table
        self.table_exc.setRowCount(0) # Clear
        num_oscillators = (len(popt) - 1) // 3
        for i in range(num_oscillators):
            f = popt[1 + i*3]
            E0 = popt[2 + i*3]
            gamma = popt[3 + i*3]
            
            # Restore locks
            # Indices in locked_mask (which includes eps at 0): 1+i*3, 2+i*3, 3+i*3
            l_f = locked_mask[1 + i*3]
            l_E0 = locked_mask[2 + i*3]
            l_g = locked_mask[3 + i*3]
            
            self.add_exciton_values(round(f, 4), round(E0, 4), round(gamma, 4), locks=(l_f, l_E0, l_g))
        
        msg = f"Fitting completed successfully!\nR-squared: {r_squared:.4f}"
        if r_squared < 0.9:
            msg += "\n\nWarning: Low fit quality. Please check initial guesses or range."
            QMessageBox.warning(self, "Fitting Result", msg)
        else:
            QMessageBox.information(self, "Success", msg)

    def on_fit_error(self, err_msg):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("Start Fitting")
        self.btn_fit.setStyleSheet("") # Reset Color
        QMessageBox.critical(self, "Fitting Error", err_msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
