import sys
import os
import re
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QDoubleSpinBox, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox)
from PyQt6.QtWidgets import QSpinBox, QProgressBar, QScrollArea
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from fitting_engine import (
    FitCancelled,
    composite_derivative_residual,
    finalize_physical_fit,
    fit_spectrum,
    guess_resonances,
    resonance_balanced_sigma,
    resonance_diagnostics,
    resonance_windows_from_parameters,
)
from optical_model import (
    HC_EV_NM,
    LAYER_MATERIALS,
    calculate_contrast_dynamic as calculate_contrast_core,
    config_with_layer_thicknesses,
    dielectric_func_lorentz as dielectric_func_lorentz_core,
    dielectric_func_voigt,
    layer_fit_parameters,
    spectral_derivative,
)
from scipy.interpolate import interp1d

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


# Keep the UI module's public function names while using the tested core model.
dielectric_func_lorentz = dielectric_func_lorentz_core
calculate_contrast_dynamic = calculate_contrast_core


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
    progress = pyqtSignal(int, str)

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
            self.progress.emit(5, "Preparing model")
            energy_all = HC_EV_NM / self.x_data
            mask = (energy_all >= self.fit_range[0]) & (energy_all <= self.fit_range[1])
            wavelengths = self.x_data[mask]
            energy = energy_all[mask]
            measured = self.y_data[mask]
            if wavelengths.size < 10:
                raise ValueError("Too few points in range to fit.")

            derivative_order = {
                'Derivative': 1,
                '2nd Derivative': 2,
            }.get(self.fit_method, 0)
            baseline_order = int(self.structure_config.get('baseline_order', 3))
            if derivative_order:
                objective_energy = np.tile(energy, derivative_order + 1)
                target = np.zeros(objective_energy.size)
            else:
                objective_energy = energy
                target = measured

            physical_count = self.p0.size
            line_shape = self.structure_config.get('line_shape', 'Lorentz')
            resonances = resonance_windows_from_parameters(self.p0, line_shape)
            balanced_sigma = resonance_balanced_sigma(energy, measured, resonances)
            structure_parameters = layer_fit_parameters(self.structure_config)
            fit_initial = self.p0.copy()
            fit_lower = np.asarray(self.bounds[0], dtype=float)
            fit_upper = np.asarray(self.bounds[1], dtype=float)
            fit_locked = self.locked_mask.copy()
            for _, value, lower, upper in structure_parameters:
                fit_initial = np.append(fit_initial, value)
                fit_lower = np.append(fit_lower, lower)
                fit_upper = np.append(fit_upper, upper)
                fit_locked = np.append(fit_locked, False)

            dielectric_model = (
                dielectric_func_voigt
                if self.structure_config.get('line_shape') == 'Voigt'
                else dielectric_func_lorentz
            )

            def physical_model(params):
                model_config = config_with_layer_thicknesses(
                    self.structure_config,
                    params[physical_count:physical_count + len(structure_parameters)],
                )
                epsilon = dielectric_model(energy, params[:physical_count])
                return calculate_contrast_dynamic(
                    wavelengths, epsilon, self.mat_loader, model_config
                )

            def model(params):
                contrast = physical_model(params)
                if derivative_order:
                    return composite_derivative_residual(
                        measured, contrast, energy,
                        maximum_order=derivative_order,
                        resonances=resonances,
                    )
                return contrast

            if derivative_order:
                self.progress.emit(20, "Warm-starting on original spectrum")
                warm_result = fit_spectrum(
                    energy, measured, physical_model, fit_initial,
                    (fit_lower, fit_upper), fit_locked, baseline_order=baseline_order,
                    robust=True, global_search=self.high_precision,
                    sigma=balanced_sigma,
                    cancel_check=lambda: self._abort,
                    max_nfev=int(self.structure_config.get('max_nfev', 8000)),
                )
                fit_initial = warm_result.params

            self.progress.emit(45, "Optimizing optical and structure parameters")
            result = fit_spectrum(
                objective_energy,
                target,
                model,
                fit_initial,
                (fit_lower, fit_upper),
                fit_locked,
                baseline_order=-1 if derivative_order else baseline_order,
                robust=True,
                global_search=self.high_precision and not derivative_order,
                sigma=np.ones_like(target) if derivative_order else balanced_sigma,
                cancel_check=lambda: self._abort,
                max_nfev=int(self.structure_config.get('max_nfev', 10000)),
            )
            if not result.success:
                raise RuntimeError(result.message)
            self.fit_result = result
            fitted_config = config_with_layer_thicknesses(
                self.structure_config,
                result.params[physical_count:physical_count + len(structure_parameters)],
            )
            self.fitted_layers = fitted_config['layers']

            physical_params = result.params[:physical_count]
            epsilon_roi = dielectric_model(energy, physical_params)
            physical_roi = calculate_contrast_dynamic(
                wavelengths, epsilon_roi, self.mat_loader, fitted_config
            )
            self.progress.emit(90, "Calculating fit statistics")
            finalize_physical_fit(
                result, energy, measured, physical_roi,
                baseline_order=baseline_order, sigma=balanced_sigma,
            )
            result.resonance_diagnostics = resonance_diagnostics(
                energy, measured, result.fitted, resonances
            )
            epsilon_full = dielectric_model(energy_all, physical_params)
            physical_full = calculate_contrast_dynamic(
                self.x_data, epsilon_full, self.mat_loader, fitted_config
            )
            fitted_full = result.add_baseline(physical_full, energy_all)
            self.progress.emit(100, "Fit complete")
            self.finished.emit(fitted_full, physical_params, result.r_squared)
        except FitCancelled:
            self.aborted.emit()
        except Exception as exc:
            self.error.emit(str(exc))


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
        struct_layout = QVBoxLayout()
        form_struct = QFormLayout()
        
        self.combo_sub = QComboBox()
        self.combo_sub.addItems(["Si", "Quartz", "Sapphire", "TiO2"])
        self.combo_sub.currentIndexChanged.connect(self.on_substrate_changed)

        self.combo_si_data = QComboBox()
        for filename in ("Si_data.csv", "Schinke.csv", "Green-2008.csv"):
            if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)):
                self.combo_si_data.addItem(filename)
        self.combo_si_data.currentTextChanged.connect(self.on_si_data_changed)
        
        # --- Helper for Row with Unit ---
        def create_unit_spinbox(spin: QDoubleSpinBox, unit: str):
            container = QWidget()
            lay = QHBoxLayout(container)
            lay.setContentsMargins(0,0,0,0)
            lay.addWidget(spin)
            if unit:
                lay.addWidget(QLabel(unit))
            return container

        # Temp
        self.lbl_temp = QLabel("Temperature:")
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(10, 300)
        self.spin_temp.setValue(298.0)
        self.container_temp = create_unit_spinbox(self.spin_temp, "K")

        self.spin_na = QDoubleSpinBox()
        self.spin_na.setRange(0.0, 0.95)
        self.spin_na.setDecimals(2)
        self.spin_na.setSingleStep(0.05)
        self.spin_na.setValue(0.0)
        self.spin_na.setToolTip(
            "Effective filled-pupil illumination NA. Change requires refitting; 0 is normal incidence."
        )
        self.spin_na.valueChanged.connect(self.on_na_changed)
        
        # Eps inf
        self.spin_eps_inf = QDoubleSpinBox()
        self.spin_eps_inf.setRange(1, 50)
        self.spin_eps_inf.setValue(12.0)
        for advanced_widget in (
            self.combo_si_data, self.lbl_temp, self.container_temp,
            self.spin_na, self.spin_eps_inf,
        ):
            advanced_widget.hide()

        form_struct.addRow("Semi-infinite substrate:", self.combo_sub)
        struct_layout.addLayout(form_struct)

        preset_row = QHBoxLayout()
        self.combo_structure_preset = QComboBox()
        self.combo_structure_preset.addItems([
            "Sample / SiO2 / Si",
            "hBN / Sample / hBN / SiO2 / Si",
            "hBN / Graphene / Sample / Graphene / hBN / SiO2 / Si",
        ])
        btn_preset = QPushButton("Apply Preset")
        btn_preset.clicked.connect(self.apply_structure_preset)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.combo_structure_preset)
        preset_row.addWidget(btn_preset)
        struct_layout.addLayout(preset_row)

        self.table_layers = CopyableTableWidget(0, 6)
        self.table_layers.setHorizontalHeaderLabels([
            "Material", "Thickness (nm)", "In reference", "Fit", "Min (nm)", "Max (nm)"
        ])
        self.table_layers.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_layers.setMinimumHeight(150)
        struct_layout.addWidget(self.table_layers)
        layer_buttons = QHBoxLayout()
        for label, callback in (
            ("Add Layer", self.add_structure_layer), ("Remove", self.remove_structure_layer),
            ("Move Up", lambda: self.move_structure_layer(-1)),
            ("Move Down", lambda: self.move_structure_layer(1)),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, fn=callback: fn())
            layer_buttons.addWidget(button)
        struct_layout.addLayout(layer_buttons)
        group_struct.setLayout(struct_layout)
        layout.addWidget(group_struct)
        self.apply_structure_preset()

        # --- Fitting Optimization ---
        group_opt = QGroupBox("Fitting Optimization")
        layout_opt = QGridLayout()
        
        self.spin_range_min = QDoubleSpinBox()
        self.spin_range_min.setRange(0, 10)
        self.spin_range_min.setValue(1.5)
        
        self.spin_range_max = QDoubleSpinBox()
        self.spin_range_max.setRange(0, 10)
        self.spin_range_max.setValue(3.0)
        
        layout_opt.addWidget(QLabel("ROI min (eV):"), 0, 0)
        layout_opt.addWidget(self.spin_range_min, 0, 1)
        layout_opt.addWidget(QLabel("ROI max (eV):"), 0, 2)
        layout_opt.addWidget(self.spin_range_max, 0, 3)
        
        self.combo_fit_method = QComboBox()
        self.combo_fit_method.addItems([
            "Robust LM (Recommended)",
            "Global + Robust LM",
            "1st Derivative + LM",
            "2nd Derivative + LM",
        ])
        layout_opt.addWidget(QLabel("Method:"), 1, 0)
        layout_opt.addWidget(self.combo_fit_method, 1, 1, 1, 3)

        self.combo_line_shape = QComboBox()
        self.combo_line_shape.addItems(["Voigt / Faddeeva (Recommended)", "Lorentz"])
        self.combo_line_shape.currentIndexChanged.connect(self.on_line_shape_changed)
        layout_opt.addWidget(QLabel("Line shape:"), 2, 0)
        layout_opt.addWidget(self.combo_line_shape, 2, 1, 1, 3)

        self.spin_e0_margin = QDoubleSpinBox()
        self.spin_e0_margin.setRange(0.001, 0.2)
        self.spin_e0_margin.setDecimals(3)
        self.spin_e0_margin.setValue(0.02)
        self.spin_e0_margin.setToolTip("Allowed movement around each initial E0")

        self.spin_baseline_order = QSpinBox()
        self.spin_baseline_order.setRange(0, 5)
        self.spin_baseline_order.setValue(3)
        self.spin_e0_margin.hide()
        self.spin_baseline_order.hide()
        
        group_opt.setLayout(layout_opt)
        layout.addWidget(group_opt)
        
        # --- Excitons ---
        group_excitons = QGroupBox("Excitons")
        vbox_exc = QVBoxLayout()
        
        # Update Table Columns: f, Lock, E0, Lock, G, Lock (Interleaved)
        self.table_exc = CopyableTableWidget(0, 8)
        self.table_exc.setHorizontalHeaderLabels(["f", "Lock", "E0", "Lock", "wL", "Lock", "wG", "Lock"])
        self.table_exc.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # Resize lock columns to be small
        for c in [1,3,5,7]:
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
        self.fit_progress = QProgressBar()
        self.fit_progress.setRange(0, 100)
        self.fit_progress.setVisible(False)
        layout.addWidget(self.fit_progress)
        
        layout.addStretch()
        panel.setMinimumWidth(560)
        scroll = QScrollArea()
        scroll.setMinimumWidth(600)
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        self.main_layout.addWidget(scroll, 1)

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

    def _table_checkbox(self, checked=False, enabled=True):
        cell = QWidget()
        checkbox = QCheckBox()
        checkbox.setChecked(bool(checked))
        checkbox.setEnabled(enabled)
        layout = QHBoxLayout(cell)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return cell

    def add_structure_layer(self, material="hBN", thickness=10.0, in_reference=True,
                            fit=False, minimum=0.0, maximum=100.0, row=None):
        row = self.table_layers.rowCount() if row is None else row
        self.table_layers.insertRow(row)
        material_box = QComboBox()
        material_box.addItems(LAYER_MATERIALS)
        material_box.setCurrentText(material)
        self.table_layers.setCellWidget(row, 0, material_box)
        for column, value in ((1, thickness), (4, minimum), (5, maximum)):
            self.table_layers.setItem(row, column, QTableWidgetItem(str(value)))
        reference_cell = self._table_checkbox(
            in_reference and material != "Sample", material != "Sample"
        )
        self.table_layers.setCellWidget(row, 2, reference_cell)
        reference_checkbox = reference_cell.findChild(QCheckBox)
        def update_reference_state(text, checkbox=reference_checkbox):
            is_sample = text == "Sample"
            if is_sample:
                checkbox.setChecked(False)
            checkbox.setEnabled(not is_sample)
        material_box.currentTextChanged.connect(update_reference_state)
        self.table_layers.setCellWidget(row, 3, self._table_checkbox(fit))

    def _layer_row_data(self, row):
        material = self.table_layers.cellWidget(row, 0).currentText()
        values = [float(self.table_layers.item(row, column).text()) for column in (1, 4, 5)]
        reference = self.table_layers.cellWidget(row, 2).findChild(QCheckBox).isChecked()
        fit = self.table_layers.cellWidget(row, 3).findChild(QCheckBox).isChecked()
        return material, values[0], reference, fit, values[1], values[2]

    def get_structure_layers(self):
        return [
            {
                "material": material, "thickness_nm": thickness,
                "in_reference": reference, "fit": fit,
                "min_nm": minimum, "max_nm": maximum,
            }
            for material, thickness, reference, fit, minimum, maximum in
            (self._layer_row_data(row) for row in range(self.table_layers.rowCount()))
        ]

    def apply_structure_preset(self):
        presets = {
            0: [("Sample", 0.65, False, False, 0.1, 2.0),
                ("SiO2", 285.0, True, True, 265.0, 305.0)],
            1: [("hBN", 10.0, False, True, 0.0, 100.0),
                ("Sample", 0.65, False, False, 0.1, 2.0),
                ("hBN", 10.0, False, True, 0.0, 100.0),
                ("SiO2", 285.0, True, True, 265.0, 305.0)],
            2: [("hBN", 10.0, False, True, 0.0, 100.0),
                ("Graphene", 0.335, False, False, 0.1, 2.0),
                ("Sample", 0.65, False, False, 0.1, 2.0),
                ("Graphene", 0.335, False, False, 0.1, 2.0),
                ("hBN", 10.0, False, True, 0.0, 100.0),
                ("SiO2", 285.0, True, True, 265.0, 305.0)],
        }
        self.combo_sub.setCurrentText("Si")
        self.table_layers.setRowCount(0)
        for row in presets[self.combo_structure_preset.currentIndex()]:
            self.add_structure_layer(*row)

    def remove_structure_layer(self):
        row = self.table_layers.currentRow()
        if row >= 0:
            self.table_layers.removeRow(row)

    def move_structure_layer(self, direction):
        row = self.table_layers.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= self.table_layers.rowCount():
            return
        data = self._layer_row_data(row)
        self.table_layers.removeRow(row)
        self.add_structure_layer(*data, row=target)
        self.table_layers.setCurrentCell(target, 1)

    def on_substrate_changed(self, index):
        # Advanced substrate settings remain code-level defaults.
        pass

    def on_line_shape_changed(self, index):
        show_gaussian = index == 0
        self.table_exc.setColumnHidden(6, not show_gaussian)
        self.table_exc.setColumnHidden(7, not show_gaussian)

    def on_si_data_changed(self, filename):
        if not filename:
            return
        self.si_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        self.init_materials()

    def on_na_changed(self, _value):
        if getattr(self, 'last_y_fit', None) is not None:
            self.status_label.setText("Illumination NA changed - run fitting again.")
            self.status_label.setStyleSheet("color: darkorange")

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

        roi = (x_ev >= self.spin_range_min.value()) & (x_ev <= self.spin_range_max.value())
        guesses = guess_resonances(x_ev[roi], y_contrast[roi])
        
        if len(guesses) == 0:
            QMessageBox.information(self, "Info", "No significant peaks found.")
            return

        # Ask user confirmation? No, just populate (easy to remove)
        reply = QMessageBox.question(self, "Auto-Guess", f"Found {len(guesses)} peaks. Replace existing excitons?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.table_exc.setRowCount(0)
            for energy, linewidth in guesses:
                self.add_exciton_values(0.1, round(energy, 4), round(linewidth, 4))

    def add_exciton_values(self, f, E0, gamma, gaussian=0.01, locks=(False, False, False, False)):
        row = self.table_exc.rowCount()
        self.table_exc.insertRow(row)
        # Values in 0, 2, 4
        self.table_exc.setItem(row, 0, QTableWidgetItem(str(f)))
        self.table_exc.setItem(row, 2, QTableWidgetItem(str(E0)))
        self.table_exc.setItem(row, 4, QTableWidgetItem(str(gamma)))
        self.table_exc.setItem(row, 6, QTableWidgetItem(str(gaussian)))
        
        # Add Checkboxes for locks in 1, 3, 5
        for i, col in enumerate([1, 3, 5, 7]):
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
            self._infer_sio2_thickness_from_filename(path)
    
    def load_sample(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Sample File", "", "Data Files (*.csv *.txt *.dat);;All Files (*)")
        if path:
            self.samp_path = path
            self.btn_samp.setText(f"Sample: {os.path.basename(path)}")
            self._infer_sio2_thickness_from_filename(path)

    def _infer_sio2_thickness_from_filename(self, path):
        filename = os.path.basename(path)
        if len(re.findall(r"hBN", filename, re.IGNORECASE)) >= 2:
            self.combo_structure_preset.setCurrentIndex(1)
            self.apply_structure_preset()
        match = re.search(r"(\d+(?:\.\d+)?)\s*nm\s*SiO2", filename, re.IGNORECASE)
        if match:
            oxide = float(match.group(1))
            for row in range(self.table_layers.rowCount()):
                if self.table_layers.cellWidget(row, 0).currentText() == "SiO2":
                    self.table_layers.item(row, 1).setText(str(oxide))
                    self.table_layers.item(row, 4).setText(str(max(0.0, oxide - 20.0)))
                    self.table_layers.item(row, 5).setText(str(oxide + 20.0))
            
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
                 wl = HC_EV_NM / x
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
            substrate_is_unit_placeholder = (
                np.ptp(sub_final) <= 1e-8 and np.allclose(sub_final, 1.0, atol=1e-8)
                and np.any(samp_final < 0)
            )
            y_contrast_exp = (
                samp_final if substrate_is_unit_placeholder
                else (samp_final - sub_final) / sub_final
            )
            
            # Convert back to eV for output
            x_ev = HC_EV_NM / wl_final
            
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
        x_exp_nm = HC_EV_NM / x_ev
            
        # UI State -> Fitting
        self.btn_fit.setText("Stop Fitting")
        self.btn_fit.setStyleSheet("background-color: #ffaaaa; color: black;") # Red tint
        self.btn_fit.setEnabled(True) # Keep enabled so user can click to stop
        self.fit_progress.setValue(0)
        self.fit_progress.setVisible(True)
    
        try:
            # 3. Prepare Config
            
            # Map Dropdown to Config
            method_str = self.combo_fit_method.currentText()
            fit_method = 'Standard'
            high_prec = False
            
            if method_str == "Global + Robust LM":
                fit_method = 'Standard'
                high_prec = True
            elif method_str == "1st Derivative + LM":
                fit_method = 'Derivative'
            elif method_str == "2nd Derivative + LM":
                fit_method = '2nd Derivative'
            
            struct_config = {
                'substrate_type': self.combo_sub.currentText(),
                'temp': self.spin_temp.value(),
                'numerical_aperture': self.spin_na.value(),
                'layers': self.get_structure_layers(),
                'contrast_definition': 'relative',
                'line_shape': 'Voigt' if self.combo_line_shape.currentIndex() == 0 else 'Lorentz',
                'fit_method': fit_method,
                'high_precision': high_prec
                ,'baseline_order': self.spin_baseline_order.value()
                ,'max_nfev': 12000 if high_prec else 8000
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
                    if struct_config['line_shape'] == 'Voigt':
                        wg = float(self.table_exc.item(r, 6).text())
                        p0.extend([f, E0, g, wg])
                    else:
                        p0.extend([f, E0, g])
                    
                    # Check Locks at 1, 3, 5
                    lock_f = self._get_lock_state(r, 1)
                    lock_E0 = self._get_lock_state(r, 3)
                    lock_g = self._get_lock_state(r, 5)
                    
                    locks = [lock_f, lock_E0, lock_g]
                    if struct_config['line_shape'] == 'Voigt':
                        locks.append(self._get_lock_state(r, 7))
                    locked_mask.extend(locks)
                    
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
            # Bounds are supplied for the full vector; the engine removes locked parameters.
            
            bounds_min = []
            bounds_max = []
            
            # Eps Inf
            bounds_min.append(1.0)
            bounds_max.append(50.0)
            
            # Oscillators
            stride = 4 if struct_config['line_shape'] == 'Voigt' else 3
            num_osc = (len(p0) - 1) // stride
            
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
            
            # Tight E0 margin (+/-0.15 eV) to prioritize user settings
            E0_margin = self.spin_e0_margin.value()
            
            for i in range(num_osc):
                base = 1 + i * stride
                if stride == 4:
                    w_l, w_g = p0[base + 2], p0[base + 3]
                    approximate_fwhm = (
                        0.5346 * w_l + np.sqrt(0.2166 * w_l**2 + w_g**2)
                    )
                    width_upper = min(
                        0.5, max(0.015, 4.0 * approximate_fwhm, 1.25 * w_l, 1.25 * w_g)
                    )
                else:
                    approximate_fwhm = p0[base + 2]
                    width_upper = min(0.5, max(0.015, 4.0 * approximate_fwhm))
                # f: 0 to 20
                bounds_min.append(0.0)
                bounds_max.append(20.0)
                
                # E0: Tight bounds around user-specified value
                user_E0 = user_E0_values[i] if i < len(user_E0_values) else 2.0
                local_e0_margin = min(E0_margin, max(0.004, 0.35 * approximate_fwhm))
                bounds_min.append(max(0.1, user_E0 - local_e0_margin))
                bounds_max.append(user_E0 + local_e0_margin)
                
                # Gamma: 0.0001 to 0.5 eV (Prevent singular or too broad)
                bounds_min.append(0.0001)
                bounds_max.append(width_upper)
                if stride == 4:
                    bounds_min.append(0.0001)
                    bounds_max.append(width_upper)
            
            # 5. Fit Range
            fit_range = (self.spin_range_min.value(), self.spin_range_max.value())
            
            # 6. Start Thread
            self.worker = FittingWorker(x_exp_nm, y_contrast_exp, self.mat_loader, p0, (bounds_min, bounds_max),
                                        struct_config, fit_range, locked_mask)
            self.worker.finished.connect(lambda y, p, r2: self.on_fit_finished(x_exp_nm, y_contrast_exp, y, p, r2, locked_mask))
            self.worker.error.connect(self.on_fit_error)
            self.worker.aborted.connect(self.on_fit_aborted)
            self.worker.progress.connect(self.on_fit_progress)
            self.worker.start()

        except Exception as e:
            self.on_fit_error(str(e))
            
    def on_fit_aborted(self):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("Start Fitting")
        self.btn_fit.setStyleSheet("")
        self.status_label.setText("Fitting stopped by user.")
        self.fit_progress.setVisible(False)

    def on_fit_progress(self, value, message):
        self.fit_progress.setValue(value)
        self.fit_progress.setFormat(f"{message} (%p%)")
        self.status_label.setText(message)

    def on_fit_finished(self, x_data, y_data, y_fit, popt, r_squared, locked_mask):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("Start Fitting")
        self.btn_fit.setStyleSheet("") # Reset Color
        self.fit_progress.setVisible(False)
        
        # Convert Wavelength (nm) to Energy (eV)
        x_ev = HC_EV_NM / x_data
        
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
        self.last_fit_result = getattr(self.worker, 'fit_result', None)
        fitted_layers = getattr(self.worker, 'fitted_layers', [])
        for row, layer in enumerate(fitted_layers):
            if row < self.table_layers.rowCount():
                self.table_layers.item(row, 1).setText(f"{layer['thickness_nm']:.6g}")
        
        # Update UI with results
        # popt[0] is eps_inf
        self.spin_eps_inf.setValue(popt[0])
        
        # Update Table
        self.table_exc.setRowCount(0) # Clear
        stride = 4 if self.worker.structure_config.get('line_shape') == 'Voigt' else 3
        num_oscillators = (len(popt) - 1) // stride
        for i in range(num_oscillators):
            base = 1 + i * stride
            f, E0, gamma = popt[base:base + 3]
            gaussian = popt[base + 3] if stride == 4 else 0.01
            
            # Restore locks
            # Indices in locked_mask (which includes eps at 0): 1+i*3, 2+i*3, 3+i*3
            l_f, l_E0, l_g = locked_mask[base:base + 3]
            l_wg = locked_mask[base + 3] if stride == 4 else False
            
            self.add_exciton_values(round(f, 4), round(E0, 4), round(gamma, 4), round(gaussian, 4), locks=(l_f, l_E0, l_g, l_wg))
        
        result = self.last_fit_result
        msg = f"Fitting completed successfully!\nR-squared: {r_squared:.4f}"
        if result is not None:
            msg += (
                f"\nRMSE: {result.rmse:.6g}"
                f"\nOptimizer: {result.method}"
                f"\nJacobian condition: {result.jacobian_condition:.3g}"
                f"\nDurbin-Watson: {result.durbin_watson:.3f}"
            )
            if result.jacobian_condition > 1e10:
                msg += "\n\nWarning: parameters are strongly correlated; reported values may not be identifiable."
            fitted_rows = [layer for layer in fitted_layers if layer['fit']]
            for offset, layer in enumerate(fitted_rows):
                error_index = len(popt) + offset
                msg += (
                    f"\n{layer['material']} thickness = {layer['thickness_nm']:.4f} +/- "
                    f"{result.standard_errors[error_index]:.2g} nm"
                )
            msg += f"\nEps_inf = {popt[0]:.5g} +/- {result.standard_errors[0]:.2g}"
            for index, diagnostic in enumerate(
                getattr(result, 'resonance_diagnostics', []), start=1
            ):
                msg += (
                    f"\nLocal peak {index} ({diagnostic['center_ev']:.4f} eV): "
                    f"R2 = {diagnostic['local_r_squared']:.4f}, amplitude = "
                    f"{diagnostic['amplitude_ratio']:.2f}x"
                )
            for i in range((len(popt) - 1) // stride):
                base = 1 + stride * i
                msg += (
                    f"\nPeak {i + 1}: E0 = {popt[base + 1]:.6g} +/- "
                    f"{result.standard_errors[base + 1]:.2g} eV, Gamma = "
                    f"{popt[base + 2]:.5g} +/- {result.standard_errors[base + 2]:.2g} eV"
                )
                if stride == 4:
                    msg += f", wG = {popt[base + 3]:.5g} +/- {result.standard_errors[base + 3]:.2g} eV"
        if r_squared < 0.9:
            msg += "\n\nWarning: Low fit quality. Please check initial guesses or range."
            QMessageBox.warning(self, "Fitting Result", msg)
        else:
            QMessageBox.information(self, "Success", msg)

    def on_fit_error(self, err_msg):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("Start Fitting")
        self.btn_fit.setStyleSheet("") # Reset Color
        self.fit_progress.setVisible(False)
        self.status_label.setText("Fit failed")
        QMessageBox.critical(self, "Fitting Error", err_msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
