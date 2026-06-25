import ast
import importlib
import os
import unittest
from pathlib import Path


class EntrypointTests(unittest.TestCase):
    def test_streamlit_uses_backward_compatible_fitting_engine_import(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        direct_imports = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "fitting_engine"
        ]
        self.assertEqual(direct_imports, [])
        self.assertIn("resonance_balanced_sigma = getattr(", source)

    def test_streamlit_invalidates_fit_results_when_fit_context_changes(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("def make_data_signature(", source)
        self.assertIn("def make_fit_context_signature(", source)
        self.assertIn("stored_context_signature", source)
        self.assertIn("Data or fitting setup changed. Run the fit again.", source)
        self.assertIn("fit_context_signature,", source)
        self.assertIn("st.session_state.fit_results[:7]", source)

    def test_streamlit_validates_roi_before_guessing_or_fitting(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("def validate_roi(", source)
        self.assertIn("ROI Min must be smaller than ROI Max.", source)
        self.assertIn("roi_mask, roi_error = validate_roi(", source)
        self.assertIn("disabled=x_exp_ev is None or roi_error is not None", source)
        self.assertIn("or structure_error is not None or roi_error is not None", source)

    def test_streamlit_flags_local_peak_fit_failures(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("LOCAL_R2_WARNING_THRESHOLD = 0.90", source)
        self.assertIn("AMPLITUDE_RATIO_WARNING_RANGE = (0.70, 1.30)", source)
        self.assertIn("def local_fit_quality_warnings(", source)
        self.assertIn("def local_fit_quality_flags(", source)
        self.assertIn("Global R2 can hide missed exciton features", source)
        self.assertIn("Download Resonance Diagnostics (CSV)", source)
        self.assertIn("\"Quality flag\": (", source)
        self.assertIn("\"Issue\": \"; \".join(", source)

    def test_streamlit_fit_plot_includes_residual_panel(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("fig, (ax, ax_residual) = plt.subplots(", source)
        self.assertIn("gridspec_kw={\"height_ratios\": [3, 1]}", source)
        self.assertIn("ax_residual.set_ylabel(\"Residual\")", source)
        self.assertIn("residual = y_exp_contrast - y_model", source)
        self.assertIn("ax_residual.axhline(0.0", source)

    def test_streamlit_spectrum_export_includes_residual_and_baseline(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("baseline_e = y_model_e - physical_model_e", source)
        self.assertIn("residual_e = y_fit_exp_res - y_model_e", source)
        self.assertIn("\"Contrast_PhysicalModel\": physical_model_e", source)
        self.assertIn("\"Baseline\": baseline_e", source)
        self.assertIn("\"Residual_ExpMinusFit\": residual_e", source)

    def test_streamlit_constrains_nonphysical_resonance_inputs(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("\"f\": st.column_config.NumberColumn(\"f (eV²)\", min_value=0.0", source)
        self.assertIn("\"E0\": st.column_config.NumberColumn(\"E0 (eV)\", min_value=0.0", source)
        self.assertIn("\"wL\": st.column_config.NumberColumn(\"wL (eV)\", min_value=0.0001", source)
        self.assertIn("\"wG\": st.column_config.NumberColumn(\"wG (eV)\", min_value=0.0001", source)

    def test_desktop_rejects_nonphysical_resonance_inputs(self):
        source = Path("gui_app.py").read_text(encoding="utf-8")
        self.assertIn("if f < 0 or E0 <= 0 or g <= 0:", source)
        self.assertIn("f must be >= 0; E0 and wL must be > 0", source)
        self.assertIn("if wg <= 0:", source)
        self.assertIn("Invalid exciton parameters in row", source)

    def test_desktop_module_imports_without_starting_event_loop(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        module = importlib.import_module("gui_app")
        self.assertTrue(hasattr(module, "MainWindow"))
        self.assertTrue(hasattr(module, "FittingWorker"))

    def test_desktop_exposes_si_sources_and_layer_table(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        module = importlib.import_module("gui_app")
        app = module.QApplication.instance() or module.QApplication([])
        window = module.MainWindow()
        self.assertFalse(window.btn_plot.isEnabled())
        self.assertFalse(window.btn_auto_guess.isEnabled())
        self.assertFalse(window.btn_fit.isEnabled())
        self.assertFalse(window.btn_export_data.isEnabled())
        sources = [window.combo_si_data.itemText(i) for i in range(window.combo_si_data.count())]
        self.assertEqual(sources, ["Si_data.csv", "Schinke.csv", "Green-2008.csv"])
        self.assertEqual(window.table_layers.rowCount(), 2)
        self.assertEqual(
            [row["material"] for row in window.get_structure_layers()],
            ["Sample", "SiO2"],
        )
        window.combo_structure_preset.setCurrentIndex(2)
        window.apply_structure_preset()
        self.assertEqual(
            [row["material"] for row in window.get_structure_layers()],
            ["hBN", "Graphene", "Sample", "Graphene", "hBN", "SiO2"],
        )
        add_layer = next(
            button for button in window.findChildren(module.QPushButton)
            if button.text() == "Add Layer"
        )
        row_count = window.table_layers.rowCount()
        add_layer.click()
        app.processEvents()
        self.assertEqual(window.table_layers.rowCount(), row_count + 1)
        self.assertEqual(window.get_structure_layers()[-1]["material"], "hBN")
        for advanced_widget in (
            window.combo_si_data, window.spin_na, window.spin_temp,
            window.spin_eps_inf, window.spin_e0_margin, window.spin_baseline_order,
        ):
            self.assertFalse(advanced_widget.isVisible())
        window.sub_path = "reference.csv"
        window.samp_path = "sample.csv"
        window._update_action_states()
        self.assertTrue(window.btn_plot.isEnabled())
        self.assertTrue(window.btn_auto_guess.isEnabled())
        self.assertTrue(window.btn_fit.isEnabled())
        self.assertFalse(window.btn_export_data.isEnabled())
        window.on_si_data_changed("Schinke.csv")
        self.assertIsNotNone(window.mat_loader)
        window.last_y_fit = [0.0]
        window.canvas.axes.plot([1.0], [0.0], label="Fit Model")
        window._updating_fit_results = True
        window.add_structure_layer()
        self.assertIsNotNone(window.last_y_fit)
        self.assertTrue(
            any(line.get_label() == "Fit Model" for line in window.canvas.axes.lines)
        )
        window._updating_fit_results = False
        window.add_structure_layer()
        self.assertIsNone(window.last_y_fit)
        self.assertIn("settings changed", window.status_label.text())
        window.last_y_fit = [0.0]
        window.spin_na.setValue(0.5)
        self.assertIn("run fitting again", window.status_label.text())
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
