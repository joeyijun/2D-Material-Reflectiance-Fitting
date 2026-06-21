import importlib
import os
import unittest


class EntrypointTests(unittest.TestCase):
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
