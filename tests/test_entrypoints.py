import importlib
import os
import unittest


class EntrypointTests(unittest.TestCase):
    def test_desktop_module_imports_without_starting_event_loop(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        module = importlib.import_module("gui_app")
        self.assertTrue(hasattr(module, "MainWindow"))
        self.assertTrue(hasattr(module, "FittingWorker"))

    def test_desktop_exposes_si_sources_and_structure_fit(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        module = importlib.import_module("gui_app")
        app = module.QApplication.instance() or module.QApplication([])
        window = module.MainWindow()
        sources = [window.combo_si_data.itemText(i) for i in range(window.combo_si_data.count())]
        self.assertEqual(sources, ["Si_data.csv", "Schinke.csv", "Green-2008.csv"])
        self.assertTrue(window.chk_fit_sio2.isChecked())
        window.on_si_data_changed("Schinke.csv")
        self.assertIsNotNone(window.mat_loader)
        window.last_y_fit = [0.0]
        window.spin_na.setValue(0.5)
        self.assertIn("run fitting again", window.status_label.text())
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
