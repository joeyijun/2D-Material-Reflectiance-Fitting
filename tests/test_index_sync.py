import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def javascript_template(source):
    had_final_newline = source.endswith("\n")
    source = "\n".join(line.rstrip() for line in source.splitlines())
    if had_final_newline:
        source += "\n"
    return source.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


class StandaloneIndexTests(unittest.TestCase):
    def test_embedded_python_sources_are_synchronized(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        sources = {
            "streamlitAppContent": "streamlit_app.py",
            "opticalModelContent": "optical_model.py",
            "fittingEngineContent": "fitting_engine.py",
            "materialsContent": "materials.py",
            "schinkeDataContent": "Schinke.csv",
            "greenDataContent": "Green-2008.csv",
        }
        for variable, filename in sources.items():
            marker = f"const {variable} = `"
            start = html.index(marker) + len(marker)
            end = html.index("`;", start)
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertEqual(html[start:end], javascript_template(source), filename)
            compile(source, filename, "exec")
            self.assertIn(f'"{filename}": {variable}', html)


if __name__ == "__main__":
    unittest.main()
