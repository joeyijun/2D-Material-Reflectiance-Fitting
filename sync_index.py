"""Synchronize Python sources embedded in the standalone stlite HTML app."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"


def javascript_template(source):
    had_final_newline = source.endswith("\n")
    source = "\n".join(line.rstrip() for line in source.splitlines())
    if had_final_newline:
        source += "\n"
    return source.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def replace_block(html, variable, source):
    start_marker = f"const {variable} = `"
    start = html.index(start_marker) + len(start_marker)
    end = html.index("`;", start)
    return html[:start] + javascript_template(source) + html[end:]


def main():
    html = INDEX.read_text(encoding="utf-8")
    html = replace_block(html, "streamlitAppContent", (ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    html = replace_block(html, "siDataContent", (ROOT / "Si_data.csv").read_text(encoding="utf-8"))

    module_marker = "        const siDataContent = `"
    generated = (
        "        const opticalModelContent = `"
        + javascript_template((ROOT / "optical_model.py").read_text(encoding="utf-8"))
        + "`;\n\n"
        + "        const fittingEngineContent = `"
        + javascript_template((ROOT / "fitting_engine.py").read_text(encoding="utf-8"))
        + "`;\n\n"
        + "        const materialsContent = `"
        + javascript_template((ROOT / "materials.py").read_text(encoding="utf-8"))
        + "`;\n\n"
        + "        const schinkeDataContent = `"
        + javascript_template((ROOT / "Schinke.csv").read_text(encoding="utf-8"))
        + "`;\n\n"
        + "        const greenDataContent = `"
        + javascript_template((ROOT / "Green-2008.csv").read_text(encoding="utf-8"))
        + "`;\n\n"
    )
    optical_marker = "        const opticalModelContent = `"
    if optical_marker in html:
        start = html.index(optical_marker)
        end = html.index(module_marker, start)
        html = html[:start] + generated + html[end:]
    else:
        html = html.replace(module_marker, generated + module_marker, 1)

    files_marker = '                    "streamlit_app.py": streamlitAppContent,'
    replacement = (
        files_marker
        + '\n                    "optical_model.py": opticalModelContent,'
        + '\n                    "fitting_engine.py": fittingEngineContent,'
    )
    if '"fitting_engine.py": fittingEngineContent' not in html:
        html = html.replace(files_marker, replacement, 1)
    fitting_file = '                    "fitting_engine.py": fittingEngineContent,'
    if '"materials.py": materialsContent' not in html:
        html = html.replace(
            fitting_file,
            fitting_file
            + '\n                    "materials.py": materialsContent,'
            + '\n                    "Schinke.csv": schinkeDataContent,'
            + '\n                    "Green-2008.csv": greenDataContent,',
            1,
        )
    INDEX.write_text(html, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
