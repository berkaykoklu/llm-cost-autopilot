import json
import xml.etree.ElementTree as ET
from pathlib import Path

from offline.figures import architecture_svg, curve_svg


def test_figures_are_valid_svg_with_measured_labels() -> None:
    curve = json.loads(Path("data/curve.json").read_text())
    for svg in (curve_svg(curve), architecture_svg(curve)):
        ET.fromstring(svg)  # raises on malformed XML
    assert "67.3%" in curve_svg(curve)
    assert "Jev" in architecture_svg(curve)
