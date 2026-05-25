from pathlib import Path

from engrave2svg.graph_trace import Polyline
from engrave2svg.svg_export import export_svg


def test_svg_export_writes_editable_polylines(tmp_path: Path):
    output = tmp_path / "out.svg"
    export_svg(
        [Polyline(points=[(1, 1), (5, 5), (9, 1)])],
        output,
        width=10,
        height=10,
    )

    text = output.read_text()
    assert "<polyline" in text
    assert "engraving-centerlines" in text
    assert "path-0000" in text
