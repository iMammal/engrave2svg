from __future__ import annotations

from pathlib import Path

import svgwrite

from .graph_trace import Polyline


def export_svg(
    polylines: list[Polyline],
    output_path: str | Path,
    width: int,
    height: int,
    stroke_width: float = 1.2,
) -> None:
    drawing = svgwrite.Drawing(
        str(output_path),
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {width} {height}",
        profile="tiny",
    )
    drawing.add(
        drawing.rect(insert=(0, 0), size=(width, height), fill="black", opacity=0)
    )
    group = drawing.g(
        id="engraving-centerlines",
        fill="none",
        stroke="white",
        stroke_width=stroke_width,
        stroke_linecap="round",
        stroke_linejoin="round",
    )
    for index, polyline in enumerate(polylines):
        if len(polyline.points) < 2:
            continue
        points = [(float(x), float(y)) for x, y in polyline.points]
        element = drawing.polyline(points=points)
        element["id"] = f"path-{index:04d}"
        if polyline.closed:
            element["points"] = " ".join(f"{x},{y}" for x, y in points + [points[0]])
        group.add(element)
    drawing.add(group)
    drawing.save(pretty=True)
