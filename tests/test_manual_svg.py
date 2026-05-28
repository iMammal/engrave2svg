import csv
import json
import random
from pathlib import Path

import networkx as nx

from engrave2svg.manual_svg import ManualSvgConfig, run_manual_svg_pipeline
from engrave2svg.null_models import (
    analyze_graphml,
    compare_null_models,
    graph_from_strokes,
    lozenge_lattice_strokes,
    random_scratch_strokes,
    save_graphml,
)


def _write_svg(path: Path, body: str) -> None:
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg"
    xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
    width="120" height="100" viewBox="0 0 120 100">
  <g inkscape:label="Background" style="display:none">
    <image href="background.png" x="0" y="0" width="120" height="100"/>
    <line x1="0" y1="50" x2="120" y2="50"/>
  </g>
  <g inkscape:label="Manual Trace">
    {body}
  </g>
</svg>
''',
    )


def _run_manual(tmp_path: Path, svg: Path, snap_radius: float = 3.0):
    return run_manual_svg_pipeline(
        input_path=svg,
        output_path=tmp_path / "manual.svg",
        debug_dir=tmp_path / "debug",
        config=ManualSvgConfig(snap_radius=snap_radius),
        metrics_path=tmp_path / "metrics.json",
        graph_path=tmp_path / "graph.graphml",
        nodes_csv_path=tmp_path / "nodes.csv",
        edges_csv_path=tmp_path / "edges.csv",
        orientation_hist_path=tmp_path / "orientation.png",
    )


def test_manual_svg_triangle_mesh_produces_cycles_and_junction(tmp_path: Path):
    svg = tmp_path / "triangle_mesh.svg"
    _write_svg(
        svg,
        """
        <path d="M 20 80 L 60 20 L 100 80 Z"/>
        <line x1="60" y1="20" x2="60" y2="55"/>
        <line x1="20" y1="80" x2="60" y2="55"/>
        <line x1="100" y1="80" x2="60" y2="55"/>
        """,
    )

    metrics = _run_manual(tmp_path, svg)
    graph_metrics = analyze_graphml(tmp_path / "graph_merged.graphml")
    payload = json.loads((tmp_path / "metrics.json").read_text())

    assert metrics.input_mode == "manual-svg"
    assert payload["graph_metrics"]["input_mode"] == "manual-svg"
    assert graph_metrics["cycle_count"] >= 3
    assert graph_metrics["junction_count"] >= 1
    assert (tmp_path / "debug" / "manual_trace_debug.svg").exists()


def test_manual_svg_lozenge_lattice_has_four_cycles_and_candidates(tmp_path: Path):
    svg = tmp_path / "lozenge.svg"
    lines = []
    for index, stroke in enumerate(lozenge_lattice_strokes(120, 100, 35)):
        (x1, y1), (x2, y2) = stroke.points
        lines.append(f'<line id="s{index}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    _write_svg(svg, "\n".join(lines))

    _run_manual(tmp_path, svg)
    graph_metrics = analyze_graphml(tmp_path / "graph_merged.graphml")

    assert graph_metrics["four_cycle_count"] > 0
    assert graph_metrics["lozenge_candidate_count"] > 0


def test_manual_svg_ignores_hidden_background_layers(tmp_path: Path):
    svg = tmp_path / "hidden.svg"
    _write_svg(svg, '<line x1="10" y1="10" x2="90" y2="10"/>')

    metrics = _run_manual(tmp_path, svg)
    rows = list(csv.DictReader((tmp_path / "edges_merged.csv").open()))

    assert metrics.paths == 1
    assert metrics.merged_node_count == 2
    assert metrics.merged_edge_count == 1
    assert len(rows) == 1


def test_manual_svg_endpoint_snapping_merges_near_touching_strokes(tmp_path: Path):
    svg = tmp_path / "snap.svg"
    _write_svg(
        svg,
        """
        <line x1="10" y1="20" x2="50" y2="20"/>
        <line x1="52" y1="20" x2="90" y2="20"/>
        """,
    )

    metrics = _run_manual(tmp_path, svg, snap_radius=3.0)

    assert metrics.raw_node_count == 4
    assert metrics.merged_node_count == 3
    assert metrics.connected_components == 1


def test_manual_svg_metrics_are_compatible_with_null_model_comparison(tmp_path: Path):
    observed_svg = tmp_path / "observed.svg"
    _write_svg(
        observed_svg,
        """
        <path d="M 20 80 L 60 20 L 100 80 Z"/>
        <line x1="60" y1="20" x2="60" y2="55"/>
        <line x1="20" y1="80" x2="60" y2="55"/>
        <line x1="100" y1="80" x2="60" y2="55"/>
        """,
    )
    _run_manual(tmp_path, observed_svg)
    controls = tmp_path / "controls"
    controls.mkdir()
    save_graphml(nx.read_graphml(tmp_path / "graph_merged.graphml"), controls / "manual_lozenge_control.graphml")
    save_graphml(
        graph_from_strokes(
            random_scratch_strokes(120, 100, 12, 500, random.Random(3))
        ),
        controls / "random_scratches.graphml",
    )

    csv_path, json_path = compare_null_models(
        observed_graph=tmp_path / "graph_merged.graphml",
        observed_metrics=tmp_path / "metrics.json",
        controls_dir=controls,
        output_dir=tmp_path / "comparison",
    )

    rows = list(csv.DictReader(csv_path.open()))
    payload = json.loads(json_path.read_text())
    assert rows
    assert payload["observed"]["node_count"] > 0
    assert payload["class_aggregates"]["lozenge"]["n_controls"] == 1
