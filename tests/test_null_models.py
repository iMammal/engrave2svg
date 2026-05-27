import csv
import json
import random
from pathlib import Path

from engrave2svg.null_models import (
    analyze_graph,
    compare_null_models,
    generate_control_set,
    graph_from_strokes,
    lozenge_grid_graph,
    random_scratch_strokes,
    save_graphml,
    write_lozenge_outputs,
)


def test_lozenge_lattice_yields_four_cycles_and_candidates(tmp_path: Path):
    graph = lozenge_grid_graph(width=180, height=140, spacing=35)
    graph_path = tmp_path / "clean_lozenge.graphml"
    save_graphml(graph, graph_path)

    metrics = analyze_graph(graph)
    candidates_csv, summary_json = write_lozenge_outputs(graph_path, tmp_path)
    rows = list(csv.DictReader(candidates_csv.open()))
    summary = json.loads(summary_json.read_text())

    assert metrics["four_cycle_count"] > 0
    assert metrics["lozenge_candidate_count"] > 0
    assert len(rows) == metrics["lozenge_candidate_count"]
    assert summary["lozenge_candidate_count"] == metrics["lozenge_candidate_count"]


def test_random_scratch_control_has_fewer_lozenge_candidates():
    lozenge = analyze_graph(lozenge_grid_graph(width=180, height=140, spacing=35))
    scratches = graph_from_strokes(
        random_scratch_strokes(
            width=180,
            height=140,
            stroke_count=24,
            target_total_length=1200,
            rng=random.Random(7),
        )
    )
    scratch_metrics = analyze_graph(scratches)

    assert scratch_metrics["lozenge_candidate_count"] < lozenge["lozenge_candidate_count"]
    assert scratch_metrics["four_cycle_count"] <= lozenge["four_cycle_count"]


def test_orientation_entropy_is_lower_for_clean_lattice_than_random_scratches():
    lozenge = analyze_graph(lozenge_grid_graph(width=180, height=140, spacing=35))
    scratches = graph_from_strokes(
        random_scratch_strokes(
            width=180,
            height=140,
            stroke_count=40,
            target_total_length=1800,
            rng=random.Random(11),
        )
    )
    scratch_metrics = analyze_graph(scratches)

    assert lozenge["orientation_entropy"] < scratch_metrics["orientation_entropy"]
    assert lozenge["dominant_orientation_families"] <= scratch_metrics["dominant_orientation_families"]


def test_compare_null_models_writes_csv_json_plots_and_lozenge_outputs(tmp_path: Path):
    controls = tmp_path / "controls"
    controls.mkdir()
    observed_graph = controls / "observed_lozenge.graphml"
    random_graph = controls / "random_scratches.graphml"
    clean_control = controls / "clean_lozenge_control.graphml"
    save_graphml(lozenge_grid_graph(width=180, height=140, spacing=35), observed_graph)
    save_graphml(lozenge_grid_graph(width=180, height=140, spacing=35), clean_control)
    save_graphml(
        graph_from_strokes(
            random_scratch_strokes(
                width=180,
                height=140,
                stroke_count=24,
                target_total_length=1200,
                rng=random.Random(3),
            )
        ),
        random_graph,
    )

    output = tmp_path / "comparison"
    csv_path, json_path = compare_null_models(
        observed_graph=observed_graph,
        observed_metrics=None,
        controls_dir=controls,
        output_dir=output,
        orientation_shuffles=3,
        seed=5,
    )
    rows = list(csv.DictReader(csv_path.open()))
    payload = json.loads(json_path.read_text())

    assert csv_path.exists()
    assert json_path.exists()
    assert rows
    assert any(row["metric"] == "lozenge_candidate_count" for row in rows)
    assert payload["comparisons"]
    assert (output / "orientation_entropy.png").exists()
    assert (output / "lozenge_candidates.csv").exists()
    assert (output / "lozenge_summary.json").exists()


def test_generate_controls_writes_png_metadata_and_svg_previews(tmp_path: Path):
    metadata = generate_control_set(
        tmp_path,
        width=120,
        height=90,
        spacing=30,
        seed=9,
        svg_previews=True,
    )

    expected = json.loads((tmp_path / "expected_metadata.json").read_text())
    assert len(metadata["controls"]) == 4
    assert len(expected["controls"]) == 4
    for control in expected["controls"]:
        assert Path(control["image"]).exists()
        assert Path(control["svg"]).exists()
        assert control["stroke_count"] > 0
