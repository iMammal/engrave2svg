import csv
import json
import random
from pathlib import Path

from engrave2svg.null_models import (
    analyze_graph,
    analyze_graphml,
    compare_null_models,
    detect_open_lozenge_candidates,
    generate_control_set,
    generate_controls_main,
    graph_from_strokes,
    lozenge_grid_graph,
    random_scratch_strokes,
    save_graphml,
    Stroke,
    write_lozenge_outputs,
)
from engrave2svg.pipeline import PipelineConfig, run_pipeline
from engrave2svg.preprocessing import PreprocessParams


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
    assert metrics["closed_lozenge_candidate_count"] == metrics["lozenge_candidate_count"]
    assert metrics["cycles_per_node"] > 0
    assert metrics["four_cycles_per_node"] > 0
    assert metrics["lozenge_candidates_per_cycle"] > 0
    assert "degree4_fraction" in metrics
    assert len(rows) == metrics["lozenge_candidate_count"]
    assert summary["lozenge_candidate_count"] == metrics["lozenge_candidate_count"]
    assert "open_lozenge_candidate_count" in summary
    assert (tmp_path / "open_lozenge_candidates.csv").exists()


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
    assert (output / "endpoints_per_1000px.png").exists()
    assert (output / "lozenge_candidates.csv").exists()
    assert (output / "lozenge_summary.json").exists()
    assert "class_aggregates" in payload
    assert payload["class_aggregates"]["lozenge"]["n_controls"] >= 1


def test_open_lozenge_detector_reports_implied_candidates_separately():
    graph = graph_from_strokes(
        [
            # Two positive-slope sides and two negative-slope sides that imply a rhombus,
            # but stop just short of several corners so no closed 4-cycle is present.
            Stroke([(12, 42), (42, 12)]),
            Stroke([(58, 88), (88, 58)]),
            Stroke([(12, 58), (42, 88)]),
            Stroke([(58, 12), (88, 42)]),
        ]
    )
    metrics = analyze_graph(graph)
    open_candidates = detect_open_lozenge_candidates(graph)

    assert metrics["lozenge_candidate_count"] == 0
    assert metrics["open_lozenge_candidate_count"] > 0
    assert open_candidates
    assert 0.0 < open_candidates[0]["confidence"] <= 1.0


def test_generate_controls_supports_many_seed_directories(tmp_path: Path):
    rc = generate_controls_main(
        [
            "--output-dir",
            str(tmp_path),
            "--width",
            "120",
            "--height",
            "90",
            "--spacing",
            "30",
            "--seed",
            "20",
            "--seed-count",
            "2",
        ]
    )

    manifest = json.loads((tmp_path / "control_seed_manifest.json").read_text())
    assert rc == 0
    assert manifest["seed_count"] == 2
    assert (tmp_path / "seed_0020" / "clean_lozenge_lattice.png").exists()
    assert (tmp_path / "seed_0021" / "random_scratches.png").exists()


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
        assert control["stroke_width"] == 3
        assert control["control_polarity"] == "bright-on-dark"


def test_clean_lozenge_control_vectorizes_to_nonzero_graph_and_lozenges(tmp_path: Path):
    generate_control_set(
        tmp_path,
        width=160,
        height=120,
        spacing=32,
        seed=4,
        stroke_width=3,
    )

    metrics = run_pipeline(
        input_path=tmp_path / "clean_lozenge_lattice.png",
        output_path=tmp_path / "clean.svg",
        debug_dir=tmp_path / "debug",
        metrics_path=tmp_path / "metrics.json",
        graph_path=tmp_path / "graph.graphml",
        config=PipelineConfig(
            preprocess=PreprocessParams(
                crop="none",
                threshold_mode="global",
                threshold_value=180,
                morph_kernel_size=1,
                min_component_size=1,
                denoise_kernel_size=1,
                clahe_clip_limit=1.0,
            ),
            simplification_epsilon=0.0,
            node_merge_radius=2.0,
        ),
    )
    graph_metrics = analyze_graphml(tmp_path / "graph_merged.graphml")

    assert metrics.paths > 0
    assert graph_metrics["node_count"] > 0
    assert graph_metrics["edge_count"] > 0
    assert graph_metrics["cycle_count"] > 0
    assert graph_metrics["lozenge_candidate_count"] > 0


def test_compare_groups_extracted_controls_once_and_reports_invalid(tmp_path: Path):
    controls = tmp_path / "controls"
    lozenge_dir = controls / "clean_lozenge_lattice"
    random_dir = controls / "random_scratches"
    invalid_dir = controls / "empty_random_scratches"
    lozenge_dir.mkdir(parents=True)
    random_dir.mkdir()
    invalid_dir.mkdir()

    lozenge_graph = lozenge_grid_graph(width=180, height=140, spacing=35)
    random_graph = graph_from_strokes(
        random_scratch_strokes(
            width=180,
            height=140,
            stroke_count=24,
            target_total_length=1200,
            rng=random.Random(12),
        )
    )
    save_graphml(lozenge_graph, lozenge_dir / "graph_merged.graphml")
    save_graphml(lozenge_graph, lozenge_dir / "graph_raw.graphml")
    save_graphml(random_graph, random_dir / "graph_merged.graphml")
    save_graphml(random_graph, random_dir / "graph_raw.graphml")
    (lozenge_dir / "metrics.json").write_text(json.dumps({"graph_metrics": {"merged_node_count": 10}}))
    (random_dir / "metrics.json").write_text(json.dumps({"graph_metrics": {"merged_node_count": 10}}))
    (invalid_dir / "metrics.json").write_text(
        json.dumps({"graph_metrics": {"merged_node_count": 0, "total_traced_length_px": 0}})
    )

    csv_path, json_path = compare_null_models(
        observed_graph=lozenge_dir / "graph_merged.graphml",
        observed_metrics=None,
        controls_dir=controls,
        output_dir=tmp_path / "comparison",
    )
    rows = list(csv.DictReader(csv_path.open()))
    payload = json.loads(json_path.read_text())
    by_metric = {row["metric"]: row for row in rows}
    classes = [control["class"] for control in payload["valid_controls"]]

    assert classes.count("lozenge") == 1
    assert classes.count("random") == 1
    assert payload["class_aggregates"]["lozenge"]["n_controls"] == 1
    assert payload["class_aggregates"]["random"]["n_controls"] == 1
    assert payload["invalid_controls"]
    assert "node_count == 0" in payload["invalid_controls"][0]["invalid_reason"]
    assert float(by_metric["total_traced_length"]["lozenge_control_mean"]) > 0
    assert float(by_metric["total_traced_length"]["random_control_mean"]) > 0
    assert (tmp_path / "comparison" / "invalid_controls.csv").exists()
