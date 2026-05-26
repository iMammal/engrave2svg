import csv
import json
from pathlib import Path

import cv2
import numpy as np

from engrave2svg.graph_metrics import (
    build_engraving_graphs,
    write_graph_exports,
    write_orientation_histogram,
)
from engrave2svg.graph_trace import trace_skeleton
from engrave2svg.pipeline import PipelineConfig, run_pipeline
from engrave2svg.preprocessing import PreprocessParams


def test_crosshatch_merges_junction_cluster_and_preserves_degrees():
    skeleton = np.zeros((31, 31), dtype=np.uint8)
    skeleton[15, 5:26] = 255
    skeleton[5:26, 15] = 255
    skeleton[14:17, 14:17] = 255

    trace = trace_skeleton(skeleton)
    bundle = build_engraving_graphs(trace, trace.polylines, node_merge_radius=2.0)

    assert bundle.metrics["raw_junction_count"] > 1
    assert bundle.metrics["merged_junction_count"] == 1
    assert bundle.metrics["merged_endpoint_count"] == 4
    assert bundle.metrics["merged_edge_count"] == 4
    junctions = [
        data
        for _, data in bundle.merged_graph.nodes(data=True)
        if data["node_type"] == "junction"
    ]
    assert junctions[0]["degree"] == 4


def test_crosshatch_image_pipeline_reports_known_node_degrees(tmp_path: Path):
    image = np.zeros((41, 41, 3), dtype=np.uint8)
    cv2.line(image, (20, 6), (20, 34), (255, 255, 255), 1)
    cv2.line(image, (6, 20), (34, 20), (255, 255, 255), 1)
    input_path = tmp_path / "crosshatch.png"
    cv2.imwrite(str(input_path), image)

    metrics = run_pipeline(
        input_path=input_path,
        output_path=tmp_path / "out.svg",
        debug_dir=tmp_path / "debug",
        config=PipelineConfig(
            preprocess=PreprocessParams(
                crop="none",
                threshold_mode="global",
                threshold_value=180,
                morph_kernel_size=1,
                min_component_size=1,
                denoise_kernel_size=1,
            ),
            simplification_epsilon=0.0,
            node_merge_radius=2.0,
        ),
    )

    assert metrics.raw_junction_count > 1
    assert metrics.merged_junction_count == 1
    assert metrics.merged_endpoint_count == 4
    assert metrics.merged_edge_count == 4
    assert (tmp_path / "debug" / "11_merged_nodes.png").exists()


def test_diamond_cross_has_two_orientation_families(tmp_path: Path):
    skeleton = np.zeros((41, 41), dtype=np.uint8)
    cv2.line(skeleton, (8, 8), (32, 32), 255, 1)
    cv2.line(skeleton, (8, 32), (32, 8), 255, 1)

    trace = trace_skeleton(skeleton)
    bundle = build_engraving_graphs(trace, trace.polylines, node_merge_radius=2.0)
    peaks = bundle.orientation["peaks"]
    peak_angles = sorted(round(peak["angle_center_deg"]) for peak in peaks[:2])

    assert peak_angles == [45, 135]
    hist_path = tmp_path / "orientation.png"
    write_orientation_histogram(hist_path, bundle.orientation)
    assert hist_path.exists()


def test_graph_exports_write_raw_and_merged_files(tmp_path: Path):
    skeleton = np.zeros((21, 21), dtype=np.uint8)
    skeleton[10, 4:17] = 255
    trace = trace_skeleton(skeleton)
    bundle = build_engraving_graphs(trace, trace.polylines, node_merge_radius=0.0)

    written = write_graph_exports(
        graph_path=tmp_path / "output.graphml",
        nodes_csv_path=tmp_path / "nodes.csv",
        edges_csv_path=tmp_path / "edges.csv",
        bundle=bundle,
    )

    assert Path(written["graph_raw"]).name == "output_raw.graphml"
    assert Path(written["graph_merged"]).name == "output_merged.graphml"
    rows = list(csv.DictReader(Path(written["nodes_merged_csv"]).open()))
    assert len(rows) == 2
    assert {row["node_type"] for row in rows} == {"endpoint"}


def test_metrics_json_shape_from_bundle():
    skeleton = np.zeros((21, 21), dtype=np.uint8)
    skeleton[10, 4:17] = 255
    trace = trace_skeleton(skeleton)
    bundle = build_engraving_graphs(trace, trace.polylines, node_merge_radius=1.0)

    encoded = json.dumps(bundle.metrics)
    decoded = json.loads(encoded)

    assert decoded["node_merge_radius"] == 1.0
    assert decoded["raw_node_count"] == 2
    assert decoded["merged_node_count"] == 2
