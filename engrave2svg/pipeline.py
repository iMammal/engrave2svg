from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .gap_bridge import bridge_gaps, save_gap_bridge_debug
from .graph_metrics import (
    build_engraving_graphs,
    save_merged_nodes_debug,
    write_graph_exports,
    write_metrics_json,
    write_orientation_histogram,
)
from .graph_trace import save_trace_debug, simplify_polylines, trace_skeleton
from .preprocessing import (
    PreprocessParams,
    load_image,
    preprocess_image,
    save_preprocess_debug,
)
from .skeleton import analyze_skeleton, save_skeleton_debug, skeletonize_binary
from .svg_export import export_svg


@dataclass(frozen=True)
class PipelineConfig:
    preprocess: PreprocessParams
    simplification_epsilon: float = 1.25
    stroke_width: float = 1.2
    node_merge_radius: float = 0.0
    bridge_gaps_radius: float = 0.0
    bridge_gaps_angle_tolerance: float = 30.0

    def to_flat_dict(self) -> dict[str, object]:
        values = asdict(self.preprocess)
        values["simplification_epsilon"] = self.simplification_epsilon
        values["stroke_width"] = self.stroke_width
        values["node_merge_radius"] = self.node_merge_radius
        values["bridge_gaps_radius"] = self.bridge_gaps_radius
        values["bridge_gaps_angle_tolerance"] = self.bridge_gaps_angle_tolerance
        return values


@dataclass(frozen=True)
class PipelineMetrics:
    output_svg: str
    debug_dir: str
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    foreground_pixels: int
    skeleton_pixels: int
    graph_nodes: int
    graph_edges: int
    paths: int
    endpoints: int
    junctions: int
    bridge_count: int
    node_merge_radius: float
    bridge_gaps_radius: float
    bridge_gaps_angle_tolerance: float
    raw_node_count: int
    raw_endpoint_count: int
    raw_junction_count: int
    raw_edge_count: int
    merged_node_count: int
    merged_endpoint_count: int
    merged_junction_count: int
    merged_edge_count: int
    connected_components: int
    average_node_degree: float
    graph_density: float
    total_traced_length_px: float
    orientation_circular_mean_deg: float
    orientation_circular_variance: float
    dominant_angle_peaks: str
    output_metrics: str = ""
    graph_raw: str = ""
    graph_merged: str = ""
    nodes_raw_csv: str = ""
    nodes_merged_csv: str = ""
    edges_raw_csv: str = ""
    edges_merged_csv: str = ""
    orientation_histogram: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    debug_dir: str | Path | None,
    config: PipelineConfig,
    metrics_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    nodes_csv_path: str | Path | None = None,
    edges_csv_path: str | Path | None = None,
    orientation_hist_path: str | Path | None = None,
) -> PipelineMetrics:
    image = load_image(input_path)
    preprocessed = preprocess_image(image, config.preprocess)
    bridge_result = bridge_gaps(
        preprocessed.cleaned,
        radius=config.bridge_gaps_radius,
        angle_tolerance=config.bridge_gaps_angle_tolerance,
    )
    skeleton = skeletonize_binary(bridge_result.bridged)
    analysis = analyze_skeleton(skeleton)
    trace = trace_skeleton(analysis.skeleton)
    polylines = simplify_polylines(trace.polylines, config.simplification_epsilon)
    graph_bundle = build_engraving_graphs(
        trace,
        polylines,
        node_merge_radius=config.node_merge_radius,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = analysis.skeleton.shape
    export_svg(
        polylines,
        output,
        width=width,
        height=height,
        stroke_width=config.stroke_width,
    )

    debug = ""
    if debug_dir:
        debug_path = Path(debug_dir)
        debug = str(debug_path)
        save_preprocess_debug(preprocessed, debug_path)
        save_gap_bridge_debug(bridge_result, debug_path)
        save_skeleton_debug(analysis, debug_path)
        save_trace_debug(analysis.skeleton, polylines, debug_path)
        cv2.imwrite(str(debug_path / "10_final_skeleton.png"), analysis.skeleton)
        save_merged_nodes_debug(analysis.skeleton, graph_bundle, debug_path)

    written = write_graph_exports(
        graph_path=graph_path,
        nodes_csv_path=nodes_csv_path,
        edges_csv_path=edges_csv_path,
        bundle=graph_bundle,
    )
    if orientation_hist_path:
        write_orientation_histogram(orientation_hist_path, graph_bundle.orientation)
        written["orientation_histogram"] = str(orientation_hist_path)

    graph_metrics = {
        **graph_bundle.metrics,
        "bridge_gaps_radius": float(config.bridge_gaps_radius),
        "bridge_gaps_angle_tolerance": float(config.bridge_gaps_angle_tolerance),
        "bridge_count": bridge_result.bridge_count,
        "bridges": bridge_result.bridges_as_dicts(),
    }
    orientation = graph_bundle.orientation
    merged = graph_metrics["merged"]
    pipeline_metrics = PipelineMetrics(
        output_svg=str(output),
        debug_dir=debug,
        crop_x=preprocessed.crop_box.x,
        crop_y=preprocessed.crop_box.y,
        crop_width=preprocessed.crop_box.width,
        crop_height=preprocessed.crop_box.height,
        foreground_pixels=int(np.count_nonzero(preprocessed.cleaned)),
        skeleton_pixels=int(np.count_nonzero(analysis.skeleton)),
        graph_nodes=trace.graph.number_of_nodes(),
        graph_edges=trace.graph.number_of_edges(),
        paths=len(polylines),
        endpoints=len(analysis.endpoints),
        junctions=len(analysis.junctions),
        bridge_count=bridge_result.bridge_count,
        node_merge_radius=config.node_merge_radius,
        bridge_gaps_radius=config.bridge_gaps_radius,
        bridge_gaps_angle_tolerance=config.bridge_gaps_angle_tolerance,
        raw_node_count=int(graph_metrics["raw_node_count"]),
        raw_endpoint_count=int(graph_metrics["raw_endpoint_count"]),
        raw_junction_count=int(graph_metrics["raw_junction_count"]),
        raw_edge_count=int(graph_metrics["raw_edge_count"]),
        merged_node_count=int(graph_metrics["merged_node_count"]),
        merged_endpoint_count=int(graph_metrics["merged_endpoint_count"]),
        merged_junction_count=int(graph_metrics["merged_junction_count"]),
        merged_edge_count=int(graph_metrics["merged_edge_count"]),
        connected_components=int(graph_metrics["connected_components"]),
        average_node_degree=float(merged["average_node_degree"]),
        graph_density=float(merged["graph_density"]),
        total_traced_length_px=float(graph_metrics["total_traced_length_px"]),
        orientation_circular_mean_deg=float(orientation["circular_mean_deg"]),
        orientation_circular_variance=float(orientation["circular_variance"]),
        dominant_angle_peaks=json_dumps_compact(graph_metrics["dominant_angle_peaks"]),
        output_metrics=str(metrics_path) if metrics_path else "",
        graph_raw=written.get("graph_raw", ""),
        graph_merged=written.get("graph_merged", ""),
        nodes_raw_csv=written.get("nodes_raw_csv", ""),
        nodes_merged_csv=written.get("nodes_merged_csv", ""),
        edges_raw_csv=written.get("edges_raw_csv", ""),
        edges_merged_csv=written.get("edges_merged_csv", ""),
        orientation_histogram=written.get("orientation_histogram", ""),
    )

    if metrics_path:
        write_metrics_json(
            metrics_path,
            pipeline_metrics=pipeline_metrics.to_dict(),
            graph_metrics=graph_metrics,
            config=config.to_flat_dict(),
        )

    return pipeline_metrics


def json_dumps_compact(value: object) -> str:
    import json

    return json.dumps(value, separators=(",", ":"))
