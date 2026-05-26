from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import networkx as nx
import numpy as np

from .graph_trace import Point, Polyline, TraceResult


@dataclass(frozen=True)
class EngravingGraphBundle:
    raw_graph: nx.MultiGraph
    merged_graph: nx.MultiGraph
    metrics: dict[str, object]
    orientation: dict[str, object]


def build_engraving_graphs(
    trace: TraceResult,
    polylines: list[Polyline],
    node_merge_radius: float = 0.0,
) -> EngravingGraphBundle:
    component_by_point = _pixel_component_lookup(trace.graph)
    raw_nodes = _raw_node_points(trace.graph, polylines, component_by_point)
    raw_graph = _build_raw_graph(raw_nodes, polylines)
    raw_stats = graph_statistics(raw_graph, count_mode="stored_type")

    cluster_by_point, merge_report = _cluster_raw_nodes(
        raw_nodes,
        radius=node_merge_radius,
        component_by_point=component_by_point,
    )
    merged_graph = _build_merged_graph(raw_nodes, polylines, cluster_by_point)
    merged_stats = graph_statistics(merged_graph, count_mode="topology")
    orientation = orientation_statistics(merged_graph)

    metrics = {
        "node_merge_radius": float(node_merge_radius),
        "raw": raw_stats,
        "merged": merged_stats,
        "merge_report": merge_report,
        "orientation": orientation,
        "raw_node_count": raw_stats["node_count"],
        "raw_endpoint_count": raw_stats["endpoints"],
        "raw_junction_count": raw_stats["junctions"],
        "raw_edge_count": raw_stats["edge_count"],
        "merged_node_count": merged_stats["node_count"],
        "merged_endpoint_count": merged_stats["endpoints"],
        "merged_junction_count": merged_stats["junctions"],
        "merged_edge_count": merged_stats["edge_count"],
        "connected_components": merged_stats["connected_components"],
        "total_traced_length_px": merged_stats["total_traced_length_px"],
        "dominant_angle_peaks": orientation["peaks"],
    }
    return EngravingGraphBundle(
        raw_graph=raw_graph,
        merged_graph=merged_graph,
        metrics=metrics,
        orientation=orientation,
    )


def graph_statistics(
    graph: nx.MultiGraph, count_mode: str = "topology"
) -> dict[str, object]:
    degrees = {node: int(degree) for node, degree in graph.degree()}
    degree_distribution: dict[str, int] = {}
    for degree in degrees.values():
        degree_distribution[str(degree)] = degree_distribution.get(str(degree), 0) + 1

    if count_mode == "stored_type":
        endpoints = sum(1 for _, data in graph.nodes(data=True) if data["node_type"] == "endpoint")
        junctions = sum(1 for _, data in graph.nodes(data=True) if data["node_type"] == "junction")
    else:
        endpoints = sum(1 for degree in degrees.values() if degree == 1)
        junctions = sum(1 for degree in degrees.values() if degree >= 3)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    simple = nx.Graph(graph)
    density = float(nx.density(simple)) if node_count > 1 else 0.0
    components = nx.number_connected_components(simple) if node_count else 0
    total_length = sum(float(data.get("length_px", 0.0)) for _, _, data in graph.edges(data=True))

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "endpoints": endpoints,
        "junctions": junctions,
        "degree_distribution": dict(sorted(degree_distribution.items(), key=lambda item: int(item[0]))),
        "connected_components": components,
        "average_node_degree": float(sum(degrees.values()) / node_count) if node_count else 0.0,
        "graph_density": density,
        "total_traced_length_px": float(total_length),
    }


def orientation_statistics(
    graph: nx.MultiGraph,
    bins: int = 18,
) -> dict[str, object]:
    edges = [
        (float(data["orientation_deg"]), float(data["length_px"]))
        for _, _, data in graph.edges(data=True)
        if float(data.get("length_px", 0.0)) > 0
    ]
    bin_edges = np.linspace(0.0, 180.0, bins + 1)
    hist = np.zeros(bins, dtype=np.float64)
    edge_counts = np.zeros(bins, dtype=np.int32)

    for angle, length in edges:
        index = min(bins - 1, int(angle / 180.0 * bins))
        hist[index] += length
        edge_counts[index] += 1

    total_weight = float(hist.sum())
    if total_weight > 0:
        doubled = np.deg2rad([angle * 2.0 for angle, _ in edges])
        weights = np.array([length for _, length in edges], dtype=np.float64)
        sin_sum = float(np.sum(np.sin(doubled) * weights))
        cos_sum = float(np.sum(np.cos(doubled) * weights))
        mean = (math.degrees(math.atan2(sin_sum, cos_sum)) / 2.0) % 180.0
        resultant = math.hypot(sin_sum, cos_sum) / total_weight
        circular_variance = 1.0 - resultant
    else:
        mean = 0.0
        circular_variance = 0.0

    peaks = _dominant_peaks(hist, edge_counts, bin_edges)
    return {
        "bin_count": bins,
        "bin_edges_deg": [float(value) for value in bin_edges.tolist()],
        "length_weighted_histogram": [float(value) for value in hist.tolist()],
        "edge_counts": [int(value) for value in edge_counts.tolist()],
        "circular_mean_deg": float(mean),
        "circular_variance": float(circular_variance),
        "peaks": peaks,
    }


def write_graph_exports(
    graph_path: str | Path | None,
    nodes_csv_path: str | Path | None,
    edges_csv_path: str | Path | None,
    bundle: EngravingGraphBundle,
) -> dict[str, str]:
    written: dict[str, str] = {}
    if graph_path:
        raw_path, merged_path = _raw_merged_paths(graph_path)
        _write_network_graph(bundle.raw_graph, raw_path)
        _write_network_graph(bundle.merged_graph, merged_path)
        written["graph_raw"] = str(raw_path)
        written["graph_merged"] = str(merged_path)

    if nodes_csv_path:
        raw_path, merged_path = _raw_merged_paths(nodes_csv_path)
        write_nodes_csv(bundle.raw_graph, raw_path)
        write_nodes_csv(bundle.merged_graph, merged_path)
        written["nodes_raw_csv"] = str(raw_path)
        written["nodes_merged_csv"] = str(merged_path)

    if edges_csv_path:
        raw_path, merged_path = _raw_merged_paths(edges_csv_path)
        write_edges_csv(bundle.raw_graph, raw_path)
        write_edges_csv(bundle.merged_graph, merged_path)
        written["edges_raw_csv"] = str(raw_path)
        written["edges_merged_csv"] = str(merged_path)

    return written


def write_nodes_csv(graph: nx.MultiGraph, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "x_px",
        "y_px",
        "x_calibrated",
        "y_calibrated",
        "degree",
        "pixel_degree",
        "node_type",
        "component_id",
        "raw_node_count",
        "raw_node_ids",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node_id, data in sorted(graph.nodes(data=True), key=lambda item: item[0]):
            writer.writerow({field: data.get(field, "") for field in fields} | {"id": node_id})


def write_edges_csv(graph: nx.MultiGraph, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "source",
        "target",
        "length_px",
        "length_calibrated",
        "orientation_deg",
        "point_count",
        "polyline",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source, target, _key, data in sorted(
            graph.edges(keys=True, data=True), key=lambda item: str(item[3].get("id", ""))
        ):
            writer.writerow(
                {
                    "id": data.get("id", ""),
                    "source": source,
                    "target": target,
                    "length_px": data.get("length_px", 0.0),
                    "length_calibrated": data.get("length_calibrated", ""),
                    "orientation_deg": data.get("orientation_deg", 0.0),
                    "point_count": data.get("point_count", 0),
                    "polyline": data.get("polyline", ""),
                }
            )


def write_metrics_json(
    path: str | Path,
    pipeline_metrics: dict[str, object],
    graph_metrics: dict[str, object],
    config: dict[str, object],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline_metrics,
        "parameters": config,
        "graph_metrics": graph_metrics,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def write_orientation_histogram(
    path: str | Path,
    orientation: dict[str, object],
    width: int = 900,
    height: int = 520,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_left = 70
    margin_right = 30
    margin_top = 35
    margin_bottom = 70
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    hist = np.array(orientation["length_weighted_histogram"], dtype=np.float64)
    max_value = float(hist.max()) if hist.size else 0.0
    cv2.rectangle(
        image,
        (margin_left, margin_top),
        (margin_left + chart_w, margin_top + chart_h),
        (30, 30, 30),
        1,
    )
    if max_value > 0:
        bar_w = chart_w / len(hist)
        for index, value in enumerate(hist):
            x0 = int(margin_left + index * bar_w)
            x1 = int(margin_left + (index + 1) * bar_w) - 2
            bar_h = int(chart_h * value / max_value)
            y0 = margin_top + chart_h - bar_h
            cv2.rectangle(image, (x0, y0), (x1, margin_top + chart_h), (42, 96, 164), -1)

    for angle in (0, 45, 90, 135, 180):
        x = int(margin_left + chart_w * angle / 180.0)
        cv2.line(image, (x, margin_top + chart_h), (x, margin_top + chart_h + 6), (0, 0, 0), 1)
        cv2.putText(
            image,
            str(angle),
            (x - 12, margin_top + chart_h + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "Length-weighted edge orientation (degrees)",
        (margin_left, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Total traced length",
        (10, margin_top + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), image)


def save_merged_nodes_debug(
    skeleton: np.ndarray,
    bundle: EngravingGraphBundle,
    debug_dir: str | Path,
) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for _, data in bundle.raw_graph.nodes(data=True):
        cv2.circle(canvas, (int(round(data["x_px"])), int(round(data["y_px"]))), 2, (0, 0, 255), -1)
    for _, data in bundle.merged_graph.nodes(data=True):
        cv2.circle(canvas, (int(round(data["x_px"])), int(round(data["y_px"]))), 4, (0, 180, 0), 1)
    cv2.imwrite(str(path / "11_merged_nodes.png"), canvas)


def _build_raw_graph(
    raw_nodes: dict[Point, dict[str, object]], polylines: list[Polyline]
) -> nx.MultiGraph:
    graph = nx.MultiGraph(
        graph_type="raw_engraving_graph",
        coordinate_units="pixels",
    )
    node_id_by_point = _node_ids(raw_nodes)
    for point, data in sorted(raw_nodes.items(), key=lambda item: _point_sort_key(item[0])):
        graph.add_node(node_id_by_point[point], **data)

    for index, polyline in enumerate(polylines):
        if len(polyline.points) < 2:
            continue
        source_point, target_point = _edge_terminal_points(polyline)
        for point in (source_point, target_point):
            if point not in node_id_by_point:
                raw_nodes[point] = _terminal_node_data(point)
                node_id_by_point[point] = f"n{len(node_id_by_point):05d}"
                graph.add_node(node_id_by_point[point], **raw_nodes[point])
        attrs = _edge_attrs(index, polyline)
        graph.add_edge(node_id_by_point[source_point], node_id_by_point[target_point], key=attrs["id"], **attrs)

    _annotate_graph_degrees(graph)
    return graph


def _build_merged_graph(
    raw_nodes: dict[Point, dict[str, object]],
    polylines: list[Polyline],
    cluster_by_point: dict[Point, str],
) -> nx.MultiGraph:
    graph = nx.MultiGraph(
        graph_type="merged_engraving_graph",
        coordinate_units="pixels",
    )
    grouped: dict[str, list[Point]] = {}
    for point, cluster_id in cluster_by_point.items():
        grouped.setdefault(cluster_id, []).append(point)

    for cluster_id, points in sorted(grouped.items()):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        raw_ids = [_format_point(point) for point in sorted(points, key=_point_sort_key)]
        pixel_degrees = [int(raw_nodes[point].get("pixel_degree", 0)) for point in points]
        graph.add_node(
            cluster_id,
            x_px=float(sum(xs) / len(xs)),
            y_px=float(sum(ys) / len(ys)),
            x_calibrated="",
            y_calibrated="",
            degree=0,
            pixel_degree=max(pixel_degrees) if pixel_degrees else 0,
            node_type="unclassified",
            component_id=str(raw_nodes[points[0]].get("component_id", "")),
            raw_node_count=len(points),
            raw_node_ids=";".join(raw_ids),
        )

    for index, polyline in enumerate(polylines):
        if len(polyline.points) < 2:
            continue
        source_point, target_point = _edge_terminal_points(polyline)
        source = cluster_by_point[source_point]
        target = cluster_by_point[target_point]
        if source == target:
            continue
        attrs = _edge_attrs(index, polyline)
        graph.add_edge(source, target, key=attrs["id"], **attrs)

    _annotate_graph_degrees(graph)
    for node_id, degree in graph.degree():
        graph.nodes[node_id]["node_type"] = _node_type_from_degree(int(degree))
    return graph


def _raw_node_points(
    pixel_graph: nx.Graph,
    polylines: list[Polyline],
    component_by_point: dict[Point, int],
) -> dict[Point, dict[str, object]]:
    raw_nodes: dict[Point, dict[str, object]] = {}
    for point, degree in pixel_graph.degree:
        if degree == 2:
            continue
        raw_nodes[point] = {
            "x_px": int(point[0]),
            "y_px": int(point[1]),
            "x_calibrated": "",
            "y_calibrated": "",
            "degree": 0,
            "pixel_degree": int(degree),
            "node_type": _raw_node_type(int(degree)),
            "component_id": str(component_by_point.get(point, -1)),
            "raw_node_count": 1,
            "raw_node_ids": _format_point(point),
        }

    for polyline in polylines:
        if len(polyline.points) < 2:
            continue
        for point in _edge_terminal_points(polyline):
            if point not in raw_nodes:
                raw_nodes[point] = _terminal_node_data(point, component_by_point.get(point, -1))
    return raw_nodes


def _cluster_raw_nodes(
    raw_nodes: dict[Point, dict[str, object]],
    radius: float,
    component_by_point: dict[Point, int],
) -> tuple[dict[Point, str], dict[str, object]]:
    points = sorted(raw_nodes, key=_point_sort_key)
    if radius <= 0:
        return (
            {point: f"m{index:05d}" for index, point in enumerate(points)},
            {
                "clusters": len(points),
                "merged_clusters": 0,
                "largest_cluster_size": 1 if points else 0,
                "merged_raw_node_count": 0,
            },
        )

    parent = {point: point for point in points}

    def find(point: Point) -> Point:
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = parent[point]
        return point

    def union(a: Point, b: Point) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for index, point in enumerate(points):
        for other in points[index + 1 :]:
            if component_by_point.get(point, -1) != component_by_point.get(other, -2):
                continue
            if math.dist(point, other) <= radius:
                union(point, other)

    grouped: dict[Point, list[Point]] = {}
    for point in points:
        grouped.setdefault(find(point), []).append(point)

    cluster_by_point: dict[Point, str] = {}
    for index, members in enumerate(sorted(grouped.values(), key=lambda group: _point_sort_key(group[0]))):
        cluster_id = f"m{index:05d}"
        for point in members:
            cluster_by_point[point] = cluster_id

    cluster_sizes = [len(members) for members in grouped.values()]
    merged_clusters = sum(1 for size in cluster_sizes if size > 1)
    return (
        cluster_by_point,
        {
            "clusters": len(grouped),
            "merged_clusters": merged_clusters,
            "largest_cluster_size": max(cluster_sizes, default=0),
            "merged_raw_node_count": sum(size for size in cluster_sizes if size > 1),
        },
    )


def _pixel_component_lookup(pixel_graph: nx.Graph) -> dict[Point, int]:
    lookup: dict[Point, int] = {}
    for index, component in enumerate(nx.connected_components(pixel_graph)):
        for point in component:
            lookup[point] = index
    return lookup


def _edge_attrs(index: int, polyline: Polyline) -> dict[str, object]:
    length = _polyline_length(polyline.points)
    orientation = _orientation(polyline.points[0], polyline.points[-1])
    return {
        "id": f"e{index:05d}",
        "length_px": float(length),
        "length_calibrated": "",
        "orientation_deg": float(orientation),
        "point_count": len(polyline.points),
        "closed": int(polyline.closed),
        "polyline": " ".join(_format_point(point) for point in polyline.points),
    }


def _edge_terminal_points(polyline: Polyline) -> tuple[Point, Point]:
    if polyline.closed:
        return polyline.points[0], polyline.points[-1]
    return polyline.points[0], polyline.points[-1]


def _polyline_length(points: Iterable[Point]) -> float:
    ordered = list(points)
    return float(
        sum(math.dist(a, b) for a, b in zip(ordered, ordered[1:]))
    )


def _orientation(a: Point, b: Point) -> float:
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    return float(angle % 180.0)


def _dominant_peaks(
    hist: np.ndarray, edge_counts: np.ndarray, bin_edges: np.ndarray
) -> list[dict[str, object]]:
    if hist.size == 0 or float(hist.max()) <= 0:
        return []
    threshold = float(hist.max()) * 0.25
    total = float(hist.sum())
    peaks: list[dict[str, object]] = []
    for index, value in enumerate(hist):
        prev_value = hist[index - 1] if index > 0 else hist[-1]
        next_value = hist[index + 1] if index < hist.size - 1 else hist[0]
        if value < threshold or value < prev_value or value < next_value:
            continue
        start = float(bin_edges[index])
        end = float(bin_edges[index + 1])
        peaks.append(
            {
                "bin_index": int(index),
                "angle_center_deg": float((start + end) / 2.0),
                "bin_start_deg": start,
                "bin_end_deg": end,
                "length_weight": float(value),
                "length_fraction": float(value / total) if total else 0.0,
                "edge_count": int(edge_counts[index]),
            }
        )
    return sorted(peaks, key=lambda peak: float(peak["length_weight"]), reverse=True)


def _annotate_graph_degrees(graph: nx.MultiGraph) -> None:
    for node_id, degree in graph.degree():
        graph.nodes[node_id]["degree"] = int(degree)


def _raw_node_type(degree: int) -> str:
    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree >= 3:
        return "junction"
    return "connector"


def _node_type_from_degree(degree: int) -> str:
    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree >= 3:
        return "junction"
    return "connector"


def _terminal_node_data(point: Point, component_id: int = -1) -> dict[str, object]:
    return {
        "x_px": int(point[0]),
        "y_px": int(point[1]),
        "x_calibrated": "",
        "y_calibrated": "",
        "degree": 0,
        "pixel_degree": 0,
        "node_type": "traced_terminal",
        "component_id": str(component_id),
        "raw_node_count": 1,
        "raw_node_ids": _format_point(point),
    }


def _node_ids(raw_nodes: dict[Point, dict[str, object]]) -> dict[Point, str]:
    return {
        point: f"n{index:05d}"
        for index, point in enumerate(sorted(raw_nodes, key=_point_sort_key))
    }


def _raw_merged_paths(path: str | Path) -> tuple[Path, Path]:
    base = Path(path)
    return (
        base.with_name(f"{base.stem}_raw{base.suffix}"),
        base.with_name(f"{base.stem}_merged{base.suffix}"),
    )


def _write_network_graph(graph: nx.MultiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gexf":
        nx.write_gexf(graph, path)
    else:
        nx.write_graphml(graph, path)


def _format_point(point: Point) -> str:
    return f"{int(point[0])},{int(point[1])}"


def _point_sort_key(point: Point) -> tuple[int, int]:
    return (int(point[1]), int(point[0]))
