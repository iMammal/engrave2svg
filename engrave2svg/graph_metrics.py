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
    del polylines
    raw_graph = _build_topology_graph(trace.graph)
    raw_stats = graph_statistics(raw_graph)
    merged_graph, merge_report = _merge_graph_nodes(raw_graph, node_merge_radius)
    merged_stats = graph_statistics(merged_graph)
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
        "node_degree_histogram": merged_stats["degree_distribution"],
        "connected_component_size_histogram": merged_stats[
            "connected_component_size_histogram"
        ],
        "largest_connected_component_fraction": merged_stats[
            "largest_connected_component_fraction"
        ],
    }
    return EngravingGraphBundle(
        raw_graph=raw_graph,
        merged_graph=merged_graph,
        metrics=metrics,
        orientation=orientation,
    )


def graph_statistics(graph: nx.MultiGraph) -> dict[str, object]:
    degrees = {node: int(degree) for node, degree in graph.degree()}
    degree_distribution: dict[str, int] = {}
    for degree in degrees.values():
        degree_distribution[str(degree)] = degree_distribution.get(str(degree), 0) + 1

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    simple = nx.Graph(graph)
    component_sizes = [
        len(component) for component in nx.connected_components(simple)
    ] if node_count else []
    component_size_histogram: dict[str, int] = {}
    for size in component_sizes:
        component_size_histogram[str(size)] = component_size_histogram.get(str(size), 0) + 1

    total_length = sum(
        float(data.get("length_px", 0.0)) for _, _, data in graph.edges(data=True)
    )
    largest = max(component_sizes, default=0)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "endpoints": sum(1 for degree in degrees.values() if degree == 1),
        "junctions": sum(1 for degree in degrees.values() if degree >= 3),
        "degree_distribution": dict(
            sorted(degree_distribution.items(), key=lambda item: int(item[0]))
        ),
        "node_degree_histogram": dict(
            sorted(degree_distribution.items(), key=lambda item: int(item[0]))
        ),
        "connected_components": len(component_sizes),
        "connected_component_size_histogram": dict(
            sorted(component_size_histogram.items(), key=lambda item: int(item[0]))
        ),
        "largest_connected_component_fraction": float(largest / node_count)
        if node_count
        else 0.0,
        "average_node_degree": float(sum(degrees.values()) / node_count)
        if node_count
        else 0.0,
        "graph_density": float(nx.density(simple)) if node_count > 1 else 0.0,
        "total_traced_length_px": float(total_length),
    }


def orientation_statistics(graph: nx.MultiGraph, bins: int = 18) -> dict[str, object]:
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

    return {
        "bin_count": bins,
        "bin_edges_deg": [float(value) for value in bin_edges.tolist()],
        "length_weighted_histogram": [float(value) for value in hist.tolist()],
        "edge_counts": [int(value) for value in edge_counts.tolist()],
        "circular_mean_deg": float(mean),
        "circular_variance": float(circular_variance),
        "peaks": _dominant_peaks(hist, edge_counts, bin_edges),
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
            graph.edges(keys=True, data=True),
            key=lambda item: str(item[3].get("id", "")),
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
    output.write_text(
        json.dumps(
            {
                "pipeline": pipeline_metrics,
                "parameters": config,
                "graph_metrics": graph_metrics,
            },
            indent=2,
        )
        + "\n"
    )


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
            y0 = margin_top + chart_h - int(chart_h * value / max_value)
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
    cv2.imwrite(str(output), image)


def save_merged_nodes_debug(
    skeleton: np.ndarray,
    bundle: EngravingGraphBundle,
    debug_dir: str | Path,
) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    component_colors = _component_colors(bundle.merged_graph)
    for source, target, data in bundle.merged_graph.edges(data=True):
        color = component_colors.get(source, (120, 120, 120))
        points = _parse_polyline(data.get("polyline", ""))
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    for node_id, data in bundle.merged_graph.nodes(data=True):
        point = (int(round(data["x_px"])), int(round(data["y_px"])))
        node_type = data.get("node_type", "")
        if node_type == "junction":
            color = (0, 0, 255)
            radius = 4
        elif node_type == "endpoint":
            color = (0, 180, 0)
            radius = 3
        else:
            color = (255, 0, 0)
            radius = 3
        cv2.circle(canvas, point, radius, color, -1)
        cv2.putText(
            canvas,
            str(data.get("degree", "")),
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path / "11_merged_nodes.png"), canvas)


def _build_topology_graph(pixel_graph: nx.Graph) -> nx.MultiGraph:
    graph = nx.MultiGraph(graph_type="skeleton_topology_graph", coordinate_units="pixels")
    if pixel_graph.number_of_nodes() == 0:
        return graph

    candidate_pixels = {point for point, degree in pixel_graph.degree if degree != 2}
    cluster_by_pixel, clusters = _cluster_candidate_pixels(pixel_graph, candidate_pixels)
    for index, pixels in enumerate(clusters):
        node_id = f"n{index:05d}"
        xs = [point[0] for point in pixels]
        ys = [point[1] for point in pixels]
        pixel_degrees = [int(pixel_graph.degree(point)) for point in pixels]
        graph.add_node(
            node_id,
            x_px=float(sum(xs) / len(xs)),
            y_px=float(sum(ys) / len(ys)),
            x_calibrated="",
            y_calibrated="",
            degree=0,
            pixel_degree=max(pixel_degrees, default=0),
            node_type="unclassified",
            component_id="",
            raw_node_count=len(pixels),
            raw_node_ids=";".join(_format_point(point) for point in sorted(pixels, key=_point_sort_key)),
        )

    visited_edges: set[frozenset[Point]] = set()
    edge_index = 0
    for pixel in sorted(cluster_by_pixel, key=_point_sort_key):
        source = cluster_by_pixel[pixel]
        for neighbor in sorted(pixel_graph.neighbors(pixel), key=_point_sort_key):
            edge_id = _pixel_edge_id(pixel, neighbor)
            if edge_id in visited_edges:
                continue
            if cluster_by_pixel.get(neighbor) == source:
                visited_edges.add(edge_id)
                continue
            path = _walk_topology_edge(pixel_graph, pixel, neighbor, cluster_by_pixel, visited_edges)
            target = cluster_by_pixel.get(path[-1])
            if target is None or target == source:
                continue
            attrs = _edge_attrs(edge_index, path)
            graph.add_edge(source, target, key=attrs["id"], **attrs)
            edge_index += 1

    edge_index = _add_cycle_components(pixel_graph, graph, cluster_by_pixel, visited_edges, edge_index)
    _annotate_topology_graph(graph)
    return graph


def _cluster_candidate_pixels(
    pixel_graph: nx.Graph, candidate_pixels: set[Point]
) -> tuple[dict[Point, str], list[list[Point]]]:
    clusters: list[list[Point]] = []
    if candidate_pixels:
        subgraph = pixel_graph.subgraph(candidate_pixels)
        for component in nx.connected_components(subgraph):
            clusters.append(sorted(component, key=_point_sort_key))
    clusters.sort(key=lambda pixels: _point_sort_key(pixels[0]))
    return (
        {
            point: f"n{index:05d}"
            for index, pixels in enumerate(clusters)
            for point in pixels
        },
        clusters,
    )


def _walk_topology_edge(
    pixel_graph: nx.Graph,
    start: Point,
    first: Point,
    cluster_by_pixel: dict[Point, str],
    visited_edges: set[frozenset[Point]],
) -> list[Point]:
    path = [start, first]
    previous = start
    current = first
    visited_edges.add(_pixel_edge_id(previous, current))
    while current not in cluster_by_pixel:
        choices = [
            neighbor
            for neighbor in sorted(pixel_graph.neighbors(current), key=_point_sort_key)
            if neighbor != previous
        ]
        if not choices:
            break
        next_point = next(
            (
                neighbor
                for neighbor in choices
                if _pixel_edge_id(current, neighbor) not in visited_edges
            ),
            choices[0],
        )
        visited_edges.add(_pixel_edge_id(current, next_point))
        path.append(next_point)
        previous, current = current, next_point
    return path


def _add_cycle_components(
    pixel_graph: nx.Graph,
    graph: nx.MultiGraph,
    cluster_by_pixel: dict[Point, str],
    visited_edges: set[frozenset[Point]],
    edge_index: int,
) -> int:
    for component in nx.connected_components(pixel_graph):
        if any(point in cluster_by_pixel for point in component):
            continue
        edges = [
            edge for edge in pixel_graph.subgraph(component).edges
            if _pixel_edge_id(*edge) not in visited_edges
        ]
        if not edges:
            continue
        path = _trace_cycle_path(pixel_graph, component, visited_edges)
        if len(path) < 3:
            continue
        node_id = f"n{graph.number_of_nodes():05d}"
        start = path[0]
        graph.add_node(
            node_id,
            x_px=float(start[0]),
            y_px=float(start[1]),
            x_calibrated="",
            y_calibrated="",
            degree=0,
            pixel_degree=2,
            node_type="cycle",
            component_id="",
            raw_node_count=1,
            raw_node_ids=_format_point(start),
        )
        attrs = _edge_attrs(edge_index, path + [start])
        graph.add_edge(node_id, node_id, key=attrs["id"], **attrs)
        edge_index += 1
    return edge_index


def _trace_cycle_path(
    pixel_graph: nx.Graph,
    component: set[Point],
    visited_edges: set[frozenset[Point]],
) -> list[Point]:
    start = sorted(component, key=_point_sort_key)[0]
    neighbors = sorted(pixel_graph.neighbors(start), key=_point_sort_key)
    if not neighbors:
        return [start]
    path = [start, neighbors[0]]
    previous = start
    current = neighbors[0]
    visited_edges.add(_pixel_edge_id(previous, current))
    while current != start:
        choices = [
            neighbor
            for neighbor in sorted(pixel_graph.neighbors(current), key=_point_sort_key)
            if neighbor != previous
        ]
        if not choices:
            break
        next_point = choices[0]
        edge_id = _pixel_edge_id(current, next_point)
        if edge_id in visited_edges and next_point != start:
            break
        visited_edges.add(edge_id)
        if next_point != start:
            path.append(next_point)
        previous, current = current, next_point
    return path


def _merge_graph_nodes(
    raw_graph: nx.MultiGraph, radius: float
) -> tuple[nx.MultiGraph, dict[str, object]]:
    if radius <= 0 or raw_graph.number_of_nodes() == 0:
        merged = raw_graph.copy()
        _annotate_topology_graph(merged)
        return (
            merged,
            {
                "clusters": raw_graph.number_of_nodes(),
                "merged_clusters": 0,
                "largest_cluster_size": 1 if raw_graph.number_of_nodes() else 0,
                "merged_raw_node_count": 0,
            },
        )

    parent = {node: node for node in raw_graph.nodes}
    components = {
        node: index
        for index, component in enumerate(nx.connected_components(nx.Graph(raw_graph)))
        for node in component
    }

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    nodes = sorted(raw_graph.nodes)
    for index, node in enumerate(nodes):
        for other in nodes[index + 1 :]:
            if components.get(node) != components.get(other):
                continue
            if _node_distance(raw_graph, node, other) <= radius:
                union(node, other)

    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(find(node), []).append(node)

    cluster_ids: dict[str, str] = {}
    for index, members in enumerate(sorted(grouped.values(), key=lambda group: group[0])):
        cluster_id = f"m{index:05d}"
        for node in members:
            cluster_ids[node] = cluster_id

    merged = nx.MultiGraph(graph_type="merged_engraving_graph", coordinate_units="pixels")
    for cluster_id in sorted(set(cluster_ids.values())):
        members = [node for node, mapped in cluster_ids.items() if mapped == cluster_id]
        datas = [raw_graph.nodes[node] for node in members]
        raw_ids = ";".join(str(data.get("raw_node_ids", "")) for data in datas)
        weights = [max(1, int(data.get("raw_node_count", 1))) for data in datas]
        total_weight = sum(weights)
        merged.add_node(
            cluster_id,
            x_px=float(sum(float(data["x_px"]) * weight for data, weight in zip(datas, weights)) / total_weight),
            y_px=float(sum(float(data["y_px"]) * weight for data, weight in zip(datas, weights)) / total_weight),
            x_calibrated="",
            y_calibrated="",
            degree=0,
            pixel_degree=max(int(data.get("pixel_degree", 0)) for data in datas),
            node_type="unclassified",
            component_id="",
            raw_node_count=sum(weights),
            raw_node_ids=raw_ids,
        )

    edge_index = 0
    for source, target, data in raw_graph.edges(data=True):
        merged_source = cluster_ids[source]
        merged_target = cluster_ids[target]
        if merged_source == merged_target and source != target:
            continue
        attrs = dict(data)
        attrs["id"] = f"e{edge_index:05d}"
        merged.add_edge(merged_source, merged_target, key=attrs["id"], **attrs)
        edge_index += 1

    _annotate_topology_graph(merged)
    cluster_sizes = [len(members) for members in grouped.values()]
    return (
        merged,
        {
            "clusters": len(grouped),
            "merged_clusters": sum(1 for size in cluster_sizes if size > 1),
            "largest_cluster_size": max(cluster_sizes, default=0),
            "merged_raw_node_count": sum(size for size in cluster_sizes if size > 1),
        },
    )


def _annotate_topology_graph(graph: nx.MultiGraph) -> None:
    for node_id, degree in graph.degree():
        graph.nodes[node_id]["degree"] = int(degree)
        graph.nodes[node_id]["node_type"] = _node_type_from_degree(int(degree))
    for index, component in enumerate(nx.connected_components(nx.Graph(graph))):
        for node_id in component:
            graph.nodes[node_id]["component_id"] = str(index)


def _edge_attrs(index: int, points: list[Point]) -> dict[str, object]:
    length = _polyline_length(points)
    return {
        "id": f"e{index:05d}",
        "length_px": float(length),
        "length_calibrated": "",
        "orientation_deg": float(_orientation(points[0], points[-1])),
        "point_count": len(points),
        "closed": int(points[0] == points[-1]),
        "polyline": " ".join(_format_point(point) for point in points),
    }


def _polyline_length(points: Iterable[Point]) -> float:
    ordered = list(points)
    return float(sum(math.dist(a, b) for a, b in zip(ordered, ordered[1:])))


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


def _node_type_from_degree(degree: int) -> str:
    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree >= 3:
        return "junction"
    return "connector"


def _component_colors(graph: nx.MultiGraph) -> dict[str, tuple[int, int, int]]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    ]
    colors: dict[str, tuple[int, int, int]] = {}
    for index, component in enumerate(nx.connected_components(nx.Graph(graph))):
        color = palette[index % len(palette)]
        for node in component:
            colors[node] = color
    return colors


def _parse_polyline(value: object) -> list[Point]:
    points: list[Point] = []
    for token in str(value).split():
        if "," not in token:
            continue
        x, y = token.split(",", 1)
        points.append((int(float(x)), int(float(y))))
    return points


def _node_distance(graph: nx.MultiGraph, a: str, b: str) -> float:
    data_a = graph.nodes[a]
    data_b = graph.nodes[b]
    return math.dist(
        (float(data_a["x_px"]), float(data_a["y_px"])),
        (float(data_b["x_px"]), float(data_b["y_px"])),
    )


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


def _pixel_edge_id(a: Point, b: Point) -> frozenset[Point]:
    return frozenset((a, b))


def _format_point(point: Point) -> str:
    return f"{int(point[0])},{int(point[1])}"


def _point_sort_key(point: Point) -> tuple[int, int]:
    return (int(point[1]), int(point[0]))
