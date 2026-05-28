from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import svgwrite

from .graph_metrics import (
    build_graph_bundle_from_raw,
    write_graph_exports,
    write_metrics_json,
    write_orientation_histogram,
)
from .pipeline import PipelineMetrics, json_dumps_compact

PointF = tuple[float, float]


@dataclass(frozen=True)
class ManualSvgConfig:
    manual_layer: str = "Manual Trace"
    snap_radius: float = 3.0
    intersection_split: bool = True
    flatten_tolerance: float = 1.0
    stroke_width: float = 1.2

    def to_dict(self) -> dict[str, object]:
        return {
            "input_mode": "manual-svg",
            "manual_layer": self.manual_layer,
            "svg_snap_radius": float(self.snap_radius),
            "svg_intersection_split": bool(self.intersection_split),
            "svg_flatten_tolerance": float(self.flatten_tolerance),
            "stroke_width": float(self.stroke_width),
        }


@dataclass(frozen=True)
class ManualPolyline:
    points: list[PointF]
    source_id: str


@dataclass(frozen=True)
class ManualSvgResult:
    polylines: list[ManualPolyline]
    raw_graph: nx.MultiGraph
    graph_bundle: object


def run_manual_svg_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    debug_dir: str | Path | None,
    config: ManualSvgConfig,
    metrics_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    nodes_csv_path: str | Path | None = None,
    edges_csv_path: str | Path | None = None,
    orientation_hist_path: str | Path | None = None,
) -> PipelineMetrics:
    result = svg_to_graph(input_path, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manual_trace_svg(output, result.graph_bundle.merged_graph)

    debug = ""
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        debug = str(debug_path)
        write_manual_trace_svg(debug_path / "manual_trace_debug.svg", result.graph_bundle.merged_graph)

    written = write_graph_exports(
        graph_path=graph_path,
        nodes_csv_path=nodes_csv_path,
        edges_csv_path=edges_csv_path,
        bundle=result.graph_bundle,
    )
    if orientation_hist_path:
        write_orientation_histogram(orientation_hist_path, result.graph_bundle.orientation)
        written["orientation_histogram"] = str(orientation_hist_path)

    graph_metrics = {
        **result.graph_bundle.metrics,
        "input_mode": "manual-svg",
        "manual_layer": config.manual_layer,
        "svg_snap_radius": float(config.snap_radius),
        "svg_intersection_split": bool(config.intersection_split),
        "svg_flatten_tolerance": float(config.flatten_tolerance),
        "interpretation_note": (
            "Manual SVG analysis uses semi-manual centerline interpretation and does not "
            "represent automatic stroke recovery."
        ),
    }
    orientation = result.graph_bundle.orientation
    merged = graph_metrics["merged"]
    bounds = _graph_bounds(result.graph_bundle.merged_graph)
    pipeline_metrics = PipelineMetrics(
        output_svg=str(output),
        debug_dir=debug,
        crop_x=0,
        crop_y=0,
        crop_width=int(math.ceil(bounds[2] - bounds[0])) if bounds else 0,
        crop_height=int(math.ceil(bounds[3] - bounds[1])) if bounds else 0,
        foreground_pixels=0,
        extraction_mode="manual-svg",
        ridge_sigmas="",
        ridge_beta=0.0,
        ridge_gamma=0.0,
        ridge_threshold=0.0,
        ridge_mask_pixels=0,
        skeleton_pixels=0,
        graph_nodes=result.raw_graph.number_of_nodes(),
        graph_edges=result.raw_graph.number_of_edges(),
        paths=len(result.polylines),
        endpoints=int(graph_metrics["raw_endpoint_count"]),
        junctions=int(graph_metrics["raw_junction_count"]),
        bridge_count=0,
        node_merge_radius=float(config.snap_radius),
        bridge_gaps_radius=0.0,
        bridge_gaps_angle_tolerance=0.0,
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
        node_degree_histogram=json_dumps_compact(graph_metrics["node_degree_histogram"]),
        connected_component_size_histogram=json_dumps_compact(
            graph_metrics["connected_component_size_histogram"]
        ),
        largest_connected_component_fraction=float(
            graph_metrics["largest_connected_component_fraction"]
        ),
        output_metrics=str(metrics_path) if metrics_path else "",
        graph_raw=written.get("graph_raw", ""),
        graph_merged=written.get("graph_merged", ""),
        nodes_raw_csv=written.get("nodes_raw_csv", ""),
        nodes_merged_csv=written.get("nodes_merged_csv", ""),
        edges_raw_csv=written.get("edges_raw_csv", ""),
        edges_merged_csv=written.get("edges_merged_csv", ""),
        orientation_histogram=written.get("orientation_histogram", ""),
        input_mode="manual-svg",
        manual_layer=config.manual_layer,
        svg_snap_radius=float(config.snap_radius),
        svg_flatten_tolerance=float(config.flatten_tolerance),
        svg_intersection_split=bool(config.intersection_split),
    )

    if metrics_path:
        write_metrics_json(
            metrics_path,
            pipeline_metrics=pipeline_metrics.to_dict(),
            graph_metrics=graph_metrics,
            config=config.to_dict(),
        )
    return pipeline_metrics


def svg_to_graph(input_path: str | Path, config: ManualSvgConfig) -> ManualSvgResult:
    polylines = extract_manual_polylines(input_path, config.manual_layer, config.flatten_tolerance)
    raw_graph = build_manual_graph(polylines, intersection_split=config.intersection_split)
    graph_bundle = build_graph_bundle_from_raw(
        raw_graph,
        node_merge_radius=config.snap_radius,
        same_component_only=False,
    )
    return ManualSvgResult(polylines=polylines, raw_graph=raw_graph, graph_bundle=graph_bundle)


def extract_manual_polylines(
    input_path: str | Path,
    manual_layer: str,
    flatten_tolerance: float,
) -> list[ManualPolyline]:
    tree = ET.parse(input_path)
    root = tree.getroot()
    _remember_parents(root)
    layer = _find_manual_layer(root, manual_layer)
    if layer is None:
        raise ValueError(f"Manual SVG layer not found: {manual_layer!r}")
    polylines: list[ManualPolyline] = []
    for element in layer.iter():
        if element is layer or _is_hidden(element) or _has_hidden_ancestor(element, layer):
            continue
        tag = _local_name(element.tag)
        transform = _combined_transform(element, stop=layer)
        source_id = element.attrib.get("id", f"{tag}_{len(polylines)}")
        for points in _element_polylines(element, flatten_tolerance):
            transformed = [_apply_transform(point, transform) for point in points]
            cleaned = _dedupe_consecutive(transformed)
            if len(cleaned) >= 2:
                polylines.append(ManualPolyline(points=cleaned, source_id=source_id))
    return polylines


def build_manual_graph(polylines: list[ManualPolyline], intersection_split: bool = True) -> nx.MultiGraph:
    graph = nx.MultiGraph(graph_type="manual_svg_topology_graph", coordinate_units="svg_user")
    split_positions: list[list[tuple[float, PointF]]] = [
        [(0.0, polyline.points[0]), (_polyline_length(polyline.points), polyline.points[-1])]
        for polyline in polylines
    ]
    segment_refs = _segment_refs(polylines)
    if intersection_split:
        for index, first in enumerate(segment_refs):
            for second in segment_refs[index + 1 :]:
                if first["polyline_index"] == second["polyline_index"] and abs(first["segment_index"] - second["segment_index"]) <= 1:
                    continue
                intersection = _segment_intersection(first["a"], first["b"], second["a"], second["b"])
                if intersection is None:
                    continue
                for ref in (first, second):
                    t = _segment_t(intersection, ref["a"], ref["b"])
                    distance = ref["offset"] + t * math.dist(ref["a"], ref["b"])
                    split_positions[ref["polyline_index"]].append((distance, intersection))

    node_by_point: dict[tuple[int, int], str] = {}
    edge_index = 0
    for polyline_index, polyline in enumerate(polylines):
        positions = _unique_positions(split_positions[polyline_index])
        for (start_distance, start), (end_distance, end) in zip(positions, positions[1:]):
            if math.dist(start, end) <= 1e-9:
                continue
            edge_points = _subpolyline(polyline.points, start_distance, end_distance, start, end)
            source = _node_for_point(graph, node_by_point, start)
            target = _node_for_point(graph, node_by_point, end)
            attrs = _manual_edge_attrs(edge_index, edge_points, polyline.source_id)
            graph.add_edge(source, target, key=attrs["id"], **attrs)
            edge_index += 1
    _annotate_manual_graph(graph)
    return graph


def write_manual_trace_svg(path: str | Path, graph: nx.MultiGraph) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = _graph_bounds(graph) or (0.0, 0.0, 100.0, 100.0)
    min_x, min_y, max_x, max_y = bounds
    pad = 10.0
    width = max(1.0, max_x - min_x + 2 * pad)
    height = max(1.0, max_y - min_y + 2 * pad)
    drawing = svgwrite.Drawing(
        str(output),
        size=(width, height),
        viewBox=f"{min_x - pad} {min_y - pad} {width} {height}",
    )
    drawing.add(drawing.rect(insert=(min_x - pad, min_y - pad), size=(width, height), fill="white"))
    for _, _, data in graph.edges(data=True):
        points = _parse_polyline(data.get("polyline", ""))
        if len(points) >= 2:
            drawing.add(drawing.polyline(points=points, fill="none", stroke="#2f6fbb", stroke_width=1.4))
    for _, data in graph.nodes(data=True):
        point = (float(data["x_px"]), float(data["y_px"]))
        node_type = data.get("node_type", "")
        color = "#d62728" if node_type == "junction" else "#2ca02c" if node_type == "endpoint" else "#555555"
        drawing.add(drawing.circle(center=point, r=2.5, fill=color))
    drawing.save()


def _find_manual_layer(root: ET.Element, manual_layer: str) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag) != "g" or _is_hidden(element):
            continue
        labels = {
            element.attrib.get("id", ""),
            element.attrib.get("label", ""),
            element.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label", ""),
        }
        if manual_layer in labels:
            return element
    return None


def _element_polylines(element: ET.Element, flatten_tolerance: float) -> list[list[PointF]]:
    tag = _local_name(element.tag)
    if tag == "path" and element.attrib.get("d"):
        return _parse_path(element.attrib["d"], flatten_tolerance)
    if tag in {"polyline", "polygon"} and element.attrib.get("points"):
        points = _parse_points(element.attrib["points"])
        if tag == "polygon" and points:
            points = points + [points[0]]
        return [points]
    if tag == "line":
        return [[
            (_float_attr(element, "x1"), _float_attr(element, "y1")),
            (_float_attr(element, "x2"), _float_attr(element, "y2")),
        ]]
    if tag == "rect":
        x = _float_attr(element, "x")
        y = _float_attr(element, "y")
        width = _float_attr(element, "width")
        height = _float_attr(element, "height")
        if width > 0 and height > 0:
            return [[(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]]
    if tag in {"circle", "ellipse"}:
        cx = _float_attr(element, "cx")
        cy = _float_attr(element, "cy")
        rx = _float_attr(element, "r") if tag == "circle" else _float_attr(element, "rx")
        ry = _float_attr(element, "r") if tag == "circle" else _float_attr(element, "ry")
        if rx > 0 and ry > 0:
            steps = max(16, int(math.ceil(2 * math.pi * max(rx, ry) / max(0.5, flatten_tolerance))))
            return [[
                (cx + math.cos(2 * math.pi * i / steps) * rx, cy + math.sin(2 * math.pi * i / steps) * ry)
                for i in range(steps + 1)
            ]]
    return []


def _parse_path(d: str, flatten_tolerance: float) -> list[list[PointF]]:
    tokens = re.findall(r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", d)
    polylines: list[list[PointF]] = []
    current: PointF = (0.0, 0.0)
    start: PointF = (0.0, 0.0)
    active: list[PointF] = []
    command = ""
    index = 0
    last_cubic: PointF | None = None
    last_quadratic: PointF | None = None

    def has_number() -> bool:
        return index < len(tokens) and not re.match(r"^[A-Za-z]$", tokens[index])

    def read_point(relative: bool) -> PointF:
        nonlocal index
        point = (float(tokens[index]), float(tokens[index + 1]))
        index += 2
        return (current[0] + point[0], current[1] + point[1]) if relative else point

    def add_point(point: PointF) -> None:
        nonlocal current
        if not active or math.dist(active[-1], point) > 1e-9:
            active.append(point)
        current = point

    while index < len(tokens):
        if re.match(r"^[A-Za-z]$", tokens[index]):
            command = tokens[index]
            index += 1
        relative = command.islower()
        op = command.upper()
        if op == "M":
            if len(active) >= 2:
                polylines.append(active)
            point = read_point(relative)
            active = [point]
            current = point
            start = point
            command = "l" if relative else "L"
            last_cubic = None
            last_quadratic = None
        elif op == "L":
            while has_number():
                add_point(read_point(relative))
            last_cubic = None
            last_quadratic = None
        elif op == "H":
            while has_number():
                value = float(tokens[index])
                index += 1
                add_point((current[0] + value, current[1]) if relative else (value, current[1]))
            last_cubic = None
            last_quadratic = None
        elif op == "V":
            while has_number():
                value = float(tokens[index])
                index += 1
                add_point((current[0], current[1] + value) if relative else (current[0], value))
            last_cubic = None
            last_quadratic = None
        elif op == "C":
            while has_number():
                c1 = read_point(relative)
                c2 = read_point(relative)
                end = read_point(relative)
                for point in _flatten_cubic(current, c1, c2, end, flatten_tolerance)[1:]:
                    add_point(point)
                last_cubic = c2
                last_quadratic = None
        elif op == "S":
            while has_number():
                c1 = _reflect(current, last_cubic) if last_cubic else current
                c2 = read_point(relative)
                end = read_point(relative)
                for point in _flatten_cubic(current, c1, c2, end, flatten_tolerance)[1:]:
                    add_point(point)
                last_cubic = c2
                last_quadratic = None
        elif op == "Q":
            while has_number():
                c = read_point(relative)
                end = read_point(relative)
                for point in _flatten_quadratic(current, c, end, flatten_tolerance)[1:]:
                    add_point(point)
                last_quadratic = c
                last_cubic = None
        elif op == "T":
            while has_number():
                c = _reflect(current, last_quadratic) if last_quadratic else current
                end = read_point(relative)
                for point in _flatten_quadratic(current, c, end, flatten_tolerance)[1:]:
                    add_point(point)
                last_quadratic = c
                last_cubic = None
        elif op == "A":
            while has_number():
                if index + 6 >= len(tokens):
                    break
                index += 5
                add_point(read_point(relative))
            last_cubic = None
            last_quadratic = None
        elif op == "Z":
            add_point(start)
            if len(active) >= 2:
                polylines.append(active)
            active = []
            current = start
            last_cubic = None
            last_quadratic = None
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")
    if len(active) >= 2:
        polylines.append(active)
    return polylines


def _segment_refs(polylines: list[ManualPolyline]) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    for polyline_index, polyline in enumerate(polylines):
        offset = 0.0
        for segment_index, (a, b) in enumerate(zip(polyline.points, polyline.points[1:])):
            length = math.dist(a, b)
            if length > 1e-9:
                refs.append(
                    {
                        "polyline_index": polyline_index,
                        "segment_index": segment_index,
                        "a": a,
                        "b": b,
                        "offset": offset,
                    }
                )
            offset += length
    return refs


def _unique_positions(positions: list[tuple[float, PointF]]) -> list[tuple[float, PointF]]:
    ordered = sorted(positions, key=lambda item: item[0])
    unique: list[tuple[float, PointF]] = []
    for distance, point in ordered:
        if unique and abs(distance - unique[-1][0]) <= 1e-6:
            continue
        unique.append((distance, point))
    return unique


def _subpolyline(points: list[PointF], start_distance: float, end_distance: float, start: PointF, end: PointF) -> list[PointF]:
    result = [start]
    offset = 0.0
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        next_offset = offset + length
        if start_distance < next_offset - 1e-9 and end_distance > offset + 1e-9:
            if start_distance < next_offset and end_distance > next_offset and math.dist(result[-1], b) > 1e-9:
                result.append(b)
        offset = next_offset
    if math.dist(result[-1], end) > 1e-9:
        result.append(end)
    return result


def _node_for_point(graph: nx.MultiGraph, node_by_point: dict[tuple[int, int], str], point: PointF) -> str:
    key = (round(point[0] * 1_000_000), round(point[1] * 1_000_000))
    if key in node_by_point:
        return node_by_point[key]
    node_id = f"n{len(node_by_point):05d}"
    node_by_point[key] = node_id
    graph.add_node(
        node_id,
        x_px=float(point[0]),
        y_px=float(point[1]),
        x_calibrated=float(point[0]),
        y_calibrated=float(point[1]),
        degree=0,
        pixel_degree=0,
        node_type="unclassified",
        component_id="",
        raw_node_count=1,
        raw_node_ids=f"{point[0]:.6g},{point[1]:.6g}",
    )
    return node_id


def _manual_edge_attrs(index: int, points: list[PointF], source_id: str) -> dict[str, object]:
    return {
        "id": f"e{index:05d}",
        "source_element": source_id,
        "length_px": float(_polyline_length(points)),
        "length_calibrated": "",
        "orientation_deg": float(_orientation(points[0], points[-1])),
        "point_count": len(points),
        "closed": int(math.dist(points[0], points[-1]) <= 1e-9),
        "polyline": " ".join(_format_point(point) for point in points),
    }


def _annotate_manual_graph(graph: nx.MultiGraph) -> None:
    for node_id, degree in graph.degree():
        graph.nodes[node_id]["degree"] = int(degree)
        graph.nodes[node_id]["pixel_degree"] = int(degree)
        graph.nodes[node_id]["node_type"] = _node_type_from_degree(int(degree))
    for index, component in enumerate(nx.connected_components(nx.Graph(graph))):
        for node_id in component:
            graph.nodes[node_id]["component_id"] = str(index)


def _combined_transform(element: ET.Element, stop: ET.Element) -> tuple[float, float, float, float, float, float]:
    transforms: list[tuple[float, float, float, float, float, float]] = []
    current = element
    while current is not stop:
        transform = current.attrib.get("transform")
        if transform:
            transforms.append(_parse_transform(transform))
        parent = _PARENTS.get(current)
        if parent is None:
            break
        current = parent
    layer_transform = stop.attrib.get("transform")
    if layer_transform:
        transforms.append(_parse_transform(layer_transform))
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for transform in reversed(transforms):
        matrix = _multiply_transform(matrix, transform)
    return matrix


def _parse_transform(value: str) -> tuple[float, float, float, float, float, float]:
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args_text in re.findall(r"([A-Za-z]+)\(([^)]*)\)", value):
        args = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", args_text)]
        if name == "matrix" and len(args) == 6:
            transform = tuple(args)  # type: ignore[assignment]
        elif name == "translate":
            transform = (1.0, 0.0, 0.0, 1.0, args[0] if args else 0.0, args[1] if len(args) > 1 else 0.0)
        elif name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            transform = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate":
            angle = math.radians(args[0] if args else 0.0)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotate = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                transform = _multiply_transform(
                    _multiply_transform((1.0, 0.0, 0.0, 1.0, cx, cy), rotate),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                transform = rotate
        else:
            continue
        matrix = _multiply_transform(matrix, transform)
    return matrix


def _multiply_transform(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_transform(point: PointF, matrix: tuple[float, float, float, float, float, float]) -> PointF:
    a, b, c, d, e, f = matrix
    return (a * point[0] + c * point[1] + e, b * point[0] + d * point[1] + f)


def _is_hidden(element: ET.Element) -> bool:
    style = element.attrib.get("style", "")
    return (
        element.attrib.get("display") == "none"
        or element.attrib.get("visibility") == "hidden"
        or "display:none" in style.replace(" ", "")
        or "visibility:hidden" in style.replace(" ", "")
    )


def _has_hidden_ancestor(element: ET.Element, stop: ET.Element) -> bool:
    current = _PARENTS.get(element)
    while current is not None and current is not stop:
        if _is_hidden(current):
            return True
        current = _PARENTS.get(current)
    return False


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _float_attr(element: ET.Element, name: str, default: float = 0.0) -> float:
    value = element.attrib.get(name)
    if value is None:
        return default
    match = re.match(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", value)
    return float(match.group(0)) if match else default


def _parse_points(value: str) -> list[PointF]:
    numbers = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", value)]
    return [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]


def _flatten_cubic(a: PointF, b: PointF, c: PointF, d: PointF, tolerance: float) -> list[PointF]:
    control_length = math.dist(a, b) + math.dist(b, c) + math.dist(c, d)
    steps = max(2, int(math.ceil(control_length / max(0.5, tolerance * 4.0))))
    return [_cubic(a, b, c, d, i / steps) for i in range(steps + 1)]


def _flatten_quadratic(a: PointF, b: PointF, c: PointF, tolerance: float) -> list[PointF]:
    control_length = math.dist(a, b) + math.dist(b, c)
    steps = max(2, int(math.ceil(control_length / max(0.5, tolerance * 4.0))))
    return [_quadratic(a, b, c, i / steps) for i in range(steps + 1)]


def _cubic(a: PointF, b: PointF, c: PointF, d: PointF, t: float) -> PointF:
    mt = 1.0 - t
    return (
        mt**3 * a[0] + 3 * mt * mt * t * b[0] + 3 * mt * t * t * c[0] + t**3 * d[0],
        mt**3 * a[1] + 3 * mt * mt * t * b[1] + 3 * mt * t * t * c[1] + t**3 * d[1],
    )


def _quadratic(a: PointF, b: PointF, c: PointF, t: float) -> PointF:
    mt = 1.0 - t
    return (
        mt * mt * a[0] + 2 * mt * t * b[0] + t * t * c[0],
        mt * mt * a[1] + 2 * mt * t * b[1] + t * t * c[1],
    )


def _reflect(origin: PointF, control: PointF | None) -> PointF:
    if control is None:
        return origin
    return (2 * origin[0] - control[0], 2 * origin[1] - control[1])


def _segment_intersection(a: PointF, b: PointF, c: PointF, d: PointF) -> PointF | None:
    denominator = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if abs(denominator) < 1e-9:
        return None
    px = ((a[0] * b[1] - a[1] * b[0]) * (c[0] - d[0]) - (a[0] - b[0]) * (c[0] * d[1] - c[1] * d[0])) / denominator
    py = ((a[0] * b[1] - a[1] * b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] * d[1] - c[1] * d[0])) / denominator
    point = (px, py)
    if _point_on_segment(point, a, b) and _point_on_segment(point, c, d):
        return point
    return None


def _segment_t(point: PointF, a: PointF, b: PointF) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom))


def _point_on_segment(point: PointF, a: PointF, b: PointF, tolerance: float = 1e-6) -> bool:
    return (
        min(a[0], b[0]) - tolerance <= point[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= point[1] <= max(a[1], b[1]) + tolerance
        and abs(math.dist(a, point) + math.dist(point, b) - math.dist(a, b)) <= tolerance
    )


def _polyline_length(points: Iterable[PointF]) -> float:
    ordered = list(points)
    return float(sum(math.dist(a, b) for a, b in zip(ordered, ordered[1:])))


def _orientation(a: PointF, b: PointF) -> float:
    return float(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0)


def _format_point(point: PointF) -> str:
    return f"{point[0]:.6g},{point[1]:.6g}"


def _parse_polyline(value: object) -> list[PointF]:
    points: list[PointF] = []
    for token in str(value).split():
        if "," not in token:
            continue
        x, y = token.split(",", 1)
        points.append((float(x), float(y)))
    return points


def _dedupe_consecutive(points: list[PointF]) -> list[PointF]:
    cleaned: list[PointF] = []
    for point in points:
        if not cleaned or math.dist(cleaned[-1], point) > 1e-9:
            cleaned.append(point)
    return cleaned


def _graph_bounds(graph: nx.MultiGraph) -> tuple[float, float, float, float] | None:
    if graph.number_of_nodes() == 0:
        return None
    xs = [float(data["x_px"]) for _, data in graph.nodes(data=True)]
    ys = [float(data["y_px"]) for _, data in graph.nodes(data=True)]
    return (min(xs), min(ys), max(xs), max(ys))


def _node_type_from_degree(degree: int) -> str:
    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree >= 3:
        return "junction"
    return "connector"


_PARENTS: dict[ET.Element, ET.Element] = {}


def _remember_parents(root: ET.Element) -> None:
    _PARENTS.clear()
    for parent in root.iter():
        for child in parent:
            _PARENTS[child] = parent
