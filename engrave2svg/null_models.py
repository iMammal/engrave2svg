from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import networkx as nx
import numpy as np
import svgwrite


PointF = tuple[float, float]


@dataclass(frozen=True)
class Stroke:
    points: list[PointF]

    @property
    def length(self) -> float:
        return float(sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:])))


NORMALIZED_METRICS = (
    "endpoints_per_1000px",
    "junctions_per_1000px",
    "cycles_per_node",
    "triangles_per_node",
    "triangles_per_cycle",
    "bisected_lozenge_candidates_per_cycle",
    "bisected_lozenge_candidates_per_triangle_pair",
    "four_cycles_per_node",
    "lozenge_candidates_per_cycle",
    "degree3_fraction",
    "degree4_fraction",
    "orientation_peak_concentration",
    "orientation_entropy",
    "largest_connected_component_fraction",
)

DEFAULT_METRICS = (
    *NORMALIZED_METRICS,
    "open_lozenge_candidate_count",
    "open_lozenge_mean_confidence",
    "closed_lozenge_candidate_count",
    "triangle_count",
    "triangle_side_length_cv_mean",
    "triangle_area_mean",
    "triangle_area_cv",
    "adjacent_triangle_pair_count",
    "bisected_lozenge_candidate_count",
    "bisected_lozenge_mean_confidence",
    "dominant_orientation_families",
    "endpoint_count",
    "junction_count",
    "degree_3_fraction",
    "connected_component_count",
    "cycle_count",
    "four_cycle_count",
    "lozenge_candidate_count",
    "total_traced_length",
)

PLOT_METRICS = (
    *NORMALIZED_METRICS,
    "triangle_count",
    "triangle_side_length_cv_mean",
    "triangle_area_mean",
    "triangle_area_cv",
    "bisected_lozenge_candidate_count",
    "bisected_lozenge_mean_confidence",
)


def generate_control_set(
    output_dir: str | Path,
    width: int = 240,
    height: int = 180,
    spacing: int = 36,
    seed: int = 42,
    svg_previews: bool = False,
    stroke_width: int = 3,
    control_polarity: str = "bright-on-dark",
) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    clean = lozenge_lattice_strokes(width, height, spacing)
    clean_meta = _write_control(
        root,
        "clean_lozenge_lattice",
        "lozenge",
        clean,
        width,
        height,
        svg_previews,
        stroke_width,
        control_polarity,
    )

    broken = broken_noisy_strokes(clean, width, height, rng)
    broken_meta = _write_control(
        root,
        "broken_noisy_lozenge_lattice",
        "lozenge",
        broken,
        width,
        height,
        svg_previews,
        stroke_width,
        control_polarity,
    )

    scratches = random_scratch_strokes(
        width=width,
        height=height,
        stroke_count=len(clean),
        target_total_length=sum(stroke.length for stroke in clean),
        rng=rng,
    )
    scratch_meta = _write_control(
        root,
        "random_scratches",
        "random",
        scratches,
        width,
        height,
        svg_previews,
        stroke_width,
        control_polarity,
    )

    curved = curved_random_scratch_strokes(
        width=width,
        height=height,
        stroke_count=len(clean),
        target_total_length=sum(stroke.length for stroke in clean),
        rng=rng,
    )
    curved_meta = _write_control(
        root,
        "curved_random_scratches",
        "random",
        curved,
        width,
        height,
        svg_previews,
        stroke_width,
        control_polarity,
    )

    metadata = {
        "seed": seed,
        "width": width,
        "height": height,
        "spacing": spacing,
        "stroke_width": stroke_width,
        "control_polarity": control_polarity,
        "recommended_vectorization": {
            "crop": "none",
            "extraction_mode": "threshold",
            "threshold_mode": "global",
            "threshold_value": 180,
            "morph_kernel_size": 1,
            "min_component_size": 1,
            "simplification_epsilon": 0.0,
        },
        "controls": [clean_meta, broken_meta, scratch_meta, curved_meta],
        "note": (
            "Synthetic lozenge controls are geometric positive controls. "
            "Random scratches preserve approximate stroke count, total length, and bounding box."
        ),
    }
    (root / "expected_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def lozenge_lattice_strokes(width: int, height: int, spacing: int) -> list[Stroke]:
    strokes: list[Stroke] = []
    slope = 1.0
    for intercept in range(-height, width + height, spacing):
        clipped = _clip_line_to_box(slope, intercept, width, height)
        if clipped:
            strokes.append(Stroke(clipped))
    for intercept in range(0, width + height * 2, spacing):
        clipped = _clip_line_to_box(-slope, intercept, width, height)
        if clipped:
            strokes.append(Stroke(clipped))
    return strokes


def broken_noisy_strokes(
    strokes: list[Stroke], width: int, height: int, rng: random.Random
) -> list[Stroke]:
    broken: list[Stroke] = []
    for stroke in strokes:
        if len(stroke.points) != 2 or rng.random() > 0.45:
            broken.append(stroke)
            continue
        start, end = stroke.points
        t0 = rng.uniform(0.35, 0.55)
        gap = rng.uniform(0.035, 0.07)
        a = _interpolate(start, end, max(0.0, t0 - gap))
        b = _interpolate(start, end, min(1.0, t0 + gap))
        broken.append(Stroke([start, a]))
        broken.append(Stroke([b, end]))
    for _ in range(max(3, len(strokes) // 5)):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        angle = rng.uniform(0, math.pi)
        length = rng.uniform(8, 22)
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2
        broken.append(Stroke([_clamp_point((x - dx, y - dy), width, height), _clamp_point((x + dx, y + dy), width, height)]))
    return broken


def random_scratch_strokes(
    width: int,
    height: int,
    stroke_count: int,
    target_total_length: float,
    rng: random.Random,
) -> list[Stroke]:
    mean_length = max(5.0, target_total_length / max(1, stroke_count))
    strokes: list[Stroke] = []
    for _ in range(stroke_count):
        length = rng.uniform(mean_length * 0.45, mean_length * 1.55)
        angle = rng.uniform(0, math.pi)
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2
        strokes.append(Stroke([_clamp_point((x - dx, y - dy), width, height), _clamp_point((x + dx, y + dy), width, height)]))
    return strokes


def curved_random_scratch_strokes(
    width: int,
    height: int,
    stroke_count: int,
    target_total_length: float,
    rng: random.Random,
) -> list[Stroke]:
    mean_length = max(8.0, target_total_length / max(1, stroke_count))
    strokes: list[Stroke] = []
    for _ in range(stroke_count):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        angle = rng.uniform(0, math.pi)
        step = mean_length / 4
        points = [(x, y)]
        for _step in range(4):
            angle += rng.uniform(-0.45, 0.45)
            x += math.cos(angle) * step
            y += math.sin(angle) * step
            points.append(_clamp_point((x, y), width, height))
        strokes.append(Stroke(points))
    return strokes


def analyze_graphml(path: str | Path) -> dict[str, object]:
    graph = nx.read_graphml(path)
    return analyze_graph(graph)


def analyze_graph(graph: nx.Graph) -> dict[str, object]:
    simple = nx.Graph(graph)
    node_count = simple.number_of_nodes()
    edge_lengths = [_edge_length(graph, u, v, data) for u, v, data in graph.edges(data=True)]
    orientations = [_edge_orientation(graph, u, v, data) for u, v, data in graph.edges(data=True)]
    hist = _orientation_histogram(orientations, edge_lengths)
    entropy = _orientation_entropy(hist)
    peaks = _dominant_orientation_peaks(hist)
    degrees = dict(simple.degree())
    components = [len(component) for component in nx.connected_components(simple)] if node_count else []
    cycles = nx.cycle_basis(simple)
    triangles = enumerate_triangles(simple)
    triangle_shape = triangle_shape_metrics(simple, triangles)
    bisected_lozenges = detect_bisected_lozenge_candidates(simple, triangles)
    four_cycles = enumerate_four_cycles(simple)
    lozenges = detect_lozenge_candidates(simple, four_cycles)
    open_lozenges = detect_open_lozenge_candidates(simple, closed_candidates=lozenges)
    metrics = {
        "orientation_peak_concentration": float(max(hist) / sum(hist)) if sum(hist) else 0.0,
        "orientation_entropy": entropy,
        "dominant_orientation_families": len(peaks),
        "endpoint_count": sum(1 for degree in degrees.values() if degree == 1),
        "junction_count": sum(1 for degree in degrees.values() if degree >= 3),
        "degree3_fraction": float(sum(1 for degree in degrees.values() if degree == 3) / node_count) if node_count else 0.0,
        "degree4_fraction": float(sum(1 for degree in degrees.values() if degree == 4) / node_count) if node_count else 0.0,
        "degree_3_fraction": float(sum(1 for degree in degrees.values() if degree == 3) / node_count) if node_count else 0.0,
        "connected_component_count": len(components),
        "largest_connected_component_fraction": float(max(components, default=0) / node_count) if node_count else 0.0,
        "cycle_count": len(cycles),
        "triangle_count": len(triangles),
        **triangle_shape,
        "adjacent_triangle_pair_count": adjacent_triangle_pair_count(triangles),
        "bisected_lozenge_candidate_count": len(bisected_lozenges),
        "bisected_lozenge_mean_confidence": _mean(
            [float(item["confidence"]) for item in bisected_lozenges]
        ),
        "four_cycle_count": len(four_cycles),
        "closed_lozenge_candidate_count": len(lozenges),
        "lozenge_candidate_count": len(lozenges),
        "open_lozenge_candidate_count": len(open_lozenges),
        "open_lozenge_mean_confidence": _mean([float(item["confidence"]) for item in open_lozenges]),
        "total_traced_length": float(sum(edge_lengths)),
        "node_count": node_count,
        "edge_count": simple.number_of_edges(),
    }
    return _with_normalized_metrics(metrics)


def enumerate_triangles(graph: nx.Graph) -> list[tuple[str, str, str]]:
    triangles: set[tuple[str, str, str]] = set()
    nodes = sorted(str(node) for node in graph.nodes)
    order = {node: index for index, node in enumerate(nodes)}
    neighbors = {node: {str(item) for item in graph.neighbors(node)} for node in nodes}
    for a in nodes:
        for b in sorted(node for node in neighbors[a] if order[node] > order[a]):
            for c in sorted(neighbors[a] & neighbors[b]):
                if order[c] > order[b]:
                    triangles.add((a, b, c))
    return sorted(triangles)


def triangle_shape_metrics(
    graph: nx.Graph, triangles: Iterable[tuple[str, str, str]]
) -> dict[str, object]:
    side_cvs: list[float] = []
    areas: list[float] = []
    for triangle in triangles:
        points = [_node_xy(graph, node) for node in triangle]
        sides = [
            math.dist(points[0], points[1]),
            math.dist(points[1], points[2]),
            math.dist(points[2], points[0]),
        ]
        mean_side = _mean(sides)
        side_cvs.append(_sd(sides) / mean_side if mean_side else 0.0)
        areas.append(abs(_polygon_area(points)))
    area_mean = _mean(areas)
    return {
        "triangle_side_length_cv_mean": _mean(side_cvs),
        "triangle_area_mean": area_mean,
        "triangle_area_cv": _sd(areas) / area_mean if area_mean else 0.0,
    }


def adjacent_triangle_pair_count(triangles: Iterable[tuple[str, str, str]]) -> int:
    return len(_adjacent_triangle_pairs(list(triangles)))


def detect_bisected_lozenge_candidates(
    graph: nx.Graph,
    triangles: Iterable[tuple[str, str, str]] | None = None,
    side_tolerance: float = 0.35,
    angle_tolerance_deg: float = 18.0,
    diagonal_tolerance: float = 0.75,
    min_area: float = 4.0,
    min_confidence: float = 0.55,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for first, second, shared in _adjacent_triangle_pairs(
        list(triangles) if triangles is not None else enumerate_triangles(graph)
    ):
        union_nodes = set(first) | set(second)
        if len(union_nodes) != 4:
            continue
        outer = [node for node in union_nodes if node not in shared]
        if len(outer) != 2:
            continue
        ordered = _order_nodes_around_centroid(graph, list(union_nodes))
        points = [_node_xy(graph, node) for node in ordered]
        area = abs(_polygon_area(points))
        if area < min_area:
            continue
        sides = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
        if min(sides) <= 0:
            continue
        side_ratio = max(sides) / min(sides)
        if side_ratio > 1.0 + side_tolerance:
            continue
        angles = [_interior_angle(points[i - 1], points[i], points[(i + 1) % 4]) for i in range(4)]
        parallel_a = _axial_angle_diff(_segment_angle(points[0], points[1]), _segment_angle(points[2], points[3]))
        parallel_b = _axial_angle_diff(_segment_angle(points[1], points[2]), _segment_angle(points[3], points[0]))
        if parallel_a > angle_tolerance_deg or parallel_b > angle_tolerance_deg:
            continue
        opposite_angle_error = max(
            abs(angles[0] - angles[2]),
            abs(angles[1] - angles[3]),
        )
        if opposite_angle_error > angle_tolerance_deg:
            continue
        diagonals = [math.dist(points[0], points[2]), math.dist(points[1], points[3])]
        diagonal_ratio = max(diagonals) / min(diagonals) if min(diagonals) else float("inf")
        if diagonal_ratio > 1.0 + diagonal_tolerance:
            continue
        parallel_score = max(0.0, 1.0 - max(parallel_a, parallel_b) / angle_tolerance_deg)
        angle_score = max(0.0, 1.0 - opposite_angle_error / angle_tolerance_deg)
        side_score = min(1.0, 1.0 / side_ratio)
        diagonal_score = min(1.0, 1.0 / diagonal_ratio) if math.isfinite(diagonal_ratio) else 0.0
        confidence = float(
            0.35 * parallel_score
            + 0.25 * angle_score
            + 0.25 * side_score
            + 0.15 * diagonal_score
        )
        if confidence < min_confidence:
            continue
        candidates.append(
            {
                "triangle_a": ";".join(first),
                "triangle_b": ";".join(second),
                "shared_edge": ";".join(sorted(shared)),
                "outer_boundary_nodes": ";".join(ordered),
                "area": float(area),
                "side_lengths": ";".join(f"{value:.6g}" for value in sides),
                "interior_angles_deg": ";".join(f"{value:.6g}" for value in angles),
                "diagonal_lengths": ";".join(f"{value:.6g}" for value in diagonals),
                "opposite_parallel_error_deg": float(max(parallel_a, parallel_b)),
                "opposite_angle_error_deg": float(opposite_angle_error),
                "side_length_ratio": float(side_ratio),
                "diagonal_length_ratio": float(diagonal_ratio),
                "confidence": confidence,
            }
        )
    candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return candidates


def enumerate_four_cycles(graph: nx.Graph) -> list[tuple[str, str, str, str]]:
    cycles: set[tuple[str, str, str, str]] = set()
    nodes = sorted(str(node) for node in graph.nodes)
    for a in nodes:
        for b in sorted(str(node) for node in graph.neighbors(a)):
            if b == a:
                continue
            for c in sorted(str(node) for node in graph.neighbors(b)):
                if c in {a, b}:
                    continue
                for d in sorted(str(node) for node in graph.neighbors(c)):
                    if d in {a, b, c}:
                        continue
                    if graph.has_edge(d, a):
                        cycles.add(_canonical_cycle((a, b, c, d)))
    return sorted(cycles)


def detect_lozenge_candidates(
    graph: nx.Graph,
    cycles: Iterable[tuple[str, str, str, str]] | None = None,
    side_tolerance: float = 0.35,
    angle_tolerance_deg: float = 18.0,
    min_area: float = 4.0,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for cycle in cycles if cycles is not None else enumerate_four_cycles(graph):
        ordered = _order_cycle(graph, cycle)
        if ordered is None:
            continue
        points = [_node_xy(graph, node) for node in ordered]
        sides = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
        if min(sides) <= 0:
            continue
        side_ratio = max(sides) / min(sides)
        angles = [_interior_angle(points[i - 1], points[i], points[(i + 1) % 4]) for i in range(4)]
        parallel_a = _axial_angle_diff(_segment_angle(points[0], points[1]), _segment_angle(points[2], points[3]))
        parallel_b = _axial_angle_diff(_segment_angle(points[1], points[2]), _segment_angle(points[3], points[0]))
        area = abs(_polygon_area(points))
        diagonals = [math.dist(points[0], points[2]), math.dist(points[1], points[3])]
        if (
            side_ratio <= 1.0 + side_tolerance
            and parallel_a <= angle_tolerance_deg
            and parallel_b <= angle_tolerance_deg
            and abs(angles[0] - angles[2]) <= angle_tolerance_deg
            and abs(angles[1] - angles[3]) <= angle_tolerance_deg
            and area >= min_area
        ):
            candidates.append(
                {
                    "cycle_nodes": ";".join(ordered),
                    "side_lengths": ";".join(f"{value:.6g}" for value in sides),
                    "interior_angles_deg": ";".join(f"{value:.6g}" for value in angles),
                    "diagonal_lengths": ";".join(f"{value:.6g}" for value in diagonals),
                    "aspect_ratio": float(max(diagonals) / min(diagonals)) if min(diagonals) else 0.0,
                    "area": float(area),
                    "opposite_parallel_error_deg": float(max(parallel_a, parallel_b)),
                    "side_length_ratio": float(side_ratio),
                }
            )
    return candidates


def detect_open_lozenge_candidates(
    graph: nx.Graph,
    closed_candidates: Iterable[dict[str, object]] | None = None,
    parallel_tolerance_deg: float = 18.0,
    angle_tolerance_deg: float = 22.0,
    extension_tolerance_px: float = 12.0,
    side_tolerance: float = 0.45,
    min_area: float = 4.0,
    min_confidence: float = 0.45,
    max_parallel_pairs: int = 400,
) -> list[dict[str, object]]:
    edges = _geometric_edges(graph)
    if len(edges) < 4:
        return []
    closed_node_sets = _closed_lozenge_node_sets(closed_candidates)
    parallel_pairs: list[tuple[dict[str, object], dict[str, object], float]] = []
    for index, first in enumerate(edges):
        for second in edges[index + 1 :]:
            if set(first["nodes"]) & set(second["nodes"]):
                continue
            parallel_error = _axial_angle_diff(float(first["angle"]), float(second["angle"]))
            if parallel_error > parallel_tolerance_deg:
                continue
            length_ratio = max(float(first["length"]), float(second["length"])) / max(
                1e-9, min(float(first["length"]), float(second["length"]))
            )
            if length_ratio > 1.0 + side_tolerance:
                continue
            parallel_pairs.append((first, second, parallel_error))
            if len(parallel_pairs) >= max_parallel_pairs:
                break
        if len(parallel_pairs) >= max_parallel_pairs:
            break

    candidates: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for pair_index, (a1, a2, error_a) in enumerate(parallel_pairs):
        nodes_a = set(a1["nodes"]) | set(a2["nodes"])
        for b1, b2, error_b in parallel_pairs[pair_index + 1 :]:
            nodes = nodes_a | set(b1["nodes"]) | set(b2["nodes"])
            if len(nodes) < 4:
                continue
            if any(nodes == closed for closed in closed_node_sets):
                continue
            family_angle = _axial_angle_diff(float(a1["angle"]), float(b1["angle"]))
            if family_angle < angle_tolerance_deg or family_angle > 180.0 - angle_tolerance_deg:
                continue
            intersections: list[PointF] = []
            support_hits = 0
            for side_a in (a1, a2):
                for side_b in (b1, b2):
                    point = _line_intersection(
                        side_a["points"][0],
                        side_a["points"][1],
                        side_b["points"][0],
                        side_b["points"][1],
                    )
                    if point is None:
                        break
                    intersections.append(point)
                    if (
                        _point_segment_distance(point, side_a["points"][0], side_a["points"][1])
                        <= extension_tolerance_px
                        and _point_segment_distance(point, side_b["points"][0], side_b["points"][1])
                        <= extension_tolerance_px
                    ):
                        support_hits += 1
                else:
                    continue
                break
            if len(intersections) != 4 or support_hits < 2:
                continue
            ordered_points = _order_points_around_centroid(intersections)
            area = abs(_polygon_area(ordered_points))
            if area < min_area:
                continue
            side_lengths = [
                math.dist(ordered_points[i], ordered_points[(i + 1) % 4])
                for i in range(4)
            ]
            if min(side_lengths) <= 0:
                continue
            side_ratio = max(side_lengths) / min(side_lengths)
            if side_ratio > 1.0 + side_tolerance:
                continue
            key = tuple(sorted(tuple(sorted(edge["nodes"])) for edge in (a1, a2, b1, b2)))
            if key in seen:
                continue
            seen.add(key)
            parallel_score = max(0.0, 1.0 - ((error_a + error_b) / 2.0) / parallel_tolerance_deg)
            support_score = support_hits / 4.0
            balance_score = min(1.0, 1.0 / side_ratio)
            angle_score = min(1.0, family_angle / 60.0, (180.0 - family_angle) / 60.0)
            confidence = float(
                0.35 * support_score
                + 0.25 * parallel_score
                + 0.25 * balance_score
                + 0.15 * angle_score
            )
            if confidence < min_confidence:
                continue
            diagonals = [math.dist(ordered_points[0], ordered_points[2]), math.dist(ordered_points[1], ordered_points[3])]
            candidates.append(
                {
                    "side_edges": ";".join("|".join(edge["nodes"]) for edge in (a1, a2, b1, b2)),
                    "corner_points": ";".join(f"{x:.3f},{y:.3f}" for x, y in ordered_points),
                    "side_lengths": ";".join(f"{value:.6g}" for value in side_lengths),
                    "diagonal_lengths": ";".join(f"{value:.6g}" for value in diagonals),
                    "area": float(area),
                    "family_angle_deg": float(family_angle),
                    "opposite_parallel_error_deg": float(max(error_a, error_b)),
                    "side_length_ratio": float(side_ratio),
                    "supporting_intersections": support_hits,
                    "confidence": confidence,
                }
            )
    candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return candidates


def write_lozenge_outputs(
    graph_path: str | Path,
    output_dir: str | Path,
    side_tolerance: float = 0.35,
    angle_tolerance_deg: float = 18.0,
) -> tuple[Path, Path]:
    graph = nx.read_graphml(graph_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cycles = enumerate_four_cycles(nx.Graph(graph))
    triangles = enumerate_triangles(nx.Graph(graph))
    candidates = detect_lozenge_candidates(
        nx.Graph(graph),
        cycles,
        side_tolerance=side_tolerance,
        angle_tolerance_deg=angle_tolerance_deg,
    )
    open_candidates = detect_open_lozenge_candidates(nx.Graph(graph), closed_candidates=candidates)
    bisected_candidates = detect_bisected_lozenge_candidates(
        nx.Graph(graph),
        triangles,
        side_tolerance=side_tolerance,
        angle_tolerance_deg=angle_tolerance_deg,
    )
    csv_path = root / "lozenge_candidates.csv"
    fields = [
        "cycle_nodes",
        "side_lengths",
        "interior_angles_deg",
        "diagonal_lengths",
        "aspect_ratio",
        "area",
        "opposite_parallel_error_deg",
        "side_length_ratio",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    open_csv_path = root / "open_lozenge_candidates.csv"
    open_fields = [
        "side_edges",
        "corner_points",
        "side_lengths",
        "diagonal_lengths",
        "area",
        "family_angle_deg",
        "opposite_parallel_error_deg",
        "side_length_ratio",
        "supporting_intersections",
        "confidence",
    ]
    with open_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=open_fields)
        writer.writeheader()
        writer.writerows(open_candidates)
    bisected_csv_path = root / "bisected_lozenge_candidates.csv"
    bisected_fields = [
        "triangle_a",
        "triangle_b",
        "shared_edge",
        "outer_boundary_nodes",
        "area",
        "side_lengths",
        "interior_angles_deg",
        "diagonal_lengths",
        "opposite_parallel_error_deg",
        "opposite_angle_error_deg",
        "side_length_ratio",
        "diagonal_length_ratio",
        "confidence",
    ]
    with bisected_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bisected_fields)
        writer.writeheader()
        writer.writerows(bisected_candidates)
    summary = {
        "graph": str(graph_path),
        "triangle_count": len(triangles),
        "adjacent_triangle_pair_count": adjacent_triangle_pair_count(triangles),
        "bisected_lozenge_candidate_count": len(bisected_candidates),
        "bisected_lozenge_mean_confidence": _mean(
            [float(item["confidence"]) for item in bisected_candidates]
        ),
        "four_cycle_count": len(cycles),
        "closed_lozenge_candidate_count": len(candidates),
        "lozenge_candidate_count": len(candidates),
        "open_lozenge_candidate_count": len(open_candidates),
        "open_lozenge_mean_confidence": _mean([float(item["confidence"]) for item in open_candidates]),
        "side_tolerance": side_tolerance,
        "angle_tolerance_deg": angle_tolerance_deg,
    }
    json_path = root / "lozenge_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def compare_null_models(
    observed_graph: str | Path | None,
    observed_metrics: str | Path | None,
    controls_dir: str | Path,
    output_dir: str | Path,
    orientation_shuffles: int = 0,
    seed: int = 42,
    include_raw_graphs: bool = False,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    observed = _load_analysis(observed_graph, observed_metrics)
    controls = _load_control_analyses(controls_dir, include_raw_graphs=include_raw_graphs)
    if observed_graph and orientation_shuffles > 0:
        for index, metrics in enumerate(
            orientation_shuffled_edge_metrics(observed_graph, orientation_shuffles, seed)
        ):
            controls.append(
                {
                    "label": f"random_orientation_shuffle_{index:04d}",
                    "class": "random",
                    "path": str(observed_graph),
                    "metrics": metrics,
                    "valid": True,
                    "invalid_reason": "",
                }
            )
    valid_controls = [item for item in controls if item["valid"]]
    invalid_controls = [item for item in controls if not item["valid"]]
    null_controls = [item for item in valid_controls if item["class"] == "random"]
    lozenge_controls = [item for item in valid_controls if item["class"] == "lozenge"]
    class_aggregates = _class_aggregates(valid_controls)

    rows: list[dict[str, object]] = []
    for metric in DEFAULT_METRICS:
        null_values = [float(item["metrics"][metric]) for item in null_controls if metric in item["metrics"]]
        lozenge_values = [
            float(item["metrics"][metric]) for item in lozenge_controls if metric in item["metrics"]
        ]
        observed_value = float(observed.get(metric, 0.0))
        rows.append(
            {
                "metric": metric,
                "observed": observed_value,
                "null_mean": _mean(null_values),
                "null_sd": _sd(null_values),
                "null_percentile": _percentile(observed_value, null_values),
                "empirical_p_greater_equal": _empirical_p(observed_value, null_values, "greater"),
                "empirical_p_less_equal": _empirical_p(observed_value, null_values, "less"),
                "lozenge_control_n": len(lozenge_values),
                "lozenge_control_mean": _mean(lozenge_values),
                "lozenge_control_sd": _sd(lozenge_values),
                "random_control_n": len(null_values),
                "random_control_mean": _mean(null_values),
                "random_control_sd": _sd(null_values),
                "closer_to": _closer_to(observed_value, lozenge_values, null_values),
            }
        )

    csv_path = root / "comparison_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "observed": observed,
        "valid_controls": valid_controls,
        "invalid_controls": invalid_controls,
        "class_aggregates": class_aggregates,
        "comparisons": rows,
        "interpretation_note": (
            "These summaries provide statistical support for or against mesh-like geometric "
            "structure relative to the supplied controls. They do not establish intention."
        ),
    }
    json_path = root / "comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_invalid_controls(root / "invalid_controls.csv", invalid_controls)
    _write_metric_plots(rows, valid_controls, root)
    if observed_graph:
        write_lozenge_outputs(observed_graph, root)
    return csv_path, json_path


def orientation_shuffled_edge_metrics(
    graph_path: str | Path,
    iterations: int = 100,
    seed: int = 42,
) -> list[dict[str, object]]:
    graph = nx.read_graphml(graph_path)
    rng = random.Random(seed)
    lengths = [_edge_length(graph, u, v, data) for u, v, data in graph.edges(data=True)]
    results: list[dict[str, object]] = []
    for _ in range(iterations):
        orientations = [rng.uniform(0, 180) for _ in lengths]
        hist = _orientation_histogram(orientations, lengths)
        results.append(
            {
                "orientation_peak_concentration": float(max(hist) / sum(hist)) if sum(hist) else 0.0,
                "orientation_entropy": _orientation_entropy(hist),
                "dominant_orientation_families": len(_dominant_orientation_peaks(hist)),
                "total_traced_length": float(sum(lengths)),
            }
        )
    return results


def graph_from_strokes(strokes: list[Stroke]) -> nx.Graph:
    graph = nx.Graph()
    for stroke_index, stroke in enumerate(strokes):
        previous = None
        for point_index, point in enumerate(stroke.points):
            node = f"s{stroke_index:04d}_{point_index:03d}"
            graph.add_node(node, x_px=float(point[0]), y_px=float(point[1]))
            if previous is not None:
                a = graph.nodes[previous]
                length = math.dist((float(a["x_px"]), float(a["y_px"])), point)
                angle = _segment_angle((float(a["x_px"]), float(a["y_px"])), point)
                graph.add_edge(previous, node, length_px=length, orientation_deg=angle)
            previous = node
    return graph


def lozenge_grid_graph(width: int = 180, height: int = 140, spacing: int = 35) -> nx.Graph:
    strokes = lozenge_lattice_strokes(width, height, spacing)
    points: dict[tuple[int, int], str] = {}
    graph = nx.Graph()
    for stroke in strokes:
        for point in _stroke_intersections(stroke, strokes, width, height):
            key = (round(point[0]), round(point[1]))
            if key not in points:
                node = f"n{len(points):05d}"
                points[key] = node
                graph.add_node(node, x_px=float(key[0]), y_px=float(key[1]))
    for stroke in strokes:
        line_points = sorted(
            [
                key for key in points
                if _point_on_segment((float(key[0]), float(key[1])), stroke.points[0], stroke.points[-1])
            ],
            key=lambda key: math.dist(stroke.points[0], (key[0], key[1])),
        )
        for a, b in zip(line_points, line_points[1:]):
            if a == b:
                continue
            node_a = points[a]
            node_b = points[b]
            length = math.dist(a, b)
            graph.add_edge(node_a, node_b, length_px=length, orientation_deg=_segment_angle(a, b))
    return graph


def save_graphml(graph: nx.Graph, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output)


def generate_controls_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic lozenge and random-scratch controls.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--spacing", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--svg-previews", action="store_true")
    parser.add_argument("--stroke-width", type=int, default=3)
    parser.add_argument(
        "--control-polarity",
        choices=("bright-on-dark", "dark-on-bright"),
        default="bright-on-dark",
    )
    args = parser.parse_args(argv)
    if args.seed_count <= 1:
        metadata = generate_control_set(
            args.output_dir,
            width=args.width,
            height=args.height,
            spacing=args.spacing,
            seed=args.seed,
            svg_previews=args.svg_previews,
            stroke_width=args.stroke_width,
            control_polarity=args.control_polarity,
        )
        print(f"Wrote {len(metadata['controls'])} controls to {args.output_dir}.")
    else:
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "seed_start": args.seed,
            "seed_count": args.seed_count,
            "controls": [],
        }
        for offset in range(args.seed_count):
            seed = args.seed + offset
            seed_dir = root / f"seed_{seed:04d}"
            metadata = generate_control_set(
                seed_dir,
                width=args.width,
                height=args.height,
                spacing=args.spacing,
                seed=seed,
                svg_previews=args.svg_previews,
                stroke_width=args.stroke_width,
                control_polarity=args.control_polarity,
            )
            manifest["controls"].append(
                {"seed": seed, "directory": str(seed_dir), "control_count": len(metadata["controls"])}
            )
        (root / "control_seed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {args.seed_count} seeded control sets to {args.output_dir}.")
    return 0


def compare_null_models_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare observed engraving graph metrics to controls/null models.")
    parser.add_argument("--observed-graph")
    parser.add_argument("--observed-metrics")
    parser.add_argument("--controls-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--orientation-shuffles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-raw-graphs",
        action="store_true",
        help="Treat *_raw.graphml files as independent controls. Default ignores raw graphs to avoid double-counting extraction runs.",
    )
    args = parser.parse_args(argv)
    if not args.observed_graph and not args.observed_metrics:
        parser.error("--observed-graph or --observed-metrics is required.")
    csv_path, json_path = compare_null_models(
        observed_graph=args.observed_graph,
        observed_metrics=args.observed_metrics,
        controls_dir=args.controls_dir,
        output_dir=args.output_dir,
        orientation_shuffles=max(0, args.orientation_shuffles),
        seed=args.seed,
        include_raw_graphs=args.include_raw_graphs,
    )
    print(f"Wrote {csv_path} and {json_path}.")
    return 0


def _write_control(
    root: Path,
    name: str,
    control_class: str,
    strokes: list[Stroke],
    width: int,
    height: int,
    svg_previews: bool,
    stroke_width: int,
    control_polarity: str,
) -> dict[str, object]:
    png_path = root / f"{name}.png"
    if control_polarity == "bright-on-dark":
        image = np.zeros((height, width, 3), dtype=np.uint8)
        stroke_color = (245, 245, 245)
        background = "black"
        foreground = "white"
    else:
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        stroke_color = (10, 10, 10)
        background = "white"
        foreground = "black"
    for stroke in strokes:
        points = np.array([(round(x), round(y)) for x, y in stroke.points], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points], False, stroke_color, max(1, stroke_width), cv2.LINE_AA)
    cv2.imwrite(str(png_path), image)
    if svg_previews:
        _write_svg_preview(root / f"{name}.svg", strokes, width, height, stroke_width, background, foreground)
    return {
        "name": name,
        "class": control_class,
        "image": str(png_path),
        "svg": str(root / f"{name}.svg") if svg_previews else "",
        "stroke_width": stroke_width,
        "control_polarity": control_polarity,
        "stroke_count": len(strokes),
        "total_length_px": float(sum(stroke.length for stroke in strokes)),
        "bounding_box": {"x": 0, "y": 0, "width": width, "height": height},
    }


def _write_svg_preview(
    path: Path,
    strokes: list[Stroke],
    width: int,
    height: int,
    stroke_width: int,
    background: str,
    foreground: str,
) -> None:
    drawing = svgwrite.Drawing(str(path), size=(width, height), viewBox=f"0 0 {width} {height}")
    drawing.add(drawing.rect(insert=(0, 0), size=(width, height), fill=background))
    for stroke in strokes:
        drawing.add(drawing.polyline(points=stroke.points, fill="none", stroke=foreground, stroke_width=stroke_width))
    drawing.save()


def _clip_line_to_box(slope: float, intercept: float, width: int, height: int) -> list[PointF] | None:
    points: list[PointF] = []
    for x in (0.0, float(width - 1)):
        y = slope * x + intercept
        if 0 <= y <= height - 1:
            points.append((x, y))
    for y in (0.0, float(height - 1)):
        x = (y - intercept) / slope
        if 0 <= x <= width - 1:
            points.append((x, y))
    unique: list[PointF] = []
    for point in points:
        if all(math.dist(point, other) > 1e-6 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    return [unique[0], unique[1]]


def _load_analysis(graph_path: str | Path | None, metrics_path: str | Path | None) -> dict[str, object]:
    values: dict[str, object] = {}
    if metrics_path:
        values.update(_metrics_from_json(metrics_path))
    if graph_path:
        values.update(analyze_graphml(graph_path))
    return values


def _load_control_analyses(
    controls_dir: str | Path, include_raw_graphs: bool = False
) -> list[dict[str, object]]:
    root = Path(controls_dir)
    metadata_classes = _metadata_classes(root)
    controls: list[dict[str, object]] = []

    consumed_metrics: set[Path] = set()
    consumed_graphs: set[Path] = set()
    for metrics_path in sorted(root.rglob("*.json")):
        if _skip_json(metrics_path):
            continue
        graph_path = _preferred_graph_for_metrics(metrics_path, include_raw_graphs)
        metrics = _metrics_from_json(metrics_path)
        if graph_path:
            metrics.update(analyze_graphml(graph_path))
            consumed_graphs.add(graph_path.resolve())
        label = _control_label(metrics_path, graph_path)
        control_class = _classify_control(label, metadata_classes)
        controls.append(_control_record(label, control_class, str(graph_path or metrics_path), metrics))
        consumed_metrics.add(metrics_path.resolve())

    for graph_path in sorted(root.rglob("*.graphml")):
        if graph_path.resolve() in consumed_graphs:
            continue
        if not include_raw_graphs and _is_raw_graph(graph_path):
            continue
        label = _control_label(None, graph_path)
        control_class = _classify_control(label, metadata_classes)
        controls.append(_control_record(label, control_class, str(graph_path), analyze_graphml(graph_path)))
    return controls


def _control_record(
    label: str, control_class: str, path: str, metrics: dict[str, object]
) -> dict[str, object]:
    node_count = float(metrics.get("node_count", 0) or 0)
    total_length = float(metrics.get("total_traced_length", 0) or 0)
    invalid_reasons: list[str] = []
    if node_count <= 0:
        invalid_reasons.append("node_count == 0")
    if total_length <= 0:
        invalid_reasons.append("total_traced_length == 0")
    if control_class == "unknown":
        invalid_reasons.append("control class could not be inferred")
    return {
        "label": label,
        "class": control_class,
        "path": path,
        "metrics": metrics,
        "valid": not invalid_reasons,
        "invalid_reason": "; ".join(invalid_reasons),
    }


def _metadata_classes(root: Path) -> dict[str, str]:
    metadata_path = root / "expected_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return {}
    classes: dict[str, str] = {}
    for control in payload.get("controls", []):
        name = str(control.get("name", ""))
        control_class = str(control.get("class", ""))
        if name and control_class:
            classes[name.lower()] = control_class
    return classes


def _classify_control(label: str, metadata_classes: dict[str, str]) -> str:
    lower = label.lower()
    for name, control_class in metadata_classes.items():
        if name in lower:
            return control_class
    if "lozenge" in lower or "rhomb" in lower or "diamond" in lower:
        return "lozenge"
    if "random" in lower or "scratch" in lower or "shuffle" in lower:
        return "random"
    return "unknown"


def _control_label(metrics_path: Path | None, graph_path: Path | None) -> str:
    primary = graph_path or metrics_path
    if primary is None:
        return "unknown"
    parent = primary.parent.name
    if parent and parent not in {".", ""} and primary.stem in {"metrics", "output_merged", "graph_merged"}:
        return parent
    stem = primary.stem
    for suffix in ("_merged", "_raw", "_metrics", "_output"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _preferred_graph_for_metrics(
    metrics_path: Path, include_raw_graphs: bool
) -> Path | None:
    candidates = [
        metrics_path.with_name("graph_merged.graphml"),
        metrics_path.with_name("output_merged.graphml"),
        metrics_path.with_name(f"{metrics_path.stem}_merged.graphml"),
    ]
    candidates.extend(sorted(metrics_path.parent.glob("*_merged.graphml")))
    if include_raw_graphs:
        candidates.extend(sorted(metrics_path.parent.glob("*_raw.graphml")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _is_raw_graph(path: Path) -> bool:
    return path.stem.endswith("_raw") or path.name == "graph_raw.graphml"


def _skip_json(path: Path) -> bool:
    return path.name in {
        "comparison_summary.json",
        "lozenge_summary.json",
        "expected_metadata.json",
        "control_seed_manifest.json",
    }


def _metrics_from_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    graph_metrics = payload.get("graph_metrics", payload)
    merged = graph_metrics.get("merged", {})
    orientation = graph_metrics.get("orientation", {})
    values = {
        "node_count": graph_metrics.get("merged_node_count", graph_metrics.get("node_count", merged.get("node_count", 0))),
        "edge_count": graph_metrics.get("merged_edge_count", graph_metrics.get("edge_count", merged.get("edge_count", 0))),
        "endpoint_count": graph_metrics.get("merged_endpoint_count", graph_metrics.get("endpoint_count", merged.get("endpoints", 0))),
        "junction_count": graph_metrics.get("merged_junction_count", graph_metrics.get("junction_count", merged.get("junctions", 0))),
        "connected_component_count": graph_metrics.get(
            "connected_components",
            graph_metrics.get("connected_component_count", merged.get("connected_components", 0)),
        ),
        "largest_connected_component_fraction": graph_metrics.get(
            "largest_connected_component_fraction",
            merged.get("largest_connected_component_fraction", 0),
        ),
        "total_traced_length": graph_metrics.get(
            "total_traced_length",
            graph_metrics.get("total_traced_length_px", merged.get("total_traced_length_px", 0)),
        ),
        "cycle_count": graph_metrics.get("cycle_count", 0),
        "triangle_count": graph_metrics.get("triangle_count", 0),
        "triangle_side_length_cv_mean": graph_metrics.get("triangle_side_length_cv_mean", 0),
        "triangle_area_mean": graph_metrics.get("triangle_area_mean", 0),
        "triangle_area_cv": graph_metrics.get("triangle_area_cv", 0),
        "adjacent_triangle_pair_count": graph_metrics.get("adjacent_triangle_pair_count", 0),
        "bisected_lozenge_candidate_count": graph_metrics.get("bisected_lozenge_candidate_count", 0),
        "bisected_lozenge_mean_confidence": graph_metrics.get("bisected_lozenge_mean_confidence", 0),
        "four_cycle_count": graph_metrics.get("four_cycle_count", 0),
        "closed_lozenge_candidate_count": graph_metrics.get(
            "closed_lozenge_candidate_count",
            graph_metrics.get("lozenge_candidate_count", 0),
        ),
        "lozenge_candidate_count": graph_metrics.get("lozenge_candidate_count", 0),
        "open_lozenge_candidate_count": graph_metrics.get("open_lozenge_candidate_count", 0),
        "open_lozenge_mean_confidence": graph_metrics.get("open_lozenge_mean_confidence", 0),
    }
    for key in ("degree3_fraction", "degree4_fraction", "degree_3_fraction"):
        if key in graph_metrics:
            values[key] = graph_metrics[key]
    degree_distribution = graph_metrics.get("degree_distribution", merged.get("degree_distribution", {}))
    node_count = float(values.get("node_count", 0) or 0)
    if isinstance(degree_distribution, dict) and node_count > 0:
        values["degree3_fraction"] = float(degree_distribution.get("3", degree_distribution.get(3, 0))) / node_count
        values["degree4_fraction"] = float(degree_distribution.get("4", degree_distribution.get(4, 0))) / node_count
    hist = orientation.get("length_weighted_histogram", [])
    if hist:
        values["orientation_peak_concentration"] = float(max(hist) / sum(hist)) if sum(hist) else 0.0
        values["orientation_entropy"] = _orientation_entropy([float(value) for value in hist])
        values["dominant_orientation_families"] = len(_dominant_orientation_peaks([float(value) for value in hist]))
    return _with_normalized_metrics(values)


def _with_normalized_metrics(metrics: dict[str, object]) -> dict[str, object]:
    total_length = float(metrics.get("total_traced_length", 0) or 0)
    node_count = float(metrics.get("node_count", 0) or 0)
    cycle_count = float(metrics.get("cycle_count", 0) or 0)
    endpoint_count = float(metrics.get("endpoint_count", 0) or 0)
    junction_count = float(metrics.get("junction_count", 0) or 0)
    triangle_count = float(metrics.get("triangle_count", 0) or 0)
    adjacent_triangle_pair_count_value = float(metrics.get("adjacent_triangle_pair_count", 0) or 0)
    bisected_lozenge_count = float(metrics.get("bisected_lozenge_candidate_count", 0) or 0)
    four_cycle_count = float(metrics.get("four_cycle_count", 0) or 0)
    lozenge_count = float(metrics.get("lozenge_candidate_count", 0) or 0)
    if "closed_lozenge_candidate_count" not in metrics:
        metrics["closed_lozenge_candidate_count"] = lozenge_count
    metrics["endpoints_per_1000px"] = endpoint_count / total_length * 1000.0 if total_length else 0.0
    metrics["junctions_per_1000px"] = junction_count / total_length * 1000.0 if total_length else 0.0
    metrics["cycles_per_node"] = cycle_count / node_count if node_count else 0.0
    metrics["triangles_per_node"] = triangle_count / node_count if node_count else 0.0
    metrics["triangles_per_cycle"] = triangle_count / cycle_count if cycle_count else 0.0
    metrics["bisected_lozenge_candidates_per_cycle"] = (
        bisected_lozenge_count / cycle_count if cycle_count else 0.0
    )
    metrics["bisected_lozenge_candidates_per_triangle_pair"] = (
        bisected_lozenge_count / adjacent_triangle_pair_count_value
        if adjacent_triangle_pair_count_value
        else 0.0
    )
    metrics["four_cycles_per_node"] = four_cycle_count / node_count if node_count else 0.0
    metrics["lozenge_candidates_per_cycle"] = lozenge_count / cycle_count if cycle_count else 0.0
    if "degree3_fraction" not in metrics:
        metrics["degree3_fraction"] = float(metrics.get("degree_3_fraction", 0) or 0)
    metrics["degree_3_fraction"] = float(metrics.get("degree3_fraction", metrics.get("degree_3_fraction", 0)) or 0)
    if "degree4_fraction" not in metrics:
        metrics["degree4_fraction"] = 0.0
    return metrics


def _write_invalid_controls(path: Path, invalid_controls: list[dict[str, object]]) -> None:
    fields = ["label", "class", "path", "invalid_reason", "node_count", "total_traced_length"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for control in invalid_controls:
            metrics = control.get("metrics", {})
            writer.writerow(
                {
                    "label": control.get("label", ""),
                    "class": control.get("class", ""),
                    "path": control.get("path", ""),
                    "invalid_reason": control.get("invalid_reason", ""),
                    "node_count": metrics.get("node_count", 0) if isinstance(metrics, dict) else 0,
                    "total_traced_length": metrics.get("total_traced_length", 0)
                    if isinstance(metrics, dict)
                    else 0,
                }
            )


def _class_aggregates(controls: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    aggregates: dict[str, dict[str, object]] = {}
    for control_class in sorted({str(control["class"]) for control in controls}):
        class_controls = [control for control in controls if control["class"] == control_class]
        metric_summaries: dict[str, dict[str, object]] = {}
        for metric in DEFAULT_METRICS:
            values = [
                float(control["metrics"][metric])
                for control in class_controls
                if metric in control["metrics"]
            ]
            metric_summaries[metric] = {
                "n": len(values),
                "mean": _mean(values),
                "sd": _sd(values),
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "values": values,
            }
        aggregates[control_class] = {
            "n_controls": len(class_controls),
            "metrics": metric_summaries,
        }
    return aggregates


def _write_metric_plots(
    rows: list[dict[str, object]],
    controls: list[dict[str, object]],
    output_dir: Path,
) -> None:
    for metric in PLOT_METRICS:
        row = next((item for item in rows if item["metric"] == metric), None)
        if not row:
            continue
        image = np.full((260, 420, 3), 255, dtype=np.uint8)
        random_values = [
            float(control["metrics"][metric])
            for control in controls
            if control["class"] == "random" and metric in control["metrics"]
        ]
        lozenge_values = [
            float(control["metrics"][metric])
            for control in controls
            if control["class"] == "lozenge" and metric in control["metrics"]
        ]
        observed = float(row["observed"])
        values = random_values + lozenge_values + [observed]
        minimum = min(values) if values else 0.0
        maximum = max(values) if values else 1.0
        if math.isclose(minimum, maximum):
            minimum -= 0.5
            maximum += 0.5
        plot_top, plot_bottom = 45, 205
        cv2.line(image, (55, plot_top), (55, plot_bottom), (40, 40, 40), 1)
        cv2.line(image, (55, plot_bottom), (380, plot_bottom), (40, 40, 40), 1)
        for x, label, class_values, color in (
            (155, "random", random_values, (70, 130, 190)),
            (275, "lozenge", lozenge_values, (70, 150, 90)),
        ):
            _draw_distribution(image, class_values, x, plot_top, plot_bottom, minimum, maximum, color)
            cv2.putText(image, label, (x - 35, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        observed_y = _plot_y(observed, plot_top, plot_bottom, minimum, maximum)
        cv2.line(image, (70, observed_y), (370, observed_y), (40, 40, 210), 1)
        cv2.putText(image, "observed", (310, max(plot_top + 12, observed_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 160), 1, cv2.LINE_AA)
        cv2.putText(image, metric[:36], (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"{metric}.png"), image)


def _draw_distribution(
    image: np.ndarray,
    values: list[float],
    x: int,
    plot_top: int,
    plot_bottom: int,
    minimum: float,
    maximum: float,
    color: tuple[int, int, int],
) -> None:
    if not values:
        cv2.putText(image, "n=0", (x - 15, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
        return
    sorted_values = sorted(values)
    q1 = _quantile(sorted_values, 0.25)
    median = _quantile(sorted_values, 0.5)
    q3 = _quantile(sorted_values, 0.75)
    y_q1 = _plot_y(q1, plot_top, plot_bottom, minimum, maximum)
    y_median = _plot_y(median, plot_top, plot_bottom, minimum, maximum)
    y_q3 = _plot_y(q3, plot_top, plot_bottom, minimum, maximum)
    y_min = _plot_y(min(sorted_values), plot_top, plot_bottom, minimum, maximum)
    y_max = _plot_y(max(sorted_values), plot_top, plot_bottom, minimum, maximum)
    cv2.line(image, (x, y_min), (x, y_max), color, 1)
    cv2.rectangle(image, (x - 20, min(y_q1, y_q3)), (x + 20, max(y_q1, y_q3)), color, 1)
    cv2.line(image, (x - 24, y_median), (x + 24, y_median), color, 2)
    for index, value in enumerate(sorted_values):
        jitter = ((index % 7) - 3) * 4
        cv2.circle(image, (x + jitter, _plot_y(value, plot_top, plot_bottom, minimum, maximum)), 3, color, -1)
    cv2.putText(image, f"n={len(values)}", (x - 18, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1, cv2.LINE_AA)


def _plot_y(value: float, plot_top: int, plot_bottom: int, minimum: float, maximum: float) -> int:
    fraction = (value - minimum) / (maximum - minimum)
    return int(round(plot_bottom - fraction * (plot_bottom - plot_top)))


def _orientation_histogram(orientations: list[float], lengths: list[float], bins: int = 18) -> list[float]:
    hist = [0.0] * bins
    for angle, length in zip(orientations, lengths):
        index = min(bins - 1, int((angle % 180.0) / 180.0 * bins))
        hist[index] += float(length)
    return hist


def _orientation_entropy(hist: list[float]) -> float:
    total = float(sum(hist))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in hist:
        if value > 0:
            p = value / total
            entropy -= p * math.log(p)
    return float(entropy / math.log(len(hist))) if len(hist) > 1 else 0.0


def _dominant_orientation_peaks(hist: list[float]) -> list[int]:
    if not hist or max(hist) <= 0:
        return []
    threshold = max(hist) * 0.25
    peaks: list[int] = []
    for index, value in enumerate(hist):
        if value >= threshold and value >= hist[index - 1] and value >= hist[(index + 1) % len(hist)]:
            peaks.append(index)
    return peaks


def _edge_length(graph: nx.Graph, u: str, v: str, data: dict[str, object]) -> float:
    if "length_px" in data:
        return float(data["length_px"])
    return math.dist(_node_xy(graph, u), _node_xy(graph, v))


def _edge_orientation(graph: nx.Graph, u: str, v: str, data: dict[str, object]) -> float:
    if "orientation_deg" in data:
        return float(data["orientation_deg"]) % 180.0
    return _segment_angle(_node_xy(graph, u), _node_xy(graph, v))


def _geometric_edges(graph: nx.Graph) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for u, v, data in graph.edges(data=True):
        a = _node_xy(graph, u)
        b = _node_xy(graph, v)
        length = _edge_length(graph, u, v, data)
        if length <= 0:
            continue
        edges.append(
            {
                "nodes": (str(u), str(v)),
                "points": (a, b),
                "length": float(length),
                "angle": _edge_orientation(graph, u, v, data),
            }
        )
    return edges


def _closed_lozenge_node_sets(candidates: Iterable[dict[str, object]] | None) -> list[set[str]]:
    closed: list[set[str]] = []
    for candidate in candidates or []:
        nodes = str(candidate.get("cycle_nodes", ""))
        if nodes:
            closed.append(set(nodes.split(";")))
    return closed


def _adjacent_triangle_pairs(
    triangles: list[tuple[str, str, str]]
) -> list[tuple[tuple[str, str, str], tuple[str, str, str], set[str]]]:
    pairs: list[tuple[tuple[str, str, str], tuple[str, str, str], set[str]]] = []
    ordered = [tuple(sorted(triangle)) for triangle in triangles]
    for index, first in enumerate(ordered):
        first_nodes = set(first)
        for second in ordered[index + 1 :]:
            shared = first_nodes & set(second)
            if len(shared) == 2:
                pairs.append((first, second, shared))
    return pairs


def _order_nodes_around_centroid(graph: nx.Graph, nodes: list[str]) -> list[str]:
    points = {node: _node_xy(graph, node) for node in nodes}
    cx = sum(point[0] for point in points.values()) / len(points)
    cy = sum(point[1] for point in points.values()) / len(points)
    return sorted(nodes, key=lambda node: math.atan2(points[node][1] - cy, points[node][0] - cx))


def _node_xy(graph: nx.Graph, node: str) -> PointF:
    data = graph.nodes[node]
    return (float(data.get("x_px", data.get("x", 0.0))), float(data.get("y_px", data.get("y", 0.0))))


def _order_cycle(graph: nx.Graph, cycle: tuple[str, str, str, str]) -> tuple[str, str, str, str] | None:
    start = cycle[0]
    path = [start]
    used = {start}
    current = start
    for _ in range(3):
        choices = [node for node in cycle if node not in used and graph.has_edge(current, node)]
        if not choices:
            return None
        current = choices[0]
        path.append(current)
        used.add(current)
    if graph.has_edge(path[-1], start):
        return tuple(path)  # type: ignore[return-value]
    return None


def _canonical_cycle(cycle: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    rotations = [cycle[i:] + cycle[:i] for i in range(4)]
    reversed_cycle = tuple(reversed(cycle))
    rotations.extend(reversed_cycle[i:] + reversed_cycle[:i] for i in range(4))
    return min(rotations)


def _segment_angle(a: PointF, b: PointF) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def _axial_angle_diff(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def _interior_angle(a: PointF, b: PointF, c: PointF) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = math.hypot(*ba) * math.hypot(*bc)
    if denom == 0:
        return 0.0
    dot = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / denom))
    return math.degrees(math.acos(dot))


def _polygon_area(points: list[PointF]) -> float:
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sd(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _percentile(value: float, distribution: list[float]) -> float:
    if not distribution:
        return 0.0
    return float(sum(1 for item in distribution if item <= value) / len(distribution))


def _empirical_p(value: float, distribution: list[float], tail: str) -> float:
    if not distribution:
        return 1.0
    if tail == "greater":
        count = sum(1 for item in distribution if item >= value)
    else:
        count = sum(1 for item in distribution if item <= value)
    return float((count + 1) / (len(distribution) + 1))


def _closer_to(value: float, lozenge_values: list[float], random_values: list[float]) -> str:
    if not lozenge_values and not random_values:
        return "undetermined"
    lozenge_distance = abs(value - _mean(lozenge_values)) if lozenge_values else float("inf")
    random_distance = abs(value - _mean(random_values)) if random_values else float("inf")
    if lozenge_distance < random_distance:
        return "lozenge_controls"
    if random_distance < lozenge_distance:
        return "random_controls"
    return "tie"


def _interpolate(a: PointF, b: PointF, t: float) -> PointF:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _clamp_point(point: PointF, width: int, height: int) -> PointF:
    return (min(max(point[0], 0.0), width - 1.0), min(max(point[1], 0.0), height - 1.0))


def _stroke_intersections(stroke: Stroke, strokes: list[Stroke], width: int, height: int) -> list[PointF]:
    points = [stroke.points[0], stroke.points[-1]]
    for other in strokes:
        point = _segment_intersection(stroke.points[0], stroke.points[-1], other.points[0], other.points[-1])
        if point and 0 <= point[0] < width and 0 <= point[1] < height:
            points.append(point)
    unique: list[PointF] = []
    for point in points:
        if all(math.dist(point, other) > 1.0 for other in unique):
            unique.append(point)
    return unique


def _segment_intersection(a: PointF, b: PointF, c: PointF, d: PointF) -> PointF | None:
    point = _line_intersection(a, b, c, d)
    if point and _point_on_segment(point, a, b) and _point_on_segment(point, c, d):
        return point
    return None


def _line_intersection(a: PointF, b: PointF, c: PointF, d: PointF) -> PointF | None:
    denominator = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if abs(denominator) < 1e-9:
        return None
    px = ((a[0] * b[1] - a[1] * b[0]) * (c[0] - d[0]) - (a[0] - b[0]) * (c[0] * d[1] - c[1] * d[0])) / denominator
    py = ((a[0] * b[1] - a[1] * b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] * d[1] - c[1] * d[0])) / denominator
    return (px, py)


def _point_segment_distance(point: PointF, a: PointF, b: PointF) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    length_sq = ab[0] * ab[0] + ab[1] * ab[1]
    if length_sq <= 0:
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((point[0] - a[0]) * ab[0] + (point[1] - a[1]) * ab[1]) / length_sq))
    projection = (a[0] + t * ab[0], a[1] + t * ab[1])
    return math.dist(point, projection)


def _order_points_around_centroid(points: list[PointF]) -> list[PointF]:
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    return sorted(points, key=lambda point: math.atan2(point[1] - cy, point[0] - cx))


def _point_on_segment(point: PointF, a: PointF, b: PointF, tolerance: float = 1.25) -> bool:
    line_distance = abs(math.dist(a, point) + math.dist(point, b) - math.dist(a, b))
    return line_distance <= tolerance
