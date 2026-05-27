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


DEFAULT_METRICS = (
    "orientation_peak_concentration",
    "orientation_entropy",
    "dominant_orientation_families",
    "endpoint_count",
    "junction_count",
    "degree_3_fraction",
    "connected_component_count",
    "largest_connected_component_fraction",
    "cycle_count",
    "four_cycle_count",
    "lozenge_candidate_count",
    "total_traced_length",
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
    four_cycles = enumerate_four_cycles(simple)
    lozenges = detect_lozenge_candidates(simple, four_cycles)
    return {
        "orientation_peak_concentration": float(max(hist) / sum(hist)) if sum(hist) else 0.0,
        "orientation_entropy": entropy,
        "dominant_orientation_families": len(peaks),
        "endpoint_count": sum(1 for degree in degrees.values() if degree == 1),
        "junction_count": sum(1 for degree in degrees.values() if degree >= 3),
        "degree_3_fraction": float(sum(1 for degree in degrees.values() if degree == 3) / node_count) if node_count else 0.0,
        "connected_component_count": len(components),
        "largest_connected_component_fraction": float(max(components, default=0) / node_count) if node_count else 0.0,
        "cycle_count": len(cycles),
        "four_cycle_count": len(four_cycles),
        "lozenge_candidate_count": len(lozenges),
        "total_traced_length": float(sum(edge_lengths)),
        "node_count": node_count,
        "edge_count": simple.number_of_edges(),
    }


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
    candidates = detect_lozenge_candidates(
        nx.Graph(graph),
        cycles,
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
    summary = {
        "graph": str(graph_path),
        "four_cycle_count": len(cycles),
        "lozenge_candidate_count": len(candidates),
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
                "lozenge_control_mean": _mean(lozenge_values),
                "random_control_mean": _mean(null_values),
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
        "comparisons": rows,
        "interpretation_note": (
            "These summaries provide statistical support for or against mesh-like geometric "
            "structure relative to the supplied controls. They do not establish intention."
        ),
    }
    json_path = root / "comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_invalid_controls(root / "invalid_controls.csv", invalid_controls)
    _write_metric_plots(rows, root)
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
    parser.add_argument("--svg-previews", action="store_true")
    parser.add_argument("--stroke-width", type=int, default=3)
    parser.add_argument(
        "--control-polarity",
        choices=("bright-on-dark", "dark-on-bright"),
        default="bright-on-dark",
    )
    args = parser.parse_args(argv)
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
    }


def _metrics_from_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    graph_metrics = payload.get("graph_metrics", payload)
    merged = graph_metrics.get("merged", {})
    orientation = graph_metrics.get("orientation", {})
    values = {
        "node_count": graph_metrics.get("merged_node_count", merged.get("node_count", 0)),
        "edge_count": graph_metrics.get("merged_edge_count", merged.get("edge_count", 0)),
        "endpoint_count": graph_metrics.get("merged_endpoint_count", merged.get("endpoints", 0)),
        "junction_count": graph_metrics.get("merged_junction_count", merged.get("junctions", 0)),
        "connected_component_count": graph_metrics.get("connected_components", merged.get("connected_components", 0)),
        "largest_connected_component_fraction": graph_metrics.get(
            "largest_connected_component_fraction",
            merged.get("largest_connected_component_fraction", 0),
        ),
        "total_traced_length": graph_metrics.get("total_traced_length_px", merged.get("total_traced_length_px", 0)),
        "cycle_count": graph_metrics.get("cycle_count", 0),
        "four_cycle_count": graph_metrics.get("four_cycle_count", 0),
        "lozenge_candidate_count": graph_metrics.get("lozenge_candidate_count", 0),
    }
    hist = orientation.get("length_weighted_histogram", [])
    if hist:
        values["orientation_peak_concentration"] = float(max(hist) / sum(hist)) if sum(hist) else 0.0
        values["orientation_entropy"] = _orientation_entropy([float(value) for value in hist])
        values["dominant_orientation_families"] = len(_dominant_orientation_peaks([float(value) for value in hist]))
    return values


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


def _write_metric_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    for metric in DEFAULT_METRICS[:6]:
        row = next((item for item in rows if item["metric"] == metric), None)
        if not row:
            continue
        image = np.full((220, 360, 3), 255, dtype=np.uint8)
        labels = ["observed", "null", "lozenge"]
        values = [
            float(row["observed"]),
            float(row["random_control_mean"]),
            float(row["lozenge_control_mean"]),
        ]
        maximum = max(values) if max(values) > 0 else 1.0
        for index, value in enumerate(values):
            x0 = 45 + index * 100
            h = int(140 * value / maximum)
            cv2.rectangle(image, (x0, 170 - h), (x0 + 48, 170), (60, 110, 180), -1)
            cv2.putText(image, labels[index], (x0 - 10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, metric[:36], (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"{metric}.png"), image)


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
    denominator = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if abs(denominator) < 1e-9:
        return None
    px = ((a[0] * b[1] - a[1] * b[0]) * (c[0] - d[0]) - (a[0] - b[0]) * (c[0] * d[1] - c[1] * d[0])) / denominator
    py = ((a[0] * b[1] - a[1] * b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] * d[1] - c[1] * d[0])) / denominator
    if _point_on_segment((px, py), a, b) and _point_on_segment((px, py), c, d):
        return (px, py)
    return None


def _point_on_segment(point: PointF, a: PointF, b: PointF, tolerance: float = 1.25) -> bool:
    line_distance = abs(math.dist(a, point) + math.dist(point, b) - math.dist(a, b))
    return line_distance <= tolerance
