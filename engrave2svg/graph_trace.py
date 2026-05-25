from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import networkx as nx
import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True)
class Polyline:
    points: list[Point]
    closed: bool = False


@dataclass(frozen=True)
class TraceResult:
    polylines: list[Polyline]
    graph: nx.Graph


def trace_skeleton(skeleton: np.ndarray) -> TraceResult:
    graph = skeleton_to_graph(skeleton)
    if graph.number_of_nodes() == 0:
        return TraceResult(polylines=[], graph=graph)

    polylines: list[Polyline] = []
    visited_edges: set[frozenset[Point]] = set()
    node_points = {node for node, degree in graph.degree if degree != 2}

    for node in sorted(node_points):
        for neighbor in sorted(graph.neighbors(node)):
            edge_id = _edge_id(node, neighbor)
            if edge_id in visited_edges:
                continue
            path = _walk_until_node(graph, node, neighbor, node_points, visited_edges)
            if len(path) >= 2:
                polylines.append(Polyline(points=path, closed=False))

    for component in nx.connected_components(graph):
        component_edges = list(graph.subgraph(component).edges)
        if not component_edges:
            continue
        if all(_edge_id(a, b) in visited_edges for a, b in component_edges):
            continue
        cycle = _trace_cycle(graph, component, visited_edges)
        if len(cycle) >= 3:
            polylines.append(Polyline(points=cycle, closed=True))

    polylines = _drop_internal_node_artifacts(polylines, node_points)
    return TraceResult(polylines=polylines, graph=graph)


def simplify_polylines(
    polylines: list[Polyline], epsilon: float
) -> list[Polyline]:
    if epsilon <= 0:
        return polylines

    simplified: list[Polyline] = []
    for polyline in polylines:
        points = np.array(polyline.points, dtype=np.int32).reshape((-1, 1, 2))
        approx = cv2.approxPolyDP(points, epsilon=epsilon, closed=polyline.closed)
        approx_points = [(int(point[0][0]), int(point[0][1])) for point in approx]
        if len(approx_points) >= 2:
            simplified.append(Polyline(points=approx_points, closed=polyline.closed))
    return simplified


def save_trace_debug(
    skeleton: np.ndarray, polylines: list[Polyline], debug_dir: str | Path
) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    colors = [
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
    for index, polyline in enumerate(polylines):
        color = colors[index % len(colors)]
        points = np.array(polyline.points, dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], polyline.closed, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(path / "09_traced_polylines.png"), canvas)


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    skel = skeleton > 0
    graph = nx.Graph()
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        point = (int(x), int(y))
        graph.add_node(point)
        for nx_, ny_ in _prior_neighbors(skel, int(x), int(y)):
            graph.add_edge(point, (nx_, ny_))
    return graph


def _walk_until_node(
    graph: nx.Graph,
    start: Point,
    first_neighbor: Point,
    node_points: set[Point],
    visited_edges: set[frozenset[Point]],
) -> list[Point]:
    path = [start, first_neighbor]
    previous = start
    current = first_neighbor
    visited_edges.add(_edge_id(previous, current))

    while current not in node_points:
        next_nodes = [
            neighbor
            for neighbor in graph.neighbors(current)
            if neighbor != previous
            and _edge_id(current, neighbor) not in visited_edges
        ]
        if not next_nodes:
            break
        next_node = sorted(next_nodes)[0]
        visited_edges.add(_edge_id(current, next_node))
        path.append(next_node)
        previous, current = current, next_node
    return path


def _trace_cycle(
    graph: nx.Graph,
    component: set[Point],
    visited_edges: set[frozenset[Point]],
) -> list[Point]:
    start = sorted(component)[0]
    neighbors = sorted(graph.neighbors(start))
    if not neighbors:
        return [start]

    path = [start, neighbors[0]]
    previous = start
    current = neighbors[0]
    visited_edges.add(_edge_id(previous, current))

    while current != start:
        candidates = [
            neighbor
            for neighbor in sorted(graph.neighbors(current))
            if neighbor != previous
        ]
        if not candidates:
            break
        next_node = candidates[0]
        edge = _edge_id(current, next_node)
        if edge in visited_edges and next_node != start:
            break
        visited_edges.add(edge)
        if next_node != start:
            path.append(next_node)
        previous, current = current, next_node
    return path


def _prior_neighbors(skel: np.ndarray, x: int, y: int) -> list[Point]:
    neighbors: list[Point] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx_ = x + dx
            ny_ = y + dy
            if (ny_, nx_) >= (y, x):
                continue
            if 0 <= ny_ < skel.shape[0] and 0 <= nx_ < skel.shape[1] and skel[ny_, nx_]:
                neighbors.append((nx_, ny_))
    return neighbors


def _edge_id(a: Point, b: Point) -> frozenset[Point]:
    return frozenset((a, b))


def _drop_internal_node_artifacts(
    polylines: list[Polyline], node_points: set[Point]
) -> list[Polyline]:
    filtered: list[Polyline] = []
    for polyline in polylines:
        if (
            not polyline.closed
            and len(polyline.points) <= 2
            and polyline.points[0] in node_points
            and polyline.points[-1] in node_points
        ):
            continue
        filtered.append(polyline)
    return filtered
