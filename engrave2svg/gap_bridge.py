from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .graph_trace import Point, skeleton_to_graph
from .skeleton import analyze_skeleton, skeletonize_binary


@dataclass(frozen=True)
class BridgeRecord:
    source: Point
    target: Point
    distance_px: float
    angle_difference_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "source": {"x": self.source[0], "y": self.source[1]},
            "target": {"x": self.target[0], "y": self.target[1]},
            "distance_px": self.distance_px,
            "angle_difference_deg": self.angle_difference_deg,
        }


@dataclass(frozen=True)
class GapBridgeResult:
    bridged: np.ndarray
    preliminary_skeleton: np.ndarray
    bridges: list[BridgeRecord]

    @property
    def bridge_count(self) -> int:
        return len(self.bridges)

    def bridges_as_dicts(self) -> list[dict[str, object]]:
        return [bridge.to_dict() for bridge in self.bridges]


def bridge_gaps(
    binary: np.ndarray,
    radius: float,
    angle_tolerance: float,
) -> GapBridgeResult:
    preliminary = skeletonize_binary(binary)
    if radius <= 0:
        return GapBridgeResult(
            bridged=binary.copy(),
            preliminary_skeleton=preliminary,
            bridges=[],
        )

    analysis = analyze_skeleton(preliminary)
    graph = skeleton_to_graph(preliminary)
    directions = {
        endpoint: direction
        for endpoint in analysis.endpoints
        if (direction := _endpoint_outward_direction(graph, endpoint)) is not None
    }
    candidates = _bridge_candidates(
        endpoints=analysis.endpoints,
        directions=directions,
        binary=binary,
        radius=radius,
        angle_tolerance=angle_tolerance,
    )
    selected = _mutual_nearest_bridges(candidates)

    bridged = binary.copy()
    for bridge in selected:
        cv2.line(bridged, bridge.source, bridge.target, 255, 1, cv2.LINE_8)

    return GapBridgeResult(
        bridged=bridged,
        preliminary_skeleton=preliminary,
        bridges=selected,
    )


def save_gap_bridge_debug(
    result: GapBridgeResult,
    debug_dir: str | Path,
) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(result.bridged, cv2.COLOR_GRAY2BGR)
    for bridge in result.bridges:
        cv2.line(canvas, bridge.source, bridge.target, (0, 165, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, bridge.source, 2, (0, 255, 0), -1)
        cv2.circle(canvas, bridge.target, 2, (0, 255, 0), -1)
    cv2.imwrite(str(path / "06a_bridged.png"), canvas)


def _bridge_candidates(
    endpoints: list[Point],
    directions: dict[Point, tuple[float, float]],
    binary: np.ndarray,
    radius: float,
    angle_tolerance: float,
) -> list[BridgeRecord]:
    candidates: list[BridgeRecord] = []
    for index, source in enumerate(sorted(endpoints, key=_point_sort_key)):
        if source not in directions:
            continue
        for target in sorted(endpoints, key=_point_sort_key)[index + 1 :]:
            if target not in directions:
                continue
            distance = math.dist(source, target)
            if distance <= 0 or distance > radius:
                continue
            gap_vector = _normalize((target[0] - source[0], target[1] - source[1]))
            if gap_vector is None:
                continue
            source_angle = _angle_between(directions[source], gap_vector)
            target_angle = _angle_between(directions[target], (-gap_vector[0], -gap_vector[1]))
            axial_difference = _axial_angle_between(directions[source], directions[target])
            if (
                source_angle <= angle_tolerance
                and target_angle <= angle_tolerance
                and axial_difference <= angle_tolerance
                and _gap_is_clear(binary, source, target)
            ):
                candidates.append(
                    BridgeRecord(
                        source=source,
                        target=target,
                        distance_px=float(distance),
                        angle_difference_deg=float(max(source_angle, target_angle, axial_difference)),
                    )
                )
    return candidates


def _mutual_nearest_bridges(candidates: list[BridgeRecord]) -> list[BridgeRecord]:
    best_by_endpoint: dict[Point, BridgeRecord] = {}
    for candidate in sorted(candidates, key=lambda bridge: (bridge.distance_px, bridge.angle_difference_deg)):
        for endpoint in (candidate.source, candidate.target):
            current = best_by_endpoint.get(endpoint)
            if current is None or (
                candidate.distance_px,
                candidate.angle_difference_deg,
            ) < (
                current.distance_px,
                current.angle_difference_deg,
            ):
                best_by_endpoint[endpoint] = candidate

    selected: list[BridgeRecord] = []
    used: set[Point] = set()
    for candidate in sorted(candidates, key=lambda bridge: (bridge.distance_px, bridge.angle_difference_deg)):
        if candidate.source in used or candidate.target in used:
            continue
        if best_by_endpoint.get(candidate.source) is candidate and best_by_endpoint.get(candidate.target) is candidate:
            selected.append(candidate)
            used.add(candidate.source)
            used.add(candidate.target)
    return selected


def _endpoint_outward_direction(
    graph,
    endpoint: Point,
    lookahead: int = 8,
) -> tuple[float, float] | None:
    neighbors = sorted(graph.neighbors(endpoint))
    if len(neighbors) != 1:
        return None

    previous = endpoint
    current = neighbors[0]
    interior = current
    for _ in range(max(1, lookahead - 1)):
        choices = [neighbor for neighbor in graph.neighbors(current) if neighbor != previous]
        if len(choices) != 1:
            break
        previous, current = current, choices[0]
        interior = current

    return _normalize((endpoint[0] - interior[0], endpoint[1] - interior[1]))


def _gap_is_clear(binary: np.ndarray, source: Point, target: Point) -> bool:
    points = _line_points(source, target)
    interior = points[2:-2] if len(points) > 4 else points[1:-1]
    if not interior:
        return True
    hit_count = sum(1 for x, y in interior if binary[y, x] > 0)
    return hit_count == 0


def _line_points(source: Point, target: Point) -> list[Point]:
    canvas = np.zeros(
        (max(source[1], target[1]) + 3, max(source[0], target[0]) + 3),
        dtype=np.uint8,
    )
    cv2.line(canvas, source, target, 255, 1, cv2.LINE_8)
    ys, xs = np.where(canvas > 0)
    return sorted(
        [(int(x), int(y)) for x, y in zip(xs, ys)],
        key=lambda point: math.dist(source, point),
    )


def _normalize(vector: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(vector[0], vector[1])
    if length == 0:
        return None
    return (float(vector[0] / length), float(vector[1] / length))


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return float(math.degrees(math.acos(dot)))


def _axial_angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    angle = _angle_between(a, b)
    return min(angle, abs(180.0 - angle))


def _point_sort_key(point: Point) -> tuple[int, int]:
    return (point[1], point[0])
