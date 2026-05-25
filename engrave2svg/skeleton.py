from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class SkeletonAnalysis:
    skeleton: np.ndarray
    endpoints: list[tuple[int, int]]
    junctions: list[tuple[int, int]]
    degree_map: np.ndarray


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    skeleton_bool = skeletonize(binary > 0)
    return (skeleton_bool.astype(np.uint8) * 255)


def analyze_skeleton(skeleton: np.ndarray) -> SkeletonAnalysis:
    skel = skeleton > 0
    degree_map = np.zeros(skeleton.shape, dtype=np.uint8)
    endpoints: list[tuple[int, int]] = []
    junctions: list[tuple[int, int]] = []

    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        degree = _neighbor_count(skel, y, x)
        degree_map[y, x] = degree
        point = (int(x), int(y))
        if degree == 1:
            endpoints.append(point)
        elif degree >= 3:
            junctions.append(point)

    return SkeletonAnalysis(
        skeleton=skeleton,
        endpoints=endpoints,
        junctions=junctions,
        degree_map=degree_map,
    )


def save_skeleton_debug(analysis: SkeletonAnalysis, debug_dir: str | Path) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / "07_skeleton.png"), analysis.skeleton)

    overlay = cv2.cvtColor(analysis.skeleton, cv2.COLOR_GRAY2BGR)
    for x, y in analysis.endpoints:
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
    for x, y in analysis.junctions:
        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(str(path / "08_nodes.png"), overlay)


def _neighbor_count(skel: np.ndarray, y: int, x: int) -> int:
    y0 = max(0, y - 1)
    y1 = min(skel.shape[0], y + 2)
    x0 = max(0, x - 1)
    x1 = min(skel.shape[1], x + 2)
    return int(np.count_nonzero(skel[y0:y1, x0:x1]) - 1)
