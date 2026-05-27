import cv2
import numpy as np

from engrave2svg.graph_metrics import build_engraving_graphs
from engrave2svg.graph_trace import trace_skeleton


def _bundle(skeleton: np.ndarray):
    trace = trace_skeleton(skeleton)
    return build_engraving_graphs(trace, trace.polylines, node_merge_radius=0.0)


def test_crossing_x_preserves_shared_junction():
    skeleton = np.zeros((41, 41), dtype=np.uint8)
    cv2.line(skeleton, (8, 8), (32, 32), 255, 1)
    cv2.line(skeleton, (8, 32), (32, 8), 255, 1)

    bundle = _bundle(skeleton)
    stats = bundle.metrics["merged"]

    assert stats["connected_components"] == 1
    assert stats["endpoints"] == 4
    assert stats["junctions"] == 1
    assert stats["degree_distribution"]["4"] == 1
    assert bundle.metrics["node_degree_histogram"]["4"] == 1
    assert bundle.metrics["connected_component_size_histogram"]["5"] == 1
    assert bundle.metrics["largest_connected_component_fraction"] == 1.0


def test_ladder_graph_preserves_rung_junctions():
    skeleton = np.zeros((45, 55), dtype=np.uint8)
    skeleton[8:37, 12] = 255
    skeleton[8:37, 42] = 255
    for y in (14, 22, 30):
        skeleton[y, 12:43] = 255

    bundle = _bundle(skeleton)
    stats = bundle.metrics["merged"]

    assert stats["connected_components"] == 1
    assert stats["endpoints"] == 4
    assert stats["junctions"] == 6
    assert stats["largest_connected_component_fraction"] == 1.0


def test_connected_zigzag_network_is_one_component():
    skeleton = np.zeros((50, 70), dtype=np.uint8)
    points = np.array([(8, 35), (18, 15), (30, 35), (42, 15), (55, 35)], dtype=np.int32)
    cv2.polylines(skeleton, [points], False, 255, 1)
    skeleton[15:42, 30] = 255
    skeleton[27, 18:43] = 255

    bundle = _bundle(skeleton)
    stats = bundle.metrics["merged"]

    assert stats["connected_components"] == 1
    assert stats["junctions"] >= 2
    assert stats["endpoints"] >= 3


def test_triangle_mesh_vertices_are_shared_nodes():
    skeleton = np.zeros((60, 70), dtype=np.uint8)
    top = (35, 8)
    left = (12, 48)
    right = (58, 48)
    mid = (35, 48)
    cv2.line(skeleton, top, left, 255, 1)
    cv2.line(skeleton, top, right, 255, 1)
    cv2.line(skeleton, left, right, 255, 1)
    cv2.line(skeleton, top, mid, 255, 1)

    bundle = _bundle(skeleton)
    stats = bundle.metrics["merged"]

    assert stats["connected_components"] == 1
    assert stats["junctions"] >= 2
    assert int(stats["degree_distribution"].get("3", 0)) >= 2
