import numpy as np

from engrave2svg.graph_trace import simplify_polylines, trace_skeleton
from engrave2svg.skeleton import analyze_skeleton


def test_traces_single_straight_line():
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    skeleton[10, 3:17] = 255

    analysis = analyze_skeleton(skeleton)
    trace = trace_skeleton(skeleton)

    assert len(analysis.endpoints) == 2
    assert len(analysis.junctions) == 0
    assert len(trace.polylines) == 1
    assert trace.polylines[0].points[0] in {(3, 10), (16, 10)}
    assert trace.polylines[0].points[-1] in {(3, 10), (16, 10)}


def test_splits_paths_at_t_junction():
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    skeleton[10, 3:17] = 255
    skeleton[4:11, 10] = 255

    analysis = analyze_skeleton(skeleton)
    trace = trace_skeleton(skeleton)

    assert len(analysis.endpoints) == 3
    assert len(analysis.junctions) >= 1
    assert len(trace.polylines) >= 3


def test_simplifies_polyline_without_dropping_endpoints():
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    for x in range(2, 18):
        y = 10 + (x % 2)
        skeleton[y, x] = 255

    trace = trace_skeleton(skeleton)
    simplified = simplify_polylines(trace.polylines, epsilon=2.0)

    assert len(simplified) == 1
    assert simplified[0].points[0] == trace.polylines[0].points[0]
    assert simplified[0].points[-1] == trace.polylines[0].points[-1]
