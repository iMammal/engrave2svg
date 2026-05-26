import json
from pathlib import Path

import cv2
import numpy as np

from engrave2svg.gap_bridge import bridge_gaps
from engrave2svg.pipeline import PipelineConfig, run_pipeline
from engrave2svg.preprocessing import PreprocessParams
from engrave2svg.skeleton import analyze_skeleton, skeletonize_binary


def test_radius_zero_leaves_binary_unchanged():
    binary = np.zeros((21, 31), dtype=np.uint8)
    binary[10, 3:12] = 255
    binary[10, 18:28] = 255

    result = bridge_gaps(binary, radius=0.0, angle_tolerance=30.0)

    assert result.bridge_count == 0
    assert np.array_equal(result.bridged, binary)


def test_broken_line_reconnects_conservatively():
    binary = np.zeros((31, 41), dtype=np.uint8)
    binary[15, 5:16] = 255
    binary[15, 22:35] = 255

    result = bridge_gaps(binary, radius=8.0, angle_tolerance=20.0)
    analysis = analyze_skeleton(skeletonize_binary(result.bridged))

    assert result.bridge_count == 1
    assert len(analysis.endpoints) == 2


def test_nearby_parallel_lines_do_not_merge():
    binary = np.zeros((31, 45), dtype=np.uint8)
    binary[10, 5:17] = 255
    binary[13, 19:32] = 255

    result = bridge_gaps(binary, radius=6.0, angle_tolerance=30.0)
    analysis = analyze_skeleton(skeletonize_binary(result.bridged))

    assert result.bridge_count == 0
    assert len(analysis.endpoints) == 4


def test_x_crossing_is_not_distorted():
    binary = np.zeros((41, 41), dtype=np.uint8)
    cv2.line(binary, (8, 8), (32, 32), 255, 1)
    cv2.line(binary, (8, 32), (32, 8), 255, 1)
    before = skeletonize_binary(binary)

    result = bridge_gaps(binary, radius=5.0, angle_tolerance=30.0)
    after = skeletonize_binary(result.bridged)

    assert result.bridge_count == 0
    assert np.array_equal(before, after)


def test_pipeline_records_bridges_in_metrics_json(tmp_path: Path):
    image = np.zeros((31, 41, 3), dtype=np.uint8)
    image[15, 5:16] = (255, 255, 255)
    image[15, 22:35] = (255, 255, 255)
    input_path = tmp_path / "broken.png"
    metrics_path = tmp_path / "metrics.json"
    cv2.imwrite(str(input_path), image)

    metrics = run_pipeline(
        input_path=input_path,
        output_path=tmp_path / "out.svg",
        debug_dir=tmp_path / "debug",
        metrics_path=metrics_path,
        config=PipelineConfig(
            preprocess=PreprocessParams(
                crop="none",
                threshold_mode="global",
                threshold_value=180,
                morph_kernel_size=1,
                min_component_size=1,
                denoise_kernel_size=1,
            ),
            simplification_epsilon=0.0,
            bridge_gaps_radius=8.0,
            bridge_gaps_angle_tolerance=20.0,
        ),
    )
    payload = json.loads(metrics_path.read_text())

    assert metrics.bridge_count == 1
    assert payload["graph_metrics"]["bridge_count"] == 1
    assert len(payload["graph_metrics"]["bridges"]) == 1
    assert (tmp_path / "debug" / "06a_bridged.png").exists()
