import json
from pathlib import Path

import cv2
import numpy as np

from engrave2svg.pipeline import PipelineConfig, run_pipeline
from engrave2svg.preprocessing import PreprocessParams
from engrave2svg.ridge_extraction import RidgeParams


def test_ridge_mode_reconnects_faint_interrupted_stroke_better_than_threshold(tmp_path: Path):
    image = np.zeros((60, 90, 3), dtype=np.uint8)
    cv2.line(image, (8, 30), (32, 30), (220, 220, 220), 3)
    cv2.line(image, (33, 30), (55, 30), (85, 85, 85), 3)
    cv2.line(image, (56, 30), (80, 30), (220, 220, 220), 3)
    input_path = tmp_path / "faint_line.png"
    cv2.imwrite(str(input_path), image)

    preprocess = PreprocessParams(
        crop="none",
        threshold_mode="global",
        threshold_value=180,
        morph_kernel_size=1,
        min_component_size=5,
        denoise_kernel_size=1,
        clahe_clip_limit=1.0,
    )
    threshold_metrics = run_pipeline(
        input_path=input_path,
        output_path=tmp_path / "threshold.svg",
        debug_dir=tmp_path / "debug_threshold",
        config=PipelineConfig(
            preprocess=preprocess,
            extraction_mode="threshold",
            simplification_epsilon=0.0,
        ),
    )
    ridge_metrics_path = tmp_path / "ridge_metrics.json"
    ridge_metrics = run_pipeline(
        input_path=input_path,
        output_path=tmp_path / "ridge.svg",
        debug_dir=tmp_path / "debug_ridge",
        metrics_path=ridge_metrics_path,
        config=PipelineConfig(
            preprocess=preprocess,
            extraction_mode="ridge",
            simplification_epsilon=0.0,
            ridge=RidgeParams(sigmas=(1.0, 2.0), beta=0.5, gamma=15.0, threshold=0.05),
        ),
    )
    payload = json.loads(ridge_metrics_path.read_text())

    assert threshold_metrics.connected_components == 2
    assert ridge_metrics.connected_components == 1
    assert ridge_metrics.total_traced_length_px > threshold_metrics.total_traced_length_px
    assert payload["graph_metrics"]["extraction_mode"] == "ridge"
    assert payload["graph_metrics"]["ridge_sigmas"] == "1,2"
    assert (tmp_path / "debug_ridge" / "06b_ridge_response.png").exists()
    assert (tmp_path / "debug_ridge" / "06c_ridge_threshold_mask.png").exists()
    assert (tmp_path / "debug_ridge" / "06d_ridge_skeleton.png").exists()
    assert (tmp_path / "debug_ridge" / "06e_ridge_overlay.png").exists()


def test_ridge_mode_does_not_collapse_nearby_parallel_lines(tmp_path: Path):
    image = np.zeros((70, 90, 3), dtype=np.uint8)
    cv2.line(image, (8, 25), (80, 25), (180, 180, 180), 3)
    cv2.line(image, (8, 35), (80, 35), (180, 180, 180), 3)
    input_path = tmp_path / "parallel.png"
    cv2.imwrite(str(input_path), image)

    metrics = run_pipeline(
        input_path=input_path,
        output_path=tmp_path / "ridge.svg",
        debug_dir=tmp_path / "debug",
        config=PipelineConfig(
            preprocess=PreprocessParams(
                crop="none",
                threshold_mode="global",
                threshold_value=220,
                morph_kernel_size=1,
                min_component_size=5,
                denoise_kernel_size=1,
                clahe_clip_limit=1.0,
            ),
            extraction_mode="ridge",
            simplification_epsilon=0.0,
            ridge=RidgeParams(sigmas=(1.0, 2.0), beta=0.5, gamma=15.0, threshold=0.05),
        ),
    )

    assert metrics.connected_components == 2
    assert metrics.merged_junction_count == 0
    assert metrics.merged_endpoint_count == 4
