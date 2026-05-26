import csv
import json
from pathlib import Path

import cv2
import numpy as np

from engrave2svg.pipeline import PipelineConfig
from engrave2svg.preprocessing import PreprocessParams
from engrave2svg.sensitivity import build_default_trials, run_sensitivity


def test_build_default_trials_records_reproducible_commands(tmp_path: Path):
    config = PipelineConfig(
        preprocess=PreprocessParams(crop="none"),
        simplification_epsilon=1.5,
    )

    trials = build_default_trials(
        config,
        input_path="input.png",
        sensitivity_dir=tmp_path / "sens",
    )

    assert len(trials) == 12
    assert trials[0].trial_id == "trial_0000"
    assert "--threshold-mode global" in trials[0].command
    assert "--simplification-epsilon 0.0" in trials[0].command
    assert trials[1].command_txt.name == "command.txt"


def test_run_sensitivity_writes_one_summary_row_per_trial(tmp_path: Path):
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.line(image, (5, 20), (55, 20), (230, 230, 230), 3)
    input_path = tmp_path / "synthetic.png"
    cv2.imwrite(str(input_path), image)

    summary_path = run_sensitivity(
        input_path=input_path,
        sensitivity_dir=tmp_path / "sensitivity",
        base_config=PipelineConfig(
            preprocess=PreprocessParams(
                crop="none",
                morph_kernel_size=1,
                min_component_size=1,
            )
        ),
        jobs=1,
    )

    rows = list(csv.DictReader(summary_path.open()))

    assert len(rows) == 12
    assert rows[0]["trial_id"] == "trial_0000"
    assert rows[0]["input_path"] == str(input_path)
    assert rows[0]["threshold_mode"] == "global"
    assert Path(rows[0]["output_svg"]).exists()
    params = json.loads(Path(rows[0]["params_json"]).read_text())
    assert params["parameters"]["crop"] == "none"
