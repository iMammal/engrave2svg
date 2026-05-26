from __future__ import annotations

import csv
import json
import multiprocessing as mp
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .pipeline import PipelineConfig, PipelineMetrics, run_pipeline
from .preprocessing import PreprocessParams


@dataclass(frozen=True)
class SensitivityTrial:
    trial_id: str
    input_path: Path
    config: PipelineConfig
    output_svg: Path
    debug_dir: Path
    params_json: Path
    command_txt: Path
    command: str


def build_default_trials(
    base_config: PipelineConfig,
    input_path: str | Path,
    sensitivity_dir: str | Path,
    launcher: str = "python engrave2svg.py",
) -> list[SensitivityTrial]:
    root = Path(sensitivity_dir)
    trials_root = root / "trials"
    variants = _default_parameter_variants(base_config.preprocess)
    trials: list[SensitivityTrial] = []

    for index, preprocess in enumerate(variants):
        trial_id = f"trial_{index:04d}"
        trial_dir = trials_root / trial_id
        output_svg = trial_dir / "output.svg"
        debug_dir = trial_dir / "debug"
        config = PipelineConfig(
            preprocess=preprocess,
            simplification_epsilon=_trial_epsilon(base_config, index),
            stroke_width=base_config.stroke_width,
        )
        command = build_trial_command(
            launcher=launcher,
            input_path=input_path,
            output_svg=output_svg,
            debug_dir=debug_dir,
            config=config,
        )
        trials.append(
            SensitivityTrial(
                trial_id=trial_id,
                input_path=Path(input_path),
                config=config,
                output_svg=output_svg,
                debug_dir=debug_dir,
                params_json=trial_dir / "params.json",
                command_txt=trial_dir / "command.txt",
                command=command,
            )
        )
    return trials


def run_sensitivity(
    input_path: str | Path,
    sensitivity_dir: str | Path,
    base_config: PipelineConfig,
    jobs: int,
) -> Path:
    root = Path(sensitivity_dir)
    root.mkdir(parents=True, exist_ok=True)
    trials = build_default_trials(base_config, input_path, root)
    _write_trial_files(trials)
    _write_manifest(root / "sensitivity_manifest.csv", trials)

    if jobs <= 1:
        rows = [_run_trial(trial) for trial in trials]
    else:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            rows = pool.map(_run_trial, trials)

    summary_path = root / "sensitivity_summary.csv"
    _write_summary(summary_path, rows)
    return summary_path


def build_trial_command(
    launcher: str,
    input_path: str | Path,
    output_svg: str | Path,
    debug_dir: str | Path,
    config: PipelineConfig,
) -> str:
    values = config.to_flat_dict()
    args = [
        *shlex.split(launcher),
        str(input_path),
        "--output",
        str(output_svg),
        "--debug",
        str(debug_dir),
        "--crop",
        str(values["crop"]),
        "--threshold-mode",
        str(values["threshold_mode"]),
        "--threshold-value",
        str(values["threshold_value"]),
        "--adaptive-block-size",
        str(values["adaptive_block_size"]),
        "--adaptive-c",
        str(values["adaptive_c"]),
        "--morph-kernel-size",
        str(values["morph_kernel_size"]),
        "--min-component-size",
        str(values["min_component_size"]),
        "--denoise-kernel-size",
        str(values["denoise_kernel_size"]),
        "--clahe-clip-limit",
        str(values["clahe_clip_limit"]),
        "--auto-crop-padding",
        str(values["auto_crop_padding"]),
        "--simplification-epsilon",
        str(values["simplification_epsilon"]),
        "--stroke-width",
        str(values["stroke_width"]),
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def _run_trial(trial: SensitivityTrial) -> dict[str, object]:
    metrics = run_pipeline(
        input_path=trial.input_path,
        output_path=trial.output_svg,
        debug_dir=trial.debug_dir,
        config=trial.config,
    )
    return _trial_row(trial, metrics)


def _write_trial_files(trials: Iterable[SensitivityTrial]) -> None:
    for trial in trials:
        trial.params_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trial_id": trial.trial_id,
            "command": trial.command,
            "input_path": str(trial.input_path),
            "parameters": trial.config.to_flat_dict(),
            "output_svg": str(trial.output_svg),
            "debug_dir": str(trial.debug_dir),
        }
        trial.params_json.write_text(json.dumps(payload, indent=2) + "\n")
        trial.command_txt.write_text(trial.command + "\n")


def _write_manifest(path: Path, trials: list[SensitivityTrial]) -> None:
    rows = [
        {
            "trial_id": trial.trial_id,
            "input_path": str(trial.input_path),
            "params_json": str(trial.params_json),
            "command_txt": str(trial.command_txt),
            "output_svg": str(trial.output_svg),
            "debug_dir": str(trial.debug_dir),
            "command": trial.command,
        }
        for trial in trials
    ]
    _write_rows(path, rows)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    _write_rows(path, sorted(rows, key=lambda row: str(row["trial_id"])))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _trial_row(
    trial: SensitivityTrial, metrics: PipelineMetrics
) -> dict[str, object]:
    row: dict[str, object] = {
        "trial_id": trial.trial_id,
        "input_path": str(trial.input_path),
        "params_json": str(trial.params_json),
        "command_txt": str(trial.command_txt),
        "command": trial.command,
    }
    row.update(trial.config.to_flat_dict())
    row.update(metrics.to_dict())
    return row


def _default_parameter_variants(base: PreprocessParams) -> list[PreprocessParams]:
    variants: list[PreprocessParams] = []
    for value in (96, 128, 160):
        variants.append(_replace(base, threshold_mode="global", threshold_value=value))
    for block_size in (25, 35, 51):
        variants.append(
            _replace(base, threshold_mode="adaptive", adaptive_block_size=block_size)
        )
    for kernel_size in (1, 3, 5):
        variants.append(
            _replace(base, threshold_mode="otsu", morph_kernel_size=kernel_size)
        )
    for min_size in (1, 12, 32):
        variants.append(_replace(base, min_component_size=min_size))
    return variants


def _trial_epsilon(base_config: PipelineConfig, index: int) -> float:
    epsilons = (
        0.0,
        base_config.simplification_epsilon,
        base_config.simplification_epsilon * 2,
    )
    return float(epsilons[index % len(epsilons)])


def _replace(base: PreprocessParams, **overrides: object) -> PreprocessParams:
    values = {
        "crop": base.crop,
        "threshold_mode": base.threshold_mode,
        "threshold_value": base.threshold_value,
        "adaptive_block_size": base.adaptive_block_size,
        "adaptive_c": base.adaptive_c,
        "morph_kernel_size": base.morph_kernel_size,
        "min_component_size": base.min_component_size,
        "denoise_kernel_size": base.denoise_kernel_size,
        "clahe_clip_limit": base.clahe_clip_limit,
        "auto_crop_padding": base.auto_crop_padding,
    }
    values.update(overrides)
    return PreprocessParams(**values)
