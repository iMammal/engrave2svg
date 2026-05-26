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
    metrics_json: Path
    graph_path: Path
    nodes_csv: Path
    edges_csv: Path
    orientation_hist: Path
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
        metrics_json = trial_dir / "metrics.json"
        graph_path = trial_dir / "graph.graphml"
        nodes_csv = trial_dir / "nodes.csv"
        edges_csv = trial_dir / "edges.csv"
        orientation_hist = trial_dir / "orientation_histogram.png"
        config = PipelineConfig(
            preprocess=preprocess,
            simplification_epsilon=_trial_epsilon(base_config, index),
            stroke_width=base_config.stroke_width,
            node_merge_radius=_trial_node_merge_radius(base_config, index),
            bridge_gaps_radius=base_config.bridge_gaps_radius,
            bridge_gaps_angle_tolerance=base_config.bridge_gaps_angle_tolerance,
        )
        command = build_trial_command(
            launcher=launcher,
            input_path=input_path,
            output_svg=output_svg,
            debug_dir=debug_dir,
            metrics_json=metrics_json,
            graph_path=graph_path,
            nodes_csv=nodes_csv,
            edges_csv=edges_csv,
            orientation_hist=orientation_hist,
            config=config,
        )
        trials.append(
            SensitivityTrial(
                trial_id=trial_id,
                input_path=Path(input_path),
                config=config,
                output_svg=output_svg,
                debug_dir=debug_dir,
                metrics_json=metrics_json,
                graph_path=graph_path,
                nodes_csv=nodes_csv,
                edges_csv=edges_csv,
                orientation_hist=orientation_hist,
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
    _write_summary_json(root / "sensitivity_summary.json", rows)
    return summary_path


def build_trial_command(
    launcher: str,
    input_path: str | Path,
    output_svg: str | Path,
    debug_dir: str | Path,
    metrics_json: str | Path,
    graph_path: str | Path,
    nodes_csv: str | Path,
    edges_csv: str | Path,
    orientation_hist: str | Path,
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
        "--metrics",
        str(metrics_json),
        "--graph",
        str(graph_path),
        "--nodes-csv",
        str(nodes_csv),
        "--edges-csv",
        str(edges_csv),
        "--orientation-hist",
        str(orientation_hist),
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
        "--node-merge-radius",
        str(values["node_merge_radius"]),
        "--bridge-gaps-radius",
        str(values["bridge_gaps_radius"]),
        "--bridge-gaps-angle-tolerance",
        str(values["bridge_gaps_angle_tolerance"]),
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def _run_trial(trial: SensitivityTrial) -> dict[str, object]:
    metrics = run_pipeline(
        input_path=trial.input_path,
        output_path=trial.output_svg,
        debug_dir=trial.debug_dir,
        config=trial.config,
        metrics_path=trial.metrics_json,
        graph_path=trial.graph_path,
        nodes_csv_path=trial.nodes_csv,
        edges_csv_path=trial.edges_csv,
        orientation_hist_path=trial.orientation_hist,
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
            "metrics_json": str(trial.metrics_json),
            "graph_path": str(trial.graph_path),
            "nodes_csv": str(trial.nodes_csv),
            "edges_csv": str(trial.edges_csv),
            "orientation_hist": str(trial.orientation_hist),
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
            "metrics_json": str(trial.metrics_json),
            "graph_path": str(trial.graph_path),
            "nodes_csv": str(trial.nodes_csv),
            "edges_csv": str(trial.edges_csv),
            "orientation_hist": str(trial.orientation_hist),
            "command": trial.command,
        }
        for trial in trials
    ]
    _write_rows(path, rows)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    _write_rows(path, sorted(rows, key=lambda row: str(row["trial_id"])))


def _write_summary_json(path: Path, rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: str(row["trial_id"]))
    payload = {
        "trials": ordered,
        "stability": {
            "raw_node_count": _numeric_stability(ordered, "raw_node_count"),
            "merged_node_count": _numeric_stability(ordered, "merged_node_count"),
            "merged_edge_count": _numeric_stability(ordered, "merged_edge_count"),
            "connected_components": _numeric_stability(ordered, "connected_components"),
            "total_traced_length_px": _numeric_stability(ordered, "total_traced_length_px"),
            "bridge_count": _numeric_stability(ordered, "bridge_count"),
            "dominant_angle_peaks": _peak_stability(ordered),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


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
    threshold_values = _unique_ints(
        max(0, base.threshold_value - 20),
        base.threshold_value,
        min(255, base.threshold_value + 20),
    )
    for value in threshold_values:
        variants.append(_replace(base, threshold_mode="global", threshold_value=value))
    for block_size in _unique_odd_ints(
        _odd_at_least(base.adaptive_block_size - 10, 3),
        _odd_at_least(base.adaptive_block_size, 3),
        _odd_at_least(base.adaptive_block_size + 10, 3),
    ):
        variants.append(
            _replace(base, threshold_mode="adaptive", adaptive_block_size=block_size)
        )
    for kernel_size in _unique_odd_ints(
        _odd_at_least(base.morph_kernel_size - 2, 1),
        _odd_at_least(base.morph_kernel_size, 1),
        _odd_at_least(base.morph_kernel_size + 2, 1),
    ):
        variants.append(
            _replace(base, threshold_mode="otsu", morph_kernel_size=kernel_size)
        )
    for min_size in _unique_ints(
        max(1, base.min_component_size // 2),
        max(1, base.min_component_size),
        max(1, base.min_component_size * 2),
    ):
        variants.append(_replace(base, min_component_size=min_size))
    return variants


def _trial_epsilon(base_config: PipelineConfig, index: int) -> float:
    epsilons = (
        0.0,
        base_config.simplification_epsilon,
        base_config.simplification_epsilon * 2,
    )
    return float(epsilons[index % len(epsilons)])


def _trial_node_merge_radius(base_config: PipelineConfig, index: int) -> float:
    radii = (
        base_config.node_merge_radius,
        base_config.node_merge_radius + 1.0,
        max(0.0, base_config.node_merge_radius - 1.0),
    )
    return float(radii[index % len(radii)])


def _numeric_stability(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "stddev": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "stddev": variance ** 0.5,
    }


def _peak_stability(rows: list[dict[str, object]]) -> dict[str, object]:
    rounded_peak_counts: dict[str, int] = {}
    for row in rows:
        try:
            peaks = json.loads(str(row.get("dominant_angle_peaks", "[]")))
        except json.JSONDecodeError:
            peaks = []
        rounded = tuple(
            round(float(peak["angle_center_deg"])) for peak in peaks[:4]
        )
        key = ",".join(str(value) for value in rounded)
        rounded_peak_counts[key] = rounded_peak_counts.get(key, 0) + 1
    return {
        "unique_peak_sets": len(rounded_peak_counts),
        "peak_set_frequencies": rounded_peak_counts,
    }


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


def _unique_ints(*values: int) -> tuple[int, ...]:
    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    while unique and len(unique) < len(values):
        unique.append(unique[-1] + 1)
    return tuple(unique)


def _unique_odd_ints(*values: int) -> tuple[int, ...]:
    unique = list(_unique_ints(*values))
    index = 0
    while index < len(unique):
        unique[index] = _odd_at_least(unique[index], 1)
        if unique.count(unique[index]) > 1:
            unique[index] = _odd_at_least(unique[index] + 2, 1)
            index = -1
        index += 1
    return tuple(unique)


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), minimum)
    return value if value % 2 else value + 1
