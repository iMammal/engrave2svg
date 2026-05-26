from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline
from .preprocessing import PreprocessParams
from .sensitivity import run_sensitivity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engrave2svg",
        description="Convert a lower-panel bitmap engraving tracing into editable SVG centerlines.",
    )
    parser.add_argument("input", help="Input raster image.")
    parser.add_argument("--output", "-o", help="Output SVG path for a single run.")
    parser.add_argument(
        "--crop",
        default="auto",
        help="Crop mode: auto, none, or manual x,y,width,height. Default: auto.",
    )
    parser.add_argument("--debug", help="Directory for diagnostic stage images.")
    parser.add_argument(
        "--threshold-mode",
        choices=("otsu", "global", "adaptive"),
        default="otsu",
    )
    parser.add_argument("--threshold-value", type=int, default=128)
    parser.add_argument("--adaptive-block-size", type=int, default=35)
    parser.add_argument("--adaptive-c", type=int, default=-5)
    parser.add_argument("--morph-kernel-size", type=int, default=3)
    parser.add_argument("--min-component-size", type=int, default=12)
    parser.add_argument("--denoise-kernel-size", type=int, default=3)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--auto-crop-padding", type=int, default=8)
    parser.add_argument("--simplification-epsilon", type=float, default=1.25)
    parser.add_argument("--stroke-width", type=float, default=1.2)
    parser.add_argument(
        "--node-merge-radius",
        type=float,
        default=0.0,
        help="Merge raw endpoint/junction nodes within this pixel radius for final graph metrics. Default: 0.0.",
    )
    parser.add_argument("--metrics", help="Output metrics JSON path.")
    parser.add_argument(
        "--graph",
        help="GraphML or GEXF base path. Writes *_raw and *_merged graph files.",
    )
    parser.add_argument(
        "--nodes-csv",
        help="Nodes CSV base path. Writes *_raw and *_merged CSV files.",
    )
    parser.add_argument(
        "--edges-csv",
        help="Edges CSV base path. Writes *_raw and *_merged CSV files.",
    )
    parser.add_argument(
        "--orientation-hist",
        help="Output PNG path for the length-weighted orientation histogram.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run the default deterministic parameter sensitivity batch.",
    )
    parser.add_argument(
        "--sensitivity-dir",
        default="sensitivity",
        help="Directory for sensitivity trial outputs and CSV summaries.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Local multiprocessing workers for --sensitivity. Default: 1.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = PreprocessParams(
        crop=args.crop,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        morph_kernel_size=args.morph_kernel_size,
        min_component_size=args.min_component_size,
        denoise_kernel_size=args.denoise_kernel_size,
        clahe_clip_limit=args.clahe_clip_limit,
        auto_crop_padding=args.auto_crop_padding,
    )
    config = PipelineConfig(
        preprocess=params,
        simplification_epsilon=args.simplification_epsilon,
        stroke_width=args.stroke_width,
        node_merge_radius=args.node_merge_radius,
    )

    if args.sensitivity:
        summary_path = run_sensitivity(
            input_path=args.input,
            sensitivity_dir=args.sensitivity_dir,
            base_config=config,
            jobs=max(1, args.jobs),
        )
        print(f"Wrote sensitivity summary to {summary_path}.")
        return 0

    if not args.output:
        build_parser().error("--output is required unless --sensitivity is set.")

    metrics = run_pipeline(
        input_path=args.input,
        output_path=Path(args.output),
        debug_dir=args.debug,
        config=config,
        metrics_path=args.metrics,
        graph_path=args.graph,
        nodes_csv_path=args.nodes_csv,
        edges_csv_path=args.edges_csv,
        orientation_hist_path=args.orientation_hist,
    )

    print(
        f"Wrote {metrics.output_svg} with {metrics.paths} paths "
        f"({metrics.merged_endpoint_count} merged endpoints, "
        f"{metrics.merged_junction_count} merged junctions)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
