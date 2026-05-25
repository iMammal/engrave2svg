from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .graph_trace import save_trace_debug, simplify_polylines, trace_skeleton
from .preprocessing import (
    PreprocessParams,
    load_image,
    preprocess_image,
    save_preprocess_debug,
)
from .skeleton import analyze_skeleton, save_skeleton_debug, skeletonize_binary
from .svg_export import export_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engrave2svg",
        description="Convert a lower-panel bitmap engraving tracing into editable SVG centerlines.",
    )
    parser.add_argument("input", help="Input raster image.")
    parser.add_argument("--output", "-o", required=True, help="Output SVG path.")
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
    parser.add_argument("--simplification-epsilon", type=float, default=1.25)
    parser.add_argument("--stroke-width", type=float, default=1.2)
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
    )

    image = load_image(args.input)
    preprocessed = preprocess_image(image, params)
    skeleton = skeletonize_binary(preprocessed.cleaned)
    analysis = analyze_skeleton(skeleton)
    trace = trace_skeleton(analysis.skeleton)
    polylines = simplify_polylines(trace.polylines, args.simplification_epsilon)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = analysis.skeleton.shape
    export_svg(
        polylines,
        output_path,
        width=width,
        height=height,
        stroke_width=args.stroke_width,
    )

    if args.debug:
        save_preprocess_debug(preprocessed, args.debug)
        save_skeleton_debug(analysis, args.debug)
        save_trace_debug(analysis.skeleton, polylines, args.debug)
        cv2.imwrite(str(Path(args.debug) / "10_final_skeleton.png"), analysis.skeleton)

    print(
        f"Wrote {output_path} with {len(polylines)} paths "
        f"({len(analysis.endpoints)} endpoints, {len(analysis.junctions)} junctions)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
