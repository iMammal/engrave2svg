from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .graph_trace import save_trace_debug, simplify_polylines, trace_skeleton
from .preprocessing import (
    PreprocessParams,
    load_image,
    preprocess_image,
    save_preprocess_debug,
)
from .skeleton import analyze_skeleton, save_skeleton_debug, skeletonize_binary
from .svg_export import export_svg


@dataclass(frozen=True)
class PipelineConfig:
    preprocess: PreprocessParams
    simplification_epsilon: float = 1.25
    stroke_width: float = 1.2

    def to_flat_dict(self) -> dict[str, object]:
        values = asdict(self.preprocess)
        values["simplification_epsilon"] = self.simplification_epsilon
        values["stroke_width"] = self.stroke_width
        return values


@dataclass(frozen=True)
class PipelineMetrics:
    output_svg: str
    debug_dir: str
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    foreground_pixels: int
    skeleton_pixels: int
    graph_nodes: int
    graph_edges: int
    paths: int
    endpoints: int
    junctions: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    debug_dir: str | Path | None,
    config: PipelineConfig,
) -> PipelineMetrics:
    image = load_image(input_path)
    preprocessed = preprocess_image(image, config.preprocess)
    skeleton = skeletonize_binary(preprocessed.cleaned)
    analysis = analyze_skeleton(skeleton)
    trace = trace_skeleton(analysis.skeleton)
    polylines = simplify_polylines(trace.polylines, config.simplification_epsilon)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = analysis.skeleton.shape
    export_svg(
        polylines,
        output,
        width=width,
        height=height,
        stroke_width=config.stroke_width,
    )

    debug = ""
    if debug_dir:
        debug_path = Path(debug_dir)
        debug = str(debug_path)
        save_preprocess_debug(preprocessed, debug_path)
        save_skeleton_debug(analysis, debug_path)
        save_trace_debug(analysis.skeleton, polylines, debug_path)
        cv2.imwrite(str(debug_path / "10_final_skeleton.png"), analysis.skeleton)

    return PipelineMetrics(
        output_svg=str(output),
        debug_dir=debug,
        crop_x=preprocessed.crop_box.x,
        crop_y=preprocessed.crop_box.y,
        crop_width=preprocessed.crop_box.width,
        crop_height=preprocessed.crop_box.height,
        foreground_pixels=int(np.count_nonzero(preprocessed.cleaned)),
        skeleton_pixels=int(np.count_nonzero(analysis.skeleton)),
        graph_nodes=trace.graph.number_of_nodes(),
        graph_edges=trace.graph.number_of_edges(),
        paths=len(polylines),
        endpoints=len(analysis.endpoints),
        junctions=len(analysis.junctions),
    )
