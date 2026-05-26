# engrave2svg

`engrave2svg` converts a white or gray bitmap line tracing on a dark background into editable SVG centerlines. It is designed for archaeological lower-panel line drawings where preserving stroke topology matters more than producing decorative smoothing.

## Install

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python engrave2svg.py --help
```

The default pipeline is CPU-only and uses OpenCV, scikit-image, numpy, svgwrite, and networkx.

## CLI

```bash
python engrave2svg.py input.png --output output.svg --crop auto --debug debug/
```

Installed console script usage is also supported:

```bash
engrave2svg input.png --output output.svg --crop auto --debug debug/
```

Crop options:

- `--crop auto`: searches the lower half of the image for bright tracing pixels.
- `--crop none`: treats the input as already cropped.
- `--crop x,y,width,height`: uses a manual crop box.

Useful tuning parameters:

```bash
--threshold-mode otsu|global|adaptive
--threshold-value 128
--adaptive-block-size 35
--adaptive-c -5
--morph-kernel-size 3
--min-component-size 12
--clahe-clip-limit 2.0
--auto-crop-padding 8
--simplification-epsilon 1.25
```

Lower `--morph-kernel-size` and `--simplification-epsilon` when fine topology is being lost. Raise `--min-component-size` when isolated noise becomes exported paths.

## Pipeline

1. Load image.
2. Crop the lower panel automatically, manually, or not at all.
3. Convert to grayscale.
4. Apply light median denoising.
5. Normalize local contrast with CLAHE.
6. Threshold foreground tracing strokes.
7. Morphologically clean and remove small components.
8. Skeletonize strokes to one-pixel centerlines.
9. Detect endpoints and junctions by 8-neighborhood degree.
10. Convert skeleton pixels to a graph.
11. Trace graph edges into polylines split at junctions.
12. Simplify with Douglas-Peucker.
13. Export editable SVG polylines.
14. Save stage-by-stage diagnostic PNGs.

Debug output names are `01_cropped.png` through `10_final_skeleton.png`.

## Sensitivity Runs

Use `--sensitivity` to run a deterministic local parameter sweep. The sweep is still CPU-only; `--jobs` controls local multiprocessing only.

```bash
python engrave2svg.py input.png \
  --sensitivity \
  --sensitivity-dir sensitivity \
  --jobs 4
```

This writes one row per trial to `sensitivity/sensitivity_summary.csv`. Every row records the full parameter set, input path, output SVG path, debug directory, measured tracing counts, and the standalone command used for that trial.

Each trial also gets its own directory:

```text
sensitivity/trials/trial_0000/params.json
sensitivity/trials/trial_0000/command.txt
sensitivity/trials/trial_0000/output.svg
sensitivity/trials/trial_0000/debug/*.png
```

The `command.txt` files are independent single-image commands so the same trial layout can later be dispatched as SLURM array jobs. See `docs/hpc_delta_slurm.md` for a CPU-only NCSA Delta example. This branch does not add GPU or HPC acceleration.

## Example

```bash
bash scripts/run_example.sh
```

To force a specific interpreter, use `PYTHON=/path/to/python bash scripts/run_example.sh`.

This creates:

- `examples/synthetic_panel.png`
- `examples/synthetic_output.svg`
- `examples/debug/*.png`

An example SVG is already included at `examples/synthetic_output.svg`.

## Tests

```bash
python -m pytest
```

The tests use synthetic line drawings to check crop behavior, skeletonization, junction splitting, simplification, and SVG export.

## Failure Modes

- Broken strokes: thresholding or morphology can split faint engraved lines. Try adaptive thresholding, lower the global threshold, reduce denoising, or use a smaller morphology kernel.
- False joins: close parallel strokes can merge during thresholding or closing. Reduce `--morph-kernel-size` and inspect `05_thresholded.png` and `06_cleaned.png`.
- Noisy intersections: anti-aliased crossings often create clusters of junction pixels rather than one clean node. The tracer preserves topology but may emit several short paths around the intersection.
- Anti-aliasing artifacts: pale edge pixels can become small side branches after skeletonization. Increase `--min-component-size`, use a slightly higher threshold, or crop more tightly.
- Over-simplification: Douglas-Peucker can move bends away from the original centerline. Lower `--simplification-epsilon` or set it to `0` for raw traced paths.
- Auto-crop misses the panel: use `--crop none` for pre-cropped images or pass `--crop x,y,width,height`.

## Notes for Field Use

The exported SVG uses one `<polyline>` per traced segment with stable IDs like `path-0000`. Coordinates are relative to the cropped panel, not the original full image. Keep the debug directory with the SVG when recording provenance; it captures the exact intermediate stages that led to the vector result.
