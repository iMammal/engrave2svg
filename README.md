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
--extraction-mode threshold|ridge
--threshold-mode otsu|global|adaptive
--threshold-value 128
--adaptive-block-size 35
--adaptive-c -5
--morph-kernel-size 3
--min-component-size 12
--clahe-clip-limit 2.0
--auto-crop-padding 8
--simplification-epsilon 1.25
--node-merge-radius 0.0
--bridge-gaps-radius 0.0
--bridge-gaps-angle-tolerance 30.0
--ridge-sigmas 1,2,3
--ridge-beta 0.5
--ridge-gamma 15.0
--ridge-threshold 0.05
```

Lower `--morph-kernel-size` and `--simplification-epsilon` when fine topology is being lost. Raise `--min-component-size` when isolated noise becomes exported paths.
Use `--bridge-gaps-radius` only when faint but visually continuous strokes break into disconnected fragments. Radius `0.0` keeps the historical behavior unchanged. When bridging is enabled, endpoints are connected only when preliminary skeleton endpoint directions face each other, local orientation is compatible, the distance is within radius, and the intervening gap is clear.

`--extraction-mode threshold` is the established default and preserves the existing global, adaptive, and Otsu threshold workflows. `--extraction-mode ridge` is experimental: it applies multi-scale Frangi ridge enhancement to the normalized grayscale image before thresholding, then feeds the ridge-derived skeleton mask into the same skeletonization, graph, metrics, and SVG stages. Ridge mode is intended for noisy archaeological imagery where engraved strokes are visually continuous but too faint for a single intensity threshold. It is conservative by design: response pixels must still have grayscale support, broad ridge masks are thinned before graph extraction, and all ridge parameters are recorded for reproducibility.

Scientific data outputs:

```bash
--metrics output_metrics.json
--graph output.graphml
--nodes-csv output_nodes.csv
--edges-csv output_edges.csv
--orientation-hist output_orientation_histogram.png
```

`--graph`, `--nodes-csv`, and `--edges-csv` write paired audit files with `_raw` and `_merged` suffixes. For example, `--graph output.graphml` writes `output_raw.graphml` and `output_merged.graphml`. The raw graph preserves endpoint and junction pixels before consolidation. The merged graph applies `--node-merge-radius` and reports the radius, raw counts, merged counts, and merge report in the metrics JSON.

## Pipeline

1. Load image.
2. Crop the lower panel automatically, manually, or not at all.
3. Convert to grayscale.
4. Apply light median denoising.
5. Normalize local contrast with CLAHE.
6. Extract foreground strokes with threshold mode, or experimentally enhance bright curvilinear ridges from the normalized grayscale image.
7. Morphologically clean threshold masks, or thin ridge masks conservatively without aggressive region filling.
8. Optionally skeletonize a preliminary mask and bridge conservative endpoint-to-endpoint gaps.
9. Skeletonize the cleaned, ridge-derived, or bridged strokes to one-pixel centerlines.
10. Detect endpoints and junctions by 8-neighborhood degree.
11. Convert skeleton pixels to a graph.
12. Trace graph edges into polylines split at junctions.
13. Simplify with Douglas-Peucker.
14. Build raw and merged engraving graphs for scientific analysis.
15. Compute node degree, connected component, length, and orientation statistics.
16. Export GraphML/GEXF, node CSV, edge CSV, metrics JSON, and orientation histogram when requested.
17. Export editable SVG polylines as a visual artifact.
18. Save stage-by-stage diagnostic PNGs.

Debug output names are `01_cropped.png` through `11_merged_nodes.png`, including `06a_bridged.png` after optional gap bridging. Ridge mode also writes `06b_ridge_response.png`, `06c_ridge_threshold_mask.png`, `06d_ridge_skeleton.png`, and `06e_ridge_overlay.png`. `11_merged_nodes.png` overlays component-colored graph edges, endpoint/junction/connector node classes, and node degrees so graph topology can be checked against the visible skeleton.

The SVG is not the canonical research output. It is useful for inspection and illustration, but the GraphML/GEXF, CSV, and JSON files are the reproducible data products for analysis, review, and downstream statistics.

## Sensitivity Runs

Use `--sensitivity` to run a deterministic local parameter sweep. The sweep is still CPU-only; `--jobs` controls local multiprocessing only.

```bash
python engrave2svg.py input.png \
  --sensitivity \
  --sensitivity-dir sensitivity \
  --extraction-mode threshold \
  --node-merge-radius 2.0 \
  --bridge-gaps-radius 0.0 \
  --jobs 4
```

This writes one row per trial to `sensitivity/sensitivity_summary.csv` and a machine-readable aggregate to `sensitivity/sensitivity_summary.json`. Every row records the full parameter set, extraction mode, ridge parameters, input path, output SVG path, debug directory, graph export paths, measured tracing counts, bridge count, raw and merged node counts, connected components, total traced length, dominant orientation peaks, and the standalone command used for that trial.

Each trial also gets its own directory:

```text
sensitivity/trials/trial_0000/params.json
sensitivity/trials/trial_0000/command.txt
sensitivity/trials/trial_0000/output.svg
sensitivity/trials/trial_0000/metrics.json
sensitivity/trials/trial_0000/graph_raw.graphml
sensitivity/trials/trial_0000/graph_merged.graphml
sensitivity/trials/trial_0000/nodes_raw.csv
sensitivity/trials/trial_0000/nodes_merged.csv
sensitivity/trials/trial_0000/edges_raw.csv
sensitivity/trials/trial_0000/edges_merged.csv
sensitivity/trials/trial_0000/orientation_histogram.png
sensitivity/trials/trial_0000/debug/*.png
```

The `command.txt` files are independent single-image commands so the same trial layout can later be dispatched as SLURM array jobs. See `docs/hpc_delta_slurm.md` for a CPU-only NCSA Delta example. This branch does not add GPU or HPC acceleration.

## Null Models And Positive Controls

`engrave2svg` includes analysis utilities for asking whether an extracted graph has more mesh-like geometric structure than randomized scratch controls. These tools operate on canonical GraphML/CSV/JSON outputs; SVG remains a visual preview only.

Generate synthetic controls:

```bash
python scripts/generate_controls.py \
  --output-dir controls/lozenge_nulls \
  --width 240 \
  --height 180 \
  --spacing 36 \
  --stroke-width 3 \
  --control-polarity bright-on-dark \
  --seed 42 \
  --svg-previews
```

This writes clean lozenge/rhombic lattice, broken/noisy lozenge lattice, random scratch, and curved random scratch PNG controls plus `expected_metadata.json`. The default controls are bright strokes on a dark background, matching the threshold pipeline polarity; use `--control-polarity dark-on-bright` only for downstream tools that explicitly expect that polarity. Clean synthetic lozenge lattices are the positive controls because their geometry is known. SandDraw/iPad sand-texture images are not positive controls unless a preprocessing step first isolates actual groove centerlines; otherwise they test rendering texture, not extracted engraving topology.

Vectorize the clean lozenge control with this recipe before using it as an extracted positive-control graph:

```bash
python engrave2svg.py controls/lozenge_nulls/clean_lozenge_lattice.png \
  --output controls/vectorized/clean_lozenge/output.svg \
  --metrics controls/vectorized/clean_lozenge/metrics.json \
  --graph controls/vectorized/clean_lozenge/graph.graphml \
  --nodes-csv controls/vectorized/clean_lozenge/nodes.csv \
  --edges-csv controls/vectorized/clean_lozenge/edges.csv \
  --debug controls/vectorized/clean_lozenge/debug \
  --crop none \
  --extraction-mode threshold \
  --threshold-mode global \
  --threshold-value 180 \
  --morph-kernel-size 1 \
  --min-component-size 1 \
  --denoise-kernel-size 1 \
  --clahe-clip-limit 1.0 \
  --simplification-epsilon 0.0 \
  --node-merge-radius 2.0
```

That recipe is intentionally plain threshold extraction. For the default generated clean lozenge control it should yield nonzero paths, graph nodes/edges, cycles, and lozenge candidates.

Compare an observed extraction against a directory of null/control GraphML or metrics JSON files:

```bash
python scripts/compare_null_models.py \
  --observed-graph output_merged.graphml \
  --observed-metrics output_metrics.json \
  --controls-dir controls/vectorized \
  --output-dir null_comparison \
  --orientation-shuffles 100 \
  --seed 42
```

The comparison writes `comparison_summary.csv`, `comparison_summary.json`, `invalid_controls.csv`, simple PNG plots for key metrics, `lozenge_candidates.csv`, and `lozenge_summary.json`. Metrics include dominant orientation peak concentration, orientation entropy, number of dominant orientation families, endpoint and junction counts, degree-3 fraction, connected components, largest connected component fraction, cycle count, 4-cycle count, conservative lozenge/rhombus candidate count, and total traced length. When a control directory contains `metrics.json`, `graph_raw.graphml`, and `graph_merged.graphml`, the comparison uses the merged graph as one control sample and ignores raw graphs by default to avoid double-counting. Controls with `node_count == 0` or `total_traced_length == 0` are reported separately as invalid and excluded from class means.

Lozenge candidates are deliberately conservative. The detector operates on graph topology and node coordinates, starts from simple 4-cycles, and checks side lengths, interior angles, diagonal lengths, aspect ratio, area, and approximate parallelism of opposite sides. A 4-cycle is not automatically called a lozenge. The output should be read as statistical support for or against mesh-like geometric structure relative to the supplied controls, not as evidence of intention.

## Example

```bash
python engrave2svg.py input.png \
  --output output.svg \
  --metrics output_metrics.json \
  --graph output.graphml \
  --nodes-csv output_nodes.csv \
  --edges-csv output_edges.csv \
  --orientation-hist output_orientation_histogram.png \
  --debug debug \
  --extraction-mode threshold \
  --threshold-mode global \
  --threshold-value 180 \
  --morph-kernel-size 1 \
  --min-component-size 12 \
  --simplification-epsilon 3.0 \
  --bridge-gaps-radius 0.0 \
  --crop auto
```

Experimental ridge extraction example:

```bash
python engrave2svg.py input.png \
  --output output_ridge.svg \
  --metrics output_ridge_metrics.json \
  --debug debug_ridge \
  --extraction-mode ridge \
  --ridge-sigmas 1,2,3 \
  --ridge-beta 0.5 \
  --ridge-gamma 15.0 \
  --ridge-threshold 0.05 \
  --threshold-mode global \
  --threshold-value 180 \
  --morph-kernel-size 1 \
  --min-component-size 12 \
  --simplification-epsilon 3.0 \
  --crop auto
```

The repository example script is still available:

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

The tests use synthetic line drawings and graph controls to check crop behavior, skeletonization, junction splitting, simplification, ridge extraction, graph metrics, node merging, orientation families, lozenge/null-model analysis, sensitivity summaries, and SVG export.

## Failure Modes

- Broken strokes: thresholding or morphology can split faint engraved lines. Try adaptive thresholding, lower the global threshold, reduce denoising, or use a smaller morphology kernel.
- Ridge artifacts: ridge mode can preserve faint continuity that thresholding misses, but low `--ridge-threshold` or broad `--ridge-sigmas` can create side branches or merge close strokes. Inspect `06b` through `06e` debug images and prefer the smallest scale range that preserves the target strokes.
- False joins: close parallel strokes can merge during thresholding or closing. Reduce `--morph-kernel-size` and inspect `05_thresholded.png` and `06_cleaned.png`.
- Noisy intersections: anti-aliased crossings often create clusters of junction pixels rather than one clean node. The tracer preserves topology but may emit several short paths around the intersection.
- Over-bridging: `--bridge-gaps-radius` is conservative, but any nonzero value can change topology. Inspect `06a_bridged.png` and the `bridges` list in the metrics JSON; lower the radius or angle tolerance if nearby strokes are incorrectly connected.
- Node consolidation risk: `--node-merge-radius` reports both raw and merged graph statistics. Keep the raw GraphML/CSV files with the merged outputs so reviewers can audit any topology changes.
- Anti-aliasing artifacts: pale edge pixels can become small side branches after skeletonization. Increase `--min-component-size`, use a slightly higher threshold, or crop more tightly.
- Over-simplification: Douglas-Peucker can move bends away from the original centerline. Lower `--simplification-epsilon` or set it to `0` for raw traced paths.
- Auto-crop misses the panel: use `--crop none` for pre-cropped images or pass `--crop x,y,width,height`.

## Notes for Field Use

The exported SVG uses one `<polyline>` per traced segment with stable IDs like `path-0000`. Coordinates are relative to the cropped panel, not the original full image. Keep the debug directory with the SVG when recording provenance; it captures the exact intermediate stages that led to the vector result.

For reviewer-facing computational archaeology work, prefer the graph outputs over the SVG. Nodes are explicit endpoints and junctions/intersections; edges are traced skeleton stroke segments with pixel coordinates, polyline geometry, length, and axial orientation. The metrics JSON reports raw and merged topology, extraction mode, ridge parameters, optional bridge parameters and bridge records, node degree histograms, connected component size histograms, largest connected component fraction, graph density where meaningful, average node degree, total traced length, circular orientation statistics, and dominant length-weighted orientation bins. Sensitivity and null-model summaries help test whether an observed extraction is stable and more consistent with mesh-like geometric structure than with randomized scratch patterns; they should not be used to overclaim intention.
