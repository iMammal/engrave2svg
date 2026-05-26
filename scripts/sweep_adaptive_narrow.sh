#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-IMG-20260424-WA0002.jpg}"
OUTROOT="${2:-sweeps/adaptive_narrow}"

mkdir -p "$OUTROOT"

echo "run,block_size,adaptive_c,min_component_size,paths,endpoints,junctions,log" > "$OUTROOT/summary.csv"

for block in 61 71 81 91 111; do
  for cval in 2 4 6 8 10; do
    for comp in 12 18 24 32; do
      run="adapt_b${block}_c${cval}_comp${comp}"
      rundir="$OUTROOT/$run"
      mkdir -p "$rundir"

      log="$rundir/run.log"

      echo "Running $run"

      python engrave2svg.py "$INPUT" \
        --output "$rundir/out.svg" \
        --crop auto \
        --debug "$rundir/debug" \
        --threshold-mode adaptive \
        --adaptive-block-size "$block" \
        --adaptive-c "$cval" \
        --morph-kernel-size 1 \
        --min-component-size "$comp" \
        --simplification-epsilon 3.0 \
        --node-merge-radius 0 \
        --metrics "$rundir/metrics.json" \
        --graph "$rundir/graph.graphml" \
        --nodes-csv "$rundir/nodes.csv" \
        --edges-csv "$rundir/edges.csv" \
        --orientation-hist "$rundir/orientation_hist.png" \
        2>&1 | tee "$log"

      line="$(grep 'Wrote ' "$log" | tail -n 1 || true)"

      paths="$(echo "$line" | sed -n 's/.*with \([0-9][0-9]*\) paths.*/\1/p')"
      endpoints="$(echo "$line" | sed -n 's/.*(\([0-9][0-9]*\).*endpoints.*/\1/p')"
      junctions="$(echo "$line" | sed -n 's/.*endpoints, \([0-9][0-9]*\).*junctions.*/\1/p')"

      echo "$run,$block,$cval,$comp,${paths:-NA},${endpoints:-NA},${junctions:-NA},$log" >> "$OUTROOT/summary.csv"
    done
  done
done

echo "Done. Summary written to $OUTROOT/summary.csv"
