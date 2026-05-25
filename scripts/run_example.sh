#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" scripts/make_synthetic_input.py
"${PYTHON_BIN}" engrave2svg.py examples/synthetic_panel.png \
  --output examples/synthetic_output.svg \
  --crop auto \
  --debug examples/debug \
  --threshold-mode otsu \
  --morph-kernel-size 1 \
  --min-component-size 8 \
  --simplification-epsilon 1.2
