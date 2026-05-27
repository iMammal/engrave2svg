#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrave2svg.null_models import generate_controls_main


if __name__ == "__main__":
    raise SystemExit(generate_controls_main())
