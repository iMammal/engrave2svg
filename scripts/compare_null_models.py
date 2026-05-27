#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrave2svg.null_models import compare_null_models_main


if __name__ == "__main__":
    raise SystemExit(compare_null_models_main())
