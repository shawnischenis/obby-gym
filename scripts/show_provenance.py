#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from obby_rl.config import validate_config  # noqa: E402
from obby_rl.provenance import collect_provenance  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(collect_provenance(validate_config()), indent=2, sort_keys=True))
