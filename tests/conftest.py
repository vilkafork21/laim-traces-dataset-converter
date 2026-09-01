from __future__ import annotations

import sys
from pathlib import Path


NODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_ROOT))
