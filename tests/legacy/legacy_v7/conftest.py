import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SHIMS = Path(__file__).resolve().parent.parent / "legacy_v7_shims"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SHIMS) not in sys.path:
    sys.path.insert(0, str(SHIMS))
