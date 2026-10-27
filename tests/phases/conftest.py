import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASES_DIR = Path(__file__).resolve().parent
for sub in PHASES_DIR.iterdir():
    if sub.is_dir():
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
