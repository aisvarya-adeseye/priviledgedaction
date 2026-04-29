"""pytest configuration helpers.

Ensure the project root (one level above `test/`) is on `sys.path`
so imports like `from core...` resolve when running `pytest`.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
