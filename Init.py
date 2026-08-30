"""FreeCAD application initializer for the MeasurePilot workbench."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir():
    source_path = str(_SRC)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
