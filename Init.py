"""FreeCAD application initializer for the MeasurePilot workbench."""

from pathlib import Path
import sys


def _measurepilot_root() -> Path:
    """Locate the workbench root without relying on ``__file__``.

    FreeCAD executes addon initializers through its module loader and may not
    define ``__file__``. Prefer the active versioned user-data directory, then
    fall back to bounded entries already present on ``sys.path`` for development
    checkouts and system-wide installations.
    """

    candidates: list[Path] = []
    try:
        import FreeCAD as App
    except Exception:
        App = None

    if App is not None:
        for getter_name in ("getUserAppDataDir", "getResourceDir"):
            try:
                base = Path(getattr(App, getter_name)())
            except (AttributeError, OSError, TypeError, ValueError):
                continue
            candidates.append(base / "Mod" / "MeasurePilot")

    for entry in tuple(sys.path):
        if not isinstance(entry, (str, bytes)):
            continue
        try:
            base = Path(entry)
        except (OSError, TypeError, ValueError):
            continue
        candidates.extend((base, base / "MeasurePilot"))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = candidate.expanduser().resolve()
        except (OSError, RuntimeError):
            normalized = candidate
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        if (normalized / "src" / "measurepilot").is_dir() and (
            normalized / "freecad_workbench"
        ).is_dir():
            return normalized

    raise ImportError(
        "MeasurePilot workbench root could not be located from FreeCAD user data or sys.path"
    )


_ROOT = _measurepilot_root()
for _runtime_dir in (_ROOT / "vendor", _ROOT / "src"):
    if _runtime_dir.is_dir():
        _runtime_path = str(_runtime_dir)
        if _runtime_path not in sys.path:
            sys.path.insert(0, _runtime_path)
