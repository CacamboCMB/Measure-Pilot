"""FreeCAD GUI initializer for the MeasurePilot workbench."""


def _measurepilot_root():
    """Locate the workbench root without relying on loader globals."""

    from pathlib import Path as _Path
    import sys as _sys

    candidates = []
    try:
        import FreeCAD as _App
    except Exception:
        _App = None

    if _App is not None:
        for getter_name in ("getUserAppDataDir", "getResourceDir"):
            try:
                base = _Path(getattr(_App, getter_name)())
            except (AttributeError, OSError, TypeError, ValueError):
                continue
            candidates.append(base / "Mod" / "MeasurePilot")

    for entry in tuple(_sys.path):
        if not isinstance(entry, (str, bytes)):
            continue
        try:
            base = _Path(entry)
        except (OSError, TypeError, ValueError):
            continue
        candidates.extend((base, base / "MeasurePilot"))

    seen = set()
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
import sys as _sys

_ROOT_PATH = str(_ROOT)
if _ROOT_PATH not in _sys.path:
    _sys.path.insert(0, _ROOT_PATH)
_ICON = str(_ROOT / "resources" / "icons" / "MeasurePilot.svg")

import FreeCADGui as _Gui


class MeasurePilotWorkbench(_Gui.Workbench):
    """Native MeasurePilot workbench."""

    MenuText = "MeasurePilot"
    ToolTip = "Measurement-guided planar reverse engineering"
    Icon = ""

    def Initialize(self):
        """Register commands only when FreeCAD first loads the workbench."""

        from freecad_workbench.commands import COMMAND_NAME, register_commands

        register_commands()
        self.appendToolbar("MeasurePilot", [COMMAND_NAME])
        self.appendMenu("MeasurePilot", [COMMAND_NAME])

    def GetClassName(self):
        return "Gui::PythonWorkbench"


MeasurePilotWorkbench.Icon = _ICON
_Gui.addWorkbench(MeasurePilotWorkbench)
