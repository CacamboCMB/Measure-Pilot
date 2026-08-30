"""FreeCAD GUI initializer for the MeasurePilot workbench."""

from pathlib import Path
import sys

import FreeCADGui as Gui


def _measurepilot_root() -> Path:
    """Locate the workbench root when FreeCAD omits ``__file__``."""

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
_ROOT_PATH = str(_ROOT)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)
_ICON = str(_ROOT / "resources" / "icons" / "MeasurePilot.svg")


class MeasurePilotWorkbench(Gui.Workbench):
    """Native MeasurePilot workbench."""

    MenuText = "MeasurePilot"
    ToolTip = "Measurement-guided planar reverse engineering"
    Icon = _ICON

    def Initialize(self):
        """Register commands only when FreeCAD first loads the workbench."""

        from freecad_workbench.commands import COMMAND_NAME, register_commands

        register_commands()
        self.appendToolbar("MeasurePilot", [COMMAND_NAME])
        self.appendMenu("MeasurePilot", [COMMAND_NAME])

    def GetClassName(self):
        return "Gui::PythonWorkbench"


Gui.addWorkbench(MeasurePilotWorkbench)
