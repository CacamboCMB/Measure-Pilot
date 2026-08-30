"""FreeCAD GUI initializer for the MeasurePilot workbench."""

from pathlib import Path

import FreeCADGui as Gui


_ROOT = Path(__file__).resolve().parent
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
