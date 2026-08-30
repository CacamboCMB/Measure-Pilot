from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys
import types
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_package_metadata_and_required_files_are_consistent() -> None:
    package = ET.parse(ROOT / "package.xml").getroot()
    namespace = {"p": "https://wiki.freecad.org/Package_Metadata"}
    assert package.findtext("p:name", namespaces=namespace) == "MeasurePilot"
    assert package.findtext("p:version", namespaces=namespace) == "0.5.0"
    workbench = package.find("p:content/p:workbench", namespace)
    assert workbench is not None
    assert workbench.findtext("p:freecadmin", namespaces=namespace) == "1.0"
    icon = package.findtext("p:icon", namespaces=namespace)
    assert icon == "resources/icons/MeasurePilot.svg"
    assert (ROOT / icon).is_file()
    assert (ROOT / "Init.py").is_file()
    assert (ROOT / "InitGui.py").is_file()


def test_init_adds_only_repository_src_to_python_path(monkeypatch) -> None:
    source = str(ROOT / "src")
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != source])
    runpy.run_path(str(ROOT / "Init.py"))
    assert sys.path[0] == source


def test_command_module_import_is_lazy_without_freecad_or_pyside(monkeypatch) -> None:
    for name in ("FreeCAD", "FreeCADGui", "Part", "Sketcher", "PySide"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module_path = ROOT / "freecad_workbench" / "commands.py"
    spec = importlib.util.spec_from_file_location("measurepilot_test_commands", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.COMMAND_NAME == "MeasurePilot_CreatePlanarModel"
    assert "FreeCAD" not in sys.modules
    assert "PySide" not in sys.modules


def test_initgui_registers_workbench_and_initialize_registers_one_command(monkeypatch) -> None:
    registered_workbenches: list[type] = []
    registered_commands: dict[str, object] = {}

    class Workbench:
        def __init__(self):
            self.toolbars = []
            self.menus = []

        def appendToolbar(self, name, commands):
            self.toolbars.append((name, list(commands)))

        def appendMenu(self, name, commands):
            self.menus.append((name, list(commands)))

    fake_gui = types.ModuleType("FreeCADGui")
    fake_gui.Workbench = Workbench
    fake_gui.addWorkbench = registered_workbenches.append
    fake_gui.addCommand = lambda name, command: registered_commands.setdefault(name, command)
    monkeypatch.setitem(sys.modules, "FreeCADGui", fake_gui)

    commands_name = "freecad_workbench.commands"
    monkeypatch.delitem(sys.modules, commands_name, raising=False)
    init_globals = runpy.run_path(str(ROOT / "InitGui.py"))
    assert registered_workbenches == [init_globals["MeasurePilotWorkbench"]]

    workbench = registered_workbenches[0]()
    workbench.Initialize()
    assert list(registered_commands) == ["MeasurePilot_CreatePlanarModel"]
    assert workbench.toolbars == [
        ("MeasurePilot", ["MeasurePilot_CreatePlanarModel"])
    ]
    assert workbench.menus == [
        ("MeasurePilot", ["MeasurePilot_CreatePlanarModel"])
    ]
    assert workbench.GetClassName() == "Gui::PythonWorkbench"


def test_command_reports_actionable_error_instead_of_raising(monkeypatch) -> None:
    from freecad_workbench import commands

    messages: list[tuple[str, str]] = []

    class FileDialog:
        @staticmethod
        def getOpenFileName(*_args):
            return ("/missing/analysis.json", "")

        @staticmethod
        def getSaveFileName(*_args):
            return ("", "")

    class MessageBox:
        Yes = 1
        No = 2

        @staticmethod
        def question(*_args):
            return MessageBox.No

        @staticmethod
        def critical(_parent, title, message):
            messages.append((title, message))

        @staticmethod
        def information(*_args):
            raise AssertionError("success message must not be shown")

    class InputDialog:
        @staticmethod
        def getDouble(*_args):
            return (2.0, True)

    widgets = types.SimpleNamespace(
        QFileDialog=FileDialog,
        QMessageBox=MessageBox,
        QInputDialog=InputDialog,
    )
    monkeypatch.setattr(commands, "_widgets", lambda: widgets)
    commands.CreatePlanarModelCommand().Activated()
    assert messages
    assert messages[0][0] == "MeasurePilot model creation failed"
    assert "missing" in messages[0][1]
