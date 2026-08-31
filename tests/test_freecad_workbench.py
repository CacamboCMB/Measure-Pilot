from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _exec_without_file(path: Path) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "measurepilot_freecad_loader_test"}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace, namespace)
    assert "__file__" not in namespace
    return namespace


def _exec_split_namespace_without_file(path: Path) -> dict[str, object]:
    globals_namespace: dict[str, object] = {
        "__builtins__": __builtins__,
        "__name__": "measurepilot_freecad_split_loader_test",
    }
    locals_namespace: dict[str, object] = {}
    exec(
        compile(path.read_bytes(), str(path), "exec"),
        globals_namespace,
        locals_namespace,
    )
    assert "__file__" not in globals_namespace
    assert "__file__" not in locals_namespace
    return locals_namespace


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
    monkeypatch.setattr(sys, "path", [str(ROOT), *[item for item in sys.path if item not in {str(ROOT), source}]])
    _exec_without_file(ROOT / "Init.py")
    assert sys.path[0] == source


def test_init_prefers_versioned_freecad_user_data_without_file(monkeypatch, tmp_path) -> None:
    user_data = tmp_path / "FreeCAD" / "v1-1"
    workbench_root = user_data / "Mod" / "MeasurePilot"
    (workbench_root / "src" / "measurepilot").mkdir(parents=True)
    (workbench_root / "freecad_workbench").mkdir()
    (workbench_root / "vendor").mkdir()

    fake_app = types.ModuleType("FreeCAD")
    fake_app.getUserAppDataDir = lambda: str(user_data)
    fake_app.getResourceDir = lambda: str(tmp_path / "resource")
    monkeypatch.setitem(sys.modules, "FreeCAD", fake_app)
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != str(ROOT)])

    namespace = _exec_without_file(ROOT / "Init.py")
    assert namespace["_ROOT"] == workbench_root.resolve()
    assert sys.path[0] == str(workbench_root / "src")
    assert sys.path[1] == str(workbench_root / "vendor")


def test_init_supports_split_loader_namespace_without_file(monkeypatch) -> None:
    source = str(ROOT / "src")
    monkeypatch.delitem(sys.modules, "FreeCAD", raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [str(ROOT), *[item for item in sys.path if item not in {str(ROOT), source}]],
    )

    namespace = _exec_split_namespace_without_file(ROOT / "Init.py")
    assert namespace["_ROOT"] == ROOT.resolve()
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


def test_initgui_registers_workbench_without_file_and_initialize_registers_one_command(monkeypatch) -> None:
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
    init_globals = _exec_without_file(ROOT / "InitGui.py")
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


def test_initgui_supports_split_loader_namespace_without_file(monkeypatch) -> None:
    registered_workbenches: list[type] = []

    class Workbench:
        pass

    fake_gui = types.ModuleType("FreeCADGui")
    fake_gui.Workbench = Workbench
    fake_gui.addWorkbench = registered_workbenches.append
    monkeypatch.setitem(sys.modules, "FreeCADGui", fake_gui)
    monkeypatch.delitem(sys.modules, "FreeCAD", raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [str(ROOT), *[item for item in sys.path if item != str(ROOT)]],
    )

    namespace = _exec_split_namespace_without_file(ROOT / "InitGui.py")
    assert registered_workbenches == [namespace["MeasurePilotWorkbench"]]
    assert registered_workbenches[0].Icon.endswith("resources/icons/MeasurePilot.svg")


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
