"""Lazy GUI commands for the native MeasurePilot FreeCAD workbench."""

from __future__ import annotations

from pathlib import Path


COMMAND_NAME = "MeasurePilot_CreatePlanarModel"
_REGISTERED = False
_ROOT = Path(__file__).resolve().parents[1]
_ICON = str(_ROOT / "resources" / "icons" / "MeasurePilot.svg")


def _widgets():
    try:
        from PySide import QtWidgets

        return QtWidgets
    except ImportError:
        from PySide import QtGui

        return QtGui


def _dialog_path(value) -> str:
    if isinstance(value, tuple):
        return str(value[0])
    return str(value)


class CreatePlanarModelCommand:
    """Create one supported native PartDesign model from canonical evidence."""

    def GetResources(self):
        return {
            "Pixmap": _ICON,
            "MenuText": "Create planar model",
            "ToolTip": (
                "Create a native Body/Sketch/Pad model from MeasurePilot M2 analysis"
            ),
        }

    def IsActive(self):
        return True

    def Activated(self):
        widgets = _widgets()
        try:
            analysis_path = _dialog_path(
                widgets.QFileDialog.getOpenFileName(
                    None,
                    "Select canonical MeasurePilot M2 analysis",
                    "",
                    "MeasurePilot analysis (*.json);;JSON files (*.json)",
                )
            )
            if not analysis_path:
                return

            graph_path = ""
            use_graph = widgets.QMessageBox.question(
                None,
                "MeasurePilot parameter graph",
                "Apply an optional M3/M4 parameter graph?",
                widgets.QMessageBox.Yes | widgets.QMessageBox.No,
                widgets.QMessageBox.No,
            )
            if use_graph == widgets.QMessageBox.Yes:
                graph_path = _dialog_path(
                    widgets.QFileDialog.getOpenFileName(
                        None,
                        "Select canonical MeasurePilot parameter graph",
                        "",
                        "MeasurePilot graph (*.json);;JSON files (*.json)",
                    )
                )
                if not graph_path:
                    return

            thickness, accepted = widgets.QInputDialog.getDouble(
                None,
                "MeasurePilot thickness",
                "Uniform thickness in mm:",
                2.0,
                0.001,
                1000.0,
                3,
            )
            if not accepted:
                return

            save_path = _dialog_path(
                widgets.QFileDialog.getSaveFileName(
                    None,
                    "Optionally save the FreeCAD document",
                    f"{Path(analysis_path).stem}.FCStd",
                    "FreeCAD document (*.FCStd)",
                )
            )
            if save_path and Path(save_path).suffix.lower() != ".fcstd":
                save_path += ".FCStd"

            from measurepilot.freecad_model import (
                build_model_plan,
                create_freecad_document,
            )

            analysis_bytes = Path(analysis_path).read_bytes()
            graph_bytes = Path(graph_path).read_bytes() if graph_path else None
            plan = build_model_plan(
                analysis_bytes,
                graph_bytes=graph_bytes,
                thickness_mm=thickness,
                document_name=Path(analysis_path).stem,
            )
            result = create_freecad_document(
                plan,
                save_path=save_path or None,
            )

            import FreeCADGui as Gui

            active_document = getattr(Gui, "activeDocument", lambda: None)()
            if active_document is not None:
                active_view = getattr(active_document, "activeView", lambda: None)()
                if active_view is not None and hasattr(active_view, "fitAll"):
                    active_view.fitAll()
            location = f"\nSaved: {result.saved_path}" if result.saved_path else ""
            widgets.QMessageBox.information(
                None,
                "MeasurePilot",
                "Native planar model created successfully." + location,
            )
        except Exception as exc:
            widgets.QMessageBox.critical(
                None,
                "MeasurePilot model creation failed",
                str(exc),
            )


def register_commands() -> None:
    """Register workbench commands idempotently."""

    global _REGISTERED
    if _REGISTERED:
        return
    import FreeCADGui as Gui

    Gui.addCommand(COMMAND_NAME, CreatePlanarModelCommand())
    _REGISTERED = True
