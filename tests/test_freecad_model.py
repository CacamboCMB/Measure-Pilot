from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from measurepilot.freecad_model import (
    FreeCADRuntimeError,
    ModelGeometryError,
    ModelInputError,
    build_model_plan,
    create_freecad_document,
)
from measurepilot.parameter_graph import import_m2_analysis
from measurepilot.provenance import canonical_json_bytes


def _analysis(
    *,
    unresolved: list[dict] | None = None,
    second_hole_x: float = 130.0,
) -> bytes:
    return canonical_json_bytes(
        {
            "configuration": {
                "features": {},
                "segmentation": {"px_per_mm": 4.0},
            },
            "contours": {
                "holes": [
                    {"id": "hole-contour-000"},
                    {"id": "hole-contour-001"},
                ],
                "outer": {"simplification_rms_mm": 0.1},
            },
            "coordinate_system": {
                "origin": "top_left_of_rectified_a4_page",
                "unit": "mm",
                "x_axis": "right",
                "y_axis": "down",
            },
            "features": {
                "circular_holes": [
                    {
                        "center_mm": [80.0, 210.0],
                        "diameter_mm": 10.0,
                        "fit_residual_mm": 0.05,
                        "id": "circular-hole-000",
                    },
                    {
                        "center_mm": [second_hole_x, 210.0],
                        "diameter_mm": 10.0,
                        "fit_residual_mm": 0.05,
                        "id": "circular-hole-001",
                    },
                ],
                "line_segments": [
                    {
                        "end_mm": [150.0, 185.0],
                        "id": "outer-line-000",
                        "length_mm": 90.0,
                        "start_mm": [60.0, 185.0],
                    },
                    {
                        "end_mm": [150.0, 235.0],
                        "id": "outer-line-001",
                        "length_mm": 50.0,
                        "start_mm": [150.0, 185.0],
                    },
                    {
                        "end_mm": [60.0, 235.0],
                        "id": "outer-line-002",
                        "length_mm": 90.0,
                        "start_mm": [150.0, 235.0],
                    },
                    {
                        "end_mm": [60.0, 185.0],
                        "id": "outer-line-003",
                        "length_mm": 50.0,
                        "start_mm": [60.0, 235.0],
                    },
                ],
            },
            "format": "measurepilot-planar-analysis",
            "image": {"height_px": 1188, "width_px": 840},
            "segmentation": {},
            "unresolved": unresolved or [],
            "version": 1,
            "warnings": [],
        }
    )


def test_plan_is_normalized_deterministic_and_contains_supported_geometry() -> None:
    analysis = _analysis()
    first = build_model_plan(
        analysis,
        thickness_mm=2.6,
        document_name="replacement plate",
    )
    second = build_model_plan(
        analysis,
        thickness_mm=2.6,
        document_name="replacement plate",
    )
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.translation_mm == (60.0, 185.0)
    assert first.document_name == "replacement_plate"
    assert first.outer_segments[0].start_mm == (0.0, 0.0)
    assert first.outer_segments[0].end_mm == (90.0, 0.0)
    assert [hole.center_mm for hole in first.holes] == [(20.0, 25.0), (70.0, 25.0)]
    assert first.thickness_mm == 2.6


def test_resolved_m3_correction_overrides_m2_hole_geometry() -> None:
    analysis = _analysis()
    graph = import_m2_analysis(analysis)
    center_observation = next(
        observation
        for observation in graph.observations
        if observation.parameter_id == "m2.hole.circular-hole-001.center_x"
    )
    diameter_observation = next(
        observation
        for observation in graph.observations
        if observation.parameter_id == "m2.hole.circular-hole-001.diameter"
    )
    graph.correct_measurement(
        center_observation.observation_id,
        value=132.0,
        uncertainty=0.1,
        note="caliper centre correction",
    )
    graph.correct_measurement(
        diameter_observation.observation_id,
        value=8.0,
        uncertainty=0.05,
        note="caliper diameter correction",
    )
    plan = build_model_plan(
        analysis,
        graph_bytes=graph.canonical_bytes(),
        thickness_mm=2.0,
    )
    second = next(hole for hole in plan.holes if hole.feature_id == "circular-hole-001")
    assert second.center_mm == (72.0, 25.0)
    assert second.diameter_mm == 8.0
    assert plan.graph_sha256 is not None


def test_graph_source_binding_and_conflicting_required_values_fail() -> None:
    analysis = _analysis()
    unrelated = import_m2_analysis(_analysis(second_hole_x=129.0))
    with pytest.raises(ModelInputError, match="not bound"):
        build_model_plan(
            analysis,
            graph_bytes=unrelated.canonical_bytes(),
            thickness_mm=2.0,
        )

    graph = import_m2_analysis(analysis)
    graph.append_measurement(
        parameter_id="m2.hole.circular-hole-000.diameter",
        quantity="length",
        value=20.0,
        uncertainty=0.01,
        note="conflicting physical value",
    )
    with pytest.raises(ModelInputError, match="CONFLICTING"):
        build_model_plan(
            analysis,
            graph_bytes=graph.canonical_bytes(),
            thickness_mm=2.0,
        )


def test_unsupported_or_invalid_geometry_is_rejected() -> None:
    with pytest.raises(ModelInputError, match="unresolved geometry"):
        build_model_plan(
            _analysis(unresolved=[{"code": "NON_CIRCULAR_HOLE"}]),
            thickness_mm=2.0,
        )
    with pytest.raises(ModelGeometryError, match="outside"):
        build_model_plan(_analysis(second_hole_x=170.0), thickness_mm=2.0)
    with pytest.raises(ModelInputError, match="positive"):
        build_model_plan(_analysis(), thickness_mm=0.0)

    value = __import__("json").loads(_analysis())
    value["features"]["line_segments"][2]["end_mm"] = [150.0, 185.0]
    with pytest.raises(ModelGeometryError):
        build_model_plan(canonical_json_bytes(value), thickness_mm=2.0)


def test_noncanonical_analysis_is_rejected() -> None:
    with pytest.raises(ModelInputError, match="not canonical"):
        build_model_plan(
            __import__("json").dumps(__import__("json").loads(_analysis()), indent=2).encode(),
            thickness_mm=2.0,
        )


@dataclass
class _Shape:
    null: bool = False

    def isNull(self):
        return self.null


class _Object:
    def __init__(self, type_id: str, name: str):
        self.TypeId = type_id
        self.Name = name
        self.Label = name
        self.properties: dict[str, str] = {}
        self.group: list[object] = []
        self.geometry: list[object] = []
        self.constraints: list[object] = []
        self.Shape = _Shape()
        self.valid = True

    def addObject(self, value):
        self.group.append(value)

    def addProperty(self, _type_name, name, _group):
        self.properties[name] = ""

    def addGeometry(self, geometry, _construction):
        if isinstance(geometry, list):
            start = len(self.geometry)
            self.geometry.extend(geometry)
            return list(range(start, len(self.geometry)))
        self.geometry.append(geometry)
        return len(self.geometry) - 1

    def addConstraint(self, constraints):
        if isinstance(constraints, list):
            self.constraints.extend(constraints)
        else:
            self.constraints.append(constraints)

    def isValid(self):
        return self.valid


class _Document:
    def __init__(self, name: str, *, invalid_pad: bool = False):
        self.Name = name
        self.objects: list[_Object] = []
        self.events: list[str] = []
        self.saved: str | None = None
        self.invalid_pad = invalid_pad

    def openTransaction(self, name):
        self.events.append(f"open:{name}")

    def commitTransaction(self):
        self.events.append("commit")

    def abortTransaction(self):
        self.events.append("abort")

    def addObject(self, type_id, name):
        obj = _Object(type_id, name)
        if type_id == "PartDesign::Pad" and self.invalid_pad:
            obj.valid = False
        self.objects.append(obj)
        return obj

    def recompute(self):
        self.events.append("recompute")

    def saveAs(self, path):
        self.saved = path
        self.events.append("save")


class _App:
    class Vector(tuple):
        def __new__(cls, x, y, z):
            return tuple.__new__(cls, (x, y, z))

    def __init__(self, *, invalid_pad: bool = False):
        self.invalid_pad = invalid_pad
        self.documents: list[_Document] = []
        self.closed: list[str] = []

    def newDocument(self, name):
        document = _Document(name, invalid_pad=self.invalid_pad)
        self.documents.append(document)
        return document

    def closeDocument(self, name):
        self.closed.append(name)


class _Part:
    @staticmethod
    def LineSegment(start, end):
        return ("line", start, end)

    @staticmethod
    def Circle(center, axis, radius):
        return ("circle", center, axis, radius)


class _Sketcher:
    @staticmethod
    def Constraint(*arguments):
        return tuple(arguments)


def test_native_runtime_creates_body_sketch_pad_metadata_and_save(tmp_path) -> None:
    plan = build_model_plan(_analysis(), thickness_mm=2.6)
    app = _App()
    destination = tmp_path / "plate.FCStd"
    result = create_freecad_document(
        plan,
        save_path=destination,
        modules=(app, _Part, _Sketcher),
    )
    assert result.body.TypeId == "PartDesign::Body"
    assert result.sketch.TypeId == "Sketcher::SketchObject"
    assert result.pad.TypeId == "PartDesign::Pad"
    assert result.pad.Profile is result.sketch
    assert result.pad.Length == 2.6
    assert len(result.sketch.geometry) == 6
    assert [item[0] for item in result.sketch.geometry].count("line") == 4
    assert [item[0] for item in result.sketch.geometry].count("circle") == 2
    assert len(result.sketch.constraints) == 4
    assert result.body.MeasurePilotAnalysisSHA256 == plan.analysis_sha256
    assert result.document.saved == str(destination)
    assert result.document.events[-1] == "commit"
    assert app.closed == []


def test_native_runtime_aborts_and_closes_document_on_failure(tmp_path) -> None:
    plan = build_model_plan(_analysis(), thickness_mm=2.0)
    app = _App(invalid_pad=True)
    with pytest.raises(FreeCADRuntimeError, match="rejected"):
        create_freecad_document(plan, modules=(app, _Part, _Sketcher))
    assert "abort" in app.documents[0].events
    assert app.closed == [plan.document_name]

    app = _App()
    with pytest.raises(FreeCADRuntimeError, match=".FCStd"):
        create_freecad_document(
            plan,
            save_path=tmp_path / "plate.step",
            modules=(app, _Part, _Sketcher),
        )
    assert "abort" in app.documents[0].events
