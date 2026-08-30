"""Native FreeCAD model planning and construction for MeasurePilot M5.

The pure planning layer validates canonical M2 evidence and optional canonical M3
parameter evidence without importing FreeCAD.  The runtime adapter imports
FreeCAD modules only when model creation is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .parameter_graph import GraphError, ParameterGraph
from .provenance import (
    ParameterStatus,
    ProvenanceError,
    canonical_json_bytes,
    finite_number,
    positive_number,
    validate_identifier,
)


MODEL_PLAN_FORMAT = "measurepilot-freecad-model-plan"
MODEL_PLAN_VERSION = 1
M2_ANALYSIS_FORMAT = "measurepilot-planar-analysis"
M2_ANALYSIS_VERSION = 1
_GEOMETRY_TOLERANCE_MM = 1e-6
_DOCUMENT_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


class FreeCADModelError(ValueError):
    """Base class for plan validation and native model construction failures."""


class ModelInputError(FreeCADModelError):
    """Raised when analysis or graph evidence cannot support the M5 model."""


class ModelGeometryError(FreeCADModelError):
    """Raised when the supported planar geometry is invalid or ambiguous."""


class FreeCADRuntimeError(FreeCADModelError):
    """Raised when native FreeCAD object construction fails."""


@dataclass(frozen=True, slots=True)
class PlanSegment:
    """One normalized straight outer-profile segment in millimetres."""

    start_mm: tuple[float, float]
    end_mm: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_mm": list(self.end_mm),
            "start_mm": list(self.start_mm),
        }


@dataclass(frozen=True, slots=True)
class PlanHole:
    """One normalized circular through-hole."""

    feature_id: str
    center_mm: tuple[float, float]
    diameter_mm: float

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_mm": list(self.center_mm),
            "diameter_mm": self.diameter_mm,
            "feature_id": self.feature_id,
        }


@dataclass(frozen=True, slots=True)
class FreeCADModelPlan:
    """Deterministic, FreeCAD-independent model construction plan."""

    analysis_sha256: str
    graph_sha256: str | None
    translation_mm: tuple[float, float]
    thickness_mm: float
    outer_segments: tuple[PlanSegment, ...]
    holes: tuple[PlanHole, ...]
    document_name: str = "MeasurePilotModel"
    body_name: str = "MeasurePilotBody"
    sketch_name: str = "MeasurePilotSketch"
    pad_name: str = "MeasurePilotPad"

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.analysis_sha256):
            raise ModelInputError("analysis_sha256 must be a lowercase SHA-256 digest")
        if self.graph_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.graph_sha256
        ):
            raise ModelInputError("graph_sha256 must be a lowercase SHA-256 digest")
        _finite_point(self.translation_mm, "translation_mm")
        _positive(self.thickness_mm, "thickness_mm")
        _validate_object_name(self.document_name, "document_name")
        _validate_object_name(self.body_name, "body_name")
        _validate_object_name(self.sketch_name, "sketch_name")
        _validate_object_name(self.pad_name, "pad_name")
        if len({self.body_name, self.sketch_name, self.pad_name}) != 3:
            raise ModelInputError("FreeCAD object names must be distinct")
        if len(self.outer_segments) < 3:
            raise ModelGeometryError("outer profile requires at least three segments")
        points = _segments_to_points(self.outer_segments)
        _validate_polygon(points)
        seen_holes: set[str] = set()
        for hole in self.holes:
            validate_identifier(hole.feature_id, "hole feature_id")
            if hole.feature_id in seen_holes:
                raise ModelGeometryError(f"duplicate hole feature_id: {hole.feature_id}")
            seen_holes.add(hole.feature_id)
            _finite_point(hole.center_mm, f"{hole.feature_id}.center_mm")
            _positive(hole.diameter_mm, f"{hole.feature_id}.diameter_mm")
        _validate_holes(points, self.holes)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "analysis_sha256": self.analysis_sha256,
            "format": MODEL_PLAN_FORMAT,
            "graph_sha256": self.graph_sha256,
            "holes": [hole.to_dict() for hole in self.holes],
            "objects": {
                "body": self.body_name,
                "document": self.document_name,
                "pad": self.pad_name,
                "sketch": self.sketch_name,
            },
            "outer_segments": [segment.to_dict() for segment in self.outer_segments],
            "thickness_mm": self.thickness_mm,
            "translation_mm": list(self.translation_mm),
            "version": MODEL_PLAN_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class FreeCADModelObjects:
    """Native objects created from a model plan."""

    document: Any
    body: Any
    sketch: Any
    pad: Any
    saved_path: Path | None


def _number(value: object, field_name: str) -> float:
    try:
        return finite_number(value, field_name)
    except ProvenanceError as exc:
        raise ModelInputError(str(exc)) from exc


def _positive(value: object, field_name: str) -> float:
    try:
        return positive_number(value, field_name)
    except ProvenanceError as exc:
        raise ModelInputError(str(exc)) from exc


def _finite_point(value: object, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ModelInputError(f"{field_name} must contain exactly two coordinates")
    return (_number(value[0], f"{field_name}[0]"), _number(value[1], f"{field_name}[1]"))


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelInputError(f"{field_name} must be an object")
    return value


def _list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelInputError(f"{field_name} must be a list")
    return value


def _validate_object_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", value):
        raise ModelInputError(
            f"{field_name} must start with a letter and contain only letters, digits, or underscore"
        )
    return value


def canonical_document_name(value: str) -> str:
    """Return a stable FreeCAD-compatible document name."""

    if not isinstance(value, str):
        raise ModelInputError("document name must be text")
    candidate = _DOCUMENT_NAME_RE.sub("_", value.strip()).strip("_")
    if not candidate:
        candidate = "MeasurePilotModel"
    if not candidate[0].isalpha():
        candidate = f"MeasurePilot_{candidate}"
    return candidate[:80]


def _close(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return math.dist(first, second) <= _GEOMETRY_TOLERANCE_MM


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    if abs(_cross(start, end, point)) > _GEOMETRY_TOLERANCE_MM:
        return False
    return (
        min(start[0], end[0]) - _GEOMETRY_TOLERANCE_MM
        <= point[0]
        <= max(start[0], end[0]) + _GEOMETRY_TOLERANCE_MM
        and min(start[1], end[1]) - _GEOMETRY_TOLERANCE_MM
        <= point[1]
        <= max(start[1], end[1]) + _GEOMETRY_TOLERANCE_MM
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    d1 = _cross(first_start, first_end, second_start)
    d2 = _cross(first_start, first_end, second_end)
    d3 = _cross(second_start, second_end, first_start)
    d4 = _cross(second_start, second_end, first_end)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return (
        (abs(d1) <= _GEOMETRY_TOLERANCE_MM and _point_on_segment(second_start, first_start, first_end))
        or (abs(d2) <= _GEOMETRY_TOLERANCE_MM and _point_on_segment(second_end, first_start, first_end))
        or (abs(d3) <= _GEOMETRY_TOLERANCE_MM and _point_on_segment(first_start, second_start, second_end))
        or (abs(d4) <= _GEOMETRY_TOLERANCE_MM and _point_on_segment(first_end, second_start, second_end))
    )


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return 0.5 * math.fsum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _segments_to_points(segments: Sequence[PlanSegment]) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    previous_end: tuple[float, float] | None = None
    for index, segment in enumerate(segments):
        start = _finite_point(segment.start_mm, f"outer_segments[{index}].start_mm")
        end = _finite_point(segment.end_mm, f"outer_segments[{index}].end_mm")
        if _close(start, end):
            raise ModelGeometryError(f"outer segment {index} has zero length")
        if previous_end is not None and not _close(previous_end, start):
            raise ModelGeometryError(
                f"outer segment chain is discontinuous before segment {index}"
            )
        points.append(start)
        previous_end = end
    if previous_end is None or not _close(previous_end, points[0]):
        raise ModelGeometryError("outer segment chain is not closed")
    return tuple(points)


def _validate_polygon(points: Sequence[tuple[float, float]]) -> None:
    if len(points) < 3:
        raise ModelGeometryError("outer polygon requires at least three vertices")
    if len(set(points)) != len(points):
        raise ModelGeometryError("outer polygon contains duplicate vertices")
    area = abs(_polygon_area(points))
    if area <= _GEOMETRY_TOLERANCE_MM:
        raise ModelGeometryError("outer polygon area is zero")
    count = len(points)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first == second or first_next == second or second_next == first:
                continue
            if first == 0 and second_next == 0:
                continue
            if _segments_intersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
            ):
                raise ModelGeometryError("outer polygon self-intersects")


def _point_in_polygon(point: tuple[float, float], points: Sequence[tuple[float, float]]) -> bool:
    inside = False
    x, y = point
    previous = points[-1]
    for current in points:
        if _point_on_segment(point, previous, current):
            return False
        if (current[1] > y) != (previous[1] > y):
            crossing_x = (previous[0] - current[0]) * (y - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 0.0:
        return math.dist(point, start)
    amount = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator
    amount = min(1.0, max(0.0, amount))
    projection = (start[0] + amount * dx, start[1] + amount * dy)
    return math.dist(point, projection)


def _validate_holes(points: Sequence[tuple[float, float]], holes: Sequence[PlanHole]) -> None:
    for hole in holes:
        if not _point_in_polygon(hole.center_mm, points):
            raise ModelGeometryError(f"hole {hole.feature_id} centre lies outside the profile")
        boundary_distance = min(
            _point_segment_distance(
                hole.center_mm,
                points[index],
                points[(index + 1) % len(points)],
            )
            for index in range(len(points))
        )
        if boundary_distance <= hole.radius_mm + _GEOMETRY_TOLERANCE_MM:
            raise ModelGeometryError(
                f"hole {hole.feature_id} intersects or touches the outer profile"
            )
    for index, first in enumerate(holes):
        for second in holes[index + 1 :]:
            if math.dist(first.center_mm, second.center_mm) <= (
                first.radius_mm + second.radius_mm + _GEOMETRY_TOLERANCE_MM
            ):
                raise ModelGeometryError(
                    f"holes {first.feature_id} and {second.feature_id} overlap or touch"
                )


def _parse_canonical_json(payload: bytes, field_name: str) -> Mapping[str, Any]:
    if not isinstance(payload, bytes):
        raise ModelInputError(f"{field_name} must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelInputError(f"{field_name} is not valid UTF-8 JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise ModelInputError(f"{field_name} JSON is not canonical")
    return _mapping(value, field_name)


def _parse_graph(payload: bytes, analysis_sha256: str) -> tuple[ParameterGraph, str]:
    value = _parse_canonical_json(payload, "parameter graph")
    try:
        graph = ParameterGraph.from_dict(value)
    except (GraphError, ValueError) as exc:
        raise ModelInputError(f"parameter graph is invalid: {exc}") from exc
    graph_sha256 = hashlib.sha256(payload).hexdigest()
    if not any(
        source.kind == "m2_analysis" and source.sha256 == analysis_sha256
        for source in graph.sources.values()
    ):
        raise ModelInputError(
            "parameter graph is not bound to the selected M2 analysis SHA-256"
        )
    return graph, graph_sha256


def _resolved_override(
    resolved: Mapping[str, Any] | None,
    parameter_id: str,
    fallback: float,
) -> float:
    if resolved is None or parameter_id not in resolved:
        return fallback
    value = resolved[parameter_id]
    if value.status in {ParameterStatus.CONFLICTING, ParameterStatus.UNRESOLVED}:
        raise ModelInputError(
            f"required geometry parameter {parameter_id} is {value.status.value}"
        )
    if value.value is None:
        raise ModelInputError(f"required geometry parameter {parameter_id} has no value")
    return _number(value.value, parameter_id)


def _line_sort_key(value: Mapping[str, Any]) -> tuple[int, str]:
    feature_id = str(value.get("id", ""))
    suffix = re.search(r"(\d+)$", feature_id)
    return (int(suffix.group(1)) if suffix else 2**31 - 1, feature_id)


def build_model_plan(
    analysis_bytes: bytes,
    *,
    thickness_mm: float,
    graph_bytes: bytes | None = None,
    document_name: str = "MeasurePilotModel",
) -> FreeCADModelPlan:
    """Build and validate a deterministic M5 model plan."""

    report = _parse_canonical_json(analysis_bytes, "M2 analysis")
    if (
        report.get("format") != M2_ANALYSIS_FORMAT
        or report.get("version") != M2_ANALYSIS_VERSION
    ):
        raise ModelInputError("unsupported M2 analysis format or version")
    coordinate_system = _mapping(report.get("coordinate_system"), "coordinate_system")
    if coordinate_system.get("unit") != "mm":
        raise ModelInputError("M2 analysis coordinate unit must be 'mm'")
    unresolved = report.get("unresolved", [])
    if not isinstance(unresolved, list):
        raise ModelInputError("unresolved must be a list")
    if unresolved:
        raise ModelInputError(
            "M2 analysis contains unresolved geometry and cannot be modelled safely"
        )

    analysis_sha256 = hashlib.sha256(analysis_bytes).hexdigest()
    graph_sha256: str | None = None
    resolved: Mapping[str, Any] | None = None
    if graph_bytes is not None:
        graph, graph_sha256 = _parse_graph(graph_bytes, analysis_sha256)
        try:
            resolved = graph.evaluate_dependencies()
        except GraphError as exc:
            raise ModelInputError(f"parameter graph cannot be resolved: {exc}") from exc

    features = _mapping(report.get("features"), "features")
    line_values = _list(features.get("line_segments"), "features.line_segments")
    if len(line_values) < 3:
        raise ModelGeometryError("M5 requires at least three straight outer line features")
    line_mappings = sorted(
        (_mapping(value, "line feature") for value in line_values),
        key=_line_sort_key,
    )

    source_segments: list[PlanSegment] = []
    seen_line_ids: set[str] = set()
    for raw in line_mappings:
        try:
            feature_id = validate_identifier(raw.get("id"), "line feature ID")
        except ProvenanceError as exc:
            raise ModelInputError(str(exc)) from exc
        if feature_id in seen_line_ids:
            raise ModelInputError(f"duplicate line feature ID: {feature_id}")
        seen_line_ids.add(feature_id)
        start = _finite_point(raw.get("start_mm"), f"{feature_id}.start_mm")
        end = _finite_point(raw.get("end_mm"), f"{feature_id}.end_mm")
        start = (
            _resolved_override(resolved, f"m2.line.{feature_id}.start_x", start[0]),
            _resolved_override(resolved, f"m2.line.{feature_id}.start_y", start[1]),
        )
        end = (
            _resolved_override(resolved, f"m2.line.{feature_id}.end_x", end[0]),
            _resolved_override(resolved, f"m2.line.{feature_id}.end_y", end[1]),
        )
        source_segments.append(PlanSegment(start_mm=start, end_mm=end))

    source_points = _segments_to_points(source_segments)
    _validate_polygon(source_points)
    translation = (
        min(point[0] for point in source_points),
        min(point[1] for point in source_points),
    )

    def normalise(point: tuple[float, float]) -> tuple[float, float]:
        return (
            round(point[0] - translation[0], 9),
            round(point[1] - translation[1], 9),
        )

    outer_segments = tuple(
        PlanSegment(start_mm=normalise(segment.start_mm), end_mm=normalise(segment.end_mm))
        for segment in source_segments
    )

    hole_values = _list(features.get("circular_holes", []), "features.circular_holes")
    contour_root = report.get("contours")
    if isinstance(contour_root, Mapping):
        contour_holes = contour_root.get("holes")
        if isinstance(contour_holes, list) and len(contour_holes) != len(hole_values):
            raise ModelInputError("M5 supports only circular enclosed contours")
    holes: list[PlanHole] = []
    seen_hole_ids: set[str] = set()
    for raw_value in sorted(
        hole_values,
        key=lambda item: str(_mapping(item, "circular hole").get("id", "")),
    ):
        raw = _mapping(raw_value, "circular hole")
        try:
            feature_id = validate_identifier(raw.get("id"), "circular-hole ID")
        except ProvenanceError as exc:
            raise ModelInputError(str(exc)) from exc
        if feature_id in seen_hole_ids:
            raise ModelInputError(f"duplicate circular-hole ID: {feature_id}")
        seen_hole_ids.add(feature_id)
        center = _finite_point(raw.get("center_mm"), f"{feature_id}.center_mm")
        diameter = _positive(raw.get("diameter_mm"), f"{feature_id}.diameter_mm")
        center = (
            _resolved_override(resolved, f"m2.hole.{feature_id}.center_x", center[0]),
            _resolved_override(resolved, f"m2.hole.{feature_id}.center_y", center[1]),
        )
        diameter = _resolved_override(
            resolved,
            f"m2.hole.{feature_id}.diameter",
            diameter,
        )
        _positive(diameter, f"m2.hole.{feature_id}.diameter")
        holes.append(
            PlanHole(
                feature_id=feature_id,
                center_mm=normalise(center),
                diameter_mm=round(diameter, 9),
            )
        )

    plan = FreeCADModelPlan(
        analysis_sha256=analysis_sha256,
        graph_sha256=graph_sha256,
        translation_mm=(round(translation[0], 9), round(translation[1], 9)),
        thickness_mm=round(_positive(thickness_mm, "thickness_mm"), 9),
        outer_segments=outer_segments,
        holes=tuple(holes),
        document_name=canonical_document_name(document_name),
    )
    plan.validate()
    return plan


def _load_freecad_modules() -> tuple[Any, Any, Any]:
    try:
        app = importlib.import_module("FreeCAD")
        part = importlib.import_module("Part")
        sketcher = importlib.import_module("Sketcher")
    except ImportError as exc:
        raise FreeCADRuntimeError(
            "FreeCAD, Part, and Sketcher modules are required inside FreeCAD"
        ) from exc
    return app, part, sketcher


def _add_metadata(body: Any, plan: FreeCADModelPlan) -> None:
    if not hasattr(body, "addProperty"):
        return
    properties = (
        ("MeasurePilotPlanFormat", MODEL_PLAN_FORMAT),
        ("MeasurePilotAnalysisSHA256", plan.analysis_sha256),
        ("MeasurePilotGraphSHA256", plan.graph_sha256 or ""),
        (
            "MeasurePilotTranslationMM",
            f"{plan.translation_mm[0]:.9g},{plan.translation_mm[1]:.9g}",
        ),
    )
    for name, value in properties:
        try:
            body.addProperty("App::PropertyString", name, "MeasurePilot")
        except Exception:
            pass
        setattr(body, name, value)


def create_freecad_document(
    plan: FreeCADModelPlan,
    *,
    save_path: str | os.PathLike[str] | None = None,
    modules: tuple[Any, Any, Any] | None = None,
) -> FreeCADModelObjects:
    """Create Body/Sketch/Pad natively and optionally save an FCStd document."""

    plan.validate()
    app, part, sketcher = modules or _load_freecad_modules()
    document = None
    transaction_open = False
    saved: Path | None = None
    try:
        document = app.newDocument(plan.document_name)
        if hasattr(document, "openTransaction"):
            document.openTransaction("Create MeasurePilot model")
            transaction_open = True

        body = document.addObject("PartDesign::Body", plan.body_name)
        sketch = document.addObject("Sketcher::SketchObject", plan.sketch_name)
        body.addObject(sketch)

        line_geometry = [
            part.LineSegment(
                app.Vector(segment.start_mm[0], segment.start_mm[1], 0.0),
                app.Vector(segment.end_mm[0], segment.end_mm[1], 0.0),
            )
            for segment in plan.outer_segments
        ]
        sketch.addGeometry(line_geometry, False)
        constraints = [
            sketcher.Constraint(
                "Coincident",
                index,
                2,
                (index + 1) % len(plan.outer_segments),
                1,
            )
            for index in range(len(plan.outer_segments))
        ]
        sketch.addConstraint(constraints)

        for hole in plan.holes:
            sketch.addGeometry(
                part.Circle(
                    app.Vector(hole.center_mm[0], hole.center_mm[1], 0.0),
                    app.Vector(0.0, 0.0, 1.0),
                    hole.radius_mm,
                ),
                False,
            )

        pad = document.addObject("PartDesign::Pad", plan.pad_name)
        body.addObject(pad)
        pad.Profile = sketch
        pad.Length = plan.thickness_mm
        if hasattr(pad, "Type"):
            pad.Type = 0
        if hasattr(pad, "Reversed"):
            pad.Reversed = False
        _add_metadata(body, plan)
        if hasattr(sketch, "Label"):
            sketch.Label = "MeasurePilot profile and through-holes"
        if hasattr(pad, "Label"):
            pad.Label = "MeasurePilot thickness"

        document.recompute()
        is_valid = getattr(pad, "isValid", None)
        if callable(is_valid) and not is_valid():
            raise FreeCADRuntimeError("FreeCAD rejected the generated PartDesign pad")
        shape = getattr(pad, "Shape", None)
        is_null = getattr(shape, "isNull", None)
        if callable(is_null) and is_null():
            raise FreeCADRuntimeError("generated PartDesign pad has a null shape")

        if save_path is not None:
            saved = Path(save_path)
            if saved.suffix.lower() != ".fcstd":
                raise FreeCADRuntimeError("save path must use the .FCStd extension")
            saved.parent.mkdir(parents=True, exist_ok=True)
            document.saveAs(str(saved))

        if transaction_open and hasattr(document, "commitTransaction"):
            document.commitTransaction()
            transaction_open = False
        return FreeCADModelObjects(
            document=document,
            body=body,
            sketch=sketch,
            pad=pad,
            saved_path=saved,
        )
    except Exception as exc:
        if document is not None and transaction_open and hasattr(document, "abortTransaction"):
            try:
                document.abortTransaction()
            except Exception:
                pass
        if document is not None and hasattr(app, "closeDocument"):
            try:
                app.closeDocument(document.Name)
            except Exception:
                pass
        if isinstance(exc, FreeCADModelError):
            raise
        raise FreeCADRuntimeError(f"native FreeCAD model creation failed: {exc}") from exc
