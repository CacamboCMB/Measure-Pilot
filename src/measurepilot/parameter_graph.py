"""Versioned deterministic parameter graph for MeasurePilot M3."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .constraints import (
    ConstraintError,
    Dependency,
    LinearConstraint,
    dependency_order,
    solve_linear_constraints,
)
from .provenance import (
    CONFLICT_NUMERICAL_FLOOR,
    CONFLICT_RULE_VERSION,
    CONFLICT_SIGMA,
    EvidenceSource,
    Observation,
    ParameterStatus,
    ProvenanceError,
    QuantityKind,
    ResolvedValue,
    UNIT_BY_QUANTITY,
    canonical_json_bytes,
    finite_number,
    positive_number,
    resolve_observations,
    validate_identifier,
    validate_supersession_graph,
)


GRAPH_FORMAT = "measurepilot-parameter-graph"
GRAPH_VERSION = 1
M2_ANALYSIS_FORMAT = "measurepilot-planar-analysis"
M2_ANALYSIS_VERSION = 1
MAX_IMAGE_UNCERTAINTY_MM = 2.0
MIN_IMAGE_UNCERTAINTY_MM = 1e-6


class GraphError(ValueError):
    """Base class for invalid M3 graph state."""


class GraphFormatError(GraphError):
    """Raised when a graph or imported analysis document is malformed."""


class CanonicalGraphError(GraphFormatError):
    """Raised when persisted JSON is not the canonical representation."""


class QuantityMismatchError(GraphError):
    """Raised when an equation mixes incompatible scalar quantity kinds."""


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """Typed scalar parameter definition."""

    parameter_id: str
    quantity: QuantityKind
    label: str = ""

    @classmethod
    def create(
        cls,
        parameter_id: str,
        quantity: QuantityKind | str,
        *,
        label: str = "",
    ) -> "ParameterDefinition":
        try:
            quantity_value = QuantityKind(quantity)
        except (TypeError, ValueError) as exc:
            raise QuantityMismatchError(f"unsupported quantity kind: {quantity}") from exc
        definition = cls(
            parameter_id=parameter_id,
            quantity=quantity_value,
            label=label,
        )
        definition.validate()
        return definition

    @property
    def unit(self) -> str:
        return UNIT_BY_QUANTITY[self.quantity]

    def validate(self) -> None:
        try:
            validate_identifier(self.parameter_id, "parameter_id")
        except ProvenanceError as exc:
            raise GraphError(str(exc)) from exc
        if not isinstance(self.quantity, QuantityKind):
            raise QuantityMismatchError("parameter quantity is unsupported")
        if not isinstance(self.label, str) or len(self.label) > 200:
            raise GraphError("parameter label must contain at most 200 characters")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "label": self.label,
            "parameter_id": self.parameter_id,
            "quantity": self.quantity.value,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterDefinition":
        if not isinstance(value, Mapping):
            raise GraphFormatError("parameter definition must be an object")
        definition = cls.create(
            value.get("parameter_id"),  # type: ignore[arg-type]
            value.get("quantity"),  # type: ignore[arg-type]
            label=value.get("label", ""),  # type: ignore[arg-type]
        )
        if value.get("unit") != definition.unit:
            raise QuantityMismatchError(
                f"unit for {definition.parameter_id} must be {definition.unit!r}"
            )
        return definition


@dataclass(slots=True)
class ParameterGraph:
    """Append-only provenance graph plus explicit linear relationships."""

    sources: dict[str, EvidenceSource] = field(default_factory=dict)
    parameters: dict[str, ParameterDefinition] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    constraints: list[LinearConstraint] = field(default_factory=list)

    def validate(self) -> None:
        for source_id, source in self.sources.items():
            source.validate()
            if source_id != source.source_id:
                raise GraphError("source mapping key does not match source_id")
        for parameter_id, definition in self.parameters.items():
            definition.validate()
            if parameter_id != definition.parameter_id:
                raise GraphError("parameter mapping key does not match parameter_id")

        observation_ids: set[str] = set()
        for observation in self.observations:
            observation.validate()
            if observation.observation_id in observation_ids:
                raise GraphError(
                    f"duplicate observation_id: {observation.observation_id}"
                )
            observation_ids.add(observation.observation_id)
            if observation.parameter_id not in self.parameters:
                raise GraphError(
                    f"observation parameter is undefined: {observation.parameter_id}"
                )
            if observation.source_id not in self.sources:
                raise GraphError(
                    f"observation source is undefined: {observation.source_id}"
                )
        try:
            validate_supersession_graph(self.observations)
        except ProvenanceError as exc:
            raise GraphError(str(exc)) from exc

        dependency_ids: set[str] = set()
        for dependency in self.dependencies:
            dependency.validate()
            if dependency.dependency_id in dependency_ids:
                raise GraphError(
                    f"duplicate dependency_id: {dependency.dependency_id}"
                )
            dependency_ids.add(dependency.dependency_id)
            target_quantity = self._parameter_quantity(
                dependency.target_parameter_id
            )
            for source_id, _coefficient in dependency.coefficients:
                if self._parameter_quantity(source_id) != target_quantity:
                    raise QuantityMismatchError(
                        "dependency mixes quantity kinds: "
                        f"{dependency.target_parameter_id} and {source_id}"
                    )
        try:
            dependency_order(self.dependencies, self.parameters)
        except ConstraintError as exc:
            raise GraphError(str(exc)) from exc

        constraint_ids: set[str] = set()
        for constraint in self.constraints:
            constraint.validate()
            if constraint.constraint_id in constraint_ids:
                raise GraphError(
                    f"duplicate constraint_id: {constraint.constraint_id}"
                )
            constraint_ids.add(constraint.constraint_id)
            quantities = {
                self._parameter_quantity(parameter_id)
                for parameter_id, _coefficient in constraint.coefficients
            }
            if len(quantities) != 1:
                raise QuantityMismatchError(
                    f"constraint {constraint.constraint_id} mixes quantity kinds"
                )

    def _parameter_quantity(self, parameter_id: str) -> QuantityKind:
        definition = self.parameters.get(parameter_id)
        if definition is None:
            raise GraphError(f"undefined parameter: {parameter_id}")
        return definition.quantity

    def ensure_parameter(
        self,
        parameter_id: str,
        quantity: QuantityKind | str,
        *,
        label: str = "",
    ) -> ParameterDefinition:
        definition = ParameterDefinition.create(parameter_id, quantity, label=label)
        existing = self.parameters.get(parameter_id)
        if existing is not None:
            if existing.quantity != definition.quantity:
                raise QuantityMismatchError(
                    f"parameter {parameter_id} already uses {existing.quantity.value}"
                )
            if label and existing.label and existing.label != label:
                raise GraphError(f"parameter {parameter_id} has a different label")
            return existing
        self.parameters[parameter_id] = definition
        return definition

    def add_source(self, source: EvidenceSource) -> EvidenceSource:
        source.validate()
        existing = self.sources.get(source.source_id)
        if existing is not None and existing != source:
            raise GraphError(f"source ID collision: {source.source_id}")
        self.sources[source.source_id] = source
        return source

    def add_observation(self, observation: Observation) -> Observation:
        observation.validate()
        if observation.parameter_id not in self.parameters:
            raise GraphError(
                f"observation parameter is undefined: {observation.parameter_id}"
            )
        if observation.source_id not in self.sources:
            raise GraphError(
                f"observation source is undefined: {observation.source_id}"
            )
        existing = next(
            (
                item
                for item in self.observations
                if item.observation_id == observation.observation_id
            ),
            None,
        )
        if existing is not None:
            if existing != observation:
                raise GraphError(
                    f"observation ID collision: {observation.observation_id}"
                )
            return existing
        candidate = [*self.observations, observation]
        try:
            validate_supersession_graph(candidate)
        except ProvenanceError as exc:
            raise GraphError(str(exc)) from exc
        self.observations.append(observation)
        return observation

    def add_dependency(self, dependency: Dependency) -> Dependency:
        dependency.validate()
        existing = next(
            (
                item
                for item in self.dependencies
                if item.dependency_id == dependency.dependency_id
            ),
            None,
        )
        if existing is not None:
            return existing
        candidate = ParameterGraph(
            sources=dict(self.sources),
            parameters=dict(self.parameters),
            observations=list(self.observations),
            dependencies=[*self.dependencies, dependency],
            constraints=list(self.constraints),
        )
        candidate.validate()
        self.dependencies.append(dependency)
        return dependency

    def add_constraint(self, constraint: LinearConstraint) -> LinearConstraint:
        constraint.validate()
        existing = next(
            (
                item
                for item in self.constraints
                if item.constraint_id == constraint.constraint_id
            ),
            None,
        )
        if existing is not None:
            return existing
        candidate = ParameterGraph(
            sources=dict(self.sources),
            parameters=dict(self.parameters),
            observations=list(self.observations),
            dependencies=list(self.dependencies),
            constraints=[*self.constraints, constraint],
        )
        candidate.validate()
        self.constraints.append(constraint)
        return constraint

    def append_measurement(
        self,
        *,
        parameter_id: str,
        quantity: QuantityKind | str,
        value: float,
        uncertainty: float,
        status: ParameterStatus | str = ParameterStatus.MEASURED,
        supersedes: str | None = None,
        note: str = "",
    ) -> Observation:
        try:
            status_value = ParameterStatus(status)
            quantity_value = QuantityKind(quantity)
            numeric_value = finite_number(value, "measurement value")
            numeric_uncertainty = positive_number(
                uncertainty,
                "measurement uncertainty",
            )
        except (TypeError, ValueError, ProvenanceError) as exc:
            raise GraphError(str(exc)) from exc
        if status_value not in {
            ParameterStatus.MEASURED,
            ParameterStatus.ASSUMED,
            ParameterStatus.LOCKED,
        }:
            raise GraphError(
                "physical append status must be MEASURED, ASSUMED, or LOCKED"
            )
        if not isinstance(note, str) or len(note) > 500:
            raise GraphError("measurement note must contain at most 500 characters")
        if supersedes is not None:
            old = next(
                (
                    item
                    for item in self.observations
                    if item.observation_id == supersedes
                ),
                None,
            )
            if old is None:
                raise GraphError(
                    f"supersedes references unknown observation: {supersedes}"
                )
            if old.parameter_id != parameter_id:
                raise GraphError(
                    "a correction may supersede only an observation of the same parameter"
                )

        definition = ParameterDefinition.create(parameter_id, quantity_value)
        measurement_payload = {
            "note": note,
            "parameter_id": parameter_id,
            "quantity": quantity_value.value,
            "status": status_value.value,
            "supersedes": supersedes,
            "uncertainty": numeric_uncertainty,
            "value": numeric_value,
        }
        measurement_bytes = canonical_json_bytes(measurement_payload)
        source = EvidenceSource.create(
            kind="physical_measurement",
            sha256=hashlib.sha256(measurement_bytes).hexdigest(),
            description="Physical scalar measurement",
            metadata={
                "parameter_id": parameter_id,
                "supersedes": supersedes,
            },
        )
        observation = Observation.create(
            parameter_id=parameter_id,
            value=numeric_value,
            uncertainty=numeric_uncertainty,
            status=status_value,
            source_id=source.source_id,
            supersedes=supersedes,
            note=note,
        )

        candidate = ParameterGraph(
            sources=dict(self.sources),
            parameters=dict(self.parameters),
            observations=list(self.observations),
            dependencies=list(self.dependencies),
            constraints=list(self.constraints),
        )
        candidate.ensure_parameter(
            definition.parameter_id,
            definition.quantity,
            label=definition.label,
        )
        candidate.add_source(source)
        candidate.add_observation(observation)
        candidate.validate()
        self.sources = candidate.sources
        self.parameters = candidate.parameters
        self.observations = candidate.observations
        return observation

    def correct_measurement(
        self,
        observation_id: str,
        *,
        value: float,
        uncertainty: float,
        status: ParameterStatus | str = ParameterStatus.MEASURED,
        note: str = "",
    ) -> Observation:
        old = next(
            (
                item
                for item in self.observations
                if item.observation_id == observation_id
            ),
            None,
        )
        if old is None:
            raise GraphError(f"cannot correct unknown observation: {observation_id}")
        definition = self.parameters[old.parameter_id]
        return self.append_measurement(
            parameter_id=old.parameter_id,
            quantity=definition.quantity,
            value=value,
            uncertainty=uncertainty,
            status=status,
            supersedes=old.observation_id,
            note=note,
        )

    def resolve_observation_values(self) -> dict[str, ResolvedValue]:
        self.validate()
        return {
            parameter_id: resolve_observations(
                parameter_id,
                self.observations,
            )
            for parameter_id in sorted(self.parameters)
        }

    def evaluate_dependencies(
        self,
        initial: Mapping[str, ResolvedValue] | None = None,
    ) -> dict[str, ResolvedValue]:
        self.validate()
        resolved = dict(initial or self.resolve_observation_values())
        try:
            ordered = dependency_order(self.dependencies, self.parameters)
        except ConstraintError as exc:
            raise GraphError(str(exc)) from exc

        for dependency in ordered:
            source_values = [
                resolved[source_id]
                for source_id, _coefficient in dependency.coefficients
            ]
            if any(
                value.status == ParameterStatus.CONFLICTING
                for value in source_values
            ):
                resolved[dependency.target_parameter_id] = ResolvedValue(
                    parameter_id=dependency.target_parameter_id,
                    status=ParameterStatus.UNRESOLVED,
                    value=None,
                    uncertainty=None,
                    reason="dependency source is conflicting",
                )
                continue
            if any(value.value is None for value in source_values):
                resolved[dependency.target_parameter_id] = ResolvedValue(
                    parameter_id=dependency.target_parameter_id,
                    status=ParameterStatus.UNRESOLVED,
                    value=None,
                    uncertainty=None,
                    reason="dependency source is unresolved",
                )
                continue

            value_by_id = {
                value.parameter_id: value
                for value in source_values
            }
            derived_value = dependency.constant + math.fsum(
                coefficient * value_by_id[source_id].value  # type: ignore[operator]
                for source_id, coefficient in dependency.coefficients
            )
            derived_uncertainty = math.sqrt(
                dependency.tolerance * dependency.tolerance
                + math.fsum(
                    (
                        coefficient
                        * (value_by_id[source_id].uncertainty or 0.0)
                    )
                    ** 2
                    for source_id, coefficient in dependency.coefficients
                )
            )
            derived_uncertainty = max(derived_uncertainty, 1e-12)
            current = resolved[dependency.target_parameter_id]
            if current.status == ParameterStatus.CONFLICTING:
                continue
            if current.value is not None:
                threshold = (
                    CONFLICT_SIGMA
                    * math.hypot(
                        current.uncertainty or 0.0,
                        derived_uncertainty,
                    )
                    + dependency.tolerance
                    + CONFLICT_NUMERICAL_FLOOR
                )
                if abs(current.value - derived_value) > threshold:
                    resolved[dependency.target_parameter_id] = ResolvedValue(
                        parameter_id=dependency.target_parameter_id,
                        status=ParameterStatus.CONFLICTING,
                        value=None,
                        uncertainty=None,
                        active_observation_ids=current.active_observation_ids,
                        reason="active observation conflicts with derived dependency",
                    )
                continue
            resolved[dependency.target_parameter_id] = ResolvedValue(
                parameter_id=dependency.target_parameter_id,
                status=ParameterStatus.DERIVED,
                value=derived_value,
                uncertainty=derived_uncertainty,
                reason=f"evaluated dependency {dependency.dependency_id}",
            )
        return resolved

    def evaluate(self) -> dict[str, ResolvedValue]:
        """Resolve observations, dependencies, and explicit linear constraints."""

        resolved = self.evaluate_dependencies()
        if not self.constraints:
            return resolved
        referenced = {
            parameter_id
            for constraint in self.constraints
            for parameter_id, _coefficient in constraint.coefficients
        }
        conflicting = sorted(
            parameter_id
            for parameter_id in referenced
            if resolved[parameter_id].status == ParameterStatus.CONFLICTING
        )
        if conflicting:
            raise GraphError(
                "cannot solve constraints with conflicting parameters: "
                + ", ".join(conflicting)
            )
        known_values = {
            parameter_id: value.value
            for parameter_id, value in resolved.items()
            if parameter_id in referenced and value.value is not None
        }
        try:
            solution = solve_linear_constraints(
                self.constraints,
                parameter_ids=self.parameters,
                known_values=known_values,
            )
        except ConstraintError as exc:
            raise GraphError(str(exc)) from exc
        for solved in solution.values:
            current = resolved[solved.parameter_id]
            if current.value is not None:
                continue
            resolved[solved.parameter_id] = ResolvedValue(
                parameter_id=solved.parameter_id,
                status=ParameterStatus.DERIVED,
                value=solved.value,
                uncertainty=solved.uncertainty,
                reason="solved from explicit linear constraints",
            )
        return resolved

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "conflict_rule_version": CONFLICT_RULE_VERSION,
            "constraints": [
                item.to_dict()
                for item in sorted(
                    self.constraints,
                    key=lambda value: value.constraint_id,
                )
            ],
            "dependencies": [
                item.to_dict()
                for item in sorted(
                    self.dependencies,
                    key=lambda value: value.dependency_id,
                )
            ],
            "format": GRAPH_FORMAT,
            "observations": [
                item.to_dict()
                for item in sorted(
                    self.observations,
                    key=lambda value: value.observation_id,
                )
            ],
            "parameters": [
                self.parameters[parameter_id].to_dict()
                for parameter_id in sorted(self.parameters)
            ],
            "sources": [
                self.sources[source_id].to_dict()
                for source_id in sorted(self.sources)
            ],
            "version": GRAPH_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterGraph":
        if not isinstance(value, Mapping):
            raise GraphFormatError("parameter graph must be an object")
        expected_keys = {
            "conflict_rule_version",
            "constraints",
            "dependencies",
            "format",
            "observations",
            "parameters",
            "sources",
            "version",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            unexpected = sorted(set(value) - expected_keys)
            raise GraphFormatError(
                f"graph keys mismatch; missing={missing}, unexpected={unexpected}"
            )
        if value.get("format") != GRAPH_FORMAT or value.get("version") != GRAPH_VERSION:
            raise GraphFormatError("unsupported parameter graph format or version")
        if value.get("conflict_rule_version") != CONFLICT_RULE_VERSION:
            raise GraphFormatError("unsupported conflict rule version")

        def require_list(name: str) -> list[Any]:
            result = value.get(name)
            if not isinstance(result, list):
                raise GraphFormatError(f"{name} must be a list")
            return result

        sources = [EvidenceSource.from_dict(item) for item in require_list("sources")]
        parameters = [
            ParameterDefinition.from_dict(item)
            for item in require_list("parameters")
        ]
        observations = [
            Observation.from_dict(item)
            for item in require_list("observations")
        ]
        dependencies = [
            Dependency.from_dict(item)
            for item in require_list("dependencies")
        ]
        constraints = [
            LinearConstraint.from_dict(item)
            for item in require_list("constraints")
        ]
        graph = cls(
            sources={item.source_id: item for item in sources},
            parameters={item.parameter_id: item for item in parameters},
            observations=observations,
            dependencies=dependencies,
            constraints=constraints,
        )
        if len(graph.sources) != len(sources):
            raise GraphFormatError("duplicate source IDs")
        if len(graph.parameters) != len(parameters):
            raise GraphFormatError("duplicate parameter IDs")
        graph.validate()
        return graph


def save_graph(graph: ParameterGraph, destination: str | os.PathLike[str]) -> Path:
    """Atomically persist canonical graph JSON."""

    graph.validate()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.canonical_bytes()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def load_graph(source: str | os.PathLike[str]) -> ParameterGraph:
    """Load and require the exact canonical graph representation."""

    path = Path(source)
    try:
        payload = path.read_bytes()
        decoded = payload.decode("utf-8")
        value = json.loads(decoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphFormatError(f"cannot read parameter graph: {path}") from exc
    graph = ParameterGraph.from_dict(value)
    if graph.canonical_bytes() != payload:
        raise CanonicalGraphError("parameter graph JSON is not canonical")
    return graph


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphFormatError(f"{field_name} must be an object")
    return value


def _list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise GraphFormatError(f"{field_name} must be a list")
    return value


def _point(value: object, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise GraphFormatError(f"{field_name} must contain two coordinates")
    try:
        return (
            finite_number(value[0], f"{field_name}[0]"),
            finite_number(value[1], f"{field_name}[1]"),
        )
    except ProvenanceError as exc:
        raise GraphFormatError(str(exc)) from exc


def _bounded_uncertainty(*candidates: float) -> float:
    finite = [
        abs(finite_number(value, "image uncertainty candidate"))
        for value in candidates
    ]
    return min(
        MAX_IMAGE_UNCERTAINTY_MM,
        max(MIN_IMAGE_UNCERTAINTY_MM, *finite),
    )


def _add_imported_observation(
    graph: ParameterGraph,
    source: EvidenceSource,
    *,
    parameter_id: str,
    value: float,
    uncertainty: float,
    label: str,
    note: str,
) -> None:
    graph.ensure_parameter(parameter_id, QuantityKind.LENGTH, label=label)
    observation = Observation.create(
        parameter_id=parameter_id,
        value=value,
        uncertainty=uncertainty,
        status=ParameterStatus.ESTIMATED,
        source_id=source.source_id,
        note=note,
    )
    graph.add_observation(observation)


def import_m2_analysis(
    report_bytes: bytes,
    *,
    graph: ParameterGraph | None = None,
) -> ParameterGraph:
    """Import canonical M2 line and circular-hole evidence with exact provenance."""

    if not isinstance(report_bytes, bytes):
        raise GraphFormatError("M2 analysis input must be bytes")
    try:
        value = json.loads(report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphFormatError("M2 analysis is not valid UTF-8 JSON") from exc
    if canonical_json_bytes(value) != report_bytes:
        raise CanonicalGraphError("M2 analysis JSON is not canonical")
    report = _mapping(value, "M2 analysis")
    if (
        report.get("format") != M2_ANALYSIS_FORMAT
        or report.get("version") != M2_ANALYSIS_VERSION
    ):
        raise GraphFormatError("unsupported M2 analysis format or version")
    coordinate_system = _mapping(
        report.get("coordinate_system"),
        "coordinate_system",
    )
    if coordinate_system.get("unit") != "mm":
        raise QuantityMismatchError("M2 analysis coordinate unit must be 'mm'")
    configuration = _mapping(report.get("configuration"), "configuration")
    segmentation = _mapping(
        configuration.get("segmentation"),
        "configuration.segmentation",
    )
    try:
        px_per_mm = positive_number(
            segmentation.get("px_per_mm"),
            "configuration.segmentation.px_per_mm",
        )
    except ProvenanceError as exc:
        raise GraphFormatError(str(exc)) from exc
    contours = _mapping(report.get("contours"), "contours")
    outer = _mapping(contours.get("outer"), "contours.outer")
    try:
        simplification_rms = abs(
            finite_number(
                outer.get("simplification_rms_mm", 0.0),
                "contours.outer.simplification_rms_mm",
            )
        )
    except ProvenanceError as exc:
        raise GraphFormatError(str(exc)) from exc
    features = _mapping(report.get("features"), "features")
    lines = _list(features.get("line_segments"), "features.line_segments")
    holes = _list(features.get("circular_holes"), "features.circular_holes")

    result = graph if graph is not None else ParameterGraph()
    source = EvidenceSource.create(
        kind="m2_analysis",
        sha256=hashlib.sha256(report_bytes).hexdigest(),
        description="Canonical MeasurePilot M2 planar analysis",
        metadata={
            "format": M2_ANALYSIS_FORMAT,
            "version": M2_ANALYSIS_VERSION,
        },
    )
    result.add_source(source)
    pixel_uncertainty = 0.5 / px_per_mm
    line_uncertainty = _bounded_uncertainty(
        pixel_uncertainty,
        simplification_rms,
    )

    for raw_line in sorted(
        lines,
        key=lambda item: str(_mapping(item, "line feature").get("id")),
    ):
        line = _mapping(raw_line, "line feature")
        try:
            feature_id = validate_identifier(line.get("id"), "line feature ID")
            start_x, start_y = _point(line.get("start_mm"), "line.start_mm")
            end_x, end_y = _point(line.get("end_mm"), "line.end_mm")
            length = positive_number(line.get("length_mm"), "line.length_mm")
        except ProvenanceError as exc:
            raise GraphFormatError(str(exc)) from exc
        prefix = f"m2.line.{feature_id}"
        imported = (
            (f"{prefix}.start_x", start_x, "Line start X"),
            (f"{prefix}.start_y", start_y, "Line start Y"),
            (f"{prefix}.end_x", end_x, "Line end X"),
            (f"{prefix}.end_y", end_y, "Line end Y"),
            (f"{prefix}.length", length, "Line length"),
        )
        for parameter_id, parameter_value, label in imported:
            _add_imported_observation(
                result,
                source,
                parameter_id=parameter_id,
                value=parameter_value,
                uncertainty=line_uncertainty,
                label=label,
                note=f"imported from M2 line feature {feature_id}",
            )

    for raw_hole in sorted(
        holes,
        key=lambda item: str(_mapping(item, "circular hole").get("id")),
    ):
        hole = _mapping(raw_hole, "circular hole")
        try:
            feature_id = validate_identifier(hole.get("id"), "circular-hole ID")
            center_x, center_y = _point(hole.get("center_mm"), "hole.center_mm")
            diameter = positive_number(hole.get("diameter_mm"), "hole.diameter_mm")
            fit_residual = abs(
                finite_number(
                    hole.get("fit_residual_mm", 0.0),
                    "hole.fit_residual_mm",
                )
            )
        except ProvenanceError as exc:
            raise GraphFormatError(str(exc)) from exc
        center_uncertainty = _bounded_uncertainty(
            pixel_uncertainty,
            fit_residual,
        )
        diameter_uncertainty = _bounded_uncertainty(
            pixel_uncertainty,
            fit_residual * 2.0,
        )
        prefix = f"m2.hole.{feature_id}"
        imported = (
            (
                f"{prefix}.center_x",
                center_x,
                center_uncertainty,
                "Hole centre X",
            ),
            (
                f"{prefix}.center_y",
                center_y,
                center_uncertainty,
                "Hole centre Y",
            ),
            (
                f"{prefix}.diameter",
                diameter,
                diameter_uncertainty,
                "Hole diameter",
            ),
        )
        for parameter_id, parameter_value, uncertainty, label in imported:
            _add_imported_observation(
                result,
                source,
                parameter_id=parameter_id,
                value=parameter_value,
                uncertainty=uncertainty,
                label=label,
                note=f"imported from M2 circular-hole feature {feature_id}",
            )
    result.validate()
    return result


def inspect_graph(
    graph: ParameterGraph,
    *,
    parameter_id: str | None = None,
) -> dict[str, Any]:
    """Return deterministic human/tool-readable graph state."""

    graph.validate()
    resolved = graph.evaluate()
    if parameter_id is not None:
        try:
            validate_identifier(parameter_id, "parameter_id")
        except ProvenanceError as exc:
            raise GraphError(str(exc)) from exc
        if parameter_id not in graph.parameters:
            raise GraphError(f"unknown parameter: {parameter_id}")
        return {
            "definition": graph.parameters[parameter_id].to_dict(),
            "observations": [
                item.to_dict()
                for item in sorted(
                    (
                        item
                        for item in graph.observations
                        if item.parameter_id == parameter_id
                    ),
                    key=lambda value: value.observation_id,
                )
            ],
            "resolution": resolved[parameter_id].to_dict(),
        }
    return {
        "counts": {
            "constraints": len(graph.constraints),
            "dependencies": len(graph.dependencies),
            "observations": len(graph.observations),
            "parameters": len(graph.parameters),
            "sources": len(graph.sources),
        },
        "format": GRAPH_FORMAT,
        "parameters": {
            parameter_id: resolved[parameter_id].to_dict()
            for parameter_id in sorted(resolved)
        },
        "version": GRAPH_VERSION,
    }
