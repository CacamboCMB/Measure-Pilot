"""Versioned parameter, observation, provenance, and constraint graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .constraints import (
    ConstraintSolution,
    LinearConstraint,
    solve_linear_constraints,
    validate_dependency_graph,
)
from .provenance import (
    EvidenceSource,
    GraphValidationError,
    Observation,
    ParameterStatus,
    QuantityKind,
    UNITS_BY_KIND,
    canonical_json_bytes,
    observations_compatible,
)

GRAPH_FORMAT = "measurepilot-parameter-graph"
GRAPH_VERSION = 1
CONFLICT_SIGMA = 3.0

_STATUS_PRIORITY: Mapping[ParameterStatus, int] = {
    ParameterStatus.ASSUMED: 1,
    ParameterStatus.ESTIMATED: 2,
    ParameterStatus.DERIVED: 3,
    ParameterStatus.MEASURED: 4,
    ParameterStatus.LOCKED: 5,
}


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    parameter_id: str
    kind: QuantityKind
    unit: str
    label: str
    dependencies: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        parameter_id: str,
        *,
        kind: QuantityKind,
        label: str | None = None,
        dependencies: Iterable[str] = (),
    ) -> "ParameterRecord":
        return cls(
            parameter_id=parameter_id,
            kind=kind,
            unit=UNITS_BY_KIND[kind],
            label=label or parameter_id,
            dependencies=tuple(sorted(set(map(str, dependencies)))),
        ).validated()

    def validated(self) -> "ParameterRecord":
        if not isinstance(self.parameter_id, str) or not self.parameter_id:
            raise GraphValidationError("parameter_id must be non-empty")
        if self.unit != UNITS_BY_KIND[self.kind]:
            raise GraphValidationError(
                f"parameter {self.parameter_id} must use unit {UNITS_BY_KIND[self.kind]}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise GraphValidationError("parameter label must be non-empty")
        if self.parameter_id in self.dependencies:
            raise GraphValidationError(
                f"parameter {self.parameter_id} cannot depend on itself"
            )
        if any(not dependency for dependency in self.dependencies):
            raise GraphValidationError("dependency IDs must be non-empty")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "dependencies": list(self.dependencies),
            "kind": self.kind.value,
            "label": self.label,
            "parameter_id": self.parameter_id,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ResolvedParameter:
    parameter_id: str
    status: ParameterStatus
    value: float | None
    uncertainty: float | None
    active_observation_ids: tuple[str, ...]
    selected_observation_ids: tuple[str, ...]
    ignored_active_observation_ids: tuple[str, ...]
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_observation_ids": list(self.active_observation_ids),
            "ignored_active_observation_ids": list(
                self.ignored_active_observation_ids
            ),
            "message": self.message,
            "parameter_id": self.parameter_id,
            "selected_observation_ids": list(self.selected_observation_ids),
            "status": self.status.value,
            "uncertainty": (
                round(self.uncertainty, 12)
                if self.uncertainty is not None
                else None
            ),
            "value": round(self.value, 12) if self.value is not None else None,
        }


@dataclass(slots=True)
class ParameterGraph:
    revision: int = 0
    sources: dict[str, EvidenceSource] = field(default_factory=dict)
    parameters: dict[str, ParameterRecord] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    constraints: dict[str, LinearConstraint] = field(default_factory=dict)

    def add_source(self, source: EvidenceSource) -> str:
        source.validated()
        existing = self.sources.get(source.source_id)
        if existing is not None and existing != source:
            raise GraphValidationError(
                f"source ID {source.source_id} refers to different content"
            )
        self.sources[source.source_id] = source
        return source.source_id

    def add_parameter(self, parameter: ParameterRecord) -> str:
        parameter.validated()
        existing = self.parameters.get(parameter.parameter_id)
        if existing is not None and existing != parameter:
            raise GraphValidationError(
                f"parameter {parameter.parameter_id} already has a different definition"
            )
        self.parameters[parameter.parameter_id] = parameter
        return parameter.parameter_id

    def ensure_parameter(
        self,
        parameter_id: str,
        *,
        kind: QuantityKind,
        label: str | None = None,
        dependencies: Iterable[str] = (),
    ) -> ParameterRecord:
        requested = ParameterRecord.create(
            parameter_id,
            kind=kind,
            label=label,
            dependencies=dependencies,
        )
        self.add_parameter(requested)
        return self.parameters[parameter_id]

    def _observation_by_id(self, observation_id: str) -> Observation:
        for observation in self.observations:
            if observation.observation_id == observation_id:
                return observation
        raise GraphValidationError(f"unknown observation ID {observation_id}")

    def add_observation(
        self,
        parameter_id: str,
        *,
        value: float,
        uncertainty: float,
        status: ParameterStatus,
        source_id: str,
        supersedes: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Observation:
        if parameter_id not in self.parameters:
            raise GraphValidationError(f"unknown parameter {parameter_id}")
        if source_id not in self.sources:
            raise GraphValidationError(f"unknown evidence source {source_id}")
        if supersedes is not None:
            prior = self._observation_by_id(supersedes)
            if prior.parameter_id != parameter_id:
                raise GraphValidationError(
                    "an observation may supersede only the same parameter"
                )
            if any(
                observation.supersedes == supersedes
                for observation in self.observations
            ):
                raise GraphValidationError(
                    f"observation {supersedes} is already superseded"
                )
        revision = self.revision + 1
        observation = Observation.create(
            parameter_id=parameter_id,
            value=value,
            uncertainty=uncertainty,
            status=status,
            source_id=source_id,
            revision=revision,
            supersedes=supersedes,
            metadata=metadata,
        )
        if any(
            existing.observation_id == observation.observation_id
            for existing in self.observations
        ):
            raise GraphValidationError(
                f"duplicate observation ID {observation.observation_id}"
            )
        self.observations.append(observation)
        self.revision = revision
        return observation

    def add_constraint(self, constraint: LinearConstraint) -> str:
        constraint.validated()
        for parameter_id in constraint.coefficients:
            if parameter_id not in self.parameters:
                raise GraphValidationError(
                    f"constraint references unknown parameter {parameter_id}"
                )
            if self.parameters[parameter_id].kind is not constraint.kind:
                raise GraphValidationError(
                    f"constraint kind does not match parameter {parameter_id}"
                )
        existing = self.constraints.get(constraint.constraint_id)
        if existing is not None and existing != constraint:
            raise GraphValidationError(
                f"constraint ID {constraint.constraint_id} has different content"
            )
        self.constraints[constraint.constraint_id] = constraint
        return constraint.constraint_id

    def active_observations(self, parameter_id: str) -> tuple[Observation, ...]:
        if parameter_id not in self.parameters:
            raise GraphValidationError(f"unknown parameter {parameter_id}")
        superseded = {
            observation.supersedes
            for observation in self.observations
            if observation.supersedes is not None
        }
        return tuple(
            observation
            for observation in self.observations
            if observation.parameter_id == parameter_id
            and observation.observation_id not in superseded
        )

    def resolve_parameter(self, parameter_id: str) -> ResolvedParameter:
        active = self.active_observations(parameter_id)
        active_ids = tuple(observation.observation_id for observation in active)
        if not active:
            return ResolvedParameter(
                parameter_id=parameter_id,
                status=ParameterStatus.UNRESOLVED,
                value=None,
                uncertainty=None,
                active_observation_ids=(),
                selected_observation_ids=(),
                ignored_active_observation_ids=(),
                message="no active observation",
            )

        selected_priority = max(_STATUS_PRIORITY[observation.status] for observation in active)
        selected = tuple(
            observation
            for observation in active
            if _STATUS_PRIORITY[observation.status] == selected_priority
        )
        ignored = tuple(
            observation
            for observation in active
            if _STATUS_PRIORITY[observation.status] != selected_priority
        )
        for first_index, first in enumerate(selected):
            for second in selected[first_index + 1 :]:
                if not observations_compatible(
                    first,
                    second,
                    sigma=CONFLICT_SIGMA,
                ):
                    return ResolvedParameter(
                        parameter_id=parameter_id,
                        status=ParameterStatus.CONFLICTING,
                        value=None,
                        uncertainty=None,
                        active_observation_ids=active_ids,
                        selected_observation_ids=tuple(
                            observation.observation_id for observation in selected
                        ),
                        ignored_active_observation_ids=tuple(
                            observation.observation_id for observation in ignored
                        ),
                        message=(
                            "active observations at the highest status priority "
                            "exceed the version-1 compatibility threshold"
                        ),
                    )

        if all(observation.uncertainty == 0.0 for observation in selected):
            value = sum(observation.value for observation in selected) / len(selected)
            uncertainty = 0.0
        else:
            weights = [
                1.0 / max(observation.uncertainty, 1e-12) ** 2
                for observation in selected
            ]
            total_weight = sum(weights)
            value = sum(
                weight * observation.value
                for weight, observation in zip(weights, selected, strict=True)
            ) / total_weight
            uncertainty = math.sqrt(1.0 / total_weight)

        return ResolvedParameter(
            parameter_id=parameter_id,
            status=selected[0].status,
            value=value,
            uncertainty=uncertainty,
            active_observation_ids=active_ids,
            selected_observation_ids=tuple(
                observation.observation_id for observation in selected
            ),
            ignored_active_observation_ids=tuple(
                observation.observation_id for observation in ignored
            ),
        )

    def resolved_parameters(self) -> dict[str, ResolvedParameter]:
        return {
            parameter_id: self.resolve_parameter(parameter_id)
            for parameter_id in sorted(self.parameters)
        }

    def evaluate_constraints(self) -> dict[str, Any]:
        dependencies = {
            parameter_id: parameter.dependencies
            for parameter_id, parameter in self.parameters.items()
            if parameter.dependencies
        }
        validate_dependency_graph(dependencies)
        resolved = self.resolved_parameters()
        known = {
            parameter_id: result.value
            for parameter_id, result in resolved.items()
            if result.value is not None
            and result.status is not ParameterStatus.CONFLICTING
        }
        if not self.constraints:
            return {
                "derived": {},
                "resolved": {
                    key: value.as_dict() for key, value in resolved.items()
                },
                "solution": None,
            }
        solution: ConstraintSolution = solve_linear_constraints(
            self.constraints.values(),
            known,
        )
        derived: dict[str, dict[str, Any]] = {}
        for parameter_id, value in sorted(solution.values.items()):
            tolerances = [
                constraint.tolerance
                for constraint in self.constraints.values()
                if parameter_id in constraint.coefficients
            ]
            derived[parameter_id] = {
                "parameter_id": parameter_id,
                "status": ParameterStatus.DERIVED.value,
                "uncertainty": max(tolerances),
                "value": value,
            }
        return {
            "derived": derived,
            "resolved": {
                key: value.as_dict() for key, value in resolved.items()
            },
            "solution": solution.as_dict(),
        }

    def validate(self) -> None:
        if not isinstance(self.revision, int) or self.revision < 0:
            raise GraphValidationError("graph revision must be non-negative")
        for key, source in self.sources.items():
            source.validated()
            if key != source.source_id:
                raise GraphValidationError("source dictionary key mismatch")
        for key, parameter in self.parameters.items():
            parameter.validated()
            if key != parameter.parameter_id:
                raise GraphValidationError("parameter dictionary key mismatch")
            for dependency in parameter.dependencies:
                if dependency not in self.parameters:
                    raise GraphValidationError(
                        f"parameter {key} depends on unknown parameter {dependency}"
                    )
        validate_dependency_graph(
            {
                key: parameter.dependencies
                for key, parameter in self.parameters.items()
                if parameter.dependencies
            }
        )
        observation_ids: set[str] = set()
        revisions: list[int] = []
        superseded_targets: set[str] = set()
        by_id: dict[str, Observation] = {}
        for observation in self.observations:
            observation.validated()
            if observation.observation_id in observation_ids:
                raise GraphValidationError("duplicate observation ID")
            observation_ids.add(observation.observation_id)
            by_id[observation.observation_id] = observation
            revisions.append(observation.revision)
            if observation.source_id not in self.sources:
                raise GraphValidationError(
                    f"observation references unknown source {observation.source_id}"
                )
            if observation.parameter_id not in self.parameters:
                raise GraphValidationError(
                    f"observation references unknown parameter {observation.parameter_id}"
                )
            if observation.supersedes is not None:
                if observation.supersedes in superseded_targets:
                    raise GraphValidationError(
                        "one observation is superseded more than once"
                    )
                superseded_targets.add(observation.supersedes)
        expected_revisions = list(range(1, len(self.observations) + 1))
        if revisions != expected_revisions or self.revision != len(self.observations):
            raise GraphValidationError(
                "observation revisions must be contiguous and match graph revision"
            )
        for observation in self.observations:
            if observation.supersedes is None:
                continue
            prior = by_id.get(observation.supersedes)
            if prior is None:
                raise GraphValidationError(
                    f"unknown superseded observation {observation.supersedes}"
                )
            if prior.parameter_id != observation.parameter_id:
                raise GraphValidationError(
                    "supersession must remain within one parameter"
                )
            if prior.revision >= observation.revision:
                raise GraphValidationError(
                    "supersession must point to an earlier revision"
                )
        for key, constraint in self.constraints.items():
            constraint.validated()
            if key != constraint.constraint_id:
                raise GraphValidationError("constraint dictionary key mismatch")
            for parameter_id in constraint.coefficients:
                if parameter_id not in self.parameters:
                    raise GraphValidationError(
                        f"constraint references unknown parameter {parameter_id}"
                    )
        # Resolution itself is part of validation because conflict is allowed,
        # but malformed priorities, values, or references are not.
        self.resolved_parameters()

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "constraints": [
                self.constraints[key].as_dict()
                for key in sorted(self.constraints)
            ],
            "format": GRAPH_FORMAT,
            "observations": [
                observation.as_dict() for observation in self.observations
            ],
            "parameters": [
                self.parameters[key].as_dict()
                for key in sorted(self.parameters)
            ],
            "revision": self.revision,
            "sources": [
                self.sources[key].as_dict() for key in sorted(self.sources)
            ],
            "version": GRAPH_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    def sha256(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterGraph":
        if payload.get("format") != GRAPH_FORMAT:
            raise GraphValidationError("unsupported parameter graph format")
        if payload.get("version") != GRAPH_VERSION:
            raise GraphValidationError("unsupported parameter graph version")
        graph = cls(revision=int(payload.get("revision", -1)))
        for item in payload.get("sources", []):
            source = EvidenceSource(
                source_id=item["source_id"],
                kind=item["kind"],
                reference=item["reference"],
                content_sha256=item.get("content_sha256"),
                metadata=dict(item.get("metadata", {})),
            ).validated()
            graph.sources[source.source_id] = source
        for item in payload.get("parameters", []):
            parameter = ParameterRecord(
                parameter_id=item["parameter_id"],
                kind=QuantityKind(item["kind"]),
                unit=item["unit"],
                label=item["label"],
                dependencies=tuple(item.get("dependencies", [])),
            ).validated()
            graph.parameters[parameter.parameter_id] = parameter
        for item in payload.get("observations", []):
            graph.observations.append(
                Observation(
                    observation_id=item["observation_id"],
                    parameter_id=item["parameter_id"],
                    value=float(item["value"]),
                    uncertainty=float(item["uncertainty"]),
                    status=ParameterStatus(item["status"]),
                    source_id=item["source_id"],
                    revision=int(item["revision"]),
                    supersedes=item.get("supersedes"),
                    metadata=dict(item.get("metadata", {})),
                ).validated()
            )
        for item in payload.get("constraints", []):
            constraint = LinearConstraint(
                constraint_id=item["constraint_id"],
                name=item["name"],
                coefficients={
                    str(key): float(value)
                    for key, value in item["coefficients"].items()
                },
                constant=float(item["constant"]),
                tolerance=float(item["tolerance"]),
                kind=QuantityKind(item["kind"]),
            ).validated()
            graph.constraints[constraint.constraint_id] = constraint
        graph.validate()
        return graph

    @classmethod
    def from_analysis_report(
        cls,
        report: Mapping[str, Any],
        *,
        content_sha256: str | None = None,
        reference: str = "M2 analysis report",
    ) -> "ParameterGraph":
        if report.get("format") != "measurepilot-planar-analysis":
            raise GraphValidationError("input is not an M2 planar analysis report")
        if report.get("version") != 1:
            raise GraphValidationError("unsupported M2 analysis report version")
        canonical_report = canonical_json_bytes(dict(report))
        digest = content_sha256 or sha256(canonical_report).hexdigest()
        source = EvidenceSource.create(
            kind="M2_ANALYSIS",
            reference=reference,
            content_sha256=digest,
            metadata={
                "analysis_format": report["format"],
                "analysis_version": report["version"],
            },
        )
        graph = cls()
        graph.add_source(source)
        try:
            px_per_mm = float(
                report["configuration"]["segmentation"]["px_per_mm"]
            )
            simplification_rms = float(
                report["contours"]["outer"]["simplification_rms_mm"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphValidationError(
                "analysis report lacks metric configuration evidence"
            ) from exc
        if not math.isfinite(px_per_mm) or px_per_mm <= 0.0:
            raise GraphValidationError("analysis px_per_mm must be positive")
        image_uncertainty = max(0.5 / px_per_mm, simplification_rms)

        def add(
            parameter_id: str,
            value: float,
            *,
            kind: QuantityKind,
            uncertainty: float,
            label: str,
            metadata: Mapping[str, Any],
        ) -> None:
            graph.ensure_parameter(parameter_id, kind=kind, label=label)
            graph.add_observation(
                parameter_id,
                value=float(value),
                uncertainty=max(float(uncertainty), 1e-9),
                status=ParameterStatus.ESTIMATED,
                source_id=source.source_id,
                metadata=metadata,
            )

        lines = sorted(
            report.get("features", {}).get("line_segments", []),
            key=lambda item: item["id"],
        )
        for line in lines:
            feature_id = str(line["id"])
            length = float(line["length_mm"])
            metadata = {"feature_id": feature_id, "feature_type": "line_segment"}
            add(
                f"feature.{feature_id}.start_x",
                float(line["start_mm"][0]),
                kind=QuantityKind.LENGTH,
                uncertainty=image_uncertainty,
                label=f"{feature_id} start X",
                metadata=metadata,
            )
            add(
                f"feature.{feature_id}.start_y",
                float(line["start_mm"][1]),
                kind=QuantityKind.LENGTH,
                uncertainty=image_uncertainty,
                label=f"{feature_id} start Y",
                metadata=metadata,
            )
            add(
                f"feature.{feature_id}.end_x",
                float(line["end_mm"][0]),
                kind=QuantityKind.LENGTH,
                uncertainty=image_uncertainty,
                label=f"{feature_id} end X",
                metadata=metadata,
            )
            add(
                f"feature.{feature_id}.end_y",
                float(line["end_mm"][1]),
                kind=QuantityKind.LENGTH,
                uncertainty=image_uncertainty,
                label=f"{feature_id} end Y",
                metadata=metadata,
            )
            add(
                f"feature.{feature_id}.length",
                length,
                kind=QuantityKind.LENGTH,
                uncertainty=image_uncertainty * 2.0,
                label=f"{feature_id} length",
                metadata=metadata,
            )
            angle_uncertainty = max(
                0.1,
                math.degrees(image_uncertainty / max(length, 1.0)),
            )
            add(
                f"feature.{feature_id}.angle",
                float(line["angle_deg"]),
                kind=QuantityKind.ANGLE,
                uncertainty=angle_uncertainty,
                label=f"{feature_id} angle",
                metadata=metadata,
            )

        holes = sorted(
            report.get("features", {}).get("circular_holes", []),
            key=lambda item: item["id"],
        )
        for hole in holes:
            feature_id = str(hole["id"])
            confidence = float(hole.get("confidence", 0.0))
            metadata = {"feature_id": feature_id, "feature_type": "circular_hole"}
            add(
                f"feature.{feature_id}.center_x",
                float(hole["center_mm"][0]),
                kind=QuantityKind.LENGTH,
                uncertainty=image_uncertainty,
                label=f"{feature_id} centre X",
                metadata=metadata,
            )
            add(
                f"feature.{feature_id}.center_y",
                float(hole["center_mm"][1]),
                kind=QuantityKind.LENGTH,
                uncertainty=image_uncertainty,
                label=f"{feature_id} centre Y",
                metadata=metadata,
            )
            add(
                f"feature.{feature_id}.diameter",
                float(hole["diameter_mm"]),
                kind=QuantityKind.LENGTH,
                uncertainty=max(
                    image_uncertainty * 2.0,
                    float(hole.get("fit_residual_mm", 0.0)),
                ),
                label=f"{feature_id} diameter",
                metadata=metadata,
            )
            add(
                f"feature.{feature_id}.circularity",
                float(hole["circularity"]),
                kind=QuantityKind.DIMENSIONLESS,
                uncertainty=max(0.001, (1.0 - confidence) * 0.1),
                label=f"{feature_id} circularity",
                metadata=metadata,
            )
        if not graph.parameters:
            raise GraphValidationError(
                "analysis report contains no supported line or circular-hole features"
            )
        graph.validate()
        return graph


def save_graph(graph: ParameterGraph, path: str | Path) -> str:
    graph.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.canonical_bytes()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return sha256(payload).hexdigest()


def load_graph(path: str | Path) -> ParameterGraph:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GraphValidationError(f"unable to read parameter graph: {source}") from exc
    if not isinstance(payload, dict):
        raise GraphValidationError("parameter graph root must be an object")
    return ParameterGraph.from_dict(payload)
