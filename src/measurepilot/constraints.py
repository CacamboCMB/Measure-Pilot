"""Deterministic linear dependencies and constraint solving for MeasurePilot M3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .provenance import (
    ProvenanceError,
    finite_number,
    positive_number,
    stable_id,
    validate_identifier,
)


class ConstraintError(ValueError):
    """Base class for dependency and linear-constraint failures."""


class MissingParameterError(ConstraintError):
    """Raised when an equation references an undefined parameter."""


class DependencyCycleError(ConstraintError):
    """Raised when derived dependencies contain a directed cycle."""


class RankDeficiencyError(ConstraintError):
    """Raised when a linear system cannot determine every unknown."""


class ExcessiveResidualError(ConstraintError):
    """Raised when an equation residual exceeds its stated tolerance."""


def _normalise_coefficients(
    coefficients: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(coefficients, Mapping) or not coefficients:
        raise ConstraintError("coefficients must be a non-empty object")
    normalised: list[tuple[str, float]] = []
    for parameter_id, raw_coefficient in coefficients.items():
        try:
            validated_id = validate_identifier(parameter_id, "coefficient parameter_id")
            coefficient = finite_number(raw_coefficient, "coefficient")
        except ProvenanceError as exc:
            raise ConstraintError(str(exc)) from exc
        if coefficient == 0.0:
            raise ConstraintError("zero coefficients must be omitted")
        normalised.append((validated_id, coefficient))
    return tuple(sorted(normalised))


def _coefficient_dict(
    coefficients: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    return {parameter_id: coefficient for parameter_id, coefficient in coefficients}


@dataclass(frozen=True, slots=True)
class Dependency:
    """Explicit derived equation: target = constant + sum(c_i * source_i)."""

    dependency_id: str
    target_parameter_id: str
    coefficients: tuple[tuple[str, float], ...]
    constant: float = 0.0
    tolerance: float = 1e-9
    name: str = ""

    @classmethod
    def create(
        cls,
        *,
        target_parameter_id: str,
        coefficients: Mapping[str, float],
        constant: float = 0.0,
        tolerance: float = 1e-9,
        name: str = "",
    ) -> "Dependency":
        normalised = _normalise_coefficients(coefficients)
        payload = {
            "coefficients": _coefficient_dict(normalised),
            "constant": float(constant),
            "name": name,
            "target_parameter_id": target_parameter_id,
            "tolerance": float(tolerance),
        }
        dependency = cls(
            dependency_id=stable_id("dep", payload),
            target_parameter_id=target_parameter_id,
            coefficients=normalised,
            constant=float(constant),
            tolerance=float(tolerance),
            name=name,
        )
        dependency.validate()
        return dependency

    def validate(self) -> None:
        try:
            validate_identifier(self.dependency_id, "dependency_id")
            validate_identifier(self.target_parameter_id, "target_parameter_id")
            coefficients = _normalise_coefficients(_coefficient_dict(self.coefficients))
            constant = finite_number(self.constant, "dependency constant")
            tolerance = positive_number(self.tolerance, "dependency tolerance")
        except ProvenanceError as exc:
            raise ConstraintError(str(exc)) from exc
        if coefficients != self.coefficients:
            raise ConstraintError("dependency coefficients are not canonical")
        if self.target_parameter_id in dict(coefficients):
            raise DependencyCycleError("dependency target cannot directly depend on itself")
        if not isinstance(self.name, str) or len(self.name) > 200:
            raise ConstraintError("dependency name must contain at most 200 characters")
        payload = {
            "coefficients": _coefficient_dict(coefficients),
            "constant": constant,
            "name": self.name,
            "target_parameter_id": self.target_parameter_id,
            "tolerance": tolerance,
        }
        if self.dependency_id != stable_id("dep", payload):
            raise ConstraintError("dependency_id does not match canonical content")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "coefficients": _coefficient_dict(self.coefficients),
            "constant": self.constant,
            "dependency_id": self.dependency_id,
            "name": self.name,
            "target_parameter_id": self.target_parameter_id,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Dependency":
        if not isinstance(value, Mapping):
            raise ConstraintError("dependency must be an object")
        dependency = cls(
            dependency_id=value.get("dependency_id"),  # type: ignore[arg-type]
            target_parameter_id=value.get("target_parameter_id"),  # type: ignore[arg-type]
            coefficients=_normalise_coefficients(value.get("coefficients", {})),  # type: ignore[arg-type]
            constant=value.get("constant", 0.0),  # type: ignore[arg-type]
            tolerance=value.get("tolerance", 1e-9),  # type: ignore[arg-type]
            name=value.get("name", ""),  # type: ignore[arg-type]
        )
        dependency.validate()
        return dependency


@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """Explicit equation: sum(c_i * parameter_i) = constant."""

    constraint_id: str
    coefficients: tuple[tuple[str, float], ...]
    constant: float
    tolerance: float
    name: str = ""

    @classmethod
    def create(
        cls,
        *,
        coefficients: Mapping[str, float],
        constant: float,
        tolerance: float,
        name: str = "",
    ) -> "LinearConstraint":
        normalised = _normalise_coefficients(coefficients)
        payload = {
            "coefficients": _coefficient_dict(normalised),
            "constant": float(constant),
            "name": name,
            "tolerance": float(tolerance),
        }
        constraint = cls(
            constraint_id=stable_id("con", payload),
            coefficients=normalised,
            constant=float(constant),
            tolerance=float(tolerance),
            name=name,
        )
        constraint.validate()
        return constraint

    def validate(self) -> None:
        try:
            validate_identifier(self.constraint_id, "constraint_id")
            coefficients = _normalise_coefficients(_coefficient_dict(self.coefficients))
            constant = finite_number(self.constant, "constraint constant")
            tolerance = positive_number(self.tolerance, "constraint tolerance")
        except ProvenanceError as exc:
            raise ConstraintError(str(exc)) from exc
        if coefficients != self.coefficients:
            raise ConstraintError("constraint coefficients are not canonical")
        if not isinstance(self.name, str) or len(self.name) > 200:
            raise ConstraintError("constraint name must contain at most 200 characters")
        payload = {
            "coefficients": _coefficient_dict(coefficients),
            "constant": constant,
            "name": self.name,
            "tolerance": tolerance,
        }
        if self.constraint_id != stable_id("con", payload):
            raise ConstraintError("constraint_id does not match canonical content")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "coefficients": _coefficient_dict(self.coefficients),
            "constant": self.constant,
            "constraint_id": self.constraint_id,
            "name": self.name,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinearConstraint":
        if not isinstance(value, Mapping):
            raise ConstraintError("linear constraint must be an object")
        constraint = cls(
            constraint_id=value.get("constraint_id"),  # type: ignore[arg-type]
            coefficients=_normalise_coefficients(value.get("coefficients", {})),  # type: ignore[arg-type]
            constant=value.get("constant"),  # type: ignore[arg-type]
            tolerance=value.get("tolerance"),  # type: ignore[arg-type]
            name=value.get("name", ""),  # type: ignore[arg-type]
        )
        constraint.validate()
        return constraint


@dataclass(frozen=True, slots=True)
class SolvedValue:
    """One value derived from a full-rank linear system."""

    parameter_id: str
    value: float
    uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_id": self.parameter_id,
            "uncertainty": self.uncertainty,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ConstraintSolution:
    """Deterministic result of a bounded linear solve."""

    values: tuple[SolvedValue, ...]
    rank: int
    residuals: tuple[tuple[str, float], ...]

    def value_map(self) -> dict[str, SolvedValue]:
        return {value.parameter_id: value for value in self.values}

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "residuals": {name: residual for name, residual in self.residuals},
            "values": [value.to_dict() for value in self.values],
        }


def dependency_order(
    dependencies: Iterable[Dependency],
    parameter_ids: Iterable[str],
) -> tuple[Dependency, ...]:
    """Return deterministic topological order or reject missing/cyclic inputs."""

    defined = {validate_identifier(item, "parameter_id") for item in parameter_ids}
    ordered = tuple(dependencies)
    by_target: dict[str, Dependency] = {}
    for dependency in ordered:
        dependency.validate()
        if dependency.target_parameter_id not in defined:
            raise MissingParameterError(
                f"dependency target is undefined: {dependency.target_parameter_id}"
            )
        if dependency.target_parameter_id in by_target:
            raise ConstraintError(
                "multiple dependencies define parameter "
                f"{dependency.target_parameter_id}"
            )
        for source_id, _coefficient in dependency.coefficients:
            if source_id not in defined:
                raise MissingParameterError(
                    f"dependency source is undefined: {source_id}"
                )
        by_target[dependency.target_parameter_id] = dependency

    targets = set(by_target)
    indegree = {
        target: sum(1 for source, _ in dependency.coefficients if source in targets)
        for target, dependency in by_target.items()
    }
    outgoing: dict[str, list[str]] = {target: [] for target in targets}
    for target, dependency in by_target.items():
        for source, _coefficient in dependency.coefficients:
            if source in targets:
                outgoing[source].append(target)
    ready = sorted(target for target, count in indegree.items() if count == 0)
    result: list[Dependency] = []
    while ready:
        target = ready.pop(0)
        result.append(by_target[target])
        for successor in sorted(outgoing[target]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(result) != len(ordered):
        cyclic = sorted(target for target, count in indegree.items() if count > 0)
        raise DependencyCycleError(
            "dependency graph contains a cycle involving: " + ", ".join(cyclic)
        )
    return tuple(result)


def _known_value(raw_value: object, parameter_id: str) -> float:
    candidate = getattr(raw_value, "value", raw_value)
    if candidate is None:
        raise ConstraintError(f"known parameter has no value: {parameter_id}")
    try:
        return finite_number(candidate, f"known value for {parameter_id}")
    except ProvenanceError as exc:
        raise ConstraintError(str(exc)) from exc


def solve_linear_constraints(
    constraints: Iterable[LinearConstraint],
    *,
    parameter_ids: Iterable[str],
    known_values: Mapping[str, object],
) -> ConstraintSolution:
    """Solve referenced unknowns and reject rank or tolerance violations."""

    defined = {validate_identifier(item, "parameter_id") for item in parameter_ids}
    ordered = tuple(sorted(constraints, key=lambda item: item.constraint_id))
    for constraint in ordered:
        constraint.validate()
    if not ordered:
        return ConstraintSolution(values=(), rank=0, residuals=())

    referenced: set[str] = set()
    for constraint in ordered:
        for parameter_id, _coefficient in constraint.coefficients:
            if parameter_id not in defined:
                raise MissingParameterError(
                    f"constraint references undefined parameter: {parameter_id}"
                )
            referenced.add(parameter_id)

    numeric_known = {
        parameter_id: _known_value(raw_value, parameter_id)
        for parameter_id, raw_value in known_values.items()
        if parameter_id in referenced
    }
    unknowns = tuple(sorted(referenced - set(numeric_known)))

    rows: list[list[float]] = []
    right_hand_side: list[float] = []
    tolerances: list[float] = []
    for constraint in ordered:
        coefficients = dict(constraint.coefficients)
        rows.append([coefficients.get(parameter_id, 0.0) for parameter_id in unknowns])
        known_contribution = math.fsum(
            coefficient * numeric_known[parameter_id]
            for parameter_id, coefficient in constraint.coefficients
            if parameter_id in numeric_known
        )
        right_hand_side.append(constraint.constant - known_contribution)
        tolerances.append(constraint.tolerance)

    if unknowns:
        matrix = np.asarray(rows, dtype=np.float64)
        vector = np.asarray(right_hand_side, dtype=np.float64)
        tolerance_array = np.asarray(tolerances, dtype=np.float64)
        weighted_matrix = matrix / tolerance_array[:, None]
        weighted_vector = vector / tolerance_array
        rank = int(np.linalg.matrix_rank(weighted_matrix))
        if rank < len(unknowns):
            raise RankDeficiencyError(
                "linear system is underdetermined: "
                f"rank {rank} for {len(unknowns)} unknowns"
            )
        solution, _residuals, _rank, _singular = np.linalg.lstsq(
            weighted_matrix,
            weighted_vector,
            rcond=None,
        )
        solved_numeric = {
            parameter_id: float(solution[index])
            for index, parameter_id in enumerate(unknowns)
        }
        normal = weighted_matrix.T @ weighted_matrix
        covariance = np.linalg.pinv(normal, hermitian=True)
        uncertainties = {
            parameter_id: math.sqrt(max(0.0, float(covariance[index, index])))
            for index, parameter_id in enumerate(unknowns)
        }
    else:
        rank = 0
        solved_numeric = {}
        uncertainties = {}

    complete = dict(numeric_known)
    complete.update(solved_numeric)
    residual_values: list[tuple[str, float]] = []
    for index, constraint in enumerate(ordered):
        left = math.fsum(
            coefficient * complete[parameter_id]
            for parameter_id, coefficient in constraint.coefficients
        )
        residual = left - constraint.constant
        residual_name = constraint.name or constraint.constraint_id
        residual_values.append((residual_name, residual))
        if abs(residual) > constraint.tolerance + 1e-12:
            raise ExcessiveResidualError(
                f"constraint {residual_name} residual {residual:.12g} "
                f"exceeds tolerance {constraint.tolerance:.12g}"
            )

    values = tuple(
        SolvedValue(
            parameter_id=parameter_id,
            value=solved_numeric[parameter_id],
            uncertainty=max(uncertainties[parameter_id], 1e-12),
        )
        for parameter_id in unknowns
    )
    return ConstraintSolution(
        values=values,
        rank=rank,
        residuals=tuple(residual_values),
    )
