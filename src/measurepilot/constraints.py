"""Explicit linear constraints and dependency validation for M3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .provenance import GraphValidationError, QuantityKind, stable_id


class ConstraintError(GraphValidationError):
    """Base class for deterministic constraint failures."""


class ConstraintUnderdeterminedError(ConstraintError):
    """Raised when the selected equations cannot determine all unknowns."""


class ConstraintInconsistentError(ConstraintError):
    """Raised when solved values exceed an equation's stated tolerance."""


class DependencyCycleError(ConstraintError):
    """Raised when derived-parameter dependencies contain a cycle."""


@dataclass(frozen=True, slots=True)
class LinearConstraint:
    constraint_id: str
    name: str
    coefficients: dict[str, float]
    constant: float
    tolerance: float
    kind: QuantityKind = QuantityKind.LENGTH

    @classmethod
    def create(
        cls,
        *,
        name: str,
        coefficients: Mapping[str, float],
        constant: float,
        tolerance: float,
        kind: QuantityKind = QuantityKind.LENGTH,
    ) -> "LinearConstraint":
        cleaned = {
            str(parameter_id): float(coefficient)
            for parameter_id, coefficient in coefficients.items()
            if float(coefficient) != 0.0
        }
        payload = {
            "coefficients": cleaned,
            "constant": float(constant),
            "kind": kind.value,
            "name": name,
            "tolerance": float(tolerance),
        }
        return cls(
            constraint_id=stable_id("constraint", payload),
            name=name,
            coefficients=cleaned,
            constant=float(constant),
            tolerance=float(tolerance),
            kind=kind,
        ).validated()

    def validated(self) -> "LinearConstraint":
        if not isinstance(self.constraint_id, str) or not self.constraint_id:
            raise ConstraintError("constraint_id must be non-empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConstraintError("constraint name must be non-empty")
        if not self.coefficients:
            raise ConstraintError("linear constraint requires at least one coefficient")
        for parameter_id, coefficient in self.coefficients.items():
            if not isinstance(parameter_id, str) or not parameter_id:
                raise ConstraintError("constraint parameter IDs must be non-empty")
            if not math.isfinite(coefficient) or coefficient == 0.0:
                raise ConstraintError("constraint coefficients must be finite and non-zero")
        if not math.isfinite(self.constant):
            raise ConstraintError("constraint constant must be finite")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ConstraintError("constraint tolerance must be finite and positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "coefficients": dict(sorted(self.coefficients.items())),
            "constant": self.constant,
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "name": self.name,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class ConstraintSolution:
    values: dict[str, float]
    residuals: dict[str, float]
    rank: int
    unknowns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "residuals": {
                key: round(value, 12)
                for key, value in sorted(self.residuals.items())
            },
            "unknowns": list(self.unknowns),
            "values": {
                key: round(value, 12)
                for key, value in sorted(self.values.items())
            },
        }


def validate_dependency_graph(
    dependencies: Mapping[str, Iterable[str]],
) -> None:
    """Reject self-dependencies and directed cycles deterministically."""

    graph = {
        str(parameter_id): tuple(sorted(set(map(str, inputs))))
        for parameter_id, inputs in dependencies.items()
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            start = path.index(node)
            cycle = path[start:] + [node]
            raise DependencyCycleError(
                "derived dependency cycle: " + " -> ".join(cycle)
            )
        visiting.add(node)
        path.append(node)
        for dependency in graph.get(node, ()):
            if dependency == node:
                raise DependencyCycleError(f"parameter {node} depends on itself")
            if dependency in graph:
                visit(dependency)
        path.pop()
        visiting.remove(node)
        visited.add(node)

    for parameter_id in sorted(graph):
        visit(parameter_id)


def solve_linear_constraints(
    constraints: Iterable[LinearConstraint],
    known_values: Mapping[str, float],
) -> ConstraintSolution:
    """Solve all unknowns in a bounded explicit linear system.

    Known values are substituted before solving. Every remaining unknown must
    be uniquely determined. The solved result is rejected if any original
    equation exceeds its own tolerance.
    """

    ordered = sorted(
        (constraint.validated() for constraint in constraints),
        key=lambda constraint: constraint.constraint_id,
    )
    if not ordered:
        raise ConstraintUnderdeterminedError("no linear constraints were supplied")
    known = {str(key): float(value) for key, value in known_values.items()}
    if not all(math.isfinite(value) for value in known.values()):
        raise ConstraintError("known constraint values must be finite")

    unknowns = tuple(
        sorted(
            {
                parameter_id
                for constraint in ordered
                for parameter_id in constraint.coefficients
                if parameter_id not in known
            }
        )
    )

    if unknowns:
        matrix_rows: list[list[float]] = []
        right_hand: list[float] = []
        for constraint in ordered:
            row = [constraint.coefficients.get(parameter_id, 0.0) for parameter_id in unknowns]
            substituted = sum(
                coefficient * known[parameter_id]
                for parameter_id, coefficient in constraint.coefficients.items()
                if parameter_id in known
            )
            matrix_rows.append(row)
            right_hand.append(constraint.constant - substituted)
        matrix = np.asarray(matrix_rows, dtype=np.float64)
        target = np.asarray(right_hand, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(matrix))
        if rank < len(unknowns):
            raise ConstraintUnderdeterminedError(
                f"constraint system rank {rank} cannot determine "
                f"{len(unknowns)} unknowns"
            )
        solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
        solved = {
            parameter_id: float(solution[index])
            for index, parameter_id in enumerate(unknowns)
        }
    else:
        rank = 0
        solved = {}

    all_values = {**known, **solved}
    residuals: dict[str, float] = {}
    violations: list[str] = []
    for constraint in ordered:
        missing = [
            parameter_id
            for parameter_id in constraint.coefficients
            if parameter_id not in all_values
        ]
        if missing:
            raise ConstraintUnderdeterminedError(
                f"constraint {constraint.name!r} has unresolved parameters: "
                + ", ".join(sorted(missing))
            )
        left = sum(
            coefficient * all_values[parameter_id]
            for parameter_id, coefficient in constraint.coefficients.items()
        )
        residual = float(left - constraint.constant)
        residuals[constraint.constraint_id] = residual
        if abs(residual) > constraint.tolerance:
            violations.append(
                f"{constraint.name}: residual {residual:.12g} exceeds "
                f"{constraint.tolerance:.12g}"
            )
    if violations:
        raise ConstraintInconsistentError(
            "inconsistent constraint system; " + "; ".join(violations)
        )
    return ConstraintSolution(
        values=solved,
        residuals=residuals,
        rank=rank,
        unknowns=unknowns,
    )
