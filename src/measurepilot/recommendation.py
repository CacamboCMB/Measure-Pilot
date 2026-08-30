"""Explainable deterministic next-best-measurement ranking for M4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .constraints import LinearConstraint
from .measurement_candidates import MeasurementCandidate
from .parameter_graph import GraphError, ParameterGraph
from .provenance import (
    Observation,
    ParameterStatus,
    QuantityKind,
    ResolvedValue,
    canonical_json_bytes,
    finite_number,
    positive_number,
)


RECOMMENDATION_FORMAT = "measurepilot-measurement-recommendations"
RECOMMENDATION_VERSION = 1
SCORE_RULE_VERSION = 1


class RecommendationError(ValueError):
    """Base class for recommendation and recording failures."""


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    """Transparent components used by the version-1 ranking formula."""

    current_rank: int
    post_measurement_rank: int
    rank_gain: int
    prior_variance: float | None
    posterior_variance: float | None
    information_gain_nats: float | None
    expected_uncertainty_reduction: float | None
    downstream_impact: int
    conflicting_parameter_count: int
    effort: float
    measurability: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflicting_parameter_count": self.conflicting_parameter_count,
            "current_rank": self.current_rank,
            "downstream_impact": self.downstream_impact,
            "effort": self.effort,
            "expected_uncertainty_reduction": (
                round(self.expected_uncertainty_reduction, 12)
                if self.expected_uncertainty_reduction is not None
                else None
            ),
            "final_score": round(self.final_score, 12),
            "information_gain_nats": (
                round(self.information_gain_nats, 12)
                if self.information_gain_nats is not None
                else None
            ),
            "measurability": self.measurability,
            "post_measurement_rank": self.post_measurement_rank,
            "posterior_variance": (
                round(self.posterior_variance, 12)
                if self.posterior_variance is not None
                else None
            ),
            "prior_variance": (
                round(self.prior_variance, 12)
                if self.prior_variance is not None
                else None
            ),
            "rank_gain": self.rank_gain,
        }


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One ranked candidate with explanation and conflict requirements."""

    rank: int
    candidate: MeasurementCandidate
    score: ScoreComponents
    reason: str
    conflicting_observation_ids: tuple[str, ...]
    requires_supersession: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "conflicting_observation_ids": list(self.conflicting_observation_ids),
            "rank": self.rank,
            "reason": self.reason,
            "requires_supersession": self.requires_supersession,
            "score": self.score.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RecommendationReport:
    """Canonical ranked recommendation report bound to exact graph bytes."""

    graph_sha256: str
    candidate_count: int
    recommendations: tuple[Recommendation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "format": RECOMMENDATION_FORMAT,
            "graph_sha256": self.graph_sha256,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "score_rule_version": SCORE_RULE_VERSION,
            "version": RECOMMENDATION_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class RecordResult:
    """Result of recording one selected candidate on a cloned graph."""

    graph: ParameterGraph
    candidate_id: str
    observation: Observation
    measured_parameter_id: str
    linking_constraint_id: str | None
    resolution: ResolvedValue

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "linking_constraint_id": self.linking_constraint_id,
            "measured_parameter_id": self.measured_parameter_id,
            "observation": self.observation.to_dict(),
            "resolution": self.resolution.to_dict(),
        }


def _resolved_without_constraint_solve(graph: ParameterGraph) -> dict[str, ResolvedValue]:
    try:
        return graph.evaluate_dependencies()
    except GraphError as exc:
        raise RecommendationError(str(exc)) from exc


def _weighted_equation_matrix(
    graph: ParameterGraph,
    resolved: Mapping[str, ResolvedValue],
) -> tuple[tuple[str, ...], np.ndarray]:
    parameter_ids = tuple(sorted(graph.parameters))
    index = {parameter_id: position for position, parameter_id in enumerate(parameter_ids)}
    rows: list[np.ndarray] = []

    for parameter_id in parameter_ids:
        result = resolved[parameter_id]
        if (
            result.value is None
            or result.uncertainty is None
            or result.uncertainty <= 0.0
            or result.status in {
                ParameterStatus.CONFLICTING,
                ParameterStatus.UNRESOLVED,
            }
        ):
            continue
        row = np.zeros(len(parameter_ids), dtype=np.float64)
        row[index[parameter_id]] = 1.0 / result.uncertainty
        rows.append(row)

    for dependency in sorted(graph.dependencies, key=lambda item: item.dependency_id):
        row = np.zeros(len(parameter_ids), dtype=np.float64)
        row[index[dependency.target_parameter_id]] = 1.0
        for source_id, coefficient in dependency.coefficients:
            row[index[source_id]] -= coefficient
        rows.append(row / dependency.tolerance)

    for constraint in sorted(graph.constraints, key=lambda item: item.constraint_id):
        row = np.zeros(len(parameter_ids), dtype=np.float64)
        for parameter_id, coefficient in constraint.coefficients:
            row[index[parameter_id]] = coefficient
        rows.append(row / constraint.tolerance)

    if rows:
        matrix = np.vstack(rows)
    else:
        matrix = np.zeros((0, len(parameter_ids)), dtype=np.float64)
    return parameter_ids, matrix


def _candidate_vector(
    candidate: MeasurementCandidate,
    parameter_ids: tuple[str, ...],
) -> np.ndarray:
    index = {parameter_id: position for position, parameter_id in enumerate(parameter_ids)}
    vector = np.zeros(len(parameter_ids), dtype=np.float64)
    for parameter_id, coefficient in candidate.coefficients:
        vector[index[parameter_id]] = coefficient
    return vector


def _rank(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    return int(np.linalg.matrix_rank(matrix))


def _finite_observable_variance(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> float | None:
    """Return finite prior variance, or None when the observable has nullspace content."""

    if not np.any(vector):
        raise RecommendationError("candidate observable is zero")
    if matrix.shape[0] == 0:
        return None
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size:
        tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
    else:
        rank = 0
    nullspace = vh[rank:, :]
    if nullspace.size:
        null_component = nullspace @ vector
        if float(np.linalg.norm(null_component)) > 1e-9 * max(
            1.0,
            float(np.linalg.norm(vector)),
        ):
            return None
    information = matrix.T @ matrix
    covariance = np.linalg.pinv(information, hermitian=True)
    variance = float(vector @ covariance @ vector)
    if variance < 0.0 and abs(variance) < 1e-12:
        variance = 0.0
    if not math.isfinite(variance) or variance < 0.0:
        raise RecommendationError("observable covariance is not finite")
    return variance


def _downstream_impact(graph: ParameterGraph, candidate: MeasurementCandidate) -> int:
    adjacency: dict[str, set[str]] = {
        parameter_id: set() for parameter_id in graph.parameters
    }
    for dependency in graph.dependencies:
        for source_id, _coefficient in dependency.coefficients:
            adjacency[source_id].add(dependency.target_parameter_id)
    for constraint in graph.constraints:
        members = [parameter_id for parameter_id, _coefficient in constraint.coefficients]
        for member in members:
            adjacency[member].update(other for other in members if other != member)

    visited = {parameter_id for parameter_id, _coefficient in candidate.coefficients}
    queue = sorted(visited)
    while queue:
        current = queue.pop(0)
        for successor in sorted(adjacency[current]):
            if successor in visited:
                continue
            visited.add(successor)
            queue.append(successor)
    return len(visited)


def _conflicting_observations(
    candidate: MeasurementCandidate,
    resolved: Mapping[str, ResolvedValue],
) -> tuple[str, ...]:
    observation_ids: set[str] = set()
    for parameter_id, _coefficient in candidate.coefficients:
        result = resolved[parameter_id]
        if result.status == ParameterStatus.CONFLICTING:
            observation_ids.update(result.active_observation_ids)
    return tuple(sorted(observation_ids))


def _reason(
    score: ScoreComponents,
    candidate: MeasurementCandidate,
) -> str:
    if score.rank_gain > 0:
        return (
            f"Adds {score.rank_gain} independent scalar equation(s), reducing "
            f"structural underdetermination and affecting {score.downstream_impact} "
            "graph parameter(s)."
        )
    if score.conflicting_parameter_count > 0:
        return (
            "Remeasures conflicting active evidence; recording requires an explicit "
            "observation to supersede."
        )
    if score.information_gain_nats is not None:
        return (
            f"Expected finite uncertainty reduction of {score.information_gain_nats:.6g} "
            f"nats across an observable affecting {score.downstream_impact} parameter(s)."
        )
    return (
        f"Provides additional explicit evidence for an observable affecting "
        f"{score.downstream_impact} parameter(s)."
    )


def _score_candidate(
    graph: ParameterGraph,
    candidate: MeasurementCandidate,
    resolved: Mapping[str, ResolvedValue],
    parameter_ids: tuple[str, ...],
    current_matrix: np.ndarray,
) -> tuple[ScoreComponents, tuple[str, ...]]:
    candidate.validate_against_graph(graph)
    vector = _candidate_vector(candidate, parameter_ids)
    current_rank = _rank(current_matrix)
    candidate_row = vector / candidate.expected_uncertainty
    post_matrix = np.vstack((current_matrix, candidate_row))
    post_rank = _rank(post_matrix)
    rank_gain = post_rank - current_rank

    prior_variance = _finite_observable_variance(current_matrix, vector)
    if prior_variance is None:
        posterior_variance = None
        information_gain = None
        uncertainty_reduction = None
    elif prior_variance <= 0.0:
        posterior_variance = 0.0
        information_gain = 0.0
        uncertainty_reduction = 0.0
    else:
        measurement_variance = candidate.expected_uncertainty**2
        posterior_variance = 1.0 / (
            1.0 / prior_variance + 1.0 / measurement_variance
        )
        information_gain = 0.5 * math.log1p(
            prior_variance / measurement_variance
        )
        uncertainty_reduction = math.sqrt(prior_variance) - math.sqrt(
            posterior_variance
        )

    downstream = _downstream_impact(graph, candidate)
    conflict_ids = _conflicting_observations(candidate, resolved)
    conflicting_parameters = sum(
        1
        for parameter_id, _coefficient in candidate.coefficients
        if resolved[parameter_id].status == ParameterStatus.CONFLICTING
    )
    base = (
        1000.0 * rank_gain
        + 100.0 * conflicting_parameters
        + 10.0 * min(information_gain or 0.0, 20.0)
        + downstream
    )
    final_score = base * candidate.measurability / candidate.effort
    components = ScoreComponents(
        current_rank=current_rank,
        post_measurement_rank=post_rank,
        rank_gain=rank_gain,
        prior_variance=prior_variance,
        posterior_variance=posterior_variance,
        information_gain_nats=information_gain,
        expected_uncertainty_reduction=uncertainty_reduction,
        downstream_impact=downstream,
        conflicting_parameter_count=conflicting_parameters,
        effort=candidate.effort,
        measurability=candidate.measurability,
        final_score=final_score,
    )
    return components, conflict_ids


def recommend_measurements(
    graph: ParameterGraph,
    candidates: Iterable[MeasurementCandidate],
    *,
    limit: int | None = None,
) -> RecommendationReport:
    """Rank candidates without mutating the graph."""

    graph.validate()
    before = graph.canonical_bytes()
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise RecommendationError("limit must be a positive integer")
    ordered_candidates = tuple(candidates)
    seen: set[str] = set()
    for candidate in ordered_candidates:
        candidate.validate_against_graph(graph)
        if candidate.candidate_id in seen:
            raise RecommendationError(
                f"duplicate candidate_id: {candidate.candidate_id}"
            )
        seen.add(candidate.candidate_id)

    resolved = _resolved_without_constraint_solve(graph)
    parameter_ids, matrix = _weighted_equation_matrix(graph, resolved)
    scored: list[
        tuple[float, str, MeasurementCandidate, ScoreComponents, tuple[str, ...]]
    ] = []
    for candidate in ordered_candidates:
        components, conflict_ids = _score_candidate(
            graph,
            candidate,
            resolved,
            parameter_ids,
            matrix,
        )
        scored.append(
            (
                components.final_score,
                candidate.candidate_id,
                candidate,
                components,
                conflict_ids,
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    if limit is not None:
        scored = scored[:limit]

    recommendations = tuple(
        Recommendation(
            rank=index + 1,
            candidate=candidate,
            score=components,
            reason=_reason(components, candidate),
            conflicting_observation_ids=conflict_ids,
            requires_supersession=bool(conflict_ids),
        )
        for index, (_score, _candidate_id, candidate, components, conflict_ids) in enumerate(scored)
    )
    if graph.canonical_bytes() != before:
        raise RecommendationError("recommendation generation mutated the graph")
    return RecommendationReport(
        graph_sha256=hashlib.sha256(before).hexdigest(),
        candidate_count=len(ordered_candidates),
        recommendations=recommendations,
    )


def find_candidate(
    candidates: Iterable[MeasurementCandidate],
    candidate_id: str,
) -> MeasurementCandidate:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise RecommendationError(f"unknown measurement candidate: {candidate_id}")


def record_measurement(
    graph: ParameterGraph,
    candidate: MeasurementCandidate,
    *,
    value: float,
    uncertainty: float,
    supersedes: str | None = None,
    note: str = "",
) -> RecordResult:
    """Apply a selected measurement to a validated clone of the source graph."""

    graph.validate()
    candidate.validate_against_graph(graph)
    try:
        numeric_value = finite_number(value, "measurement value")
        numeric_uncertainty = positive_number(
            uncertainty,
            "measurement uncertainty",
        )
    except ValueError as exc:
        raise RecommendationError(str(exc)) from exc
    clone = ParameterGraph.from_dict(graph.to_dict())
    resolved_before = _resolved_without_constraint_solve(clone)

    if candidate.is_direct:
        parameter_id = candidate.direct_parameter_id
        assert parameter_id is not None
        current = resolved_before[parameter_id]
        if current.status == ParameterStatus.CONFLICTING:
            if supersedes is None:
                raise RecommendationError(
                    "conflicting direct measurement requires --supersedes"
                )
            if supersedes not in current.active_observation_ids:
                raise RecommendationError(
                    "supersedes must identify an active conflicting observation"
                )
        elif supersedes is not None:
            if supersedes not in current.active_observation_ids:
                raise RecommendationError(
                    "supersedes must identify an active observation"
                )
        observation = clone.append_measurement(
            parameter_id=parameter_id,
            quantity=candidate.quantity,
            value=numeric_value,
            uncertainty=numeric_uncertainty,
            status=ParameterStatus.MEASURED,
            supersedes=supersedes,
            note=note or f"recorded candidate {candidate.candidate_id}",
        )
        resolution = clone.resolve_observation_values()[parameter_id]
        return RecordResult(
            graph=clone,
            candidate_id=candidate.candidate_id,
            observation=observation,
            measured_parameter_id=parameter_id,
            linking_constraint_id=None,
            resolution=resolution,
        )

    if supersedes is not None:
        raise RecommendationError(
            "composite measurements cannot supersede a scalar source observation"
        )
    measured_parameter_id = f"measurement.{candidate.candidate_id}"
    clone.ensure_parameter(
        measured_parameter_id,
        candidate.quantity,
        label=candidate.prompt,
    )
    observation = clone.append_measurement(
        parameter_id=measured_parameter_id,
        quantity=candidate.quantity,
        value=numeric_value,
        uncertainty=numeric_uncertainty,
        status=ParameterStatus.MEASURED,
        note=note or f"recorded composite candidate {candidate.candidate_id}",
    )
    coefficients = {measured_parameter_id: 1.0}
    for parameter_id, coefficient in candidate.coefficients:
        coefficients[parameter_id] = -coefficient
    constraint = LinearConstraint.create(
        coefficients=coefficients,
        constant=0.0,
        tolerance=numeric_uncertainty,
        name=f"measurement link {candidate.candidate_id}",
    )
    clone.add_constraint(constraint)
    clone.validate()
    resolution = clone.resolve_observation_values()[measured_parameter_id]
    return RecordResult(
        graph=clone,
        candidate_id=candidate.candidate_id,
        observation=observation,
        measured_parameter_id=measured_parameter_id,
        linking_constraint_id=constraint.constraint_id,
        resolution=resolution,
    )
