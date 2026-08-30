from __future__ import annotations

import pytest

from measurepilot.constraints import Dependency, LinearConstraint
from measurepilot.measurement_candidates import (
    MeasurementCandidate,
    generate_default_candidates,
)
from measurepilot.parameter_graph import ParameterGraph
from measurepilot.provenance import ParameterStatus, QuantityKind
from measurepilot.recommendation import (
    RecommendationError,
    recommend_measurements,
    record_measurement,
)


def _direct(parameter_id: str, *, uncertainty: float = 0.1) -> MeasurementCandidate:
    return MeasurementCandidate.create(
        coefficients={parameter_id: 1.0},
        quantity=QuantityKind.LENGTH,
        expected_uncertainty=uncertainty,
        effort=1.0,
        measurability=1.0,
        prompt=f"Measure {parameter_id}.",
        provenance="test_candidate_v1",
    )


def test_rank_gain_is_prioritised_over_marginal_precision_improvement() -> None:
    graph = ParameterGraph()
    for parameter_id in ("x", "y", "z"):
        graph.ensure_parameter(parameter_id, QuantityKind.LENGTH)
    graph.append_measurement(
        parameter_id="z",
        quantity=QuantityKind.LENGTH,
        value=5.0,
        uncertainty=1.0,
    )
    graph.add_constraint(
        LinearConstraint.create(
            coefficients={"x": 1.0, "y": 1.0},
            constant=10.0,
            tolerance=0.1,
            name="sum",
        )
    )
    x_candidate = _direct("x")
    z_candidate = _direct("z")
    report = recommend_measurements(graph, (z_candidate, x_candidate))
    assert report.recommendations[0].candidate == x_candidate
    assert report.recommendations[0].score.rank_gain == 1
    assert report.recommendations[1].score.rank_gain == 0
    assert report.recommendations[1].score.information_gain_nats is not None
    assert report.recommendations[1].score.expected_uncertainty_reduction > 0.0
    assert report.recommendations[1].score.posterior_variance < report.recommendations[1].score.prior_variance


def test_downstream_dependency_impact_breaks_equal_rank_gain() -> None:
    graph = ParameterGraph()
    for parameter_id in ("p", "q", "r", "s"):
        graph.ensure_parameter(parameter_id, QuantityKind.LENGTH)
    graph.add_dependency(
        Dependency.create(target_parameter_id="q", coefficients={"p": 1.0})
    )
    graph.add_dependency(
        Dependency.create(target_parameter_id="r", coefficients={"q": 1.0})
    )
    p_candidate = _direct("p", uncertainty=0.25)
    s_candidate = _direct("s", uncertainty=0.25)
    report = recommend_measurements(graph, (s_candidate, p_candidate))
    assert report.recommendations[0].candidate == p_candidate
    assert report.recommendations[0].score.rank_gain == 1
    assert report.recommendations[1].score.rank_gain == 1
    assert report.recommendations[0].score.downstream_impact == 3
    assert report.recommendations[1].score.downstream_impact == 1


def test_conflict_requires_explicit_active_supersession_and_then_resolves() -> None:
    graph = ParameterGraph()
    old = graph.append_measurement(
        parameter_id="plate.pitch",
        quantity=QuantityKind.LENGTH,
        value=52.8,
        uncertainty=0.1,
        note="first reading",
    )
    graph.append_measurement(
        parameter_id="plate.pitch",
        quantity=QuantityKind.LENGTH,
        value=54.4,
        uncertainty=0.1,
        note="second reading",
    )
    candidate = next(
        item
        for item in generate_default_candidates(graph)
        if item.direct_parameter_id == "plate.pitch"
    )
    report = recommend_measurements(graph, (candidate,))
    recommendation = report.recommendations[0]
    assert recommendation.requires_supersession is True
    assert old.observation_id in recommendation.conflicting_observation_ids

    with pytest.raises(RecommendationError, match="requires --supersedes"):
        record_measurement(
            graph,
            candidate,
            value=54.4,
            uncertainty=0.1,
        )
    assert len(graph.observations) == 2

    recorded = record_measurement(
        graph,
        candidate,
        value=54.4,
        uncertainty=0.1,
        supersedes=old.observation_id,
    )
    assert len(graph.observations) == 2
    assert len(recorded.graph.observations) == 3
    assert recorded.resolution.status == ParameterStatus.MEASURED
    assert recorded.resolution.value == pytest.approx(54.4)


def test_composite_record_creates_observable_and_link_constraint_on_clone() -> None:
    graph = ParameterGraph()
    graph.ensure_parameter("hole.left_x", QuantityKind.LENGTH)
    graph.ensure_parameter("hole.right_x", QuantityKind.LENGTH)
    before = graph.canonical_bytes()
    candidate = MeasurementCandidate.create(
        coefficients={"hole.left_x": -1.0, "hole.right_x": 1.0},
        quantity=QuantityKind.LENGTH,
        expected_uncertainty=0.1,
        effort=1.0,
        measurability=0.9,
        prompt="Measure horizontal hole pitch.",
        provenance="test_composite_v1",
    )
    recorded = record_measurement(
        graph,
        candidate,
        value=54.4,
        uncertainty=0.1,
    )
    assert graph.canonical_bytes() == before
    assert recorded.measured_parameter_id.startswith("measurement.measure-")
    assert recorded.linking_constraint_id is not None
    assert recorded.measured_parameter_id in recorded.graph.parameters
    assert len(recorded.graph.constraints) == 1
    assert recorded.resolution.status == ParameterStatus.MEASURED
    assert recorded.resolution.value == pytest.approx(54.4)


def test_recommendation_is_byte_deterministic_and_read_only() -> None:
    graph = ParameterGraph()
    graph.ensure_parameter("x", QuantityKind.LENGTH)
    candidate = _direct("x")
    before = graph.canonical_bytes()
    first = recommend_measurements(graph, (candidate,))
    second = recommend_measurements(graph, (candidate,))
    assert first.canonical_bytes() == second.canonical_bytes()
    assert graph.canonical_bytes() == before
    assert first.graph_sha256 == second.graph_sha256
