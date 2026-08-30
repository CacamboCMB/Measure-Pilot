from __future__ import annotations

import hashlib

import pytest

from measurepilot.provenance import (
    EvidenceSource,
    Observation,
    ParameterStatus,
    QuantityKind,
    active_observations,
    canonical_json_bytes,
    resolve_observations,
    validate_supersession_graph,
)


def _source(label: str) -> EvidenceSource:
    payload = label.encode("utf-8")
    return EvidenceSource.create(
        kind="test_evidence",
        sha256=hashlib.sha256(payload).hexdigest(),
        description="test source",
        metadata={"label": label},
    )


def test_content_ids_and_canonical_json_are_deterministic() -> None:
    first = _source("caliper")
    second = _source("caliper")
    assert first == second
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'

    observation_a = Observation.create(
        parameter_id="plate.pitch",
        value=52.8,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=first.source_id,
        note="first",
    )
    observation_b = Observation.create(
        parameter_id="plate.pitch",
        value=52.8,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=first.source_id,
        note="first",
    )
    assert observation_a == observation_b


def test_quantity_enum_exposes_required_units() -> None:
    assert QuantityKind.LENGTH.value == "length"
    assert QuantityKind.ANGLE.value == "angle"
    assert QuantityKind.DIMENSIONLESS.value == "dimensionless"


def test_supersession_keeps_history_and_resolves_54_4() -> None:
    source = _source("physical")
    old = Observation.create(
        parameter_id="plate.pitch",
        value=52.8,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=source.source_id,
    )
    new = Observation.create(
        parameter_id="plate.pitch",
        value=54.4,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=source.source_id,
        supersedes=old.observation_id,
        note="corrected",
    )
    observations = (old, new)
    validate_supersession_graph(observations)
    assert len(observations) == 2
    assert active_observations(observations, parameter_id="plate.pitch") == (new,)
    resolution = resolve_observations("plate.pitch", observations)
    assert resolution.status == ParameterStatus.MEASURED
    assert resolution.value == pytest.approx(54.4)
    assert resolution.active_observation_ids == (new.observation_id,)


def test_compatible_evidence_uses_inverse_variance_fusion() -> None:
    first_source = _source("image")
    second_source = _source("caliper")
    first = Observation.create(
        parameter_id="hole.diameter",
        value=10.0,
        uncertainty=0.2,
        status=ParameterStatus.ESTIMATED,
        source_id=first_source.source_id,
    )
    second = Observation.create(
        parameter_id="hole.diameter",
        value=10.1,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=second_source.source_id,
    )
    resolution = resolve_observations("hole.diameter", (first, second))
    assert resolution.status == ParameterStatus.MEASURED
    assert resolution.value == pytest.approx(10.08)
    assert resolution.uncertainty == pytest.approx((1 / 125) ** 0.5)


def test_materially_incompatible_evidence_is_conflicting() -> None:
    source_a = _source("a")
    source_b = _source("b")
    observations = (
        Observation.create(
            parameter_id="plate.width",
            value=80.0,
            uncertainty=0.1,
            status=ParameterStatus.MEASURED,
            source_id=source_a.source_id,
        ),
        Observation.create(
            parameter_id="plate.width",
            value=82.0,
            uncertainty=0.1,
            status=ParameterStatus.ESTIMATED,
            source_id=source_b.source_id,
        ),
    )
    resolution = resolve_observations("plate.width", observations)
    assert resolution.status == ParameterStatus.CONFLICTING
    assert resolution.value is None
    assert resolution.uncertainty is None


def test_compatible_locked_evidence_is_authoritative() -> None:
    source_a = _source("locked")
    source_b = _source("estimate")
    locked = Observation.create(
        parameter_id="plate.height",
        value=40.0,
        uncertainty=0.05,
        status=ParameterStatus.LOCKED,
        source_id=source_a.source_id,
    )
    estimate = Observation.create(
        parameter_id="plate.height",
        value=40.05,
        uncertainty=0.2,
        status=ParameterStatus.ESTIMATED,
        source_id=source_b.source_id,
    )
    resolution = resolve_observations("plate.height", (locked, estimate))
    assert resolution.status == ParameterStatus.LOCKED
    assert resolution.value == pytest.approx(40.0)
    assert resolution.uncertainty == pytest.approx(0.05)
