from __future__ import annotations

import math

import pytest

from measurepilot.provenance import (
    EvidenceSource,
    GraphValidationError,
    Observation,
    ParameterStatus,
    canonical_json_bytes,
    observations_compatible,
    stable_id,
)


def _observation(value: float, uncertainty: float) -> Observation:
    return Observation.create(
        parameter_id="hole_pitch_x",
        value=value,
        uncertainty=uncertainty,
        status=ParameterStatus.MEASURED,
        source_id="source-test",
        revision=1,
    )


def test_canonical_json_and_stable_ids_ignore_mapping_order() -> None:
    first = {"b": 2, "a": {"y": 4, "x": 3}}
    second = {"a": {"x": 3, "y": 4}, "b": 2}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert stable_id("test", first) == stable_id("test", second)
    assert canonical_json_bytes(first).endswith(b"\n")


def test_evidence_source_id_is_content_deterministic() -> None:
    first = EvidenceSource.create(
        kind="PHYSICAL_MEASUREMENT",
        reference="digital caliper",
        metadata={"operator": "user", "tool": "caliper"},
    )
    second = EvidenceSource.create(
        kind="PHYSICAL_MEASUREMENT",
        reference="digital caliper",
        metadata={"tool": "caliper", "operator": "user"},
    )
    assert first == second
    assert first.source_id.startswith("source-")


def test_non_finite_provenance_is_rejected() -> None:
    with pytest.raises(GraphValidationError, match="non-finite"):
        canonical_json_bytes({"value": math.nan})
    with pytest.raises(GraphValidationError, match="finite"):
        _observation(math.inf, 0.1)


def test_zero_uncertainty_is_reserved_for_locked_values() -> None:
    with pytest.raises(GraphValidationError, match="reserved"):
        _observation(10.0, 0.0)
    locked = Observation.create(
        parameter_id="datum",
        value=0.0,
        uncertainty=0.0,
        status=ParameterStatus.LOCKED,
        source_id="source-test",
        revision=1,
    )
    assert locked.uncertainty == 0.0


def test_version_one_compatibility_uses_combined_uncertainty() -> None:
    first = _observation(10.0, 0.1)
    compatible = _observation(10.3, 0.1)
    conflicting = _observation(10.5, 0.1)
    assert observations_compatible(first, compatible)
    assert not observations_compatible(first, conflicting)
