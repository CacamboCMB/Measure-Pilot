from __future__ import annotations

from hashlib import sha256

import cv2
import pytest

from measurepilot.analysis import analyze_rectified_image, canonical_analysis_json
from measurepilot.calibration import render_calibration_image
from measurepilot.constraints import LinearConstraint
from measurepilot.parameter_graph import (
    ParameterGraph,
    ParameterRecord,
    load_graph,
    save_graph,
)
from measurepilot.provenance import (
    EvidenceSource,
    GraphValidationError,
    ParameterStatus,
    QuantityKind,
)


def _source(reference: str) -> EvidenceSource:
    return EvidenceSource.create(
        kind="PHYSICAL_MEASUREMENT",
        reference=reference,
    )


def _length_graph(parameter_id: str = "hole_pitch_x") -> ParameterGraph:
    graph = ParameterGraph()
    graph.add_parameter(
        ParameterRecord.create(parameter_id, kind=QuantityKind.LENGTH)
    )
    return graph


def _m2_report() -> dict:
    image = render_calibration_image(4.0)
    cv2.rectangle(image, (240, 740), (600, 940), 35, -1)
    cv2.circle(image, (320, 840), 20, 255, -1)
    cv2.circle(image, (520, 840), 20, 255, -1)
    return analyze_rectified_image(image, px_per_mm=4.0).report


def test_measurement_correction_retains_52_8_history_and_resolves_54_4() -> None:
    graph = _length_graph()
    first_source = _source("initial caliper reading")
    second_source = _source("caliper recheck")
    graph.add_source(first_source)
    graph.add_source(second_source)
    first = graph.add_observation(
        "hole_pitch_x",
        value=52.8,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=first_source.source_id,
    )
    second = graph.add_observation(
        "hole_pitch_x",
        value=54.4,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=second_source.source_id,
        supersedes=first.observation_id,
    )

    resolved = graph.resolve_parameter("hole_pitch_x")
    assert resolved.status is ParameterStatus.MEASURED
    assert resolved.value == pytest.approx(54.4)
    assert resolved.selected_observation_ids == (second.observation_id,)
    assert len(graph.observations) == 2
    assert graph.observations[0].value == 52.8
    assert graph.revision == 2


def test_compatible_measurements_are_inverse_variance_fused() -> None:
    graph = _length_graph("diameter")
    first_source = _source("caliper A")
    second_source = _source("caliper B")
    graph.add_source(first_source)
    graph.add_source(second_source)
    graph.add_observation(
        "diameter",
        value=10.0,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=first_source.source_id,
    )
    graph.add_observation(
        "diameter",
        value=10.1,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=second_source.source_id,
    )
    resolved = graph.resolve_parameter("diameter")
    assert resolved.value == pytest.approx(10.05)
    assert resolved.uncertainty == pytest.approx(0.1 / 2**0.5)


def test_materially_incompatible_measurements_are_conflicting() -> None:
    graph = _length_graph("diameter")
    for reference, value in (("caliper A", 10.0), ("caliper B", 11.0)):
        source = _source(reference)
        graph.add_source(source)
        graph.add_observation(
            "diameter",
            value=value,
            uncertainty=0.1,
            status=ParameterStatus.MEASURED,
            source_id=source.source_id,
        )
    resolved = graph.resolve_parameter("diameter")
    assert resolved.status is ParameterStatus.CONFLICTING
    assert resolved.value is None
    assert resolved.uncertainty is None


def test_measurement_priority_does_not_delete_estimate() -> None:
    graph = _length_graph("diameter")
    estimate_source = EvidenceSource.create(
        kind="M2_ANALYSIS",
        reference="analysis.json",
    )
    measured_source = _source("caliper")
    graph.add_source(estimate_source)
    graph.add_source(measured_source)
    estimate = graph.add_observation(
        "diameter",
        value=9.8,
        uncertainty=0.4,
        status=ParameterStatus.ESTIMATED,
        source_id=estimate_source.source_id,
    )
    measurement = graph.add_observation(
        "diameter",
        value=10.0,
        uncertainty=0.05,
        status=ParameterStatus.MEASURED,
        source_id=measured_source.source_id,
    )
    resolved = graph.resolve_parameter("diameter")
    assert resolved.value == pytest.approx(10.0)
    assert resolved.selected_observation_ids == (measurement.observation_id,)
    assert resolved.ignored_active_observation_ids == (estimate.observation_id,)


def test_constraint_evaluation_derives_only_unresolved_parameter() -> None:
    graph = ParameterGraph()
    graph.add_parameter(ParameterRecord.create("left", kind=QuantityKind.LENGTH))
    graph.add_parameter(
        ParameterRecord.create(
            "right",
            kind=QuantityKind.LENGTH,
            dependencies=("left",),
        )
    )
    source = _source("left measurement")
    graph.add_source(source)
    graph.add_observation(
        "left",
        value=20.0,
        uncertainty=0.1,
        status=ParameterStatus.MEASURED,
        source_id=source.source_id,
    )
    graph.add_constraint(
        LinearConstraint.create(
            name="total pitch",
            coefficients={"left": 1.0, "right": 1.0},
            constant=54.4,
            tolerance=0.01,
        )
    )
    evaluation = graph.evaluate_constraints()
    assert evaluation["derived"]["right"]["value"] == pytest.approx(34.4)
    assert evaluation["derived"]["right"]["status"] == "DERIVED"


def test_constraint_evaluation_cannot_hide_evidence_conflict() -> None:
    graph = ParameterGraph()
    graph.add_parameter(ParameterRecord.create("x", kind=QuantityKind.LENGTH))
    graph.add_parameter(ParameterRecord.create("y", kind=QuantityKind.LENGTH))
    for reference, value in (("A", 10.0), ("B", 12.0)):
        source = _source(reference)
        graph.add_source(source)
        graph.add_observation(
            "x",
            value=value,
            uncertainty=0.1,
            status=ParameterStatus.MEASURED,
            source_id=source.source_id,
        )
    graph.add_constraint(
        LinearConstraint.create(
            name="sum",
            coefficients={"x": 1.0, "y": 1.0},
            constant=20.0,
            tolerance=0.01,
        )
    )
    with pytest.raises(GraphValidationError, match="conflicting parameter x"):
        graph.evaluate_constraints()


def test_m2_import_binds_exact_report_hash_and_estimated_parameters() -> None:
    report = _m2_report()
    raw = canonical_analysis_json(report)
    digest = sha256(raw).hexdigest()
    graph = ParameterGraph.from_analysis_report(
        report,
        content_sha256=digest,
        reference="analysis.json",
    )
    assert len(graph.sources) == 1
    assert next(iter(graph.sources.values())).content_sha256 == digest
    assert len(graph.parameters) == 32
    assert len(graph.observations) == 32
    assert all(
        observation.status is ParameterStatus.ESTIMATED
        for observation in graph.observations
    )
    diameter = graph.resolve_parameter(
        "feature.circular-hole-000.diameter"
    )
    assert diameter.value == pytest.approx(10.0, abs=0.6)


def test_graph_round_trip_preserves_canonical_bytes(tmp_path) -> None:
    graph = ParameterGraph.from_analysis_report(_m2_report())
    path = tmp_path / "part.graph.json"
    first_digest = save_graph(graph, path)
    loaded = load_graph(path)
    second_digest = save_graph(loaded, path)
    assert loaded.canonical_bytes() == graph.canonical_bytes()
    assert first_digest == second_digest == graph.sha256()
    assert not list(tmp_path.glob("*.tmp"))
