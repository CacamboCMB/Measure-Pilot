from __future__ import annotations

import json

import pytest

from measurepilot.measurement_candidates import (
    CanonicalCatalogError,
    CandidateError,
    MeasurementCandidate,
    MeasurementCatalog,
    combine_candidates,
    generate_default_candidates,
    load_catalog,
)
from measurepilot.parameter_graph import ParameterGraph, import_m2_analysis
from measurepilot.provenance import QuantityKind, canonical_json_bytes


def _two_hole_analysis() -> bytes:
    return canonical_json_bytes(
        {
            "configuration": {
                "features": {},
                "segmentation": {"px_per_mm": 4.0},
            },
            "contours": {"outer": {"simplification_rms_mm": 0.1}},
            "coordinate_system": {"unit": "mm"},
            "features": {
                "circular_holes": [
                    {
                        "center_mm": [80.0, 210.0],
                        "diameter_mm": 10.0,
                        "fit_residual_mm": 0.05,
                        "id": "circular-hole-000",
                    },
                    {
                        "center_mm": [134.4, 250.0],
                        "diameter_mm": 10.0,
                        "fit_residual_mm": 0.05,
                        "id": "circular-hole-001",
                    },
                ],
                "line_segments": [],
            },
            "format": "measurepilot-planar-analysis",
            "version": 1,
        }
    )


def test_default_candidates_are_deterministic_and_generate_linear_hole_pitch() -> None:
    graph = import_m2_analysis(_two_hole_analysis())
    first = generate_default_candidates(graph)
    second = generate_default_candidates(graph)
    assert first == second
    assert [item.candidate_id for item in first] == sorted(
        item.candidate_id for item in first
    )

    pitches = [
        candidate
        for candidate in first
        if candidate.provenance == "generated_m2_hole_pitch_v1"
    ]
    assert len(pitches) == 2
    assert any("horizontal" in candidate.prompt for candidate in pitches)
    assert any("vertical" in candidate.prompt for candidate in pitches)
    assert all(len(candidate.coefficients) == 2 for candidate in pitches)
    assert all(
        sorted(coefficient for _parameter_id, coefficient in candidate.coefficients)
        == [-1.0, 1.0]
        for candidate in pitches
    )
    assert all("Euclidean" not in candidate.prompt for candidate in first)


def test_catalog_round_trip_is_canonical_and_validated_against_graph(tmp_path) -> None:
    graph = ParameterGraph()
    graph.ensure_parameter("plate.width", QuantityKind.LENGTH)
    candidate = MeasurementCandidate.create(
        coefficients={"plate.width": 1.0},
        quantity=QuantityKind.LENGTH,
        expected_uncertainty=0.05,
        effort=1.5,
        measurability=0.9,
        prompt="Measure the plate width with a caliper.",
        provenance="user_catalog_v1",
    )
    catalog = MeasurementCatalog.create((candidate,))
    path = tmp_path / "catalog.json"
    path.write_bytes(catalog.canonical_bytes())
    loaded = load_catalog(path)
    assert loaded == catalog
    assert combine_candidates(graph, catalog=loaded)

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
    with pytest.raises(CanonicalCatalogError, match="not canonical"):
        load_catalog(pretty)


def test_catalog_rejects_undefined_parameters_and_quantity_mixing() -> None:
    graph = ParameterGraph()
    graph.ensure_parameter("plate.width", QuantityKind.LENGTH)
    undefined = MeasurementCandidate.create(
        coefficients={"missing": 1.0},
        quantity=QuantityKind.LENGTH,
        expected_uncertainty=0.1,
        effort=1.0,
        measurability=1.0,
        prompt="Measure missing.",
        provenance="user_catalog_v1",
    )
    with pytest.raises(CandidateError, match="undefined"):
        combine_candidates(graph, catalog=MeasurementCatalog.create((undefined,)))

    graph.ensure_parameter("plate.angle", QuantityKind.ANGLE)
    mixed = MeasurementCandidate.create(
        coefficients={"plate.width": 1.0, "plate.angle": 1.0},
        quantity=QuantityKind.LENGTH,
        expected_uncertainty=0.1,
        effort=1.0,
        measurability=1.0,
        prompt="Invalid mixed measurement.",
        provenance="user_catalog_v1",
    )
    with pytest.raises(CandidateError, match="mixes or misdeclares"):
        mixed.validate_against_graph(graph)


def test_zero_observable_and_duplicate_catalog_ids_are_rejected() -> None:
    with pytest.raises(CandidateError, match="zero"):
        MeasurementCandidate.create(
            coefficients={"x": 0.0},
            quantity=QuantityKind.LENGTH,
            expected_uncertainty=0.1,
            effort=1.0,
            measurability=1.0,
            prompt="Invalid.",
            provenance="test",
        )

    candidate = MeasurementCandidate.create(
        coefficients={"x": 1.0},
        quantity=QuantityKind.LENGTH,
        expected_uncertainty=0.1,
        effort=1.0,
        measurability=1.0,
        prompt="Measure x.",
        provenance="test",
    )
    with pytest.raises(CandidateError, match="duplicate"):
        MeasurementCatalog((candidate, candidate)).validate()
