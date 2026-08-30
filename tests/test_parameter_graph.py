from __future__ import annotations

import hashlib
import json

import pytest

from measurepilot.constraints import Dependency, LinearConstraint
from measurepilot.parameter_graph import (
    CanonicalGraphError,
    GraphError,
    ParameterGraph,
    QuantityMismatchError,
    import_m2_analysis,
    load_graph,
    save_graph,
)
from measurepilot.provenance import (
    ParameterStatus,
    QuantityKind,
    canonical_json_bytes,
)


def _analysis_payload() -> bytes:
    report = {
        "configuration": {
            "features": {"simplify_tolerance_mm": 0.35},
            "segmentation": {"px_per_mm": 4.0},
        },
        "contours": {"outer": {"simplification_rms_mm": 0.08}},
        "coordinate_system": {
            "origin": "top_left_of_rectified_a4_page",
            "unit": "mm",
            "x_axis": "right",
            "y_axis": "down",
        },
        "features": {
            "circular_holes": [
                {
                    "center_mm": [80.0, 210.0],
                    "diameter_mm": 10.0,
                    "fit_residual_mm": 0.04,
                    "id": "circular-hole-000",
                }
            ],
            "line_segments": [
                {
                    "end_mm": [150.0, 185.0],
                    "id": "outer-line-000",
                    "length_mm": 90.0,
                    "start_mm": [60.0, 185.0],
                }
            ],
        },
        "format": "measurepilot-planar-analysis",
        "image": {"height_px": 1188, "width_px": 840},
        "segmentation": {"polarity": "dark"},
        "unresolved": [],
        "version": 1,
        "warnings": [],
    }
    return canonical_json_bytes(report)


def test_m2_import_records_exact_hash_and_estimated_values() -> None:
    payload = _analysis_payload()
    graph = import_m2_analysis(payload)
    assert len(graph.sources) == 1
    source = next(iter(graph.sources.values()))
    assert source.sha256 == hashlib.sha256(payload).hexdigest()
    assert source.kind == "m2_analysis"
    assert len(graph.parameters) == 8
    assert all(
        observation.status == ParameterStatus.ESTIMATED
        for observation in graph.observations
    )
    resolution = graph.resolve_observation_values()[
        "m2.hole.circular-hole-000.diameter"
    ]
    assert resolution.status == ParameterStatus.ESTIMATED
    assert resolution.value == pytest.approx(10.0)
    assert 0.0 < resolution.uncertainty <= 2.0


def test_52_8_to_54_4_correction_is_append_only() -> None:
    graph = ParameterGraph()
    old = graph.append_measurement(
        parameter_id="plate.hole_pitch",
        value=52.8,
        uncertainty=0.1,
        quantity=QuantityKind.LENGTH,
        note="initial reading",
    )
    new = graph.correct_measurement(
        old.observation_id,
        value=54.4,
        uncertainty=0.1,
        note="corrected screw pitch",
    )
    assert len(graph.observations) == 2
    assert graph.observations[0] == old
    assert new.supersedes == old.observation_id
    resolution = graph.resolve_observation_values()["plate.hole_pitch"]
    assert resolution.status == ParameterStatus.MEASURED
    assert resolution.value == pytest.approx(54.4)
    assert resolution.active_observation_ids == (new.observation_id,)


def test_graph_round_trip_is_byte_deterministic(tmp_path) -> None:
    graph = import_m2_analysis(_analysis_payload())
    graph.append_measurement(
        parameter_id="plate.thickness",
        value=2.6,
        uncertainty=0.05,
        quantity=QuantityKind.LENGTH,
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_graph(graph, first)
    loaded = load_graph(first)
    save_graph(loaded, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == graph.canonical_bytes()


def test_dependencies_and_constraints_produce_derived_values() -> None:
    graph = ParameterGraph()
    graph.append_measurement(
        parameter_id="a",
        value=2.0,
        uncertainty=0.1,
        quantity=QuantityKind.LENGTH,
    )
    graph.append_measurement(
        parameter_id="b",
        value=3.0,
        uncertainty=0.1,
        quantity=QuantityKind.LENGTH,
    )
    graph.ensure_parameter("sum", QuantityKind.LENGTH)
    graph.ensure_parameter("remaining", QuantityKind.LENGTH)
    graph.add_dependency(
        Dependency.create(
            target_parameter_id="sum",
            coefficients={"a": 1.0, "b": 1.0},
            tolerance=0.01,
        )
    )
    graph.add_constraint(
        LinearConstraint.create(
            coefficients={"sum": 1.0, "remaining": 1.0},
            constant=10.0,
            tolerance=0.01,
        )
    )
    result = graph.evaluate()
    assert result["sum"].status == ParameterStatus.DERIVED
    assert result["sum"].value == pytest.approx(5.0)
    assert result["remaining"].status == ParameterStatus.DERIVED
    assert result["remaining"].value == pytest.approx(5.0)


def test_material_conflict_is_exposed_and_propagates_unresolved() -> None:
    graph = ParameterGraph()
    graph.append_measurement(
        parameter_id="width",
        value=80.0,
        uncertainty=0.1,
        quantity=QuantityKind.LENGTH,
    )
    graph.append_measurement(
        parameter_id="width",
        value=82.0,
        uncertainty=0.1,
        quantity=QuantityKind.LENGTH,
        note="independent reading",
    )
    graph.ensure_parameter("double_width", QuantityKind.LENGTH)
    graph.add_dependency(
        Dependency.create(
            target_parameter_id="double_width",
            coefficients={"width": 2.0},
        )
    )
    result = graph.evaluate()
    assert result["width"].status == ParameterStatus.CONFLICTING
    assert result["width"].value is None
    assert result["double_width"].status == ParameterStatus.UNRESOLVED
    assert "conflicting" in result["double_width"].reason


def test_quantity_mixing_is_rejected() -> None:
    graph = ParameterGraph()
    graph.ensure_parameter("length", QuantityKind.LENGTH)
    graph.ensure_parameter("angle", QuantityKind.ANGLE)
    graph.ensure_parameter("target", QuantityKind.LENGTH)
    with pytest.raises(QuantityMismatchError, match="mixes quantity"):
        graph.add_dependency(
            Dependency.create(
                target_parameter_id="target",
                coefficients={"length": 1.0, "angle": 1.0},
            )
        )


def test_noncanonical_analysis_is_rejected() -> None:
    report = json.loads(_analysis_payload())
    noncanonical = json.dumps(report, indent=2).encode("utf-8")
    with pytest.raises(CanonicalGraphError, match="not canonical"):
        import_m2_analysis(noncanonical)


def test_constraint_rejects_conflicting_known_parameter() -> None:
    graph = ParameterGraph()
    graph.append_measurement(
        parameter_id="x",
        value=1.0,
        uncertainty=0.01,
        quantity=QuantityKind.LENGTH,
    )
    graph.append_measurement(
        parameter_id="x",
        value=2.0,
        uncertainty=0.01,
        quantity=QuantityKind.LENGTH,
        note="conflict",
    )
    graph.ensure_parameter("y", QuantityKind.LENGTH)
    graph.add_constraint(
        LinearConstraint.create(
            coefficients={"x": 1.0, "y": 1.0},
            constant=3.0,
            tolerance=0.1,
        )
    )
    with pytest.raises(GraphError, match="conflicting"):
        graph.evaluate()


def test_rejected_updates_do_not_mutate_graph() -> None:
    graph = ParameterGraph()
    graph.ensure_parameter("length", QuantityKind.LENGTH)
    graph.ensure_parameter("angle", QuantityKind.ANGLE)
    before = graph.canonical_bytes()
    with pytest.raises(QuantityMismatchError):
        graph.add_constraint(
            LinearConstraint.create(
                coefficients={"length": 1.0, "angle": 1.0},
                constant=1.0,
                tolerance=0.1,
            )
        )
    assert graph.canonical_bytes() == before

    with pytest.raises(Exception, match="uncertainty"):
        graph.append_measurement(
            parameter_id="new.parameter",
            quantity=QuantityKind.LENGTH,
            value=1.0,
            uncertainty=0.0,
        )
    assert "new.parameter" not in graph.parameters
