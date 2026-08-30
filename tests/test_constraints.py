from __future__ import annotations

import pytest

from measurepilot.constraints import (
    Dependency,
    DependencyCycleError,
    ExcessiveResidualError,
    LinearConstraint,
    RankDeficiencyError,
    dependency_order,
    solve_linear_constraints,
)


def test_full_rank_linear_system_is_solved_deterministically() -> None:
    constraints = (
        LinearConstraint.create(
            coefficients={"x": 1.0, "y": 1.0},
            constant=10.0,
            tolerance=1e-6,
            name="sum",
        ),
        LinearConstraint.create(
            coefficients={"x": 1.0, "y": -1.0},
            constant=2.0,
            tolerance=1e-6,
            name="difference",
        ),
    )
    solution = solve_linear_constraints(
        constraints,
        parameter_ids={"x", "y"},
        known_values={},
    )
    values = solution.value_map()
    assert values["x"].value == pytest.approx(6.0)
    assert values["y"].value == pytest.approx(4.0)
    assert values["x"].uncertainty > 0.0
    assert solution.rank == 2


def test_rank_deficiency_is_reported_explicitly() -> None:
    constraint = LinearConstraint.create(
        coefficients={"x": 1.0, "y": 1.0},
        constant=10.0,
        tolerance=0.1,
    )
    with pytest.raises(RankDeficiencyError, match="underdetermined"):
        solve_linear_constraints(
            (constraint,),
            parameter_ids={"x", "y"},
            known_values={},
        )


def test_excessive_residual_is_rejected_for_known_values() -> None:
    constraint = LinearConstraint.create(
        coefficients={"x": 1.0},
        constant=6.0,
        tolerance=0.1,
    )
    with pytest.raises(ExcessiveResidualError, match="exceeds tolerance"):
        solve_linear_constraints(
            (constraint,),
            parameter_ids={"x"},
            known_values={"x": 5.0},
        )


def test_dependency_cycle_is_rejected_before_evaluation() -> None:
    first = Dependency.create(
        target_parameter_id="a",
        coefficients={"b": 1.0},
    )
    second = Dependency.create(
        target_parameter_id="b",
        coefficients={"a": 1.0},
    )
    with pytest.raises(DependencyCycleError, match="cycle"):
        dependency_order((first, second), {"a", "b"})


def test_dependency_order_is_stable() -> None:
    first = Dependency.create(
        target_parameter_id="sum",
        coefficients={"a": 1.0, "b": 1.0},
    )
    second = Dependency.create(
        target_parameter_id="double_sum",
        coefficients={"sum": 2.0},
    )
    ordered = dependency_order(
        (second, first),
        {"a", "b", "sum", "double_sum"},
    )
    assert [item.target_parameter_id for item in ordered] == ["sum", "double_sum"]


def test_overdetermined_inconsistent_system_rejects_residual() -> None:
    constraints = (
        LinearConstraint.create(
            coefficients={"x": 1.0},
            constant=1.0,
            tolerance=0.01,
            name="first",
        ),
        LinearConstraint.create(
            coefficients={"x": 1.0},
            constant=2.0,
            tolerance=0.01,
            name="second",
        ),
    )
    with pytest.raises(ExcessiveResidualError, match="exceeds tolerance"):
        solve_linear_constraints(
            constraints,
            parameter_ids={"x"},
            known_values={},
        )
