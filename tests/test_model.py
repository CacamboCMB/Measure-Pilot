from __future__ import annotations

import pytest

from measurepilot.errors import ProjectValidationError
from measurepilot.model import MeasurePilotProject

PROJECT_ID = "12345678-1234-5678-9234-567812345678"
TIMESTAMP = "2026-08-30T17:00:00Z"


def test_create_builds_valid_minimal_project() -> None:
    project = MeasurePilotProject.create(
        "Reference plate", project_id=PROJECT_ID, timestamp=TIMESTAMP
    )

    assert project.project_id == PROJECT_ID
    assert project.units == "mm"
    assert project.created_at == TIMESTAMP
    assert project.modified_at == TIMESTAMP
    assert project.history == [
        {
            "event_id": f"{PROJECT_ID}:created",
            "kind": "project_created",
            "timestamp": TIMESTAMP,
        }
    ]


@pytest.mark.parametrize("name", ["", " ", " padded", "padded "])
def test_invalid_project_names_are_rejected(name: str) -> None:
    with pytest.raises(ProjectValidationError):
        MeasurePilotProject.create(name, project_id=PROJECT_ID, timestamp=TIMESTAMP)


def test_noncanonical_uuid_is_rejected() -> None:
    with pytest.raises(ProjectValidationError, match="canonical lowercase"):
        MeasurePilotProject.create(
            "Part",
            project_id="ABCDEFAB-CDEF-4ABC-8DEF-ABCDEFABCDEF",
            timestamp=TIMESTAMP,
        )


def test_internal_unit_is_fixed_to_millimetres() -> None:
    project = MeasurePilotProject.create("Part", project_id=PROJECT_ID, timestamp=TIMESTAMP)
    project.units = "inch"

    with pytest.raises(ProjectValidationError, match="units must be 'mm'"):
        project.validate()


def test_non_finite_values_are_rejected() -> None:
    project = MeasurePilotProject.create("Part", project_id=PROJECT_ID, timestamp=TIMESTAMP)
    project.geometry["radius"] = float("nan")

    with pytest.raises(ProjectValidationError, match="non-finite"):
        project.validate()
