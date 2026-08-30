"""Version-1 in-memory project model.

The model deliberately contains only the M0 data required to create, validate,
and persist a MeasurePilot project. Image and CAD concepts are introduced by
later work orders rather than being speculated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from .errors import ProjectValidationError

FORMAT_NAME = "measurepilot-project"
SCHEMA_VERSION = 1
INTERNAL_LENGTH_UNIT = "mm"


def utc_now_iso() -> str:
    """Return a second-precision UTC timestamp in canonical ISO-8601 form."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProjectValidationError(f"{field_name} must be a UTC ISO-8601 string ending in 'Z'")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProjectValidationError(f"{field_name} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo != UTC:
        raise ProjectValidationError(f"{field_name} must use UTC")
    if parsed.microsecond:
        raise ProjectValidationError(f"{field_name} must not contain fractional seconds")


def _validate_json_value(value: Any, path: str = "$") -> None:
    """Reject values that cannot be represented by the canonical JSON writer."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ProjectValidationError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProjectValidationError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ProjectValidationError(f"{path} contains unsupported value type {type(value).__name__}")


@dataclass(slots=True)
class MeasurePilotProject:
    """Minimal version-1 project aggregate."""

    project_id: str
    name: str
    created_at: str
    modified_at: str
    units: str = INTERNAL_LENGTH_UNIT
    measurements: list[dict[str, Any]] = field(default_factory=list)
    geometry: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        project_id: str | None = None,
        timestamp: str | None = None,
    ) -> "MeasurePilotProject":
        canonical_timestamp = timestamp or utc_now_iso()
        canonical_project_id = project_id or str(uuid4())
        project = cls(
            project_id=canonical_project_id,
            name=name,
            created_at=canonical_timestamp,
            modified_at=canonical_timestamp,
            history=[
                {
                    "event_id": f"{canonical_project_id}:created",
                    "kind": "project_created",
                    "timestamp": canonical_timestamp,
                }
            ],
        )
        project.validate()
        return project

    def validate(self) -> None:
        try:
            parsed_uuid = UUID(self.project_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProjectValidationError("project_id must be a canonical UUID string") from exc
        if str(parsed_uuid) != self.project_id:
            raise ProjectValidationError("project_id must use canonical lowercase UUID formatting")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ProjectValidationError("name must contain non-whitespace text")
        if self.name != self.name.strip():
            raise ProjectValidationError("name must not have leading or trailing whitespace")
        if len(self.name) > 200:
            raise ProjectValidationError("name must be at most 200 characters")

        _validate_timestamp(self.created_at, "created_at")
        _validate_timestamp(self.modified_at, "modified_at")
        if self.modified_at < self.created_at:
            raise ProjectValidationError("modified_at must not precede created_at")
        if self.units != INTERNAL_LENGTH_UNIT:
            raise ProjectValidationError(f"units must be '{INTERNAL_LENGTH_UNIT}'")

        if not isinstance(self.measurements, list):
            raise ProjectValidationError("measurements must be a list")
        if not isinstance(self.geometry, dict):
            raise ProjectValidationError("geometry must be an object")
        if not isinstance(self.hypotheses, list):
            raise ProjectValidationError("hypotheses must be a list")
        if not isinstance(self.history, list):
            raise ProjectValidationError("history must be a list")

        _validate_json_value(self.measurements, "$.measurements")
        _validate_json_value(self.geometry, "$.geometry")
        _validate_json_value(self.hypotheses, "$.hypotheses")
        _validate_json_value(self.history, "$.history")
