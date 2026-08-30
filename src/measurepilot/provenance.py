"""Deterministic provenance primitives for the MeasurePilot parameter graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping

from .errors import MeasurePilotError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class GraphValidationError(MeasurePilotError):
    """Raised when graph provenance or records violate the M3 contract."""


class ParameterStatus(StrEnum):
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    LOCKED = "LOCKED"
    CONFLICTING = "CONFLICTING"
    UNRESOLVED = "UNRESOLVED"


class QuantityKind(StrEnum):
    LENGTH = "LENGTH"
    ANGLE = "ANGLE"
    DIMENSIONLESS = "DIMENSIONLESS"


UNITS_BY_KIND: Mapping[QuantityKind, str] = {
    QuantityKind.LENGTH: "mm",
    QuantityKind.ANGLE: "deg",
    QuantityKind.DIMENSIONLESS: "1",
}


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise GraphValidationError(f"{path} contains a non-string key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise GraphValidationError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite JSON with stable ordering and one trailing newline."""

    _validate_json_value(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def stable_id(prefix: str, payload: Any) -> str:
    """Create a readable deterministic identifier from canonical content."""

    if not prefix or any(character.isspace() for character in prefix):
        raise GraphValidationError("stable ID prefix must be non-empty and whitespace-free")
    digest = sha256(canonical_json_bytes(payload)).hexdigest()[:24]
    return f"{prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    source_id: str
    kind: str
    reference: str
    content_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        reference: str,
        content_sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvidenceSource":
        payload = {
            "content_sha256": content_sha256,
            "kind": kind,
            "metadata": dict(metadata or {}),
            "reference": reference,
        }
        return cls(
            source_id=stable_id("source", payload),
            kind=kind,
            reference=reference,
            content_sha256=content_sha256,
            metadata=dict(metadata or {}),
        ).validated()

    def validated(self) -> "EvidenceSource":
        if not isinstance(self.source_id, str) or not self.source_id:
            raise GraphValidationError("source_id must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise GraphValidationError("source kind must be a non-empty string")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise GraphValidationError("source reference must be a non-empty string")
        if self.content_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.content_sha256
        ):
            raise GraphValidationError("content_sha256 must be 64 lowercase hex characters")
        _validate_json_value(self.metadata, "$.metadata")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "content_sha256": self.content_sha256,
            "kind": self.kind,
            "metadata": self.metadata,
            "reference": self.reference,
            "source_id": self.source_id,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    parameter_id: str
    value: float
    uncertainty: float
    status: ParameterStatus
    source_id: str
    revision: int
    supersedes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        parameter_id: str,
        value: float,
        uncertainty: float,
        status: ParameterStatus,
        source_id: str,
        revision: int,
        supersedes: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Observation":
        payload = {
            "metadata": dict(metadata or {}),
            "parameter_id": parameter_id,
            "revision": revision,
            "source_id": source_id,
            "status": status.value,
            "supersedes": supersedes,
            "uncertainty": uncertainty,
            "value": value,
        }
        return cls(
            observation_id=stable_id("observation", payload),
            parameter_id=parameter_id,
            value=float(value),
            uncertainty=float(uncertainty),
            status=status,
            source_id=source_id,
            revision=revision,
            supersedes=supersedes,
            metadata=dict(metadata or {}),
        ).validated()

    def validated(self) -> "Observation":
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise GraphValidationError("observation_id must be non-empty")
        if not isinstance(self.parameter_id, str) or not self.parameter_id:
            raise GraphValidationError("parameter_id must be non-empty")
        if not math.isfinite(self.value):
            raise GraphValidationError("observation value must be finite")
        if not math.isfinite(self.uncertainty) or self.uncertainty < 0.0:
            raise GraphValidationError("observation uncertainty must be finite and non-negative")
        if self.uncertainty == 0.0 and self.status is not ParameterStatus.LOCKED:
            raise GraphValidationError("zero uncertainty is reserved for LOCKED observations")
        if self.status in (ParameterStatus.CONFLICTING, ParameterStatus.UNRESOLVED):
            raise GraphValidationError(
                "CONFLICTING and UNRESOLVED are resolution states, not observations"
            )
        if not isinstance(self.source_id, str) or not self.source_id:
            raise GraphValidationError("source_id must be non-empty")
        if not isinstance(self.revision, int) or self.revision < 1:
            raise GraphValidationError("observation revision must be a positive integer")
        if self.supersedes is not None and (
            not isinstance(self.supersedes, str) or not self.supersedes
        ):
            raise GraphValidationError("supersedes must be null or a non-empty ID")
        _validate_json_value(self.metadata, "$.metadata")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "metadata": self.metadata,
            "observation_id": self.observation_id,
            "parameter_id": self.parameter_id,
            "revision": self.revision,
            "source_id": self.source_id,
            "status": self.status.value,
            "supersedes": self.supersedes,
            "uncertainty": self.uncertainty,
            "value": self.value,
        }


def observations_compatible(
    first: Observation,
    second: Observation,
    *,
    sigma: float = 3.0,
    absolute_floor: float = 1e-9,
) -> bool:
    """Return whether two observations overlap under the version-1 rule."""

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise GraphValidationError("compatibility sigma must be finite and positive")
    if not math.isfinite(absolute_floor) or absolute_floor < 0.0:
        raise GraphValidationError("absolute_floor must be finite and non-negative")
    combined = math.hypot(first.uncertainty, second.uncertainty)
    limit = max(absolute_floor, sigma * combined)
    return abs(first.value - second.value) <= limit
