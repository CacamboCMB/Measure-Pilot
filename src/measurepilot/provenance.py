"""Deterministic provenance and observation resolution for MeasurePilot M3.

The records in this module are intentionally scalar and append-only.  Physical
measurements, image-derived estimates, assumptions, and locks all retain an
explicit evidence source.  Corrections create a new observation with a
``supersedes`` edge; the prior observation is never removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONFLICT_RULE_VERSION = 1
CONFLICT_SIGMA = 3.0
CONFLICT_NUMERICAL_FLOOR = 1e-9


class ProvenanceError(ValueError):
    """Base class for invalid provenance or observation state."""


class SupersessionError(ProvenanceError):
    """Raised when an append-only supersession graph is invalid."""


class QuantityKind(StrEnum):
    """Supported scalar quantity kinds."""

    LENGTH = "length"
    ANGLE = "angle"
    DIMENSIONLESS = "dimensionless"


UNIT_BY_QUANTITY: dict[QuantityKind, str] = {
    QuantityKind.LENGTH: "mm",
    QuantityKind.ANGLE: "deg",
    QuantityKind.DIMENSIONLESS: "1",
}


class ParameterStatus(StrEnum):
    """Evidence and resolved parameter states required by the M3 contract."""

    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    LOCKED = "LOCKED"
    CONFLICTING = "CONFLICTING"
    UNRESOLVED = "UNRESOLVED"


OBSERVATION_STATUSES = frozenset(
    {
        ParameterStatus.MEASURED,
        ParameterStatus.ESTIMATED,
        ParameterStatus.DERIVED,
        ParameterStatus.ASSUMED,
        ParameterStatus.LOCKED,
    }
)


def validate_identifier(value: object, field_name: str = "identifier") -> str:
    """Return a validated stable identifier."""

    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ProvenanceError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern!r}"
        )
    return value


def validate_sha256(value: object, field_name: str = "sha256") -> str:
    """Return a validated lowercase SHA-256 digest."""

    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProvenanceError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def finite_number(value: object, field_name: str) -> float:
    """Return a finite float, rejecting booleans and non-numeric values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProvenanceError(f"{field_name} must be a finite number")
    return result


def positive_number(value: object, field_name: str) -> float:
    """Return a finite strictly positive float."""

    result = finite_number(value, field_name)
    if result <= 0.0:
        raise ProvenanceError(f"{field_name} must be positive")
    return result


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically and reject non-finite numbers."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("value is not canonical JSON data") from exc
    return (text + "\n").encode("utf-8")


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    """Create a content-addressed identifier from canonical logical content."""

    validate_identifier(prefix, "ID prefix")
    digest = hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()
    return f"{prefix}-{digest[:24]}"


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """Versioned evidence source bound to an exact SHA-256 digest."""

    source_id: str
    kind: str
    sha256: str
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        sha256: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvidenceSource":
        payload = {
            "description": description,
            "kind": kind,
            "metadata": dict(metadata or {}),
            "sha256": sha256,
        }
        source = cls(
            source_id=stable_id("src", payload),
            kind=kind,
            sha256=sha256,
            description=description,
            metadata=dict(metadata or {}),
        )
        source.validate()
        return source

    def validate(self) -> None:
        validate_identifier(self.source_id, "source_id")
        validate_identifier(self.kind, "source kind")
        validate_sha256(self.sha256)
        if not isinstance(self.description, str) or len(self.description) > 500:
            raise ProvenanceError("source description must contain at most 500 characters")
        if not isinstance(self.metadata, Mapping):
            raise ProvenanceError("source metadata must be an object")
        payload = {
            "description": self.description,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "sha256": self.sha256,
        }
        expected = stable_id("src", payload)
        if self.source_id != expected:
            raise ProvenanceError("source_id does not match canonical source content")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "description": self.description,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "sha256": self.sha256,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceSource":
        if not isinstance(value, Mapping):
            raise ProvenanceError("evidence source must be an object")
        source = cls(
            source_id=value.get("source_id"),  # type: ignore[arg-type]
            kind=value.get("kind"),  # type: ignore[arg-type]
            sha256=value.get("sha256"),  # type: ignore[arg-type]
            description=value.get("description", ""),  # type: ignore[arg-type]
            metadata=value.get("metadata", {}),  # type: ignore[arg-type]
        )
        source.validate()
        return source


@dataclass(frozen=True, slots=True)
class Observation:
    """Immutable scalar observation with optional supersession edge."""

    observation_id: str
    parameter_id: str
    value: float
    uncertainty: float
    status: ParameterStatus
    source_id: str
    supersedes: str | None = None
    note: str = ""

    @classmethod
    def create(
        cls,
        *,
        parameter_id: str,
        value: float,
        uncertainty: float,
        status: ParameterStatus | str,
        source_id: str,
        supersedes: str | None = None,
        note: str = "",
    ) -> "Observation":
        status_value = ParameterStatus(status)
        payload = {
            "note": note,
            "parameter_id": parameter_id,
            "source_id": source_id,
            "status": status_value.value,
            "supersedes": supersedes,
            "uncertainty": float(uncertainty),
            "value": float(value),
        }
        observation = cls(
            observation_id=stable_id("obs", payload),
            parameter_id=parameter_id,
            value=float(value),
            uncertainty=float(uncertainty),
            status=status_value,
            source_id=source_id,
            supersedes=supersedes,
            note=note,
        )
        observation.validate()
        return observation

    def validate(self) -> None:
        validate_identifier(self.observation_id, "observation_id")
        validate_identifier(self.parameter_id, "parameter_id")
        validate_identifier(self.source_id, "source_id")
        finite_number(self.value, "observation value")
        positive_number(self.uncertainty, "observation uncertainty")
        if self.status not in OBSERVATION_STATUSES:
            raise ProvenanceError(
                f"{self.status.value} is a resolved state, not an observation state"
            )
        if self.supersedes is not None:
            validate_identifier(self.supersedes, "supersedes")
            if self.supersedes == self.observation_id:
                raise SupersessionError("an observation cannot supersede itself")
        if not isinstance(self.note, str) or len(self.note) > 500:
            raise ProvenanceError("observation note must contain at most 500 characters")
        payload = {
            "note": self.note,
            "parameter_id": self.parameter_id,
            "source_id": self.source_id,
            "status": self.status.value,
            "supersedes": self.supersedes,
            "uncertainty": self.uncertainty,
            "value": self.value,
        }
        expected = stable_id("obs", payload)
        if self.observation_id != expected:
            raise ProvenanceError(
                "observation_id does not match canonical observation content"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "note": self.note,
            "observation_id": self.observation_id,
            "parameter_id": self.parameter_id,
            "source_id": self.source_id,
            "status": self.status.value,
            "supersedes": self.supersedes,
            "uncertainty": self.uncertainty,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        if not isinstance(value, Mapping):
            raise ProvenanceError("observation must be an object")
        try:
            status = ParameterStatus(value.get("status"))
        except (TypeError, ValueError) as exc:
            raise ProvenanceError("observation status is unsupported") from exc
        observation = cls(
            observation_id=value.get("observation_id"),  # type: ignore[arg-type]
            parameter_id=value.get("parameter_id"),  # type: ignore[arg-type]
            value=value.get("value"),  # type: ignore[arg-type]
            uncertainty=value.get("uncertainty"),  # type: ignore[arg-type]
            status=status,
            source_id=value.get("source_id"),  # type: ignore[arg-type]
            supersedes=value.get("supersedes"),  # type: ignore[arg-type]
            note=value.get("note", ""),  # type: ignore[arg-type]
        )
        observation.validate()
        return observation


@dataclass(frozen=True, slots=True)
class ResolvedValue:
    """Resolved scalar value or an explicit unresolved/conflicting state."""

    parameter_id: str
    status: ParameterStatus
    value: float | None
    uncertainty: float | None
    active_observation_ids: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_observation_ids": list(self.active_observation_ids),
            "parameter_id": self.parameter_id,
            "reason": self.reason,
            "status": self.status.value,
            "uncertainty": self.uncertainty,
            "value": self.value,
        }


def validate_supersession_graph(observations: Iterable[Observation]) -> None:
    """Validate references, same-parameter correction edges, and acyclicity."""

    ordered = tuple(observations)
    by_id: dict[str, Observation] = {}
    for observation in ordered:
        observation.validate()
        if observation.observation_id in by_id:
            raise SupersessionError(
                f"duplicate observation_id: {observation.observation_id}"
            )
        by_id[observation.observation_id] = observation

    for observation in ordered:
        if observation.supersedes is None:
            continue
        target = by_id.get(observation.supersedes)
        if target is None:
            raise SupersessionError(
                f"supersedes references unknown observation: {observation.supersedes}"
            )
        if target.parameter_id != observation.parameter_id:
            raise SupersessionError(
                "a correction may supersede only an observation of the same parameter"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(observation_id: str) -> None:
        if observation_id in visited:
            return
        if observation_id in visiting:
            raise SupersessionError("observation supersession graph contains a cycle")
        visiting.add(observation_id)
        parent = by_id[observation_id].supersedes
        if parent is not None:
            visit(parent)
        visiting.remove(observation_id)
        visited.add(observation_id)

    for observation_id in sorted(by_id):
        visit(observation_id)


def active_observations(
    observations: Iterable[Observation],
    *,
    parameter_id: str | None = None,
) -> tuple[Observation, ...]:
    """Return observations not superseded by any later append-only record."""

    ordered = tuple(observations)
    validate_supersession_graph(ordered)
    superseded_ids = {
        observation.supersedes
        for observation in ordered
        if observation.supersedes is not None
    }
    active = [
        observation
        for observation in ordered
        if observation.observation_id not in superseded_ids
        and (parameter_id is None or observation.parameter_id == parameter_id)
    ]
    return tuple(sorted(active, key=lambda item: item.observation_id))


def observations_are_compatible(first: Observation, second: Observation) -> bool:
    """Apply conflict rule version 1 to two active observations."""

    combined = math.hypot(first.uncertainty, second.uncertainty)
    threshold = CONFLICT_SIGMA * combined + CONFLICT_NUMERICAL_FLOOR
    return abs(first.value - second.value) <= threshold


def _inverse_variance_fusion(
    observations: Iterable[Observation],
) -> tuple[float, float]:
    ordered = tuple(observations)
    if not ordered:
        raise ProvenanceError("cannot fuse an empty observation set")
    weights = [1.0 / (item.uncertainty * item.uncertainty) for item in ordered]
    total_weight = math.fsum(weights)
    value = math.fsum(
        weight * observation.value
        for weight, observation in zip(weights, ordered, strict=True)
    ) / total_weight
    uncertainty = math.sqrt(1.0 / total_weight)
    return value, uncertainty


def resolve_observations(
    parameter_id: str,
    observations: Iterable[Observation],
) -> ResolvedValue:
    """Resolve active evidence without silently choosing conflicting values."""

    validate_identifier(parameter_id, "parameter_id")
    active = active_observations(observations, parameter_id=parameter_id)
    active_ids = tuple(item.observation_id for item in active)
    if not active:
        return ResolvedValue(
            parameter_id=parameter_id,
            status=ParameterStatus.UNRESOLVED,
            value=None,
            uncertainty=None,
            reason="no active observations",
        )

    for index, first in enumerate(active):
        for second in active[index + 1 :]:
            if not observations_are_compatible(first, second):
                return ResolvedValue(
                    parameter_id=parameter_id,
                    status=ParameterStatus.CONFLICTING,
                    value=None,
                    uncertainty=None,
                    active_observation_ids=active_ids,
                    reason=(
                        "active observations exceed conflict rule version "
                        f"{CONFLICT_RULE_VERSION}"
                    ),
                )

    locked = tuple(item for item in active if item.status == ParameterStatus.LOCKED)
    if locked:
        value, uncertainty = _inverse_variance_fusion(locked)
        return ResolvedValue(
            parameter_id=parameter_id,
            status=ParameterStatus.LOCKED,
            value=value,
            uncertainty=uncertainty,
            active_observation_ids=active_ids,
            reason="compatible active lock is authoritative",
        )

    value, uncertainty = _inverse_variance_fusion(active)
    if any(item.status == ParameterStatus.MEASURED for item in active):
        status = ParameterStatus.MEASURED
    elif any(item.status == ParameterStatus.ESTIMATED for item in active):
        status = ParameterStatus.ESTIMATED
    elif any(item.status == ParameterStatus.DERIVED for item in active):
        status = ParameterStatus.DERIVED
    else:
        status = ParameterStatus.ASSUMED
    return ResolvedValue(
        parameter_id=parameter_id,
        status=status,
        value=value,
        uncertainty=uncertainty,
        active_observation_ids=active_ids,
        reason="compatible observations fused by inverse variance",
    )
