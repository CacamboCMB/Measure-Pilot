"""Versioned linear physical-measurement candidates for MeasurePilot M4."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .parameter_graph import ParameterGraph
from .provenance import (
    ParameterStatus,
    ProvenanceError,
    QuantityKind,
    UNIT_BY_QUANTITY,
    canonical_json_bytes,
    finite_number,
    positive_number,
    stable_id,
    validate_identifier,
)


CATALOG_FORMAT = "measurepilot-measurement-catalog"
CATALOG_VERSION = 1
_HOLE_CENTER_RE = re.compile(r"^m2\.hole\.(?P<hole>.+)\.center_(?P<axis>[xy])$")


class CandidateError(ValueError):
    """Base class for invalid measurement candidates."""


class CatalogFormatError(CandidateError):
    """Raised when a measurement catalog is malformed."""


class CanonicalCatalogError(CatalogFormatError):
    """Raised when catalog bytes are not their canonical representation."""


def _normalise_coefficients(
    coefficients: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(coefficients, Mapping) or not coefficients:
        raise CandidateError("candidate coefficients must be a non-empty object")
    result: list[tuple[str, float]] = []
    for parameter_id, raw_coefficient in coefficients.items():
        try:
            canonical_id = validate_identifier(parameter_id, "candidate parameter_id")
            coefficient = finite_number(raw_coefficient, "candidate coefficient")
        except ProvenanceError as exc:
            raise CandidateError(str(exc)) from exc
        if coefficient == 0.0:
            raise CandidateError("zero candidate coefficients must be omitted")
        result.append((canonical_id, coefficient))
    return tuple(sorted(result))


def _coefficient_dict(
    coefficients: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    return {parameter_id: coefficient for parameter_id, coefficient in coefficients}


@dataclass(frozen=True, slots=True)
class MeasurementCandidate:
    """One explicit linear scalar measurement that a user can perform."""

    candidate_id: str
    coefficients: tuple[tuple[str, float], ...]
    quantity: QuantityKind
    unit: str
    expected_uncertainty: float
    effort: float
    measurability: float
    prompt: str
    provenance: str

    @classmethod
    def create(
        cls,
        *,
        coefficients: Mapping[str, float],
        quantity: QuantityKind | str,
        expected_uncertainty: float,
        effort: float,
        measurability: float,
        prompt: str,
        provenance: str,
    ) -> "MeasurementCandidate":
        try:
            quantity_value = QuantityKind(quantity)
        except (TypeError, ValueError) as exc:
            raise CandidateError(f"unsupported candidate quantity: {quantity}") from exc
        normalised = _normalise_coefficients(coefficients)
        payload = {
            "coefficients": _coefficient_dict(normalised),
            "effort": float(effort),
            "expected_uncertainty": float(expected_uncertainty),
            "measurability": float(measurability),
            "prompt": prompt,
            "provenance": provenance,
            "quantity": quantity_value.value,
            "unit": UNIT_BY_QUANTITY[quantity_value],
        }
        candidate = cls(
            candidate_id=stable_id("measure", payload),
            coefficients=normalised,
            quantity=quantity_value,
            unit=UNIT_BY_QUANTITY[quantity_value],
            expected_uncertainty=float(expected_uncertainty),
            effort=float(effort),
            measurability=float(measurability),
            prompt=prompt,
            provenance=provenance,
        )
        candidate.validate()
        return candidate

    @property
    def is_direct(self) -> bool:
        return len(self.coefficients) == 1 and self.coefficients[0][1] == 1.0

    @property
    def direct_parameter_id(self) -> str | None:
        return self.coefficients[0][0] if self.is_direct else None

    def validate(self) -> None:
        try:
            validate_identifier(self.candidate_id, "candidate_id")
            normalised = _normalise_coefficients(_coefficient_dict(self.coefficients))
            uncertainty = positive_number(
                self.expected_uncertainty,
                "expected_uncertainty",
            )
            effort = positive_number(self.effort, "effort")
            measurability = positive_number(self.measurability, "measurability")
        except ProvenanceError as exc:
            raise CandidateError(str(exc)) from exc
        if normalised != self.coefficients:
            raise CandidateError("candidate coefficients are not canonical")
        if measurability > 1.0:
            raise CandidateError("measurability must be in (0, 1]")
        if self.unit != UNIT_BY_QUANTITY[self.quantity]:
            raise CandidateError(
                f"candidate unit must be {UNIT_BY_QUANTITY[self.quantity]!r}"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise CandidateError("candidate prompt must be non-empty")
        if len(self.prompt) > 500:
            raise CandidateError("candidate prompt must contain at most 500 characters")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise CandidateError("candidate provenance must be non-empty")
        payload = {
            "coefficients": _coefficient_dict(normalised),
            "effort": effort,
            "expected_uncertainty": uncertainty,
            "measurability": measurability,
            "prompt": self.prompt,
            "provenance": self.provenance,
            "quantity": self.quantity.value,
            "unit": self.unit,
        }
        if self.candidate_id != stable_id("measure", payload):
            raise CandidateError("candidate_id does not match canonical content")

    def validate_against_graph(self, graph: ParameterGraph) -> None:
        self.validate()
        graph.validate()
        quantities: set[QuantityKind] = set()
        for parameter_id, _coefficient in self.coefficients:
            definition = graph.parameters.get(parameter_id)
            if definition is None:
                raise CandidateError(
                    f"candidate references undefined parameter: {parameter_id}"
                )
            quantities.add(definition.quantity)
        if quantities != {self.quantity}:
            raise CandidateError("candidate mixes or misdeclares quantity kinds")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "coefficients": _coefficient_dict(self.coefficients),
            "effort": self.effort,
            "expected_uncertainty": self.expected_uncertainty,
            "measurability": self.measurability,
            "prompt": self.prompt,
            "provenance": self.provenance,
            "quantity": self.quantity.value,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurementCandidate":
        if not isinstance(value, Mapping):
            raise CatalogFormatError("catalog candidate must be an object")
        expected_keys = {
            "candidate_id",
            "coefficients",
            "effort",
            "expected_uncertainty",
            "measurability",
            "prompt",
            "provenance",
            "quantity",
            "unit",
        }
        if set(value) != expected_keys:
            raise CatalogFormatError("catalog candidate keys do not match version 1")
        try:
            quantity = QuantityKind(value.get("quantity"))
        except (TypeError, ValueError) as exc:
            raise CatalogFormatError("catalog candidate quantity is unsupported") from exc
        candidate = cls(
            candidate_id=value.get("candidate_id"),  # type: ignore[arg-type]
            coefficients=_normalise_coefficients(value.get("coefficients", {})),  # type: ignore[arg-type]
            quantity=quantity,
            unit=value.get("unit"),  # type: ignore[arg-type]
            expected_uncertainty=value.get("expected_uncertainty"),  # type: ignore[arg-type]
            effort=value.get("effort"),  # type: ignore[arg-type]
            measurability=value.get("measurability"),  # type: ignore[arg-type]
            prompt=value.get("prompt"),  # type: ignore[arg-type]
            provenance=value.get("provenance"),  # type: ignore[arg-type]
        )
        candidate.validate()
        return candidate


@dataclass(frozen=True, slots=True)
class MeasurementCatalog:
    """Canonical optional collection of user-defined linear candidates."""

    candidates: tuple[MeasurementCandidate, ...]

    def validate(self) -> None:
        seen: set[str] = set()
        for candidate in self.candidates:
            candidate.validate()
            if candidate.candidate_id in seen:
                raise CandidateError(
                    f"duplicate candidate_id: {candidate.candidate_id}"
                )
            seen.add(candidate.candidate_id)
        if tuple(sorted(self.candidates, key=lambda item: item.candidate_id)) != self.candidates:
            raise CandidateError("catalog candidates must use canonical ID order")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "format": CATALOG_FORMAT,
            "version": CATALOG_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def create(
        cls,
        candidates: Iterable[MeasurementCandidate],
    ) -> "MeasurementCatalog":
        catalog = cls(tuple(sorted(candidates, key=lambda item: item.candidate_id)))
        catalog.validate()
        return catalog

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurementCatalog":
        if not isinstance(value, Mapping):
            raise CatalogFormatError("measurement catalog must be an object")
        if set(value) != {"candidates", "format", "version"}:
            raise CatalogFormatError("catalog keys do not match version 1")
        if value.get("format") != CATALOG_FORMAT or value.get("version") != CATALOG_VERSION:
            raise CatalogFormatError("unsupported measurement catalog format or version")
        raw_candidates = value.get("candidates")
        if not isinstance(raw_candidates, list):
            raise CatalogFormatError("catalog candidates must be a list")
        catalog = cls(tuple(MeasurementCandidate.from_dict(item) for item in raw_candidates))
        catalog.validate()
        return catalog


def load_catalog(source: str | os.PathLike[str]) -> MeasurementCatalog:
    path = Path(source)
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogFormatError(f"cannot read measurement catalog: {path}") from exc
    catalog = MeasurementCatalog.from_dict(value)
    if catalog.canonical_bytes() != payload:
        raise CanonicalCatalogError("measurement catalog JSON is not canonical")
    return catalog


def _direct_profile(
    parameter_id: str,
    quantity: QuantityKind,
    uncertainty: float | None,
) -> tuple[float, float, float, str]:
    label = parameter_id
    if parameter_id.endswith(".diameter"):
        return 0.1, 1.0, 0.95, f"Measure the diameter represented by {label}."
    if parameter_id.endswith(".length"):
        return 0.1, 1.0, 0.95, f"Measure the length represented by {label}."
    if parameter_id.endswith(".center_x"):
        return 0.2, 1.2, 0.80, f"Measure the horizontal centre coordinate represented by {label}."
    if parameter_id.endswith(".center_y"):
        return 0.2, 1.2, 0.80, f"Measure the vertical centre coordinate represented by {label}."
    defaults = {
        QuantityKind.LENGTH: 0.25,
        QuantityKind.ANGLE: 0.5,
        QuantityKind.DIMENSIONLESS: 0.01,
    }
    expected = defaults[quantity]
    if uncertainty is not None and math.isfinite(uncertainty) and uncertainty > 0.0:
        expected = max(1e-6, min(expected, uncertainty))
    return expected, 1.0, 0.75, f"Measure {label} directly."


def generate_default_candidates(graph: ParameterGraph) -> tuple[MeasurementCandidate, ...]:
    """Generate conservative direct and M2 hole-pitch candidates."""

    graph.validate()
    resolved = graph.evaluate_dependencies()
    candidates: dict[str, MeasurementCandidate] = {}
    actionable = {
        ParameterStatus.ESTIMATED,
        ParameterStatus.ASSUMED,
        ParameterStatus.CONFLICTING,
        ParameterStatus.UNRESOLVED,
    }
    for parameter_id in sorted(graph.parameters):
        result = resolved[parameter_id]
        if result.status not in actionable:
            continue
        definition = graph.parameters[parameter_id]
        expected, effort, measurability, prompt = _direct_profile(
            parameter_id,
            definition.quantity,
            result.uncertainty,
        )
        candidate = MeasurementCandidate.create(
            coefficients={parameter_id: 1.0},
            quantity=definition.quantity,
            expected_uncertainty=expected,
            effort=effort,
            measurability=measurability,
            prompt=prompt,
            provenance="generated_direct_v1",
        )
        candidates[candidate.candidate_id] = candidate

    holes_by_axis: dict[str, list[tuple[str, str]]] = {"x": [], "y": []}
    for parameter_id, definition in sorted(graph.parameters.items()):
        match = _HOLE_CENTER_RE.fullmatch(parameter_id)
        if match is None or definition.quantity is not QuantityKind.LENGTH:
            continue
        holes_by_axis[match.group("axis")].append((match.group("hole"), parameter_id))
    for axis, entries in holes_by_axis.items():
        for (first_hole, first_id), (second_hole, second_id) in itertools.combinations(entries, 2):
            direction = "horizontal" if axis == "x" else "vertical"
            candidate = MeasurementCandidate.create(
                coefficients={first_id: -1.0, second_id: 1.0},
                quantity=QuantityKind.LENGTH,
                expected_uncertainty=0.15,
                effort=1.2,
                measurability=0.85,
                prompt=(
                    f"Measure the {direction} centre-to-centre pitch between "
                    f"holes {first_hole} and {second_hole}."
                ),
                provenance="generated_m2_hole_pitch_v1",
            )
            candidates[candidate.candidate_id] = candidate
    return tuple(candidates[candidate_id] for candidate_id in sorted(candidates))


def combine_candidates(
    graph: ParameterGraph,
    *,
    catalog: MeasurementCatalog | None = None,
) -> tuple[MeasurementCandidate, ...]:
    """Combine defaults and an optional catalog, rejecting duplicate IDs."""

    candidates = {candidate.candidate_id: candidate for candidate in generate_default_candidates(graph)}
    if catalog is not None:
        catalog.validate()
        for candidate in catalog.candidates:
            candidate.validate_against_graph(graph)
            if candidate.candidate_id in candidates:
                raise CandidateError(
                    f"duplicate candidate_id across generated/catalog candidates: {candidate.candidate_id}"
                )
            candidates[candidate.candidate_id] = candidate
    for candidate in candidates.values():
        candidate.validate_against_graph(graph)
    return tuple(candidates[candidate_id] for candidate_id in sorted(candidates))
