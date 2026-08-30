"""Deterministic user corrections for immutable M2 detection results."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .detection import (
    MEASURED_STATUS,
    USER_CORRECTED_STATUS,
    CircleFeature,
    DetectionError,
    DetectionResult,
    _canonical_polygon,
    _finite_positive,
    _point_tuple,
    _validate_feature_id,
    read_detection,
    write_detection,
)

CORRECTION_SCHEMA_VERSION = 1


def _canonical_payload_bytes(value: dict[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise DetectionError(f"correction payload is not valid JSON: {exc}") from exc


def read_corrections(source: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectionError(f"cannot read correction JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DetectionError("correction JSON must contain an object")
    return value


def apply_corrections(
    detection: DetectionResult,
    payload: dict[str, Any],
) -> DetectionResult:
    """Return a new result with explicit corrections and provenance history."""

    detection.validate()
    if payload.get("schema_version") != CORRECTION_SCHEMA_VERSION:
        raise DetectionError("correction schema_version must be 1")
    correction_id = _validate_feature_id(payload.get("correction_id"), "correction_id")
    if any(event.get("correction_id") == correction_id for event in detection.history):
        raise DetectionError(f"correction_id already exists in history: {correction_id}")
    note = payload.get("note", "")
    if not isinstance(note, str) or len(note) > 500:
        raise DetectionError("correction note must be text with at most 500 characters")

    profile = detection.profile_points_mm
    profile_uncertainty = detection.profile_uncertainty_mm
    profile_status = detection.profile_status
    if "profile_points_mm" in payload:
        profile = _canonical_polygon(payload["profile_points_mm"], field_name="profile_points_mm")
        profile_uncertainty = _finite_positive(
            payload.get("profile_uncertainty_mm", detection.profile_uncertainty_mm),
            "profile_uncertainty_mm",
        )
        profile_status = payload.get("profile_status", USER_CORRECTED_STATUS)
        if profile_status not in {USER_CORRECTED_STATUS, MEASURED_STATUS}:
            raise DetectionError("corrected profile status must be user_corrected or measured")

    circles = {circle.feature_id: circle for circle in detection.circles}
    cutouts = {cutout.feature_id: cutout for cutout in detection.cutouts}
    remove_ids = payload.get("remove_feature_ids", [])
    if not isinstance(remove_ids, list):
        raise DetectionError("remove_feature_ids must be a list")
    for feature_id_value in remove_ids:
        feature_id = _validate_feature_id(feature_id_value)
        if feature_id not in circles and feature_id not in cutouts:
            raise DetectionError(f"cannot remove unknown feature: {feature_id}")
        circles.pop(feature_id, None)
        cutouts.pop(feature_id, None)

    upserts = payload.get("upsert_circles", [])
    if not isinstance(upserts, list):
        raise DetectionError("upsert_circles must be a list")
    seen_upserts: set[str] = set()
    for value in upserts:
        if not isinstance(value, dict):
            raise DetectionError("each upsert_circles item must be an object")
        feature_id = _validate_feature_id(value.get("feature_id"))
        if feature_id in seen_upserts:
            raise DetectionError(f"duplicate circle upsert: {feature_id}")
        seen_upserts.add(feature_id)
        if feature_id in cutouts:
            raise DetectionError(f"circle ID conflicts with polygon cutout: {feature_id}")
        status = value.get("status", USER_CORRECTED_STATUS)
        if status not in {USER_CORRECTED_STATUS, MEASURED_STATUS}:
            raise DetectionError("corrected circle status must be user_corrected or measured")
        existing = circles.get(feature_id)
        uncertainty_default = existing.uncertainty_mm if existing is not None else 0.1
        circle = CircleFeature(
            feature_id=feature_id,
            center_mm=_point_tuple(value["center_mm"], f"{feature_id}.center_mm"),
            radius_mm=_finite_positive(value["radius_mm"], f"{feature_id}.radius_mm"),
            uncertainty_mm=_finite_positive(
                value.get("uncertainty_mm", uncertainty_default),
                f"{feature_id}.uncertainty_mm",
            ),
            status=status,
        )
        circle.validate()
        circles[feature_id] = circle

    payload_hash = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    event = {
        "event_id": f"correction:{correction_id}",
        "kind": "user_correction",
        "correction_id": correction_id,
        "payload_sha256": payload_hash,
        "note": note,
        "replaced_profile": "profile_points_mm" in payload,
        "removed_feature_ids": sorted(remove_ids),
        "upserted_circle_ids": sorted(seen_upserts),
    }
    corrected = detection.with_updates(
        profile_points_mm=profile,
        profile_uncertainty_mm=profile_uncertainty,
        profile_status=profile_status,
        circles=tuple(sorted(circles.values(), key=lambda item: item.feature_id)),
        cutouts=tuple(sorted(cutouts.values(), key=lambda item: item.feature_id)),
        history=(*detection.history, event),
    )
    return corrected


def correct_detection_file(
    detection_source: str | os.PathLike[str],
    correction_source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> tuple[Path, DetectionResult]:
    detection_path = Path(detection_source)
    correction_path = Path(correction_source)
    output_path = Path(destination)
    identities = {
        detection_path.resolve(strict=False),
        correction_path.resolve(strict=False),
        output_path.resolve(strict=False),
    }
    if len(identities) != 3:
        raise DetectionError("detection, correction, and output must use different paths")
    corrected = apply_corrections(read_detection(detection_path), read_corrections(correction_path))
    write_detection(corrected, output_path)
    return output_path, corrected
