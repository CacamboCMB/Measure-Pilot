from __future__ import annotations

import json
from pathlib import Path

import pytest

from measurepilot.cli import main
from measurepilot.corrections import apply_corrections, correct_detection_file
from measurepilot.detection import (
    CircleFeature,
    DetectionError,
    DetectionResult,
    PolygonFeature,
    canonical_detection_bytes,
    write_detection,
)


def _detection() -> DetectionResult:
    return DetectionResult(
        source_sha256="a" * 64,
        px_per_mm=4.0,
        profile_points_mm=((10.0, 10.0), (30.0, 10.0), (30.0, 25.0), (10.0, 25.0)),
        profile_uncertainty_mm=0.125,
        profile_status="estimated",
        circles=(
            CircleFeature("circle-001", (15.0, 15.0), 2.0, 0.125),
        ),
        cutouts=(
            PolygonFeature(
                "cutout-001",
                ((20.0, 16.0), (24.0, 16.0), (24.0, 19.0), (20.0, 19.0)),
                0.125,
            ),
        ),
        history=({"event_id": "detection-0001", "kind": "automatic_detection"},),
    )


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "correction_id": "manual-pass-001",
        "note": "Caliper-confirmed profile and circles",
        "profile_points_mm": [[9.8, 10.1], [30.2, 10.1], [30.2, 25.0], [9.8, 25.0]],
        "profile_uncertainty_mm": 0.05,
        "profile_status": "measured",
        "remove_feature_ids": ["cutout-001"],
        "upsert_circles": [
            {
                "feature_id": "circle-001",
                "center_mm": [15.1, 15.0],
                "radius_mm": 2.1,
                "uncertainty_mm": 0.05,
                "status": "measured",
            },
            {
                "feature_id": "circle-002",
                "center_mm": [25.0, 20.0],
                "radius_mm": 1.5,
                "uncertainty_mm": 0.1,
            },
        ],
    }


def test_correction_returns_new_result_and_preserves_original() -> None:
    original = _detection()
    original_bytes = canonical_detection_bytes(original)

    corrected = apply_corrections(original, _payload())

    assert canonical_detection_bytes(original) == original_bytes
    assert corrected is not original
    assert corrected.profile_status == "measured"
    assert corrected.profile_uncertainty_mm == 0.05
    assert [circle.feature_id for circle in corrected.circles] == ["circle-001", "circle-002"]
    assert corrected.circles[0].center_mm == (15.1, 15.0)
    assert corrected.circles[1].status == "user_corrected"
    assert corrected.cutouts == ()
    assert corrected.history[-1]["correction_id"] == "manual-pass-001"
    assert len(corrected.history[-1]["payload_sha256"]) == 64


def test_same_correction_is_deterministic() -> None:
    first = apply_corrections(_detection(), _payload())
    second = apply_corrections(_detection(), _payload())

    assert canonical_detection_bytes(first) == canonical_detection_bytes(second)


def test_duplicate_correction_id_is_rejected() -> None:
    corrected = apply_corrections(_detection(), _payload())

    with pytest.raises(DetectionError, match="already exists"):
        apply_corrections(corrected, _payload())


def test_unknown_feature_removal_is_rejected() -> None:
    payload = _payload()
    payload["remove_feature_ids"] = ["circle-999"]

    with pytest.raises(DetectionError, match="unknown feature"):
        apply_corrections(_detection(), payload)


def test_invalid_corrected_circle_is_rejected() -> None:
    payload = _payload()
    payload["upsert_circles"] = [
        {"feature_id": "circle-001", "center_mm": [1.0, 2.0], "radius_mm": -1.0}
    ]

    with pytest.raises(DetectionError, match="positive finite"):
        apply_corrections(_detection(), payload)



def test_self_intersecting_profile_correction_is_rejected() -> None:
    payload = _payload()
    payload["profile_points_mm"] = [[10.0, 10.0], [30.0, 25.0], [30.0, 10.0], [10.0, 25.0]]

    with pytest.raises(DetectionError, match="self-intersect"):
        apply_corrections(_detection(), payload)

def test_file_and_cli_correction_leave_source_immutable(tmp_path: Path, capsys: object) -> None:
    detection_path = tmp_path / "detection.json"
    correction_path = tmp_path / "correction.json"
    output_path = tmp_path / "corrected.json"
    write_detection(_detection(), detection_path)
    original_bytes = detection_path.read_bytes()
    correction_path.write_text(json.dumps(_payload()), encoding="utf-8")

    correct_detection_file(detection_path, correction_path, output_path)
    first_output = output_path.read_bytes()
    assert detection_path.read_bytes() == original_bytes

    output_path.unlink()
    assert main(
        [
            "analysis",
            "correct",
            str(detection_path),
            str(correction_path),
            str(output_path),
        ]
    ) == 0

    assert output_path.read_bytes() == first_output
    assert json.loads(capsys.readouterr().out) == json.loads(output_path.read_text(encoding="utf-8"))
    assert detection_path.read_bytes() == original_bytes


def test_correction_paths_must_be_distinct(tmp_path: Path) -> None:
    detection_path = tmp_path / "detection.json"
    correction_path = tmp_path / "correction.json"
    write_detection(_detection(), detection_path)
    correction_path.write_text(json.dumps(_payload()), encoding="utf-8")

    with pytest.raises(DetectionError, match="different paths"):
        correct_detection_file(detection_path, correction_path, detection_path)
