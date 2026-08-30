from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

from measurepilot.calibration import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    ARUCO_DICTIONARY_NAME,
    CALIBRATION_LAYOUT_VERSION,
    DEFAULT_LAYOUT,
    MARKER_IDS,
    CalibrationError,
    calibration_pdf_bytes,
    render_calibration_image,
    write_calibration_pdf,
)
from measurepilot.rectification import detect_required_markers


def test_layout_v1_has_exact_a4_marker_geometry() -> None:
    assert DEFAULT_LAYOUT.version == CALIBRATION_LAYOUT_VERSION == 1
    assert DEFAULT_LAYOUT.dictionary_name == ARUCO_DICTIONARY_NAME == "DICT_4X4_50"
    assert DEFAULT_LAYOUT.page_width_mm == A4_WIDTH_MM == 210.0
    assert DEFAULT_LAYOUT.page_height_mm == A4_HEIGHT_MM == 297.0
    assert tuple(DEFAULT_LAYOUT.marker_origins_mm) == MARKER_IDS
    np.testing.assert_allclose(
        DEFAULT_LAYOUT.marker_corners_mm(0),
        ((15.0, 15.0), (39.0, 15.0), (39.0, 39.0), (15.0, 39.0)),
    )
    np.testing.assert_allclose(
        DEFAULT_LAYOUT.marker_corners_mm(2),
        ((171.0, 258.0), (195.0, 258.0), (195.0, 282.0), (171.0, 282.0)),
    )


def test_calibration_pdf_is_byte_deterministic() -> None:
    first = calibration_pdf_bytes()
    second = calibration_pdf_bytes()
    assert first == second
    assert sha256(first).hexdigest() == sha256(second).hexdigest()
    assert first.startswith(b"%PDF-1.4\n")
    assert first.endswith(b"%%EOF\n")


def test_calibration_pdf_contains_print_contract() -> None:
    payload = calibration_pdf_bytes()
    assert b"100 mm verification ruler" in payload
    assert b"Print at 100% scale" in payload
    assert b"DICT_4X4_50" in payload
    assert b"/MediaBox [0 0 595.2756 841.8898]" in payload


def test_write_calibration_pdf_matches_generated_bytes(tmp_path) -> None:
    destination = tmp_path / "calibration.pdf"
    digest = write_calibration_pdf(destination)
    payload = destination.read_bytes()
    assert payload == calibration_pdf_bytes()
    assert digest == sha256(payload).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))


def test_metric_raster_contains_all_required_markers() -> None:
    image = render_calibration_image(4.0)
    assert image.shape == (1188, 840)
    detected = detect_required_markers(image)
    assert tuple(detected) == MARKER_IDS


def test_invalid_raster_resolution_is_rejected() -> None:
    with pytest.raises(CalibrationError, match="px_per_mm"):
        render_calibration_image(0.0)
