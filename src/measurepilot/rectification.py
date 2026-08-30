"""ArUco-based metric rectification for calibration layout version 1."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import cv2
import numpy as np

from .calibration import (
    ARUCO_DICTIONARY_NAME,
    CALIBRATION_LAYOUT_VERSION,
    DEFAULT_LAYOUT,
    MARKER_IDS,
    CalibrationError,
    CalibrationLayout,
    aruco_dictionary,
)
from .quality import DEFAULT_SHARPNESS_THRESHOLD, evaluate_capture_quality, to_grayscale

DEFAULT_PX_PER_MM = 4.0
MAX_REPROJECTION_ERROR_PX = 2.0
MIN_MARKER_AREA_PX2 = 25.0


@dataclass(frozen=True, slots=True)
class RectificationResult:
    image: np.ndarray
    report: dict[str, Any]


def _detector() -> Any:
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = aruco_dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def _raw_detections(gray: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
    detector = _detector()
    if isinstance(detector, tuple):  # pragma: no cover - older OpenCV compatibility
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            detector[0],
            parameters=detector[1],
        )
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    return list(corners), ids


def _validate_detected_markers(
    detections: Mapping[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    missing = [marker_id for marker_id in MARKER_IDS if marker_id not in detections]
    duplicates = [
        marker_id
        for marker_id in MARKER_IDS
        if marker_id in detections and len(detections[marker_id]) != 1
    ]
    if duplicates:
        joined = ", ".join(str(marker_id) for marker_id in duplicates)
        raise CalibrationError(f"duplicate required marker IDs: {joined}")
    if missing:
        joined = ", ".join(str(marker_id) for marker_id in missing)
        raise CalibrationError(f"missing required marker IDs: {joined}")

    validated: dict[int, np.ndarray] = {}
    centres: list[np.ndarray] = []
    for marker_id in MARKER_IDS:
        corners = np.asarray(detections[marker_id][0], dtype=np.float64).reshape(4, 2)
        if not np.all(np.isfinite(corners)):
            raise CalibrationError(f"marker {marker_id} contains non-finite corners")
        contour = corners.astype(np.float32).reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(contour)))
        if area < MIN_MARKER_AREA_PX2:
            raise CalibrationError(f"marker {marker_id} is too small for reliable rectification")
        if not cv2.isContourConvex(contour):
            raise CalibrationError(f"marker {marker_id} has inconsistent corner geometry")
        validated[marker_id] = corners
        centres.append(np.mean(corners, axis=0))

    centre_contour = np.asarray(centres, dtype=np.float32).reshape(-1, 1, 2)
    if abs(float(cv2.contourArea(centre_contour))) < MIN_MARKER_AREA_PX2 * 4.0:
        raise CalibrationError("required marker centres do not span a usable plane")
    if not cv2.isContourConvex(centre_contour):
        raise CalibrationError("required marker IDs have inconsistent page ordering")
    return validated


def detect_required_markers(image: np.ndarray) -> dict[int, np.ndarray]:
    """Detect and validate exactly one instance of each required marker."""

    gray = to_grayscale(image)
    corners, ids = _raw_detections(gray)
    grouped: dict[int, list[np.ndarray]] = {}
    if ids is not None:
        flat_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        for marker_corners, marker_id_value in zip(corners, flat_ids, strict=True):
            marker_id = int(marker_id_value)
            if marker_id in MARKER_IDS:
                grouped.setdefault(marker_id, []).append(np.asarray(marker_corners))
    return _validate_detected_markers(grouped)


def _correspondences(
    markers: Mapping[int, np.ndarray],
    *,
    layout: CalibrationLayout,
    px_per_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.concatenate([markers[marker_id] for marker_id in MARKER_IDS]).astype(np.float64)
    destination = np.concatenate(
        [layout.marker_corners_mm(marker_id) * px_per_mm for marker_id in MARKER_IDS]
    ).astype(np.float64)
    return source, destination


def _normalise_homography(homography: np.ndarray) -> np.ndarray:
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise CalibrationError("homography solution is invalid")
    scale = float(matrix[2, 2])
    if abs(scale) < 1e-12:
        raise CalibrationError("homography solution is singular")
    matrix = matrix / scale
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise CalibrationError("homography solution is singular")
    return matrix


def _round_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 12) for value in row] for row in matrix]


def rectify_image(
    image: np.ndarray,
    *,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
    max_reprojection_error_px: float = MAX_REPROJECTION_ERROR_PX,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
) -> RectificationResult:
    """Rectify the complete printable plane and return auditable evidence."""

    layout.validate()
    if not math.isfinite(px_per_mm) or px_per_mm <= 0.0:
        raise CalibrationError("px_per_mm must be finite and positive")
    if not math.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0.0:
        raise CalibrationError("max_reprojection_error_px must be finite and positive")

    quality = evaluate_capture_quality(image, sharpness_threshold=sharpness_threshold)
    markers = detect_required_markers(image)
    source, destination = _correspondences(markers, layout=layout, px_per_mm=px_per_mm)
    homography, _ = cv2.findHomography(source, destination, method=0)
    if homography is None:
        raise CalibrationError("unable to solve calibration-plane homography")
    homography = _normalise_homography(homography)

    reprojected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2).astype(np.float64),
        homography,
    ).reshape(-1, 2)
    residuals = np.linalg.norm(reprojected - destination, axis=1)
    reprojection_error = float(np.sqrt(np.mean(np.square(residuals))))
    if not math.isfinite(reprojection_error) or reprojection_error > max_reprojection_error_px:
        raise CalibrationError(
            "marker geometry is inconsistent: reprojection error "
            f"{reprojection_error:.3f}px exceeds {max_reprojection_error_px:.3f}px"
        )

    output_width = int(round(layout.page_width_mm * px_per_mm))
    output_height = int(round(layout.page_height_mm * px_per_mm))
    if output_width < 1 or output_height < 1:
        raise CalibrationError("requested output resolution is too small")
    rectified = cv2.warpPerspective(
        np.ascontiguousarray(image),
        homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 255),
    )

    marker_report = {
        str(marker_id): [
            [round(float(x), 6), round(float(y), 6)]
            for x, y in markers[marker_id]
        ]
        for marker_id in MARKER_IDS
    }
    report: dict[str, Any] = {
        "dictionary": ARUCO_DICTIONARY_NAME,
        "homography": _round_matrix(homography),
        "input_image": {
            "height_px": quality.height_px,
            "width_px": quality.width_px,
        },
        "layout_version": CALIBRATION_LAYOUT_VERSION,
        "marker_corners_px": marker_report,
        "marker_ids": list(MARKER_IDS),
        "output_image": {
            "height_px": output_height,
            "width_px": output_width,
        },
        "px_per_mm": round(float(px_per_mm), 9),
        "reprojection_error_px": round(reprojection_error, 9),
        "sharpness": round(quality.sharpness, 6),
        "warnings": list(quality.warnings),
    }
    return RectificationResult(image=rectified, report=report)


def _normalised_path(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _validate_distinct_paths(input_path: Path, output_path: Path, report_path: Path) -> None:
    normalised = {
        _normalised_path(input_path),
        _normalised_path(output_path),
        _normalised_path(report_path),
    }
    if len(normalised) != 3:
        raise CalibrationError("input image, rectified PNG, and JSON report paths must be distinct")


def _write_temp(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        return Path(temporary.name)


def rectify_file(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
    max_reprojection_error_px: float = MAX_REPROJECTION_ERROR_PX,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    """Rectify one image and atomically publish its PNG and JSON evidence."""

    source = Path(input_path)
    output = Path(output_path)
    report_destination = Path(report_path)
    _validate_distinct_paths(source, output, report_destination)

    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise CalibrationError(f"unable to decode input image: {source}")
    result = rectify_image(
        image,
        px_per_mm=px_per_mm,
        sharpness_threshold=sharpness_threshold,
        max_reprojection_error_px=max_reprojection_error_px,
        layout=layout,
    )
    encoded_ok, encoded_png = cv2.imencode(
        ".png",
        result.image,
        (cv2.IMWRITE_PNG_COMPRESSION, 9),
    )
    if not encoded_ok:
        raise CalibrationError("OpenCV failed to encode the rectified PNG")
    report_bytes = (
        json.dumps(
            result.report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    temporary_output: Path | None = None
    temporary_report: Path | None = None
    try:
        temporary_output = _write_temp(output, encoded_png.tobytes())
        temporary_report = _write_temp(report_destination, report_bytes)
        os.replace(temporary_output, output)
        temporary_output = None
        os.replace(temporary_report, report_destination)
        temporary_report = None
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)
    return result.report
