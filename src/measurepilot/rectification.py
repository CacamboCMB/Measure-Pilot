"""ArUco detection and metric perspective rectification."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .calibration import (
    ARUCO_DICTIONARY_NAME,
    DEFAULT_LAYOUT,
    REQUIRED_MARKER_IDS,
    CalibrationError,
    CalibrationLayout,
    aruco_dictionary,
)
from .quality import (
    DEFAULT_QUALITY_THRESHOLDS,
    QualityThresholds,
    quality_warnings,
    sharpness_score,
    to_grayscale,
)


@dataclass(frozen=True, slots=True)
class RectificationReport:
    layout_version: str
    dictionary: str
    marker_ids: tuple[int, ...]
    source_width_px: int
    source_height_px: int
    rectified_width_px: int
    rectified_height_px: int
    output_px_per_mm: float
    input_px_per_mm_estimate: float
    sharpness_score: float
    reprojection_error_px: float
    homography: tuple[tuple[float, float, float], ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_version": self.layout_version,
            "dictionary": self.dictionary,
            "marker_ids": list(self.marker_ids),
            "source_image": {
                "width_px": self.source_width_px,
                "height_px": self.source_height_px,
            },
            "rectified_image": {
                "width_px": self.rectified_width_px,
                "height_px": self.rectified_height_px,
                "px_per_mm": self.output_px_per_mm,
            },
            "input_px_per_mm_estimate": self.input_px_per_mm_estimate,
            "sharpness_score": self.sharpness_score,
            "reprojection_error_px": self.reprojection_error_px,
            "homography": [list(row) for row in self.homography],
            "warnings": list(self.warnings),
        }


def _detector() -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dictionary(), parameters)


def _polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _build_marker_map(
    corners: Iterable[np.ndarray], ids: np.ndarray | None
) -> dict[int, np.ndarray]:
    if ids is None:
        raise CalibrationError("required markers were not detected")
    flat_ids = [int(item) for item in np.asarray(ids).reshape(-1)]
    corner_list = list(corners)
    if len(flat_ids) != len(corner_list):
        raise CalibrationError("detector returned inconsistent marker IDs and corners")
    if len(flat_ids) != len(set(flat_ids)):
        duplicates = sorted({marker_id for marker_id in flat_ids if flat_ids.count(marker_id) > 1})
        raise CalibrationError(f"duplicate marker IDs detected: {duplicates}")

    marker_map: dict[int, np.ndarray] = {}
    for marker_id, raw_corners in zip(flat_ids, corner_list, strict=True):
        points = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(points).all():
            raise CalibrationError(f"marker {marker_id} has non-finite corners")
        if abs(_polygon_signed_area(points)) < 16.0:
            raise CalibrationError(f"marker {marker_id} is too small or degenerate")
        marker_map[marker_id] = points
    return marker_map


def detect_required_markers(
    image: np.ndarray,
    *,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
) -> dict[int, np.ndarray]:
    """Detect and validate the four version-1 calibration markers."""

    layout.validate()
    gray = to_grayscale(image)
    corners, ids, _rejected = _detector().detectMarkers(gray)
    marker_map = _build_marker_map(corners, ids)
    missing = sorted(set(REQUIRED_MARKER_IDS) - set(marker_map))
    if missing:
        raise CalibrationError(f"missing required marker IDs: {missing}")
    return {marker_id: marker_map[marker_id] for marker_id in REQUIRED_MARKER_IDS}


def _point_sets(
    marker_map: dict[int, np.ndarray],
    *,
    layout: CalibrationLayout,
    output_px_per_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_points: list[np.ndarray] = []
    destination_points: list[np.ndarray] = []
    for marker_id in REQUIRED_MARKER_IDS:
        source_points.append(marker_map[marker_id])
        destination_points.append(layout.placement(marker_id).corners_mm() * output_px_per_mm)
    return np.vstack(source_points).astype(np.float64), np.vstack(destination_points).astype(np.float64)


def _rms_reprojection_error(
    source_points: np.ndarray, destination_points: np.ndarray, homography: np.ndarray
) -> float:
    projected = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    squared = np.sum((projected - destination_points) ** 2, axis=1)
    return float(math.sqrt(float(np.mean(squared))))


def _input_px_per_mm(marker_map: dict[int, np.ndarray], marker_size_mm: float) -> float:
    estimates: list[float] = []
    for points in marker_map.values():
        lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        estimates.extend(float(length / marker_size_mm) for length in lengths)
    return float(np.median(np.asarray(estimates, dtype=np.float64)))


def _validate_homography_geometry(
    homography: np.ndarray,
    *,
    source_shape: tuple[int, ...],
    layout: CalibrationLayout,
    output_px_per_mm: float,
) -> None:
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise CalibrationError("homography is invalid")
    if abs(float(np.linalg.det(homography))) < 1e-12:
        raise CalibrationError("homography is singular")
    if np.linalg.cond(homography) > 1e9:
        raise CalibrationError("homography is numerically unstable")

    inverse = np.linalg.inv(homography)
    page_corners = np.asarray(
        [
            [0.0, 0.0],
            [layout.page_width_mm * output_px_per_mm, 0.0],
            [layout.page_width_mm * output_px_per_mm, layout.page_height_mm * output_px_per_mm],
            [0.0, layout.page_height_mm * output_px_per_mm],
        ],
        dtype=np.float64,
    )
    source_page = cv2.perspectiveTransform(page_corners.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if not cv2.isContourConvex(source_page.astype(np.float32)):
        raise CalibrationError("detected page geometry is not convex")
    area = abs(_polygon_signed_area(source_page))
    image_area = float(source_shape[0] * source_shape[1])
    if area < max(1_000.0, image_area * 0.01):
        raise CalibrationError("detected page occupies an implausibly small image area")


def rectify_image(
    image: np.ndarray,
    *,
    output_px_per_mm: float = 4.0,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> tuple[np.ndarray, RectificationReport]:
    """Rectify an image to the full A4 metric page coordinate system."""

    layout.validate()
    thresholds.validate()
    if not np.isfinite(output_px_per_mm) or output_px_per_mm <= 0:
        raise CalibrationError("output_px_per_mm must be a positive finite number")
    if output_px_per_mm > 20.0:
        raise CalibrationError("output_px_per_mm exceeds the version-1 safety limit of 20")

    marker_map = detect_required_markers(image, layout=layout)
    source_points, destination_points = _point_sets(
        marker_map, layout=layout, output_px_per_mm=output_px_per_mm
    )
    homography, _mask = cv2.findHomography(source_points, destination_points, method=0)
    if homography is None:
        raise CalibrationError("could not solve calibration homography")
    _validate_homography_geometry(
        homography,
        source_shape=image.shape,
        layout=layout,
        output_px_per_mm=output_px_per_mm,
    )

    reprojection = _rms_reprojection_error(source_points, destination_points, homography)
    if reprojection > thresholds.max_reprojection_error_px:
        raise CalibrationError(
            "marker geometry is inconsistent with the versioned layout: "
            f"reprojection error {reprojection:.3f} px exceeds "
            f"{thresholds.max_reprojection_error_px:.3f} px"
        )

    output_width = int(round(layout.page_width_mm * output_px_per_mm))
    output_height = int(round(layout.page_height_mm * output_px_per_mm))
    border_value: int | tuple[int, int, int]
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    rectified = cv2.warpPerspective(
        image,
        homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    input_resolution = _input_px_per_mm(marker_map, layout.marker_size_mm)
    sharpness = sharpness_score(image)
    warnings = quality_warnings(
        sharpness=sharpness,
        input_px_per_mm=input_resolution,
        reprojection_error_px=reprojection,
        thresholds=thresholds,
    )
    report = RectificationReport(
        layout_version=layout.version,
        dictionary=ARUCO_DICTIONARY_NAME,
        marker_ids=REQUIRED_MARKER_IDS,
        source_width_px=int(image.shape[1]),
        source_height_px=int(image.shape[0]),
        rectified_width_px=output_width,
        rectified_height_px=output_height,
        output_px_per_mm=float(output_px_per_mm),
        input_px_per_mm_estimate=input_resolution,
        sharpness_score=sharpness,
        reprojection_error_px=reprojection,
        homography=tuple(tuple(float(value) for value in row) for row in homography),
        warnings=tuple(warnings),
    )
    return rectified, report


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def rectify_image_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    report_destination: str | os.PathLike[str] | None = None,
    output_px_per_mm: float = 4.0,
) -> tuple[Path, Path, RectificationReport]:
    source_path = Path(source)
    output_path = Path(destination)
    report_path = (
        Path(report_destination)
        if report_destination is not None
        else output_path.with_suffix(".json")
    )

    if output_path.suffix.lower() != ".png":
        raise CalibrationError("rectified output must use the .png extension")
    if output_path.exists() and output_path.is_dir():
        raise CalibrationError(f"rectified output is a directory: {output_path}")
    if report_path.exists() and report_path.is_dir():
        raise CalibrationError(f"report output is a directory: {report_path}")

    source_identity = source_path.resolve(strict=False)
    output_identity = output_path.resolve(strict=False)
    report_identity = report_path.resolve(strict=False)
    if source_identity == output_identity:
        raise CalibrationError("rectified output must not overwrite the input image")
    if source_identity == report_identity:
        raise CalibrationError("report output must not overwrite the input image")
    if output_identity == report_identity:
        raise CalibrationError("rectified image and report must use different paths")

    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise CalibrationError(f"cannot read input image: {source_path}")
    rectified, report = rectify_image(image, output_px_per_mm=output_px_per_mm)

    success, encoded = cv2.imencode(".png", rectified)
    if not success:
        raise CalibrationError("OpenCV could not encode the rectified PNG")
    _atomic_write_bytes(output_path, encoded.tobytes())

    report_bytes = (
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(report_path, report_bytes)
    return output_path, report_path, report
