"""Deterministic M2 segmentation and corrigible 2D feature extraction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .calibration import CALIBRATION_LAYOUT_VERSION, DEFAULT_LAYOUT, CalibrationLayout
from .errors import MeasurePilotError
from .quality import to_grayscale

DETECTION_SCHEMA_VERSION = 1
DETECTION_MODEL_VERSION = "measurepilot-detection-v1"
ESTIMATED_STATUS = "estimated"
USER_CORRECTED_STATUS = "user_corrected"
MEASURED_STATUS = "measured"
_ALLOWED_STATUSES = {ESTIMATED_STATUS, USER_CORRECTED_STATUS, MEASURED_STATUS}
_CIRCULARITY_THRESHOLD = 0.82
_MIN_PART_AREA_MM2 = 100.0
_MIN_INTERNAL_FEATURE_AREA_MM2 = 3.0
_AMBIGUITY_AREA_RATIO = 0.35


class DetectionError(MeasurePilotError):
    """Raised when a defensible M2 detection cannot be produced."""


def _finite_positive(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DetectionError(f"{field_name} must be a positive finite number")
    return number


def _point_tuple(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise DetectionError(f"{field_name} must contain exactly two coordinates")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise DetectionError(f"{field_name} coordinates must be finite")
    return (x, y)


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    values = (orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b))
    epsilon = 1e-9
    if any(abs(value) <= epsilon for value in values):
        # Collinear contact between non-adjacent edges is invalid for this model.
        def within(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
            return (
                min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon
                and min(p[1], r[1]) - epsilon <= q[1] <= max(p[1], r[1]) + epsilon
            )

        return (
            (abs(values[0]) <= epsilon and within(a, c, b))
            or (abs(values[1]) <= epsilon and within(a, d, b))
            or (abs(values[2]) <= epsilon and within(c, a, d))
            or (abs(values[3]) <= epsilon and within(c, b, d))
        )
    return (values[0] > 0) != (values[1] > 0) and (values[2] > 0) != (values[3] > 0)


def _validate_simple_polygon(points: Sequence[tuple[float, float]], field_name: str) -> None:
    count = len(points)
    for first_index in range(count):
        first_next = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if (
                first_index == second_index
                or first_next == second_index
                or second_next == first_index
            ):
                continue
            if _segments_intersect(
                points[first_index],
                points[first_next],
                points[second_index],
                points[second_next],
            ):
                raise DetectionError(f"{field_name} must not self-intersect")


def _canonical_polygon(
    points: Iterable[Sequence[float]], *, field_name: str
) -> tuple[tuple[float, float], ...]:
    canonical = [_point_tuple(point, field_name) for point in points]
    if len(canonical) > 1 and canonical[0] == canonical[-1]:
        canonical.pop()
    if len(canonical) < 3 or len(set(canonical)) < 3:
        raise DetectionError(f"{field_name} must contain at least three distinct points")
    _validate_simple_polygon(canonical, field_name)
    area = _polygon_area(canonical)
    if abs(area) < 1e-6:
        raise DetectionError(f"{field_name} must have non-zero area")
    if area < 0:
        canonical.reverse()
    start = min(range(len(canonical)), key=lambda index: (canonical[index][1], canonical[index][0]))
    canonical = canonical[start:] + canonical[:start]
    return tuple(canonical)


def _validate_feature_id(value: str, field_name: str = "feature_id") -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DetectionError(f"{field_name} must be non-empty trimmed text")
    if len(value) > 100:
        raise DetectionError(f"{field_name} must be at most 100 characters")
    return value


def _validate_status(value: str) -> str:
    if value not in _ALLOWED_STATUSES:
        raise DetectionError(f"unsupported feature status: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class CircleFeature:
    feature_id: str
    center_mm: tuple[float, float]
    radius_mm: float
    uncertainty_mm: float
    status: str = ESTIMATED_STATUS

    def validate(self) -> None:
        _validate_feature_id(self.feature_id)
        _point_tuple(self.center_mm, "center_mm")
        _finite_positive(self.radius_mm, "radius_mm")
        _finite_positive(self.uncertainty_mm, "uncertainty_mm")
        _validate_status(self.status)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "feature_id": self.feature_id,
            "kind": "circle",
            "center_mm": list(self.center_mm),
            "radius_mm": self.radius_mm,
            "uncertainty_mm": self.uncertainty_mm,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CircleFeature":
        if value.get("kind") != "circle":
            raise DetectionError("circle feature kind must be 'circle'")
        feature = cls(
            feature_id=value["feature_id"],
            center_mm=_point_tuple(value["center_mm"], "center_mm"),
            radius_mm=float(value["radius_mm"]),
            uncertainty_mm=float(value["uncertainty_mm"]),
            status=value["status"],
        )
        feature.validate()
        return feature


@dataclass(frozen=True, slots=True)
class PolygonFeature:
    feature_id: str
    points_mm: tuple[tuple[float, float], ...]
    uncertainty_mm: float
    status: str = ESTIMATED_STATUS

    def validate(self) -> None:
        _validate_feature_id(self.feature_id)
        canonical = _canonical_polygon(self.points_mm, field_name=f"{self.feature_id}.points_mm")
        if canonical != self.points_mm:
            raise DetectionError(f"{self.feature_id}.points_mm must use canonical ordering")
        _finite_positive(self.uncertainty_mm, "uncertainty_mm")
        _validate_status(self.status)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "feature_id": self.feature_id,
            "kind": "polygon",
            "points_mm": [list(point) for point in self.points_mm],
            "uncertainty_mm": self.uncertainty_mm,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PolygonFeature":
        if value.get("kind") != "polygon":
            raise DetectionError("polygon feature kind must be 'polygon'")
        feature = cls(
            feature_id=value["feature_id"],
            points_mm=_canonical_polygon(value["points_mm"], field_name="points_mm"),
            uncertainty_mm=float(value["uncertainty_mm"]),
            status=value["status"],
        )
        feature.validate()
        return feature


@dataclass(frozen=True, slots=True)
class DetectionResult:
    source_sha256: str
    px_per_mm: float
    profile_points_mm: tuple[tuple[float, float], ...]
    profile_uncertainty_mm: float
    profile_status: str
    circles: tuple[CircleFeature, ...] = ()
    cutouts: tuple[PolygonFeature, ...] = ()
    history: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: int = DETECTION_SCHEMA_VERSION
    model_version: str = DETECTION_MODEL_VERSION
    layout_version: str = CALIBRATION_LAYOUT_VERSION

    def validate(self) -> None:
        if self.schema_version != DETECTION_SCHEMA_VERSION:
            raise DetectionError(f"unsupported detection schema_version: {self.schema_version!r}")
        if self.model_version != DETECTION_MODEL_VERSION:
            raise DetectionError(f"unsupported detection model_version: {self.model_version!r}")
        if self.layout_version != CALIBRATION_LAYOUT_VERSION:
            raise DetectionError(f"unsupported layout_version: {self.layout_version!r}")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise DetectionError("source_sha256 must be a lowercase SHA-256 hex digest")
        _finite_positive(self.px_per_mm, "px_per_mm")
        canonical_profile = _canonical_polygon(
            self.profile_points_mm, field_name="profile_points_mm"
        )
        if canonical_profile != self.profile_points_mm:
            raise DetectionError("profile_points_mm must use canonical ordering")
        _finite_positive(self.profile_uncertainty_mm, "profile_uncertainty_mm")
        _validate_status(self.profile_status)
        feature_ids: list[str] = []
        for circle in self.circles:
            circle.validate()
            feature_ids.append(circle.feature_id)
        for cutout in self.cutouts:
            cutout.validate()
            feature_ids.append(cutout.feature_id)
        if len(feature_ids) != len(set(feature_ids)):
            raise DetectionError("feature IDs must be unique")
        if tuple(sorted(self.circles, key=lambda item: item.feature_id)) != self.circles:
            raise DetectionError("circles must be sorted by feature_id")
        if tuple(sorted(self.cutouts, key=lambda item: item.feature_id)) != self.cutouts:
            raise DetectionError("cutouts must be sorted by feature_id")
        if not all(isinstance(item, str) for item in self.warnings):
            raise DetectionError("warnings must contain strings")
        if not all(isinstance(item, dict) for item in self.history):
            raise DetectionError("history must contain JSON objects")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "layout_version": self.layout_version,
            "source_sha256": self.source_sha256,
            "px_per_mm": self.px_per_mm,
            "profile": {
                "feature_id": "profile",
                "kind": "polygon",
                "points_mm": [list(point) for point in self.profile_points_mm],
                "uncertainty_mm": self.profile_uncertainty_mm,
                "status": self.profile_status,
            },
            "circles": [circle.to_dict() for circle in self.circles],
            "cutouts": [cutout.to_dict() for cutout in self.cutouts],
            "warnings": list(self.warnings),
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectionResult":
        try:
            profile = value["profile"]
            result = cls(
                schema_version=int(value["schema_version"]),
                model_version=value["model_version"],
                layout_version=value["layout_version"],
                source_sha256=value["source_sha256"],
                px_per_mm=float(value["px_per_mm"]),
                profile_points_mm=_canonical_polygon(
                    profile["points_mm"], field_name="profile.points_mm"
                ),
                profile_uncertainty_mm=float(profile["uncertainty_mm"]),
                profile_status=profile["status"],
                circles=tuple(
                    sorted(
                        (CircleFeature.from_dict(item) for item in value.get("circles", [])),
                        key=lambda item: item.feature_id,
                    )
                ),
                cutouts=tuple(
                    sorted(
                        (PolygonFeature.from_dict(item) for item in value.get("cutouts", [])),
                        key=lambda item: item.feature_id,
                    )
                ),
                warnings=tuple(value.get("warnings", [])),
                history=tuple(value.get("history", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectionError(f"invalid detection JSON: {exc}") from exc
        result.validate()
        return result

    def with_updates(self, **changes: Any) -> "DetectionResult":
        updated = replace(self, **changes)
        updated.validate()
        return updated


def canonical_detection_bytes(result: DetectionResult) -> bytes:
    return (
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    if path.exists() and path.is_dir():
        raise DetectionError(f"output is a directory: {path}")
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


def write_detection(result: DetectionResult, destination: str | os.PathLike[str]) -> Path:
    target = Path(destination)
    if target.suffix.lower() != ".json":
        raise DetectionError("detection output must use the .json extension")
    _atomic_write_bytes(target, canonical_detection_bytes(result))
    return target


def read_detection(source: str | os.PathLike[str]) -> DetectionResult:
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectionError(f"cannot read detection JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DetectionError("detection JSON must contain an object")
    return DetectionResult.from_dict(value)


def _source_image_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(image.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _component_candidates(
    binary: np.ndarray, px_per_mm: float
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area_px = max(1, int(round(_MIN_PART_AREA_MM2 * px_per_mm * px_per_mm)))
    candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area >= minimum_area_px:
            candidates.append((label, area))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates, labels, statistics


def _contour_to_page_mm(
    contour: np.ndarray,
    *,
    crop_x_px: int,
    crop_y_px: int,
    px_per_mm: float,
    epsilon_mm: float,
) -> tuple[tuple[float, float], ...]:
    epsilon_px = max(0.5, epsilon_mm * px_per_mm)
    approximation = cv2.approxPolyDP(contour, epsilon_px, True).reshape(-1, 2)
    points = [
        ((float(x) + crop_x_px) / px_per_mm, (float(y) + crop_y_px) / px_per_mm)
        for x, y in approximation
    ]
    return _canonical_polygon(points, field_name="detected contour")


def _child_contour_indices(hierarchy: np.ndarray, outer_index: int) -> list[int]:
    children: list[int] = []
    child = int(hierarchy[0, outer_index, 2])
    while child != -1:
        children.append(child)
        child = int(hierarchy[0, child, 0])
    return children


def detect_planar_part(
    image: np.ndarray,
    *,
    px_per_mm: float,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
    source_sha256: str | None = None,
) -> DetectionResult:
    """Detect one supported high-contrast planar part in a rectified A4 image."""

    layout.validate()
    resolution = _finite_positive(px_per_mm, "px_per_mm")
    gray = to_grayscale(image)
    expected_width = int(round(layout.page_width_mm * resolution))
    expected_height = int(round(layout.page_height_mm * resolution))
    if abs(gray.shape[1] - expected_width) > 1 or abs(gray.shape[0] - expected_height) > 1:
        raise DetectionError(
            "rectified image dimensions do not match the versioned A4 layout at the supplied px/mm"
        )

    inner_margin_mm = 2.0
    crop_x_px = int(round((layout.work_area_x_mm + inner_margin_mm) * resolution))
    crop_y_px = int(round((layout.work_area_y_mm + inner_margin_mm) * resolution))
    crop_right_px = int(
        round((layout.work_area_x_mm + layout.work_area_width_mm - inner_margin_mm) * resolution)
    )
    crop_bottom_px = int(
        round((layout.work_area_y_mm + layout.work_area_height_mm - inner_margin_mm) * resolution)
    )
    crop = gray[crop_y_px:crop_bottom_px, crop_x_px:crop_right_px]
    if crop.size == 0:
        raise DetectionError("versioned work area is empty at the supplied resolution")

    blurred = cv2.GaussianBlur(crop, (5, 5), 0.0)
    _threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    candidates, labels, statistics = _component_candidates(binary, resolution)
    if not candidates:
        raise DetectionError("no supported planar part was detected in the work area")
    if len(candidates) > 1 and candidates[1][1] >= candidates[0][1] * _AMBIGUITY_AREA_RATIO:
        raise DetectionError("multiple plausible parts were detected; the capture is ambiguous")

    selected_label, _selected_area = candidates[0]
    left = int(statistics[selected_label, cv2.CC_STAT_LEFT])
    top = int(statistics[selected_label, cv2.CC_STAT_TOP])
    width = int(statistics[selected_label, cv2.CC_STAT_WIDTH])
    height = int(statistics[selected_label, cv2.CC_STAT_HEIGHT])
    if left <= 1 or top <= 1 or left + width >= crop.shape[1] - 1 or top + height >= crop.shape[0] - 1:
        raise DetectionError("detected part touches the work-area boundary and may be clipped")

    component = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    closing_side = max(1, int(round(0.4 * resolution)))
    if closing_side % 2 == 0:
        closing_side += 1
    component = cv2.morphologyEx(
        component,
        cv2.MORPH_CLOSE,
        np.ones((closing_side, closing_side), dtype=np.uint8),
    )
    contours, hierarchy = cv2.findContours(
        component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None or not contours:
        raise DetectionError("part component did not produce a valid contour hierarchy")
    outer_candidates = [
        index for index in range(len(contours)) if int(hierarchy[0, index, 3]) == -1
    ]
    if not outer_candidates:
        raise DetectionError("part component has no external contour")
    outer_index = max(outer_candidates, key=lambda index: abs(cv2.contourArea(contours[index])))

    uncertainty_mm = max(0.1, 0.5 / resolution)
    profile = _contour_to_page_mm(
        contours[outer_index],
        crop_x_px=crop_x_px,
        crop_y_px=crop_y_px,
        px_per_mm=resolution,
        epsilon_mm=0.25,
    )

    circles: list[CircleFeature] = []
    cutouts: list[PolygonFeature] = []
    minimum_internal_area_px = _MIN_INTERNAL_FEATURE_AREA_MM2 * resolution * resolution
    circle_candidates: list[tuple[float, float, float, float]] = []
    polygon_candidates: list[tuple[tuple[tuple[float, float], ...], float, float]] = []
    for child_index in _child_contour_indices(hierarchy, outer_index):
        contour = contours[child_index]
        area_px = abs(float(cv2.contourArea(contour)))
        if area_px < minimum_internal_area_px:
            continue
        perimeter_px = float(cv2.arcLength(contour, True))
        if perimeter_px <= 0:
            continue
        circularity = float(4.0 * math.pi * area_px / (perimeter_px * perimeter_px))
        moments = cv2.moments(contour)
        if circularity >= _CIRCULARITY_THRESHOLD and moments["m00"] != 0:
            center_x_px = moments["m10"] / moments["m00"]
            center_y_px = moments["m01"] / moments["m00"]
            radius_mm = math.sqrt(area_px / math.pi) / resolution
            circle_candidates.append(
                (
                    (center_x_px + crop_x_px) / resolution,
                    (center_y_px + crop_y_px) / resolution,
                    radius_mm,
                    circularity,
                )
            )
        else:
            polygon = _contour_to_page_mm(
                contour,
                crop_x_px=crop_x_px,
                crop_y_px=crop_y_px,
                px_per_mm=resolution,
                epsilon_mm=0.25,
            )
            centroid = np.mean(np.asarray(polygon, dtype=np.float64), axis=0)
            polygon_candidates.append((polygon, float(centroid[1]), float(centroid[0])))

    circle_candidates.sort(key=lambda item: (item[1], item[0], item[2]))
    for index, (center_x, center_y, radius, _circularity) in enumerate(circle_candidates, start=1):
        circles.append(
            CircleFeature(
                feature_id=f"circle-{index:03d}",
                center_mm=(center_x, center_y),
                radius_mm=radius,
                uncertainty_mm=uncertainty_mm,
            )
        )
    polygon_candidates.sort(key=lambda item: (item[1], item[2]))
    for index, (polygon, _centroid_y, _centroid_x) in enumerate(polygon_candidates, start=1):
        cutouts.append(
            PolygonFeature(
                feature_id=f"cutout-{index:03d}",
                points_mm=polygon,
                uncertainty_mm=uncertainty_mm,
            )
        )

    result = DetectionResult(
        source_sha256=source_sha256 or _source_image_sha256(image),
        px_per_mm=resolution,
        profile_points_mm=profile,
        profile_uncertainty_mm=uncertainty_mm,
        profile_status=ESTIMATED_STATUS,
        circles=tuple(circles),
        cutouts=tuple(cutouts),
        history=(
            {
                "event_id": "detection-0001",
                "kind": "automatic_detection",
                "model_version": DETECTION_MODEL_VERSION,
            },
        ),
    )
    result.validate()
    return result


def render_detection_overlay(
    image: np.ndarray, result: DetectionResult
) -> np.ndarray:
    result.validate()
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        overlay = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        overlay = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise DetectionError(f"unsupported overlay image shape: {image.shape}")

    def pixels(points: Sequence[tuple[float, float]]) -> np.ndarray:
        return np.asarray(
            [[round(x * result.px_per_mm), round(y * result.px_per_mm)] for x, y in points],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

    cv2.polylines(overlay, [pixels(result.profile_points_mm)], True, (0, 0, 255), 2)
    for cutout in result.cutouts:
        cv2.polylines(overlay, [pixels(cutout.points_mm)], True, (255, 0, 255), 2)
    for circle in result.circles:
        center = (
            round(circle.center_mm[0] * result.px_per_mm),
            round(circle.center_mm[1] * result.px_per_mm),
        )
        radius = max(1, round(circle.radius_mm * result.px_per_mm))
        cv2.circle(overlay, center, radius, (0, 255, 0), 2)
    return overlay


def detect_image_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    px_per_mm: float,
    overlay_destination: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path | None, DetectionResult]:
    source_path = Path(source)
    output_path = Path(destination)
    overlay_path = Path(overlay_destination) if overlay_destination is not None else None
    identities = [source_path.resolve(strict=False), output_path.resolve(strict=False)]
    if overlay_path is not None:
        identities.append(overlay_path.resolve(strict=False))
    if len(identities) != len(set(identities)):
        raise DetectionError("source, detection JSON, and overlay must use different paths")
    if overlay_path is not None and overlay_path.suffix.lower() != ".png":
        raise DetectionError("detection overlay must use the .png extension")

    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise DetectionError(f"cannot read input image: {source_path}") from exc
    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DetectionError(f"cannot decode input image: {source_path}")
    result = detect_planar_part(
        image,
        px_per_mm=px_per_mm,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )

    encoded_overlay: bytes | None = None
    if overlay_path is not None:
        success, encoded = cv2.imencode(".png", render_detection_overlay(image, result))
        if not success:
            raise DetectionError("OpenCV could not encode the detection overlay")
        encoded_overlay = encoded.tobytes()

    write_detection(result, output_path)
    if overlay_path is not None and encoded_overlay is not None:
        _atomic_write_bytes(overlay_path, encoded_overlay)
    return output_path, overlay_path, result
