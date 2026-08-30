"""Metric contour and primitive extraction for the M2 planar slice."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np

from .segmentation import AnalysisError


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Physical tolerances for deterministic M2 feature classification."""

    simplify_tolerance_mm: float = 0.35
    min_line_length_mm: float = 2.0
    min_hole_area_mm2: float = 3.0
    min_circle_circularity: float = 0.84
    max_circle_residual_mm: float = 0.30
    max_circle_residual_ratio: float = 0.04

    def validate(self) -> None:
        positive = {
            "simplify_tolerance_mm": self.simplify_tolerance_mm,
            "min_line_length_mm": self.min_line_length_mm,
            "min_hole_area_mm2": self.min_hole_area_mm2,
            "max_circle_residual_mm": self.max_circle_residual_mm,
            "max_circle_residual_ratio": self.max_circle_residual_ratio,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise AnalysisError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.min_circle_circularity)
            or not 0.0 < self.min_circle_circularity <= 1.0
        ):
            raise AnalysisError("min_circle_circularity must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class FeatureExtractionResult:
    """Contours, recognised primitives, and unresolved evidence."""

    outer_contour_px: np.ndarray
    simplified_outer_px: np.ndarray
    hole_contours_px: tuple[np.ndarray, ...]
    report: dict[str, Any]


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        raise AnalysisError("foreground mask must be a non-empty 2D array")
    binary = np.where(mask != 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        raise AnalysisError("foreground mask is empty")
    return np.ascontiguousarray(binary)


def _rotate_to_smallest(points: np.ndarray) -> np.ndarray:
    keys = [(float(point[0]), float(point[1]), index) for index, point in enumerate(points)]
    _, _, start = min(keys)
    return np.concatenate((points[start:], points[:start]), axis=0)


def _canonical_points(points: np.ndarray, *, closed: bool = True) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(array) < 2:
        raise AnalysisError("contour contains too few points")
    if np.array_equal(array[0], array[-1]):
        array = array[:-1]
    forward = _rotate_to_smallest(array)
    reverse = _rotate_to_smallest(array[::-1])
    forward_key = tuple(map(tuple, forward.tolist()))
    reverse_key = tuple(map(tuple, reverse.tolist()))
    selected = forward if forward_key <= reverse_key else reverse
    if closed:
        selected = np.vstack((selected, selected[0]))
    return selected


def _points_mm(points_px: np.ndarray, px_per_mm: float) -> list[list[float]]:
    return [
        [round(float(x) / px_per_mm, 6), round(float(y) / px_per_mm, 6)]
        for x, y in points_px
    ]


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator <= 1e-18:
        return float(np.linalg.norm(point - start))
    amount = float(np.dot(point - start, vector) / denominator)
    amount = min(1.0, max(0.0, amount))
    projection = start + amount * vector
    return float(np.linalg.norm(point - projection))


def _polygon_rms_distance(points: np.ndarray, polygon: np.ndarray) -> float:
    if np.array_equal(polygon[0], polygon[-1]):
        vertices = polygon[:-1]
    else:
        vertices = polygon
    squared: list[float] = []
    for point in points:
        distance = min(
            _point_segment_distance(point, vertices[index], vertices[(index + 1) % len(vertices)])
            for index in range(len(vertices))
        )
        squared.append(distance * distance)
    return math.sqrt(sum(squared) / len(squared)) if squared else 0.0


def _fit_circle(points_mm: np.ndarray) -> tuple[float, float, float, float]:
    x = points_mm[:, 0]
    y = points_mm[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    target = x * x + y * y
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise AnalysisError("hole contour cannot support a circle fit")
    center_x, center_y, constant = map(float, solution)
    radius_squared = constant + center_x * center_x + center_y * center_y
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise AnalysisError("hole contour produced an invalid circle fit")
    radius = math.sqrt(radius_squared)
    radii = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    residual = float(np.sqrt(np.mean(np.square(radii - radius))))
    return center_x, center_y, radius, residual


def _line_features(
    simplified_px: np.ndarray,
    original_px: np.ndarray,
    px_per_mm: float,
    config: FeatureConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    vertices = simplified_px[:-1] if np.array_equal(simplified_px[0], simplified_px[-1]) else simplified_px
    rms_px = _polygon_rms_distance(original_px, vertices)
    rms_mm = rms_px / px_per_mm
    fit_confidence = max(0.0, min(1.0, 1.0 - rms_mm / config.simplify_tolerance_mm))
    lines: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for index in range(len(vertices)):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]
        delta_mm = (end - start) / px_per_mm
        length_mm = float(np.linalg.norm(delta_mm))
        angle_deg = math.degrees(math.atan2(float(delta_mm[1]), float(delta_mm[0])))
        if angle_deg < 0.0:
            angle_deg += 360.0
        evidence = {
            "end_mm": [round(float(end[0]) / px_per_mm, 6), round(float(end[1]) / px_per_mm, 6)],
            "length_mm": round(length_mm, 6),
            "source_edge_index": index,
            "start_mm": [round(float(start[0]) / px_per_mm, 6), round(float(start[1]) / px_per_mm, 6)],
        }
        if length_mm < config.min_line_length_mm:
            unresolved.append(
                {
                    "code": "SHORT_OUTER_EDGE",
                    "evidence": evidence,
                    "message": "simplified outer edge is too short for line classification",
                }
            )
            continue
        length_confidence = min(1.0, length_mm / (config.min_line_length_mm * 3.0))
        lines.append(
            {
                "angle_deg": round(angle_deg, 6),
                "confidence": round(fit_confidence * length_confidence, 6),
                "end_mm": evidence["end_mm"],
                "id": f"outer-line-{index:03d}",
                "length_mm": evidence["length_mm"],
                "provenance": "derived_from_simplified_outer_contour",
                "start_mm": evidence["start_mm"],
                "type": "line_segment",
            }
        )
    return lines, unresolved, rms_mm


def _hole_features(
    hole_contours: Iterable[np.ndarray],
    px_per_mm: float,
    config: FeatureConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    circles: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    contour_reports: list[dict[str, Any]] = []
    for index, contour_px in enumerate(hole_contours):
        open_points = contour_px[:-1] if np.array_equal(contour_px[0], contour_px[-1]) else contour_px
        cv_contour = open_points.astype(np.float32).reshape(-1, 1, 2)
        area_mm2 = abs(float(cv2.contourArea(cv_contour))) / px_per_mm**2
        perimeter_mm = float(cv2.arcLength(cv_contour, True)) / px_per_mm
        circularity = (
            4.0 * math.pi * area_mm2 / (perimeter_mm * perimeter_mm)
            if perimeter_mm > 0.0
            else 0.0
        )
        contour_reports.append(
            {
                "area_mm2": round(area_mm2, 6),
                "id": f"hole-contour-{index:03d}",
                "perimeter_mm": round(perimeter_mm, 6),
                "points_mm": _points_mm(contour_px, px_per_mm),
            }
        )
        if area_mm2 < config.min_hole_area_mm2:
            unresolved.append(
                {
                    "code": "HOLE_TOO_SMALL",
                    "contour_id": f"hole-contour-{index:03d}",
                    "message": "enclosed contour is below the supported hole area",
                }
            )
            continue
        try:
            center_x, center_y, radius, residual = _fit_circle(open_points / px_per_mm)
        except AnalysisError as exc:
            unresolved.append(
                {
                    "code": "HOLE_CIRCLE_FIT_FAILED",
                    "contour_id": f"hole-contour-{index:03d}",
                    "message": str(exc),
                }
            )
            continue
        diameter = radius * 2.0
        residual_limit = max(
            config.max_circle_residual_mm,
            config.max_circle_residual_ratio * diameter,
        )
        residual_score = max(0.0, min(1.0, 1.0 - residual / residual_limit))
        circularity_score = max(
            0.0,
            min(
                1.0,
                (circularity - config.min_circle_circularity)
                / max(1e-9, 1.0 - config.min_circle_circularity),
            ),
        )
        if circularity < config.min_circle_circularity or residual > residual_limit:
            unresolved.append(
                {
                    "circularity": round(circularity, 6),
                    "code": "NON_CIRCULAR_HOLE",
                    "contour_id": f"hole-contour-{index:03d}",
                    "fit_residual_mm": round(residual, 6),
                    "message": "enclosed contour is retained but not classified as a circular hole",
                }
            )
            continue
        circles.append(
            {
                "center_mm": [round(center_x, 6), round(center_y, 6)],
                "circularity": round(circularity, 6),
                "confidence": round(0.5 * residual_score + 0.5 * circularity_score, 6),
                "contour_id": f"hole-contour-{index:03d}",
                "diameter_mm": round(diameter, 6),
                "fit_residual_mm": round(residual, 6),
                "id": f"circular-hole-{len(circles):03d}",
                "provenance": "derived_from_enclosed_contour",
                "type": "circular_hole",
            }
        )
    return circles, unresolved, contour_reports


def extract_features(
    mask: np.ndarray,
    *,
    px_per_mm: float,
    config: FeatureConfig = FeatureConfig(),
) -> FeatureExtractionResult:
    """Extract one outer contour, direct holes, and conservative primitives."""

    if not math.isfinite(px_per_mm) or px_per_mm <= 0.0:
        raise AnalysisError("px_per_mm must be finite and positive")
    config.validate()
    binary = _validate_mask(mask)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        raise AnalysisError("foreground mask contains no contour")
    relationships = hierarchy[0]
    external = [index for index, values in enumerate(relationships) if int(values[3]) == -1]
    if len(external) != 1:
        raise AnalysisError("foreground mask must contain exactly one external contour")
    outer_index = external[0]
    outer_raw = np.asarray(contours[outer_index], dtype=np.float64).reshape(-1, 2)
    if len(outer_raw) < 4:
        raise AnalysisError("outer contour contains too few points")
    outer_px = _canonical_points(outer_raw)

    epsilon_px = config.simplify_tolerance_mm * px_per_mm
    simplified_raw = cv2.approxPolyDP(
        outer_raw.astype(np.float32).reshape(-1, 1, 2),
        epsilon_px,
        True,
    ).reshape(-1, 2)
    if len(simplified_raw) < 3:
        raise AnalysisError("outer contour simplification collapsed the part")
    simplified_px = _canonical_points(simplified_raw)

    hole_indices = [
        index
        for index, values in enumerate(relationships)
        if int(values[3]) == outer_index
    ]
    canonical_holes = tuple(
        _canonical_points(np.asarray(contours[index], dtype=np.float64).reshape(-1, 2))
        for index in hole_indices
    )
    canonical_holes = tuple(
        sorted(
            canonical_holes,
            key=lambda contour: (
                float(np.mean(contour[:-1, 0])),
                float(np.mean(contour[:-1, 1])),
            ),
        )
    )

    lines, line_unresolved, simplification_rms_mm = _line_features(
        simplified_px,
        outer_px[:-1],
        px_per_mm,
        config,
    )
    circles, hole_unresolved, hole_reports = _hole_features(
        canonical_holes,
        px_per_mm,
        config,
    )

    outer_cv = outer_px[:-1].astype(np.float32).reshape(-1, 1, 2)
    outer_area_mm2 = abs(float(cv2.contourArea(outer_cv))) / px_per_mm**2
    outer_perimeter_mm = float(cv2.arcLength(outer_cv, True)) / px_per_mm
    report: dict[str, Any] = {
        "contours": {
            "holes": hole_reports,
            "outer": {
                "area_mm2": round(outer_area_mm2, 6),
                "perimeter_mm": round(outer_perimeter_mm, 6),
                "points_mm": _points_mm(outer_px, px_per_mm),
                "simplified_points_mm": _points_mm(simplified_px, px_per_mm),
                "simplification_rms_mm": round(simplification_rms_mm, 6),
            },
        },
        "features": {
            "circular_holes": circles,
            "line_segments": lines,
        },
        "unresolved": line_unresolved + hole_unresolved,
    }
    return FeatureExtractionResult(
        outer_contour_px=outer_px,
        simplified_outer_px=simplified_px,
        hole_contours_px=canonical_holes,
        report=report,
    )
