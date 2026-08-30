"""Deterministic foreground segmentation for rectified MeasurePilot captures.

M2 deliberately supports one dominant planar foreground component. It does not
infer hidden geometry and it does not silently choose between ambiguous
components or foreground polarities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import cv2
import numpy as np

from .calibration import DEFAULT_LAYOUT, MARKER_IDS, CalibrationLayout
from .errors import MeasurePilotError
from .quality import to_grayscale

ForegroundPolarity = Literal["auto", "dark", "light"]


class AnalysisError(MeasurePilotError):
    """Raised when a rectified capture cannot be analysed safely."""


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Physical and decision thresholds for one M2 segmentation run."""

    px_per_mm: float
    polarity: ForegroundPolarity = "auto"
    page_margin_mm: float = 5.0
    marker_exclusion_margin_mm: float = 3.0
    morphology_radius_mm: float = 0.5
    min_component_area_mm2: float = 100.0
    max_secondary_area_ratio: float = 0.35
    max_foreground_fraction: float = 0.75
    auto_score_margin: float = 0.20

    def validate(self) -> None:
        finite_positive = {
            "px_per_mm": self.px_per_mm,
            "min_component_area_mm2": self.min_component_area_mm2,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise AnalysisError(f"{name} must be finite and positive")
        finite_non_negative = {
            "page_margin_mm": self.page_margin_mm,
            "marker_exclusion_margin_mm": self.marker_exclusion_margin_mm,
            "morphology_radius_mm": self.morphology_radius_mm,
        }
        for name, value in finite_non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise AnalysisError(f"{name} must be finite and non-negative")
        ratios = {
            "max_secondary_area_ratio": self.max_secondary_area_ratio,
            "max_foreground_fraction": self.max_foreground_fraction,
            "auto_score_margin": self.auto_score_margin,
        }
        for name, value in ratios.items():
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise AnalysisError(f"{name} must be between 0 and 1")
        if self.polarity not in ("auto", "dark", "light"):
            raise AnalysisError("polarity must be 'auto', 'dark', or 'light'")


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Selected component and the evidence used to select it."""

    mask: np.ndarray
    polarity: Literal["dark", "light"]
    threshold: float
    area_px: int
    area_mm2: float
    bbox_px: tuple[int, int, int, int]
    bbox_mm: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    centroid_mm: tuple[float, float]
    candidate_areas_mm2: tuple[float, ...]
    foreground_fraction: float
    dominance: float
    warnings: tuple[str, ...] = ()

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "area_mm2": round(self.area_mm2, 6),
            "area_px": self.area_px,
            "bbox_mm": [round(value, 6) for value in self.bbox_mm],
            "bbox_px": list(self.bbox_px),
            "candidate_areas_mm2": [
                round(value, 6) for value in self.candidate_areas_mm2
            ],
            "centroid_mm": [round(value, 6) for value in self.centroid_mm],
            "centroid_px": [round(value, 6) for value in self.centroid_px],
            "dominance": round(self.dominance, 9),
            "foreground_fraction": round(self.foreground_fraction, 9),
            "polarity": self.polarity,
            "threshold": round(self.threshold, 6),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    result: SegmentationResult
    score: float


def _expected_shape(
    config: SegmentationConfig,
    layout: CalibrationLayout,
) -> tuple[int, int]:
    return (
        int(round(layout.page_height_mm * config.px_per_mm)),
        int(round(layout.page_width_mm * config.px_per_mm)),
    )


def _valid_region(
    image_shape: tuple[int, int],
    config: SegmentationConfig,
    layout: CalibrationLayout,
) -> np.ndarray:
    height, width = image_shape
    valid = np.full((height, width), 255, dtype=np.uint8)

    margin = int(round(config.page_margin_mm * config.px_per_mm))
    if margin * 2 >= min(width, height):
        raise AnalysisError("page margin removes the complete image")
    if margin:
        valid[:margin, :] = 0
        valid[height - margin :, :] = 0
        valid[:, :margin] = 0
        valid[:, width - margin :] = 0

    extra = config.marker_exclusion_margin_mm
    for marker_id in MARKER_IDS:
        corners = layout.marker_corners_mm(marker_id)
        x0 = max(
            0,
            int(
                math.floor(
                    (float(corners[:, 0].min()) - extra) * config.px_per_mm
                )
            ),
        )
        y0 = max(
            0,
            int(
                math.floor(
                    (float(corners[:, 1].min()) - extra) * config.px_per_mm
                )
            ),
        )
        x1 = min(
            width,
            int(
                math.ceil(
                    (float(corners[:, 0].max()) + extra) * config.px_per_mm
                )
            ),
        )
        y1 = min(
            height,
            int(
                math.ceil(
                    (float(corners[:, 1].max()) + extra) * config.px_per_mm
                )
            ),
        )
        valid[y0:y1, x0:x1] = 0
    return valid


def _morphology(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _touches_invalid_region(component: np.ndarray, valid: np.ndarray) -> bool:
    boundary = cv2.dilate(
        (valid == 0).astype(np.uint8),
        np.ones((3, 3), np.uint8),
    )
    return bool(np.any((component != 0) & (boundary != 0)))


def _component_candidate(
    thresholded: np.ndarray,
    *,
    polarity: Literal["dark", "light"],
    threshold: float,
    valid: np.ndarray,
    config: SegmentationConfig,
) -> _Candidate:
    mask = cv2.bitwise_and(thresholded, valid)
    radius_px = int(round(config.morphology_radius_mm * config.px_per_mm))
    mask = _morphology(mask, radius_px)
    mask = cv2.bitwise_and(mask, valid)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    minimum_px = config.min_component_area_mm2 * config.px_per_mm**2
    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum_px:
            components.append((label, area))
    components.sort(key=lambda item: (-item[1], item[0]))
    if not components:
        raise AnalysisError(f"no qualifying {polarity} foreground component")

    selected_label, selected_area = components[0]
    selected = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    if _touches_invalid_region(selected, valid):
        raise AnalysisError(
            f"dominant {polarity} component touches an excluded boundary"
        )

    valid_area = int(np.count_nonzero(valid))
    fraction = selected_area / valid_area if valid_area else 1.0
    if fraction > config.max_foreground_fraction:
        raise AnalysisError(
            f"dominant {polarity} component occupies an implausible page fraction"
        )

    second_area = components[1][1] if len(components) > 1 else 0
    secondary_ratio = second_area / selected_area if selected_area else 1.0
    if secondary_ratio > config.max_secondary_area_ratio:
        raise AnalysisError(
            f"ambiguous {polarity} foreground: second component is "
            f"{secondary_ratio:.3f} of the dominant area"
        )
    dominance = 1.0 - secondary_ratio

    x = int(stats[selected_label, cv2.CC_STAT_LEFT])
    y = int(stats[selected_label, cv2.CC_STAT_TOP])
    width = int(stats[selected_label, cv2.CC_STAT_WIDTH])
    height = int(stats[selected_label, cv2.CC_STAT_HEIGHT])
    centroid_x, centroid_y = map(float, centroids[selected_label])
    scale = config.px_per_mm
    candidate_areas = tuple(area / scale**2 for _, area in components)
    result = SegmentationResult(
        mask=selected,
        polarity=polarity,
        threshold=threshold,
        area_px=selected_area,
        area_mm2=selected_area / scale**2,
        bbox_px=(x, y, width, height),
        bbox_mm=(x / scale, y / scale, width / scale, height / scale),
        centroid_px=(centroid_x, centroid_y),
        centroid_mm=(centroid_x / scale, centroid_y / scale),
        candidate_areas_mm2=candidate_areas,
        foreground_fraction=fraction,
        dominance=dominance,
    )
    area_preference = max(0.0, 1.0 - abs(fraction - 0.20) / 0.55)
    score = 0.75 * dominance + 0.25 * area_preference
    return _Candidate(result=result, score=score)


def segment_foreground(
    image: np.ndarray,
    config: SegmentationConfig,
    *,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
) -> SegmentationResult:
    """Select one explicit dominant foreground component.

    The image must already represent the complete metric A4 plane at the
    configured pixel density. Auto polarity is accepted only when one valid
    interpretation exists or one candidate has a material score advantage.
    """

    config.validate()
    layout.validate()
    gray = to_grayscale(image)
    expected_height, expected_width = _expected_shape(config, layout)
    if gray.shape != (expected_height, expected_width):
        raise AnalysisError(
            "rectified image dimensions do not match A4 layout at px_per_mm: "
            f"expected {expected_width}x{expected_height}, got "
            f"{gray.shape[1]}x{gray.shape[0]}"
        )

    threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    valid = _valid_region(gray.shape, config, layout)
    polarity_masks = {
        "dark": np.where(gray <= threshold, 255, 0).astype(np.uint8),
        "light": np.where(gray > threshold, 255, 0).astype(np.uint8),
    }

    def evaluate(polarity: Literal["dark", "light"]) -> _Candidate:
        return _component_candidate(
            polarity_masks[polarity],
            polarity=polarity,
            threshold=float(threshold),
            valid=valid,
            config=config,
        )

    if config.polarity in ("dark", "light"):
        return evaluate(config.polarity).result

    candidates: list[_Candidate] = []
    failures: list[str] = []
    for polarity in ("dark", "light"):
        try:
            candidates.append(evaluate(polarity))
        except AnalysisError as exc:
            failures.append(f"{polarity}: {exc}")
    if not candidates:
        raise AnalysisError(
            "auto polarity found no valid foreground; " + "; ".join(failures)
        )
    if len(candidates) == 1:
        return candidates[0].result

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    if candidates[0].score - candidates[1].score < config.auto_score_margin:
        raise AnalysisError(
            "auto polarity is ambiguous; select 'dark' or 'light' explicitly"
        )
    return candidates[0].result
