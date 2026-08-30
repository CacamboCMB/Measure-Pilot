"""Capture quality metrics and bounded warning generation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CalibrationError


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_sharpness_score: float = 50.0
    min_input_px_per_mm: float = 1.5
    max_reprojection_error_px: float = 1.5
    reprojection_warning_ratio: float = 0.5

    def validate(self) -> None:
        if self.min_sharpness_score < 0:
            raise CalibrationError("min_sharpness_score must not be negative")
        if self.min_input_px_per_mm <= 0:
            raise CalibrationError("min_input_px_per_mm must be positive")
        if self.max_reprojection_error_px <= 0:
            raise CalibrationError("max_reprojection_error_px must be positive")
        if not 0 < self.reprojection_warning_ratio < 1:
            raise CalibrationError("reprojection_warning_ratio must be between zero and one")


DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise CalibrationError("image must be a non-empty NumPy array")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise CalibrationError(f"unsupported image shape: {image.shape}")


def sharpness_score(image: np.ndarray) -> float:
    """Return variance of the Laplacian as a deterministic focus proxy."""

    gray = to_grayscale(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_warnings(
    *,
    sharpness: float,
    input_px_per_mm: float,
    reprojection_error_px: float,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> list[str]:
    thresholds.validate()
    warnings: list[str] = []
    if sharpness < thresholds.min_sharpness_score:
        warnings.append(
            f"LOW_SHARPNESS: {sharpness:.3f} < {thresholds.min_sharpness_score:.3f}"
        )
    if input_px_per_mm < thresholds.min_input_px_per_mm:
        warnings.append(
            "LOW_METRIC_RESOLUTION: "
            f"{input_px_per_mm:.3f} px/mm < {thresholds.min_input_px_per_mm:.3f} px/mm"
        )
    warning_limit = thresholds.max_reprojection_error_px * thresholds.reprojection_warning_ratio
    if reprojection_error_px > warning_limit:
        warnings.append(
            "ELEVATED_REPROJECTION_ERROR: "
            f"{reprojection_error_px:.3f} px > {warning_limit:.3f} px"
        )
    return warnings
