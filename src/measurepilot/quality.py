"""Capture-quality measurements used by calibrated rectification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from .calibration import CalibrationError

DEFAULT_SHARPNESS_THRESHOLD = 80.0


@dataclass(frozen=True, slots=True)
class CaptureQuality:
    """Deterministic image-quality evidence for one input capture."""

    width_px: int
    height_px: int
    sharpness: float
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "height_px": self.height_px,
            "sharpness": round(self.sharpness, 6),
            "warnings": list(self.warnings),
            "width_px": self.width_px,
        }


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise CalibrationError("input image must be a NumPy array")
    if image.size == 0 or image.ndim not in (2, 3):
        raise CalibrationError("input image is empty or has an unsupported shape")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise CalibrationError("input image must be grayscale, BGR, or BGRA")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise CalibrationError("input image is too small")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a supported image to contiguous 8-bit grayscale."""

    _validate_image(image)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if not np.all(np.isfinite(image)):
                raise CalibrationError("input image contains non-finite values")
            maximum = float(np.max(image))
            scale = 255.0 if maximum <= 1.0 else 1.0
            image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return np.ascontiguousarray(cv2.cvtColor(image, conversion))


def evaluate_capture_quality(
    image: np.ndarray,
    *,
    sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
) -> CaptureQuality:
    """Measure sharpness and emit bounded warnings without guessing validity."""

    if not math.isfinite(sharpness_threshold) or sharpness_threshold < 0.0:
        raise CalibrationError("sharpness_threshold must be finite and non-negative")
    gray = to_grayscale(image)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings: list[str] = []
    if sharpness < sharpness_threshold:
        warnings.append(
            "LOW_SHARPNESS: capture may be unsuitable for later contour extraction"
        )
    return CaptureQuality(
        width_px=int(gray.shape[1]),
        height_px=int(gray.shape[0]),
        sharpness=sharpness,
        warnings=tuple(warnings),
    )
