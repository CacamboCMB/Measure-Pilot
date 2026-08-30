"""End-to-end M2 planar analysis and standalone module CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np

from .features import FeatureConfig, FeatureExtractionResult, extract_features
from .segmentation import (
    AnalysisError,
    ForegroundPolarity,
    SegmentationConfig,
    SegmentationResult,
    segment_foreground,
)

ANALYSIS_FORMAT = "measurepilot-planar-analysis"
ANALYSIS_VERSION = 1


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Machine-readable M2 result and deterministic visual evidence."""

    report: dict[str, Any]
    overlay: np.ndarray
    mask: np.ndarray


def _config_dict(
    segmentation: SegmentationConfig,
    features: FeatureConfig,
) -> dict[str, Any]:
    return {
        "features": {
            "max_circle_residual_mm": features.max_circle_residual_mm,
            "max_circle_residual_ratio": features.max_circle_residual_ratio,
            "min_circle_circularity": features.min_circle_circularity,
            "min_hole_area_mm2": features.min_hole_area_mm2,
            "min_line_length_mm": features.min_line_length_mm,
            "simplify_tolerance_mm": features.simplify_tolerance_mm,
        },
        "segmentation": {
            "auto_score_margin": segmentation.auto_score_margin,
            "marker_exclusion_margin_mm": segmentation.marker_exclusion_margin_mm,
            "max_foreground_fraction": segmentation.max_foreground_fraction,
            "max_secondary_area_ratio": segmentation.max_secondary_area_ratio,
            "min_component_area_mm2": segmentation.min_component_area_mm2,
            "morphology_radius_mm": segmentation.morphology_radius_mm,
            "page_margin_mm": segmentation.page_margin_mm,
            "polarity": segmentation.polarity,
            "px_per_mm": segmentation.px_per_mm,
        },
    }


def _base_overlay(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image.copy())
    raise AnalysisError("overlay source has an unsupported image shape")


def _draw_overlay(
    image: np.ndarray,
    segmentation: SegmentationResult,
    features: FeatureExtractionResult,
) -> np.ndarray:
    overlay = _base_overlay(image)
    outer = np.rint(features.outer_contour_px).astype(np.int32).reshape(-1, 1, 2)
    simplified = np.rint(features.simplified_outer_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [outer], True, (0, 180, 0), 2, cv2.LINE_8)
    cv2.polylines(overlay, [simplified], True, (0, 180, 255), 1, cv2.LINE_8)
    for contour in features.hole_contours_px:
        hole = np.rint(contour).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [hole], True, (0, 0, 220), 2, cv2.LINE_8)
    x, y, width, height = segmentation.bbox_px
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (220, 120, 0), 1)
    center = tuple(int(round(value)) for value in segmentation.centroid_px)
    cv2.drawMarker(
        overlay,
        center,
        (220, 120, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=1,
        line_type=cv2.LINE_8,
    )
    return overlay


def analyze_rectified_image(
    image: np.ndarray,
    *,
    px_per_mm: float,
    polarity: ForegroundPolarity = "auto",
    segmentation_config: SegmentationConfig | None = None,
    feature_config: FeatureConfig = FeatureConfig(),
) -> AnalysisResult:
    """Analyse one controlled, already-rectified planar capture."""

    if segmentation_config is None:
        segmentation_config = SegmentationConfig(
            px_per_mm=px_per_mm,
            polarity=polarity,
        )
    else:
        if not math.isclose(segmentation_config.px_per_mm, px_per_mm, rel_tol=0.0, abs_tol=1e-12):
            raise AnalysisError("px_per_mm conflicts with segmentation_config")
        if polarity != "auto" and polarity != segmentation_config.polarity:
            raise AnalysisError("polarity conflicts with segmentation_config")

    segmentation = segment_foreground(image, segmentation_config)
    extracted = extract_features(
        segmentation.mask,
        px_per_mm=px_per_mm,
        config=feature_config,
    )
    report: dict[str, Any] = {
        "configuration": _config_dict(segmentation_config, feature_config),
        "coordinate_system": {
            "origin": "top_left_of_rectified_a4_page",
            "unit": "mm",
            "x_axis": "right",
            "y_axis": "down",
        },
        "format": ANALYSIS_FORMAT,
        "image": {
            "height_px": int(image.shape[0]),
            "width_px": int(image.shape[1]),
        },
        "segmentation": segmentation.evidence_dict(),
        "version": ANALYSIS_VERSION,
        "warnings": list(segmentation.warnings),
    }
    report.update(extracted.report)
    overlay = _draw_overlay(image, segmentation, extracted)
    return AnalysisResult(report=report, overlay=overlay, mask=segmentation.mask.copy())


def canonical_analysis_json(report: dict[str, Any]) -> bytes:
    """Encode a report deterministically and reject non-finite values."""

    try:
        text = json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisError("analysis report is not canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def _normalised(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _validate_paths(input_path: Path, report_path: Path, overlay_path: Path | None) -> None:
    paths = [_normalised(input_path), _normalised(report_path)]
    if overlay_path is not None:
        paths.append(_normalised(overlay_path))
    if len(set(paths)) != len(paths):
        raise AnalysisError("input, report, and overlay paths must be distinct")


def _write_temp(destination: Path, payload: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        return Path(temporary.name)


def analyze_file(
    input_path: str | Path,
    report_path: str | Path,
    *,
    overlay_path: str | Path | None = None,
    px_per_mm: float,
    polarity: ForegroundPolarity = "auto",
    segmentation_config: SegmentationConfig | None = None,
    feature_config: FeatureConfig = FeatureConfig(),
) -> dict[str, Any]:
    """Analyse a PNG/JPEG and atomically publish JSON and optional overlay."""

    source = Path(input_path)
    report_destination = Path(report_path)
    overlay_destination = Path(overlay_path) if overlay_path is not None else None
    _validate_paths(source, report_destination, overlay_destination)
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AnalysisError(f"unable to decode input image: {source}")
    result = analyze_rectified_image(
        image,
        px_per_mm=px_per_mm,
        polarity=polarity,
        segmentation_config=segmentation_config,
        feature_config=feature_config,
    )
    report_payload = canonical_analysis_json(result.report)
    overlay_payload: bytes | None = None
    if overlay_destination is not None:
        encoded, png = cv2.imencode(
            ".png",
            result.overlay,
            (cv2.IMWRITE_PNG_COMPRESSION, 9),
        )
        if not encoded:
            raise AnalysisError("OpenCV failed to encode the analysis overlay")
        overlay_payload = png.tobytes()

    temporary_report: Path | None = None
    temporary_overlay: Path | None = None
    try:
        temporary_report = _write_temp(report_destination, report_payload)
        if overlay_destination is not None and overlay_payload is not None:
            temporary_overlay = _write_temp(overlay_destination, overlay_payload)
        os.replace(temporary_report, report_destination)
        temporary_report = None
        if overlay_destination is not None and temporary_overlay is not None:
            os.replace(temporary_overlay, overlay_destination)
            temporary_overlay = None
    finally:
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)
        if temporary_overlay is not None:
            temporary_overlay.unlink(missing_ok=True)
    return result.report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m measurepilot.analysis")
    parser.add_argument("input", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--px-per-mm", type=float, required=True)
    parser.add_argument("--polarity", choices=("auto", "dark", "light"), default="auto")
    parser.add_argument("--page-margin-mm", type=float, default=5.0)
    parser.add_argument("--marker-exclusion-margin-mm", type=float, default=3.0)
    parser.add_argument("--morphology-radius-mm", type=float, default=0.5)
    parser.add_argument("--min-component-area-mm2", type=float, default=100.0)
    parser.add_argument("--max-secondary-area-ratio", type=float, default=0.35)
    parser.add_argument("--simplify-tolerance-mm", type=float, default=0.35)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    segmentation = SegmentationConfig(
        px_per_mm=arguments.px_per_mm,
        polarity=arguments.polarity,
        page_margin_mm=arguments.page_margin_mm,
        marker_exclusion_margin_mm=arguments.marker_exclusion_margin_mm,
        morphology_radius_mm=arguments.morphology_radius_mm,
        min_component_area_mm2=arguments.min_component_area_mm2,
        max_secondary_area_ratio=arguments.max_secondary_area_ratio,
    )
    features = FeatureConfig(simplify_tolerance_mm=arguments.simplify_tolerance_mm)
    try:
        report = analyze_file(
            arguments.input,
            arguments.report,
            overlay_path=arguments.overlay,
            px_per_mm=arguments.px_per_mm,
            segmentation_config=segmentation,
            feature_config=features,
        )
    except AnalysisError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
