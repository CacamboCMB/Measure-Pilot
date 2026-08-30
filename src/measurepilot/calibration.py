"""Versioned A4 calibration layout and deterministic sheet generation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.colors import black
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from .errors import MeasurePilotError

CALIBRATION_LAYOUT_VERSION = "measurepilot-a4-v1"
ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50
REQUIRED_MARKER_IDS = (0, 1, 2, 3)


class CalibrationError(MeasurePilotError):
    """Raised when calibration assets or inputs cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class MarkerPlacement:
    marker_id: int
    x_mm: float
    y_mm: float
    size_mm: float

    def corners_mm(self) -> np.ndarray:
        """Return TL, TR, BR, BL corners in page coordinates."""

        x0, y0, size = self.x_mm, self.y_mm, self.size_mm
        return np.asarray(
            [
                [x0, y0],
                [x0 + size, y0],
                [x0 + size, y0 + size],
                [x0, y0 + size],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class CalibrationLayout:
    version: str = CALIBRATION_LAYOUT_VERSION
    page_width_mm: float = 210.0
    page_height_mm: float = 297.0
    marker_size_mm: float = 30.0
    marker_margin_mm: float = 15.0
    work_area_x_mm: float = 52.0
    work_area_y_mm: float = 72.0
    work_area_width_mm: float = 106.0
    work_area_height_mm: float = 160.0
    ruler_x_mm: float = 55.0
    ruler_y_mm: float = 266.0
    ruler_length_mm: float = 100.0

    @property
    def marker_placements(self) -> tuple[MarkerPlacement, ...]:
        right = self.page_width_mm - self.marker_margin_mm - self.marker_size_mm
        bottom = self.page_height_mm - self.marker_margin_mm - self.marker_size_mm
        return (
            MarkerPlacement(0, self.marker_margin_mm, self.marker_margin_mm, self.marker_size_mm),
            MarkerPlacement(1, right, self.marker_margin_mm, self.marker_size_mm),
            MarkerPlacement(2, right, bottom, self.marker_size_mm),
            MarkerPlacement(3, self.marker_margin_mm, bottom, self.marker_size_mm),
        )

    def placement(self, marker_id: int) -> MarkerPlacement:
        for placement in self.marker_placements:
            if placement.marker_id == marker_id:
                return placement
        raise CalibrationError(f"marker {marker_id} is not part of layout {self.version}")

    def validate(self) -> None:
        if self.version != CALIBRATION_LAYOUT_VERSION:
            raise CalibrationError(f"unsupported calibration layout version: {self.version}")
        if (self.page_width_mm, self.page_height_mm) != (210.0, 297.0):
            raise CalibrationError("version-1 calibration layout must use A4 dimensions")
        if tuple(item.marker_id for item in self.marker_placements) != REQUIRED_MARKER_IDS:
            raise CalibrationError("version-1 marker IDs must be 0, 1, 2, and 3")
        if self.marker_size_mm <= 0 or self.marker_margin_mm <= 0:
            raise CalibrationError("marker size and margin must be positive")
        if self.ruler_length_mm != 100.0:
            raise CalibrationError("version-1 verification ruler must be exactly 100 mm")


DEFAULT_LAYOUT = CalibrationLayout()


def aruco_dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)


def marker_modules(marker_id: int) -> np.ndarray:
    """Return the exact 6x6 black/white module grid for a version-1 marker."""

    if marker_id not in REQUIRED_MARKER_IDS:
        raise CalibrationError(f"unsupported marker ID: {marker_id}")
    return cv2.aruco.generateImageMarker(
        aruco_dictionary(), marker_id, 6, borderBits=1
    )


def render_calibration_image(
    *, layout: CalibrationLayout = DEFAULT_LAYOUT, px_per_mm: float = 4.0
) -> np.ndarray:
    """Render a metric raster reference used by tests and diagnostics."""

    layout.validate()
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        raise CalibrationError("px_per_mm must be a positive finite number")
    width = int(round(layout.page_width_mm * px_per_mm))
    height = int(round(layout.page_height_mm * px_per_mm))
    image = np.full((height, width), 255, dtype=np.uint8)
    for placement in layout.marker_placements:
        side = int(round(placement.size_mm * px_per_mm))
        marker = cv2.aruco.generateImageMarker(
            aruco_dictionary(), placement.marker_id, side, borderBits=1
        )
        x = int(round(placement.x_mm * px_per_mm))
        y = int(round(placement.y_mm * px_per_mm))
        image[y : y + side, x : x + side] = marker
    return image


def _draw_vector_marker(
    pdf: canvas.Canvas, placement: MarkerPlacement, layout: CalibrationLayout
) -> None:
    modules = marker_modules(placement.marker_id)
    module_mm = placement.size_mm / modules.shape[0]
    pdf.setFillColor(black)
    for row in range(modules.shape[0]):
        for column in range(modules.shape[1]):
            if modules[row, column] != 0:
                continue
            x_mm = placement.x_mm + column * module_mm
            top_mm = placement.y_mm + row * module_mm
            y_mm = layout.page_height_mm - top_mm - module_mm
            pdf.rect(x_mm * mm, y_mm * mm, module_mm * mm, module_mm * mm, stroke=0, fill=1)


def _draw_text_from_top(
    pdf: canvas.Canvas,
    layout: CalibrationLayout,
    x_mm: float,
    y_mm: float,
    text: str,
    *,
    font: str = "Helvetica",
    size: float = 9.0,
    centered: bool = False,
) -> None:
    pdf.setFont(font, size)
    baseline = (layout.page_height_mm - y_mm) * mm
    if centered:
        pdf.drawCentredString(x_mm * mm, baseline, text)
    else:
        pdf.drawString(x_mm * mm, baseline, text)


def _draw_ruler(pdf: canvas.Canvas, layout: CalibrationLayout) -> None:
    x0 = layout.ruler_x_mm
    y = layout.ruler_y_mm
    pdf.setStrokeColor(black)
    pdf.setLineWidth(0.35)
    pdf.line(x0 * mm, (layout.page_height_mm - y) * mm, (x0 + 100.0) * mm, (layout.page_height_mm - y) * mm)
    for index in range(101):
        tick = 3.0 if index % 10 == 0 else 1.6 if index % 5 == 0 else 0.9
        x = x0 + index
        pdf.line(
            x * mm,
            (layout.page_height_mm - y) * mm,
            x * mm,
            (layout.page_height_mm - y + tick) * mm,
        )
        if index % 10 == 0:
            _draw_text_from_top(pdf, layout, x, y - 4.0, str(index), size=6.5, centered=True)
    _draw_text_from_top(
        pdf,
        layout,
        x0,
        y + 4.5,
        "100 mm verification ruler",
        size=7.0,
    )


def _write_calibration_pdf(path: Path, layout: CalibrationLayout) -> None:
    pdf = canvas.Canvas(
        str(path),
        pagesize=A4,
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle("MeasurePilot Calibration Sheet")
    pdf.setSubject(f"{layout.version} | {ARUCO_DICTIONARY_NAME} | units: mm")
    pdf.setCreator("MeasurePilot")

    for placement in layout.marker_placements:
        _draw_vector_marker(pdf, placement, layout)

    _draw_text_from_top(
        pdf,
        layout,
        layout.page_width_mm / 2.0,
        45.0,
        "MeasurePilot Calibration Sheet",
        font="Helvetica-Bold",
        size=16.0,
        centered=True,
    )
    _draw_text_from_top(
        pdf,
        layout,
        layout.page_width_mm / 2.0,
        52.0,
        "Print at 100% scale. Do not use Fit to page.",
        size=8.0,
        centered=True,
    )
    _draw_text_from_top(
        pdf,
        layout,
        layout.page_width_mm / 2.0,
        68.0,
        "Place the planar part inside this area without covering any marker.",
        size=8.0,
        centered=True,
    )

    pdf.setDash(5, 4)
    pdf.setLineWidth(0.5)
    pdf.rect(
        layout.work_area_x_mm * mm,
        (layout.page_height_mm - layout.work_area_y_mm - layout.work_area_height_mm) * mm,
        layout.work_area_width_mm * mm,
        layout.work_area_height_mm * mm,
        stroke=1,
        fill=0,
    )
    pdf.setDash()

    center_x = layout.page_width_mm / 2.0
    center_y = layout.work_area_y_mm + layout.work_area_height_mm / 2.0
    pdf.setLineWidth(0.35)
    pdf.line((center_x - 8.0) * mm, (layout.page_height_mm - center_y) * mm, (center_x + 8.0) * mm, (layout.page_height_mm - center_y) * mm)
    pdf.line(center_x * mm, (layout.page_height_mm - center_y - 8.0) * mm, center_x * mm, (layout.page_height_mm - center_y + 8.0) * mm)

    _draw_text_from_top(pdf, layout, 15.0, 49.0, "Marker 0", size=7.0)
    _draw_text_from_top(pdf, layout, 165.0, 49.0, "Marker 1", size=7.0)
    _draw_text_from_top(pdf, layout, 165.0, 289.0, "Marker 2", size=7.0)
    _draw_text_from_top(pdf, layout, 15.0, 289.0, "Marker 3", size=7.0)
    _draw_ruler(pdf, layout)
    _draw_text_from_top(
        pdf,
        layout,
        layout.page_width_mm / 2.0,
        294.0,
        f"Layout {layout.version} | {ARUCO_DICTIONARY_NAME} | units: mm",
        size=6.5,
        centered=True,
    )

    pdf.showPage()
    pdf.save()


def generate_calibration_pdf(
    destination: str | os.PathLike[str], *, layout: CalibrationLayout = DEFAULT_LAYOUT
) -> Path:
    """Atomically generate the deterministic printable calibration sheet."""

    layout.validate()
    target = Path(destination)
    if target.exists() and target.is_dir():
        raise CalibrationError(f"destination is a directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_calibration_pdf(temporary, layout)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
