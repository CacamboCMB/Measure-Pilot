"""Versioned calibration-sheet geometry and deterministic rendering."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .errors import MeasurePilotError

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
CALIBRATION_LAYOUT_VERSION = 1
ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
MARKER_IDS = (0, 1, 2, 3)
MARKER_SIZE_MM = 24.0
MARKER_MARGIN_X_MM = 15.0
MARKER_MARGIN_Y_MM = 15.0
VERIFICATION_RULER_MM = 100.0
POINTS_PER_MM = 72.0 / 25.4


class CalibrationError(MeasurePilotError):
    """Raised when calibration geometry or output is invalid."""


@dataclass(frozen=True, slots=True)
class CalibrationLayout:
    """Physical geometry of one printable calibration-sheet revision."""

    version: int = CALIBRATION_LAYOUT_VERSION
    page_width_mm: float = A4_WIDTH_MM
    page_height_mm: float = A4_HEIGHT_MM
    marker_size_mm: float = MARKER_SIZE_MM
    dictionary_name: str = ARUCO_DICTIONARY_NAME

    @property
    def marker_origins_mm(self) -> dict[int, tuple[float, float]]:
        right = self.page_width_mm - MARKER_MARGIN_X_MM - self.marker_size_mm
        bottom = self.page_height_mm - MARKER_MARGIN_Y_MM - self.marker_size_mm
        return {
            0: (MARKER_MARGIN_X_MM, MARKER_MARGIN_Y_MM),
            1: (right, MARKER_MARGIN_Y_MM),
            2: (right, bottom),
            3: (MARKER_MARGIN_X_MM, bottom),
        }

    def marker_corners_mm(self, marker_id: int) -> np.ndarray:
        """Return TL, TR, BR, BL marker corners in top-left page coordinates."""

        try:
            x, y = self.marker_origins_mm[marker_id]
        except KeyError as exc:
            raise CalibrationError(f"marker ID {marker_id} is not part of layout v{self.version}") from exc
        size = self.marker_size_mm
        return np.array(
            ((x, y), (x + size, y), (x + size, y + size), (x, y + size)),
            dtype=np.float64,
        )

    def validate(self) -> None:
        if self.version != CALIBRATION_LAYOUT_VERSION:
            raise CalibrationError(f"unsupported calibration layout version {self.version}")
        if self.dictionary_name != ARUCO_DICTIONARY_NAME:
            raise CalibrationError(f"unsupported ArUco dictionary {self.dictionary_name!r}")
        values = (self.page_width_mm, self.page_height_mm, self.marker_size_mm)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise CalibrationError("calibration dimensions must be finite and positive")
        for marker_id in MARKER_IDS:
            corners = self.marker_corners_mm(marker_id)
            if np.any(corners < 0.0):
                raise CalibrationError("marker lies outside the printable page")
            if np.any(corners[:, 0] > self.page_width_mm) or np.any(
                corners[:, 1] > self.page_height_mm
            ):
                raise CalibrationError("marker lies outside the printable page")


DEFAULT_LAYOUT = CalibrationLayout()


def aruco_dictionary() -> cv2.aruco.Dictionary:
    """Return the immutable predefined dictionary used by layout version 1."""

    dictionary_id = getattr(cv2.aruco, ARUCO_DICTIONARY_NAME, None)
    if dictionary_id is None:
        raise CalibrationError(
            "OpenCV ArUco support is unavailable; install opencv-contrib-python-headless"
        )
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _marker_image(marker_id: int, side_pixels: int) -> np.ndarray:
    if marker_id not in MARKER_IDS:
        raise CalibrationError(f"unsupported marker ID {marker_id}")
    if side_pixels < 6:
        raise CalibrationError("marker rendering requires at least 6 pixels per side")
    dictionary = aruco_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels, borderBits=1)
    else:  # pragma: no cover - compatibility with older OpenCV releases
        image = np.empty((side_pixels, side_pixels), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, side_pixels, image, 1)
    return np.ascontiguousarray(image, dtype=np.uint8)


def render_calibration_image(
    px_per_mm: float = 4.0,
    *,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Render a metric raster used by synthetic tests and capture diagnostics."""

    layout.validate()
    if not math.isfinite(px_per_mm) or px_per_mm <= 0.0:
        raise CalibrationError("px_per_mm must be finite and positive")
    width = int(round(layout.page_width_mm * px_per_mm))
    height = int(round(layout.page_height_mm * px_per_mm))
    if width < 1 or height < 1:
        raise CalibrationError("requested raster resolution is too small")

    page = np.full((height, width), 255, dtype=np.uint8)
    marker_pixels = max(6, int(round(layout.marker_size_mm * px_per_mm)))
    for marker_id, (x_mm, y_mm) in layout.marker_origins_mm.items():
        marker = _marker_image(marker_id, marker_pixels)
        x = int(round(x_mm * px_per_mm))
        y = int(round(y_mm * px_per_mm))
        x2 = min(width, x + marker_pixels)
        y2 = min(height, y + marker_pixels)
        if x < 0 or y < 0 or x2 - x != marker_pixels or y2 - y != marker_pixels:
            raise CalibrationError("marker raster falls outside the page")
        page[y:y2, x:x2] = marker

    ruler_x = int(round(55.0 * px_per_mm))
    ruler_y = int(round(148.5 * px_per_mm))
    ruler_length = int(round(VERIFICATION_RULER_MM * px_per_mm))
    thickness = max(1, int(round(0.35 * px_per_mm)))
    cv2.line(page, (ruler_x, ruler_y), (ruler_x + ruler_length, ruler_y), 0, thickness)
    for tick_mm in range(0, int(VERIFICATION_RULER_MM) + 1, 10):
        x = ruler_x + int(round(tick_mm * px_per_mm))
        tick = int(round((4.0 if tick_mm in (0, 100) else 2.5) * px_per_mm))
        cv2.line(page, (x, ruler_y - tick), (x, ruler_y + tick), 0, thickness)
    return page


def _pdf_number(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _pdf_xy(x_mm: float, y_from_top_mm: float) -> tuple[float, float]:
    return x_mm * POINTS_PER_MM, (A4_HEIGHT_MM - y_from_top_mm) * POINTS_PER_MM


def _pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(layout: CalibrationLayout) -> bytes:
    commands: list[str] = ["0 0 0 rg", "0 0 0 RG"]
    module_mm = layout.marker_size_mm / 6.0
    for marker_id, (origin_x, origin_y) in layout.marker_origins_mm.items():
        marker = _marker_image(marker_id, 6)
        for row in range(6):
            for column in range(6):
                if int(marker[row, column]) != 0:
                    continue
                x_mm = origin_x + column * module_mm
                y_top_mm = origin_y + row * module_mm
                x_pt = x_mm * POINTS_PER_MM
                y_pt = (layout.page_height_mm - y_top_mm - module_mm) * POINTS_PER_MM
                size_pt = module_mm * POINTS_PER_MM
                commands.append(
                    f"{_pdf_number(x_pt)} {_pdf_number(y_pt)} "
                    f"{_pdf_number(size_pt)} {_pdf_number(size_pt)} re f"
                )
        label_x, label_y = _pdf_xy(origin_x, origin_y + layout.marker_size_mm + 5.0)
        commands.append(
            f"BT /F1 8 Tf {_pdf_number(label_x)} {_pdf_number(label_y)} Td "
            f"(ID {marker_id}) Tj ET"
        )

    title_x, title_y = _pdf_xy(62.0, 68.0)
    commands.append(
        f"BT /F1 18 Tf {_pdf_number(title_x)} {_pdf_number(title_y)} Td "
        f"({_pdf_text('MeasurePilot Calibration A4 v1')}) Tj ET"
    )
    subtitle_x, subtitle_y = _pdf_xy(55.0, 78.0)
    commands.append(
        f"BT /F1 10 Tf {_pdf_number(subtitle_x)} {_pdf_number(subtitle_y)} Td "
        f"({_pdf_text('Print at 100% scale - do not fit to page')}) Tj ET"
    )

    ruler_x_mm = 55.0
    ruler_y_top_mm = 148.5
    ruler_x, ruler_y = _pdf_xy(ruler_x_mm, ruler_y_top_mm)
    ruler_end_x = (ruler_x_mm + VERIFICATION_RULER_MM) * POINTS_PER_MM
    commands.extend(
        [
            "0.8 w",
            f"{_pdf_number(ruler_x)} {_pdf_number(ruler_y)} m "
            f"{_pdf_number(ruler_end_x)} {_pdf_number(ruler_y)} l S",
        ]
    )
    for tick_mm in range(0, int(VERIFICATION_RULER_MM) + 1, 10):
        x_pt = (ruler_x_mm + tick_mm) * POINTS_PER_MM
        tick_mm_high = 4.0 if tick_mm in (0, 100) else 2.5
        commands.append(
            f"{_pdf_number(x_pt)} {_pdf_number(ruler_y - tick_mm_high * POINTS_PER_MM)} m "
            f"{_pdf_number(x_pt)} {_pdf_number(ruler_y + tick_mm_high * POINTS_PER_MM)} l S"
        )
    ruler_label_x, ruler_label_y = _pdf_xy(86.0, 141.0)
    commands.append(
        f"BT /F1 11 Tf {_pdf_number(ruler_label_x)} {_pdf_number(ruler_label_y)} Td "
        f"({_pdf_text('100 mm verification ruler')}) Tj ET"
    )

    instruction_x, instruction_y = _pdf_xy(59.0, 174.0)
    commands.append(
        f"BT /F1 10 Tf {_pdf_number(instruction_x)} {_pdf_number(instruction_y)} Td "
        f"({_pdf_text('Keep all four markers visible in the photograph.')}) Tj ET"
    )
    version_x, version_y = _pdf_xy(76.0, 184.0)
    commands.append(
        f"BT /F1 9 Tf {_pdf_number(version_x)} {_pdf_number(version_y)} Td "
        f"({_pdf_text('Layout v1 - DICT_4X4_50 - IDs 0, 1, 2, 3')}) Tj ET"
    )
    return ("\n".join(commands) + "\n").encode("ascii")


def calibration_pdf_bytes(*, layout: CalibrationLayout = DEFAULT_LAYOUT) -> bytes:
    """Return a byte-stable, vector A4 calibration PDF."""

    layout.validate()
    stream = _content_stream(layout)
    media_width = layout.page_width_mm * POINTS_PER_MM
    media_height = layout.page_height_mm * POINTS_PER_MM
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
            + _pdf_number(media_width).encode("ascii")
            + b" "
            + _pdf_number(media_height).encode("ascii")
            + b"] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    output = bytearray(b"%PDF-1.4\n% MeasurePilot deterministic calibration sheet\n")
    offsets = [0]
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R "
            f"/Info << /Producer (MeasurePilot) /CreationDate (D:20000101000000Z) >> >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def calibration_pdf_sha256(*, layout: CalibrationLayout = DEFAULT_LAYOUT) -> str:
    return sha256(calibration_pdf_bytes(layout=layout)).hexdigest()


def write_calibration_pdf(
    path: str | Path,
    *,
    layout: CalibrationLayout = DEFAULT_LAYOUT,
) -> str:
    """Atomically write the calibration PDF and return its SHA-256 digest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = calibration_pdf_bytes(layout=layout)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return sha256(payload).hexdigest()
