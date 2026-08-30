"""MeasurePilot core package."""

from .calibration import (
    CALIBRATION_LAYOUT_VERSION,
    DEFAULT_LAYOUT,
    generate_calibration_pdf,
)
from .corrections import apply_corrections, correct_detection_file
from .detection import (
    DETECTION_MODEL_VERSION,
    CircleFeature,
    DetectionResult,
    PolygonFeature,
    detect_image_file,
    detect_planar_part,
)
from .model import INTERNAL_LENGTH_UNIT, SCHEMA_VERSION, MeasurePilotProject
from .project import load_project, save_project
from .rectification import RectificationReport, rectify_image, rectify_image_file

__all__ = [
    "CALIBRATION_LAYOUT_VERSION",
    "DEFAULT_LAYOUT",
    "DETECTION_MODEL_VERSION",
    "CircleFeature",
    "DetectionResult",
    "PolygonFeature",
    "INTERNAL_LENGTH_UNIT",
    "SCHEMA_VERSION",
    "MeasurePilotProject",
    "RectificationReport",
    "apply_corrections",
    "correct_detection_file",
    "detect_image_file",
    "detect_planar_part",
    "generate_calibration_pdf",
    "load_project",
    "rectify_image",
    "rectify_image_file",
    "save_project",
]

__version__ = "0.3.0"
