"""Run inside FreeCAD to smoke-test the native MeasurePilot M5 model path."""

from __future__ import annotations

from measurepilot.freecad_model import build_model_plan, create_freecad_document
from measurepilot.provenance import canonical_json_bytes


REPORT = canonical_json_bytes(
    {
        "configuration": {"features": {}, "segmentation": {"px_per_mm": 4.0}},
        "contours": {
            "holes": [{"id": "hole-contour-000"}, {"id": "hole-contour-001"}],
            "outer": {"simplification_rms_mm": 0.0},
        },
        "coordinate_system": {"unit": "mm"},
        "features": {
            "circular_holes": [
                {
                    "center_mm": [20.0, 25.0],
                    "diameter_mm": 10.0,
                    "fit_residual_mm": 0.0,
                    "id": "circular-hole-000",
                },
                {
                    "center_mm": [70.0, 25.0],
                    "diameter_mm": 10.0,
                    "fit_residual_mm": 0.0,
                    "id": "circular-hole-001",
                },
            ],
            "line_segments": [
                {"id": "outer-line-000", "start_mm": [0.0, 0.0], "end_mm": [90.0, 0.0], "length_mm": 90.0},
                {"id": "outer-line-001", "start_mm": [90.0, 0.0], "end_mm": [90.0, 50.0], "length_mm": 50.0},
                {"id": "outer-line-002", "start_mm": [90.0, 50.0], "end_mm": [0.0, 50.0], "length_mm": 90.0},
                {"id": "outer-line-003", "start_mm": [0.0, 50.0], "end_mm": [0.0, 0.0], "length_mm": 50.0},
            ],
        },
        "format": "measurepilot-planar-analysis",
        "unresolved": [],
        "version": 1,
    }
)

plan = build_model_plan(REPORT, thickness_mm=2.0, document_name="MeasurePilotSmoke")
objects = create_freecad_document(plan)
assert objects.body is not None
assert objects.sketch is not None
assert objects.pad is not None
assert objects.pad.isValid()
assert not objects.pad.Shape.isNull()
print("MEASUREPILOT_FREECAD_SMOKE_PASS")
