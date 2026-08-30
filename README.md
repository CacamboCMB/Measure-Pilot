# MeasurePilot

MeasurePilot is a local-first FreeCAD extension and analysis engine for
measurement-guided reverse engineering. Its target workflow is to turn
calibrated photographs and a small number of trustworthy measurements into an
editable parametric CAD model while keeping measured, estimated, derived, and
unresolved values distinct.

The repository currently implements **M0 and M1**. M0 provides the deterministic
`.mpilot` project foundation. M1 adds a versioned A4 calibration sheet, local
ArUco detection, metric perspective rectification, and a machine-readable
capture-quality report. Contour recognition and FreeCAD model generation are
not implemented yet.

## Current capabilities

- versioned, deterministic `.mpilot` project files;
- atomic project writes and strict archive validation;
- deterministic A4 calibration PDF using OpenCV `DICT_4X4_50` markers 0–3;
- calibrated image rectification at a requested pixels-per-millimetre value;
- explicit failure for missing, duplicate, or inconsistent marker geometry;
- sharpness and reprojection quality reporting;
- a local command-line interface with no network access.

## Install for development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Project commands

```bash
measurepilot project create example.mpilot --name "Example part"
measurepilot project inspect example.mpilot
measurepilot project validate example.mpilot
```

## Calibration commands

Generate the version-1 A4 sheet and print it at exactly 100% scale:

```bash
measurepilot calibration sheet calibration-a4.pdf
```

Rectify a photograph and write both the metric PNG and JSON quality report:

```bash
measurepilot calibration rectify capture.jpg rectified.png \
  --report capture-report.json \
  --px-per-mm 4
```

The sheet includes a 100 mm verification ruler. Measure that ruler after
printing; do not use a printer setting such as “fit to page”. A rectification
failure never produces a guessed metric result.

All internal length values use millimetres. The package requires Python 3.11
or newer and processes data locally.

## Product direction

The first product slice is intentionally limited to planar or mostly planar
parts with uniform thickness. The next milestone adds contour and primitive
recognition. Measurement recommendation and native FreeCAD Sketcher/PartDesign
output follow only after calibrated capture and feature extraction are reliable.
