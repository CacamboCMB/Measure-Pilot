# MeasurePilot

MeasurePilot is a local-first FreeCAD extension and analysis engine for
measurement-guided reverse engineering. Its target workflow is to turn
calibrated photographs and a small number of trustworthy measurements into an
editable parametric CAD model while keeping measured, estimated, derived, and
unresolved values distinct.

The repository is currently at **M2**. It provides a deterministic project
foundation, calibrated A4 capture, metric perspective rectification, and
corrigible recognition of supported planar part profiles and internal features.
It does not yet solve dimensional constraints or generate CAD geometry.

## Current capabilities

- versioned, deterministic `.mpilot` project files;
- atomic project writes and strict archive validation;
- deterministic A4 calibration PDF with four ArUco markers and a 100 mm ruler;
- explicit marker validation using `DICT_4X4_50`, IDs 0–3;
- perspective rectification to a requested pixel-per-millimetre resolution;
- machine-readable quality reports with sharpness, metric resolution,
  homography, reprojection error, and warnings;
- deterministic high-contrast part segmentation inside the versioned work area;
- millimetre-based polygon profiles, circular holes, and polygonal cut-outs;
- immutable source-bound detections with explicit uncertainty and correction history;
- fully local command-line operation.

## Install for development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The package requires Python 3.11 or newer. All internal length values use
millimetres and no command performs network access.

## Project CLI

```bash
measurepilot project create example.mpilot --name "Example part"
measurepilot project inspect example.mpilot
measurepilot project validate example.mpilot
```

## Calibration CLI

Generate the version-1 A4 sheet and print it at **100% scale**:

```bash
measurepilot calibration sheet measurepilot-calibration-a4-v1.pdf
```

Rectify a photograph to the full A4 metric coordinate system at 4 px/mm:

```bash
measurepilot calibration rectify capture.jpg rectified.png \
  --report rectified.json --px-per-mm 4
```

The command fails rather than producing guessed output when required markers
are missing, duplicated, degenerate, or inconsistent with the versioned layout.
Low sharpness or low image resolution is retained as a warning in the report.


## Analysis CLI

Detect one supported high-contrast planar part in an already rectified image:

```bash
measurepilot analysis detect rectified.png detection.json \
  --px-per-mm 4 --overlay detection-overlay.png
```

Apply a deterministic correction document without modifying the original detection:

```bash
measurepilot analysis correct detection.json corrections.json corrected.json
```

M2 rejects missing, clipped, implausibly small, or ambiguous parts rather than
emitting guessed geometry. Automatic values remain `estimated`; explicit
corrections become `user_corrected` or `measured` and append provenance history.

## Project direction

The first product slice is intentionally limited to planar or mostly planar
parts with uniform thickness and visible high contrast against the calibration
sheet. The next milestone adds a parameter and constraint engine. Measurement
recommendation and native FreeCAD model generation follow only after geometry
and uncertainty remain stable under correction.
