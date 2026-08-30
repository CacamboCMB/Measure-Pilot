# MeasurePilot

MeasurePilot is a local-first FreeCAD extension and analysis engine for
measurement-guided reverse engineering. Its target workflow is to turn
calibrated photographs and a small number of trustworthy measurements into an
editable parametric CAD model while keeping measured, estimated, derived, and
unresolved values distinct.

The repository is currently at **M0**. It provides the deterministic project
foundation used by later calibration, image-analysis, and FreeCAD workbench
milestones. It does not yet generate CAD geometry.

## Current capabilities

- versioned `.mpilot` project files;
- deterministic ZIP and canonical JSON output;
- atomic project writes;
- strict archive, identity, unit, and content-hash validation;
- a minimal local CLI.

## Install for development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## CLI

```bash
measurepilot project create example.mpilot --name "Example part"
measurepilot project inspect example.mpilot
measurepilot project validate example.mpilot
```

All internal length values use millimetres. The package requires Python 3.11
or newer and performs no network access.

## Project direction

The first product slice is intentionally limited to planar or mostly planar
parts with uniform thickness. The next milestone adds an A4 calibration sheet,
ArUco detection, and metric perspective rectification. Contour extraction,
measurement recommendation, and native FreeCAD model generation follow only
after calibrated capture is reliable.
