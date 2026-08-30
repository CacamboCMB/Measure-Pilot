# MeasurePilot

MeasurePilot is a local-first FreeCAD workbench and analysis engine for
measurement-guided reverse engineering. It turns calibrated photographs and
explicit physical measurements into auditable geometry while keeping measured,
estimated, derived, conflicting, and unresolved values distinct.

The repository is currently at **M5**. It contains the deterministic project,
calibration, contour-analysis, parameter-graph, next-measurement, and native
FreeCAD planar-model slices. The supported model remains intentionally narrow:
one straight-segment planar profile, circular through-holes, and uniform
thickness.

## Current capabilities

- deterministic `.mpilot` project containers with atomic writes;
- A4 ArUco calibration sheet and metric perspective rectification;
- deterministic planar contour and circular-hole recognition;
- versioned M3 parameter/provenance graph with corrections and conflict states;
- explicit linear dependencies and constraints;
- explainable next-best-measurement ranking and recording;
- native FreeCAD `PartDesign::Body` + `Sketcher::SketchObject` +
  `PartDesign::Pad` output;
- optional application of resolved M3/M4 line and hole corrections;
- fully local/offline processing.

## Install the FreeCAD workbench on Windows

FreeCAD 1.0 or newer and Git are required.

```powershell
git clone https://github.com/CacamboCMB/Measure-Pilot.git `
  "$env:APPDATA\FreeCAD\Mod\MeasurePilot"
```

Restart FreeCAD and select **MeasurePilot** from the workbench selector. Use
**Create planar model**, choose a canonical M2 analysis JSON, optionally choose
an M3/M4 graph, enter the thickness, and optionally save an `.FCStd` document.

The root `Init.py` adds only the checkout's `src` folder to FreeCAD's Python
path, so a separate engine installation is not required for the workbench.

## Install the engine for CLI use

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The package requires Python 3.11 or newer. All internal length values use
millimetres and no command performs network access.

## Core CLI examples

```bash
measurepilot project create example.mpilot --name "Example part"
measurepilot calibration sheet calibration.pdf
measurepilot calibration rectify capture.jpg rectified.png \
  --report rectified.json --px-per-mm 4
python -m measurepilot.analysis rectified.png analysis.json \
  --overlay overlay.png --px-per-mm 4
python -m measurepilot.graph_cli import-analysis analysis.json graph.json
python -m measurepilot.recommendation_cli recommend \
  graph.json recommendations.json --limit 5
```

## Supported FreeCAD model boundary

M5 requires a continuous closed polygon made from straight line features, no
unresolved geometry, circular holes fully inside the profile, and positive
uniform thickness. The planner rejects self-intersection, broken chains,
unsupported non-circular cut-outs, conflicting required graph values, incorrect
source binding, and overlapping holes instead of guessing.

An actual FreeCAD 1.0/1.1 runtime smoke remains the material environment check
before M6 produces a distributable release.
