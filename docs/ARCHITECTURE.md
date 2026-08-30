# Architecture

## Components

1. **FreeCAD workbench** - later GUI and native Sketcher/PartDesign adapter.
2. **MeasurePilot engine** - local calibration, image, geometry, constraint, and uncertainty processing.
3. **Neutral project model** - versioned `.mpilot` data exchanged between components.

M0 implements the neutral project model and CLI. M1 adds calibrated capture to
the engine without introducing FreeCAD or contour-recognition dependencies.

## `.mpilot` version 1

A project is a ZIP container with exactly these root entries:

- `manifest.json`
- `project.json`
- `measurements.json`
- `geometry.json`
- `hypotheses.json`
- `history.json`

JSON is UTF-8, sorted, compact, newline-terminated, and rejects non-finite
numbers. ZIP entries are stored without compression, use fixed metadata, and
are written in lexical order. The manifest binds project identity, schema,
units, entry names, and SHA-256 hashes.

The writer serialises to a temporary file in the target directory, flushes it,
and atomically replaces the destination. The reader rejects unexpected,
nested, duplicate, compressed, oversized, malformed, or hash-mismatched
members before constructing a project object.

## Calibration layout `measurepilot-a4-v1`

- Page: A4, 210 x 297 mm.
- Coordinate origin: top-left of the page; x grows right, y grows down.
- Dictionary: OpenCV `DICT_4X4_50`.
- Required IDs: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left.
- Marker size: 30 mm.
- Marker margin: 15 mm from the relevant page edges.
- Verification ruler: 100 mm.

The printable PDF uses vector marker modules and deterministic ReportLab output.
The diagnostic raster uses the same layout coordinates, so tests and runtime
share one canonical geometric source.

## Rectification

OpenCV detects all marker corners. Sixteen source observations are matched to
their versioned page coordinates, and one projective homography is solved. The
engine validates finite/stable matrix values, convex page geometry, plausible
page area, marker completeness, duplicate IDs, and RMS reprojection error before
warping the full A4 plane.

The output report records:

- layout and dictionary versions;
- detected marker IDs;
- source and rectified image dimensions;
- requested output px/mm and estimated input px/mm;
- sharpness score;
- RMS reprojection error;
- the 3 x 3 homography;
- bounded quality warnings.

The PNG and JSON report are written atomically. No contour or CAD information is
inferred in M1.

## Invariants

- Internal length unit: `mm`.
- Current project schema version: `1`.
- No network dependency.
- Same validated project model produces the same archive bytes.
- Same calibration layout produces byte-identical PDF output.
- No metric result is emitted after a hard calibration failure.
