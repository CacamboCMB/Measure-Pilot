# Architecture

## Components

1. **FreeCAD workbench** - later GUI and native Sketcher/PartDesign adapter.
2. **MeasurePilot engine** - local calibration, image, geometry, constraint, and uncertainty processing.
3. **Neutral project model** - versioned `.mpilot` data exchanged between components.

M0 implements the neutral project model and CLI. M1 adds calibrated capture to
the engine. M2 adds deterministic, corrigible 2D detection without introducing
FreeCAD, constraint solving, or machine-learning dependencies.

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


## M2 detection model

M2 consumes an already rectified A4 image with an explicit px/mm value. The
engine crops to the inner versioned work area, thresholds the high-contrast
foreground, removes implausibly small calibration-line components, and selects
one unambiguous connected part. A part touching the work-area boundary is
rejected because its contour may be clipped.

The selected component is represented as:

- one canonical polygonal outer profile;
- circular child contours above an explicit circularity threshold;
- remaining supported child contours as polygonal cut-outs;
- a source-image SHA-256 binding;
- pixel-derived boundary uncertainty in millimetres;
- status and immutable provenance history.

Polygon orientation, start vertex, feature IDs, and feature ordering are
canonicalised before deterministic JSON serialization. Optional overlays are
rendered from the same millimetre model.

Corrections are separate version-1 JSON documents. They may replace the profile,
add or update circles, remove detected features, and attach a note. Applying a
correction returns a new detection result, preserves the source result, and
records the canonical correction-payload hash. No constraint or CAD semantics
are inferred in M2.

## Invariants

- Internal length unit: `mm`.
- Current project schema version: `1`.
- No network dependency.
- Same validated project model produces the same archive bytes.
- Same calibration layout produces byte-identical PDF output.
- No metric result is emitted after a hard calibration failure.
- No geometry result is emitted for a missing, clipped, or ambiguous M2 part.
- Automatic geometry is never relabelled as measured without an explicit correction.
