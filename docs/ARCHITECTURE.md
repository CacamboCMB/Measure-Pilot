# Architecture

## Components

1. **FreeCAD workbench** - later GUI and native Sketcher/PartDesign adapter.
2. **MeasurePilot engine** - local image, geometry, constraint, and uncertainty processing.
3. **Neutral project model** - versioned `.mpilot` data exchanged between components.

M0 implements only the neutral project model and CLI.

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

## Invariants

- Internal length unit: `mm`.
- Current schema version: `1`.
- No network dependency.
- Same validated model produces the same archive bytes.
- Original source data is never silently overwritten in future revisions.
