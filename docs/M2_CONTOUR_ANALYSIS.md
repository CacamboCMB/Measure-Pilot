# M2 controlled planar contour analysis

M2 consumes the complete A4 image produced by M1 rectification. It does not
perform another perspective correction and it does not create CAD geometry.

## Supported capture

- one dominant planar foreground part;
- an already rectified A4 plane;
- an explicit `px_per_mm` value matching the rectified image;
- dark-on-light, light-on-dark, or conservative automatic polarity;
- an outer silhouette with directly enclosed holes;
- straight outer edges and near-circular holes.

The calibration-marker rectangles and a configurable page margin are excluded
before connected-component selection. Printed calibration graphics below the
minimum component area are ignored. The foreground must not touch an excluded
boundary.

## Unsupported and explicit failures

M2 rejects rather than guesses when:

- the image dimensions do not match A4 at the selected pixel density;
- no component exceeds the physical minimum area;
- two components have materially similar areas;
- a component touches the page or a marker-exclusion boundary;
- automatic foreground polarity remains ambiguous;
- more than one external contour survives selection.

Non-circular enclosed contours and very short simplified outer edges remain in
the report under `unresolved`; they are not discarded or promoted to CAD
features.

## Coordinate contract

All machine-readable geometry is expressed in millimetres with the origin at
the top-left of the rectified A4 page. Positive X points right and positive Y
points down. Pixel evidence is retained for the selected component and visual
overlay.

## Module CLI

```bash
python -m measurepilot.analysis \
  rectified.png \
  analysis.json \
  --overlay overlay.png \
  --px-per-mm 4 \
  --polarity auto
```

The JSON is canonical: UTF-8, sorted keys, compact separators, finite numbers,
and one trailing newline. Output paths must be distinct from the source. The
optional overlay draws the original contour, simplified contour, holes,
bounding box, and centroid without changing the source image.

## M2 boundary

This milestone does not migrate the `.mpilot` schema, solve constraints,
recommend a next measurement, call FreeCAD, or use a machine-learning model.
Those capabilities require separate canonical work orders after this result is
published and reconciled.
