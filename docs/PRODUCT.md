# Product definition

## Problem

A photograph alone does not contain enough trustworthy metric information to
reconstruct a mechanical part. Existing photo-to-3D approaches tend to emit a
mesh or conceal uncertainty. MeasurePilot instead combines a calibrated image,
explicit measurements, geometric constraints, and targeted follow-up questions.

## Product form

The primary user interface will be a FreeCAD workbench. Image processing runs
in a separate local engine process so that dependency updates and failures do
not destabilise FreeCAD. The neutral project format is independent of FreeCAD.

## Current M1 slice

M1 establishes a trustworthy metric image before any part geometry is inferred:

1. Generate the versioned A4 calibration sheet.
2. Print it without scaling and verify the physical 100 mm ruler.
3. Photograph the sheet with the planar part inside the work area.
4. Detect marker IDs 0–3 and reject incomplete or contradictory captures.
5. Rectify the photographed plane into A4 millimetre coordinates.
6. Persist a quality report alongside the PNG.

Warnings preserve potentially usable captures with low sharpness or low metric
resolution. Missing markers, duplicate IDs, impossible geometry, and excessive
reprojection error are hard failures because no defensible metric output exists.


## Current M2 slice

M2 turns the trustworthy rectified image into a corrigible geometric proposal:

1. Restrict processing to the inner versioned work area.
2. Select one sufficiently large, high-contrast connected part.
3. Reject boundary-clipped or ambiguous captures.
4. Simplify the external contour in millimetres.
5. Classify circular internal contours as holes and retain other supported
   internal contours as polygonal cut-outs.
6. Bind the result to the source image and record uncertainty.
7. Apply explicit correction documents as immutable revisions.

This remains intentionally deterministic and classical. Machine learning is not
required for the supported high-contrast plate workflow, and no inferred value
is presented as a physical measurement.

## MVP boundary

Supported first:

- planar and 2.5D plates, covers, and simple brackets;
- lines, circular arcs, radii, chamfers, circular holes, and slots;
- uniform material thickness;
- symmetry and regular hole patterns;
- native parametric FreeCAD output in a later milestone.

Not initially supported:

- organic free-form surfaces;
- deep undercuts and multi-cavity castings;
- threads and gears;
- arbitrary photogrammetry or hidden geometry inference.

## Differentiator

MeasurePilot must not silently guess missing dimensions. It records provenance
and uncertainty, maintains competing hypotheses when needed, and ultimately
asks for the next physical measurement with the highest useful information gain.
