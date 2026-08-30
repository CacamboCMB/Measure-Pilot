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
