# M5 native FreeCAD workbench

M5 is the first directly usable FreeCAD integration. The repository itself is a
FreeCAD workbench package: place or clone it into the user `Mod/MeasurePilot`
directory and restart FreeCAD.

## Supported model

The command **Create planar model** accepts:

1. canonical M2 `measurepilot-planar-analysis` JSON;
2. optionally, a canonical M3 parameter graph bound to the exact M2 SHA-256;
3. a positive uniform thickness;
4. optionally, an `.FCStd` save path.

The M2 topology provides an ordered closed straight-segment outer profile and
circular holes. When an M3/M4 graph is supplied, resolved line endpoint and hole
centre/diameter parameters replace the image-derived values. Required
`CONFLICTING` or `UNRESOLVED` parameters stop generation.

Before FreeCAD is touched, the pure planner validates:

- canonical JSON and source binding;
- finite millimetre coordinates and positive thickness;
- continuous closed line chain;
- non-zero, non-self-intersecting polygon;
- circular holes completely inside the polygon;
- no overlapping or touching holes;
- no unresolved or unsupported non-circular geometry.

The plan is translated so the outer profile begins near the FreeCAD origin. The
original translation and exact evidence hashes remain attached as body metadata.

## Native document structure

```text
MeasurePilotModel
└── MeasurePilotBody             PartDesign::Body
    ├── MeasurePilotSketch       Sketcher::SketchObject
    │   ├── straight outer edges
    │   ├── coincident closure constraints
    │   └── circular inner wires
    └── MeasurePilotPad          PartDesign::Pad
```

The pad length is the entered thickness. The sketch and pad remain editable in
FreeCAD; circular inner wires become through-holes in the pad.

## Manual installation on Windows

```powershell
git clone https://github.com/CacamboCMB/Measure-Pilot.git `
  "$env:APPDATA\FreeCAD\Mod\MeasurePilot"
```

Restart FreeCAD and select **MeasurePilot** in the workbench selector. The root
`Init.py` adds only this checkout's `src` directory to FreeCAD's Python path.

## Runtime smoke check

Inside FreeCAD's Python console:

```python
exec(open(r"<MeasurePilot checkout>\tools\freecad_smoke.py", encoding="utf-8").read())
```

The script creates a reference 90 × 50 × 2 mm plate with two 10 mm through-holes
and verifies that Body, Sketch, Pad, and a non-null shape exist.

## M5 boundary

M5 does not generate curves, slots, non-circular cut-outs, assemblies, STEP/STL
exports, or a distributable release. Actual FreeCAD 1.0/1.1 smoke validation is
the material environment check before M6 packaging.
