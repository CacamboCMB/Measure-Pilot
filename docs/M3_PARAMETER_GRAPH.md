# M3 parameter, provenance, and constraint graph

M3 separates immutable evidence from the current resolved view of a parameter.
A value is never overwritten in place. A correction appends a new observation
whose `supersedes` field points to the earlier observation.

## Quantity contract

| Kind | Unit |
|---|---|
| `LENGTH` | `mm` |
| `ANGLE` | `deg` |
| `DIMENSIONLESS` | `1` |

A parameter definition fixes its kind and unit. A later observation cannot
change either.

## Observation states

Observations may be `ASSUMED`, `ESTIMATED`, `DERIVED`, `MEASURED`, or `LOCKED`.
`UNRESOLVED` and `CONFLICTING` are calculated resolution states and cannot be
stored as observations.

Active observations are selected by status priority:

```text
LOCKED > MEASURED > DERIVED > ESTIMATED > ASSUMED
```

Lower-priority active observations remain visible but do not dilute a physical
measurement. Compatible observations at the selected priority are fused by
inverse variance. Version 1 defines compatibility as:

```text
abs(a - b) <= max(1e-9, 3 * hypot(uncertainty_a, uncertainty_b))
```

If that condition fails, the resolved state is `CONFLICTING` and has no value.
A constraint evaluation must not replace that conflict with a derived result.

## Provenance

Each observation references a versioned evidence source. Importing an M2 report
records the SHA-256 of the exact report bytes when available. Line endpoints,
line lengths, angles, circular-hole centres, diameters, and circularity values
are imported as `ESTIMATED` observations with bounded image-derived
uncertainty.

Stable IDs are hashes of canonical logical content. Canonical JSON uses UTF-8,
sorted keys, compact separators, finite numbers, and one trailing newline.

## Linear constraints

A constraint is an explicit equation:

```text
sum(coefficient[parameter] * parameter) = constant
```

Every constraint declares a quantity kind and tolerance. Known resolved values
are substituted. The remaining system is solved only when it has full column
rank. Rank deficiency, missing values, dependency cycles, and residuals above
tolerance are explicit failures.

## Module CLI

Create a graph from M2 analysis:

```bash
python -m measurepilot.graph_cli import-analysis analysis.json part.graph.json
```

Append a measurement:

```bash
python -m measurepilot.graph_cli add-measurement \
  part.graph.json feature.circular-hole-000.diameter 10.02 \
  --uncertainty 0.05 \
  --source "digital caliper"
```

Append a correction while retaining history:

```bash
python -m measurepilot.graph_cli add-measurement \
  part.graph.json hole_pitch_x 54.4 \
  --uncertainty 0.1 \
  --source "digital caliper recheck" \
  --supersedes observation-...
```

Inspect or validate:

```bash
python -m measurepilot.graph_cli inspect part.graph.json
python -m measurepilot.graph_cli validate part.graph.json
```

M3 does not recommend measurements, perform nonlinear geometry optimisation,
migrate `.mpilot`, or generate FreeCAD objects.
