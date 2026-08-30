# M3 versioned parameter, provenance, and constraint graph

M3 converts canonical M2 analysis evidence and later physical measurements into
an auditable scalar parameter graph. It remains local, offline, deterministic,
and independent of FreeCAD.

## Canonical graph document

A graph uses format `measurepilot-parameter-graph`, version `1`, and contains
sorted records for:

- evidence sources;
- typed parameter definitions;
- append-only observations;
- derived dependencies;
- linear constraints.

Canonical JSON is UTF-8, has sorted object keys, compact separators, finite
numbers, and exactly one trailing newline. Identical logical input therefore
produces identical bytes and content-addressed IDs.

Length parameters use `mm`, angle parameters use `deg`, and dimensionless
parameters use `1`. A constraint or dependency cannot silently mix quantity
kinds.

## Provenance and corrections

Every source records a lowercase SHA-256 digest. M2 imports hash the exact
canonical report bytes. Physical measurements hash their canonical logical
payload. Source and observation IDs are derived from canonical content.

Observations are immutable and append-only. A correction creates a new
observation with an explicit `supersedes` link. The old observation remains in
history but is no longer active. Supersession links must reference the same
parameter and must be acyclic.

For example, correcting a screw pitch from `52.8 mm` to `54.4 mm` retains both
records and resolves only the new active observation.

## Parameter states and conflict rule

The domain exposes these states:

- `MEASURED`
- `ESTIMATED`
- `DERIVED`
- `ASSUMED`
- `LOCKED`
- `CONFLICTING`
- `UNRESOLVED`

Conflict rule version 1 compares every pair of active observations. Two values
are compatible when their absolute difference is no greater than three times
their combined standard uncertainty, plus a fixed numerical floor of `1e-9`.
Materially incompatible active evidence produces `CONFLICTING` with no selected
value.

Compatible non-locked observations use inverse-variance fusion:

```text
weight_i = 1 / uncertainty_i²
value = sum(weight_i * value_i) / sum(weight_i)
uncertainty = sqrt(1 / sum(weight_i))
```

A compatible active `LOCKED` observation remains authoritative and is reported
explicitly as locked.

## Dependencies and linear constraints

A dependency is an explicit equation:

```text
target = constant + sum(coefficient_i * source_i)
```

Dependencies are evaluated in deterministic topological order. Missing
parameters, multiple dependencies for one target, and dependency cycles fail
explicitly.

A linear constraint is an explicit equation:

```text
sum(coefficient_i * parameter_i) = constant
```

Unknown variables are sorted by parameter ID and solved through a deterministic
linear least-squares system. Rank deficiency, undefined parameters, conflicting
inputs, and residuals beyond each equation's stated tolerance are errors rather
than guessed values. Derived uncertainty is computed from the tolerance-weighted
normal system.

## M2 import

`import_m2_analysis` accepts only canonical M2 analysis JSON version 1. It
imports:

- line start X/Y, end X/Y, and length;
- circular-hole centre X/Y and diameter.

Every imported value is `ESTIMATED`, references the exact report SHA-256, and
uses a bounded image uncertainty derived from pixel density, simplification
residual, or circle-fit residual. The M3 cap is `2.0 mm`; uncertainty is never
zero.

M3 does not perform image recognition, nonlinear geometry optimisation,
next-measurement recommendation, `.mpilot` migration, CAD generation, GUI work,
or network access.

## Module CLI

Create a graph from M2 evidence:

```bash
python -m measurepilot.graph_cli import-analysis analysis.json graph.json
```

Append a physical measurement:

```bash
python -m measurepilot.graph_cli append-measurement \
  graph.json plate.hole_pitch 52.8 0.1 --quantity length
```

Append a correction without deleting history:

```bash
python -m measurepilot.graph_cli correct-measurement \
  graph.json OBSERVATION_ID 54.4 0.1 --note "corrected screw pitch"
```

Inspect or validate:

```bash
python -m measurepilot.graph_cli inspect graph.json
python -m measurepilot.graph_cli inspect graph.json --parameter plate.hole_pitch
python -m measurepilot.graph_cli validate graph.json
```

All successful CLI output is deterministic JSON. Expected domain failures use
exit code `2`, print an `ERROR:` message, and do not silently replace evidence.
