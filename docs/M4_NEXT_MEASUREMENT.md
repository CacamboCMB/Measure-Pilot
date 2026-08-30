# M4 deterministic next-best-measurement recommendation

M4 ranks physical measurements that can improve a version-1 M3 parameter graph.
It does not infer geometry, invoke FreeCAD, or hide decisions behind a learned
model. Every candidate is an explicit linear scalar observable over typed graph
parameters.

## Measurement candidates

A candidate contains:

- canonical coefficients for `sum(c_i * parameter_i)`;
- one quantity kind and its canonical unit;
- expected standard uncertainty;
- effort and measurability factors;
- a human-readable prompt;
- provenance describing whether it was generated or user supplied.

Default candidates are conservative. Direct measurements are generated for
`ESTIMATED`, `ASSUMED`, `CONFLICTING`, and `UNRESOLVED` parameters. Imported M2
hole centres additionally produce horizontal and vertical pairwise pitch
candidates. Euclidean point distance is nonlinear and is not generated in M4.

A canonical optional catalog can add further linear candidates. Undefined
parameters, quantity mixing, zero observables, duplicate IDs, non-positive
uncertainty/effort, and noncanonical catalog JSON are rejected.

## Transparent ranking

For each candidate MeasurePilot builds the current weighted linear evidence
matrix from finite resolved observations, dependencies, and constraints. It then
adds the candidate as one prospective equation and reports:

- current structural rank;
- post-measurement structural rank;
- structural rank gain;
- finite prior and posterior observable variance, when identifiable;
- Gaussian information gain in nats, when the prior variance is finite;
- downstream dependency/constraint impact;
- conflicting active observations;
- effort and measurability.

The version-1 score is:

```text
base = 1000 * rank_gain
     + 100 * conflicting_parameter_count
     + 10 * min(information_gain_nats, 20)
     + downstream_impact

score = base * measurability / effort
```

The large rank term intentionally prefers removing structural
underdetermination over marginally refining an already precise value. Ties are
resolved by canonical candidate ID.

## Recording

A direct candidate appends an M3 physical observation. A conflicting parameter
requires an explicit active observation ID to supersede; M4 never chooses one
silently.

A composite candidate creates a deterministic measured-observable parameter,
appends its physical observation, and adds one explicit linking constraint:

```text
measured_observable - sum(c_i * parameter_i) = 0
```

All recording occurs on an in-memory clone and is persisted atomically only
after complete validation. Failed operations leave the graph bytes unchanged.

## Module CLI

```bash
python -m measurepilot.recommendation_cli recommend \
  graph.json recommendations.json --limit 5

python -m measurepilot.recommendation_cli inspect-candidate \
  graph.json CANDIDATE_ID

python -m measurepilot.recommendation_cli record \
  graph.json CANDIDATE_ID 54.4 0.1 \
  --supersedes OBSERVATION_ID --note "caliper remeasurement"
```

An optional `--catalog catalog.json` augments generated candidates. Successful
output is canonical JSON. Expected domain failures return exit code `2` and an
`ERROR:` message.
