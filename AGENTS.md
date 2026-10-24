# Benchmark project guide

This repository owns the RelArena task catalog, its generator, the supervised
trial runner, the lab verification suite and loop, and recorded figures. RelFlow
is an installed dependency pinned in `pyproject.toml`: import it as `relflow`,
never add a `src/relflow` tree, and never patch the library from a benchmark.

## Raw structure only

Hand-crafted features are a non-starter. A task query joins tables into nested
contexts and carries columns as they are stored:

- the entity row's columns, typed by what they are;
- every related table as a branch of its rows before the anchor, ordered by
  time and capped by recency, each row carrying `occurred_at` and `age_days`;
- many-to-one relations looked up into the row that refers to them, the latest
  row before the anchor when the table is timed, the unique row otherwise;
- one-to-many relations below a child row as nested branches;
- identifiers that recur across fields hashed under one namespace per table.

Ages of timestamps relative to the anchor are the only derived values. No
counts, windows, rates, distinct counts, first or last ages, peer contexts, or
rolling summaries belong in SQL; RelFlow learns them. Change tasks by editing
the specifications in `results/raw-schema-20261022/raw.py` and regenerating;
the generated `benchmark.yaml`, `model.yaml`, and `observations.sql` are never
edited by hand.

Known limits that shaped the generator: DuckDB 1.5 raises an internal binder
error on a timeless lateral lookup under a `QUALIFY` window, so static lookups
join directly on their unique key; a window-deduplicated lookup inside a nested
branch crashed on real Avito data, so nested branches hang off the capped
parent rows; `CASE` cannot return a fixed-size array, so cast after it; Parquet
cannot store a null fixed-size list inside a nested record, so an unknown
event's word vector is zeros while its null context marks it unknown.

## Temporal contract

Anchors define the cutoff. Every branch, lookup, and root row is censored by
its timestamp against the anchor; rows without timestamps keep source order.
Labels never enter features, and nothing is selected by held-out test scores.
Preserve the batch and anchor row identity through materialization and
prediction, and score AUROC or MAE over the complete declared split.

## Modules

- `benchmarks/relarena/spec.py`, `store.py`, `dataset.py`, `resources.py`:
  task manifests, DuckDB snapshots of the declared source columns, bounded
  materialization into nested Arrow observations, and DuckDB budgets.
- `catalog.py`: discovery, starter configurations from `profiles.yaml`, and
  target binding (`supervise`).
- `smoke.py`, `smoke_queue.py`: sampled runtime checks on real data.
- `supervised.py`, `supervised_recovery.py`, `checkpoint_recovery.py`,
  `evaluation.py`, `tracking.py`: the trial runner, exact full-validation
  checkpoint selection, recovery, text-cache policy, and MLflow tracking.
- `information_suite.py`, `information_check.py`, `information_loop.py`,
  `priority_trial.py`: the lab verification suite, source audits, and the
  validation-only loop with recipe priors.
- `token_budget.py`, `text_cache_throughput.py`, `throughput_*.py`,
  `hurdle.py`: feature probes and shared helpers the suite imports.
- `benchmarks/provenance.py`: fingerprints of this source and of the installed
  RelFlow, recorded separately in every report.

## Testing

```bash
uv run --with-editable ../relflow --with "polars>=2,<3" --with transformers python -m pytest
```

Tests own their fixtures; never import another repository's tests. The catalog
tests encode the audited contracts per task (entity, anchor column, target,
timestamp column, history branch, root value) in `tests/test_relarena_catalog.py`;
extend those tables when a task's raw structure changes. One known failure
predates the catalog rebuild: `test_relarena_throughput_search.py` expects two
spawn workers to reproduce the single-process row order, which RelFlow 0.3.5's
worker sharding changed.

## External comparisons

Use only reproduced RelArena results. Read `PriorLabs/relarena`'s
`baseline_results/results.csv` filtered to `selected=True`; exclude
`reference_results.csv`, publication-transcribed figures, and `_MR` variants.
Validation scores are diagnostics, never test claims.

## Hardware

Sweeps run one task per GPU through `benchmarks/relarena/modal_sweep.py`, or
on any single-GPU host through `sweep.py` with a flock file serializing GPU
work. RelBench data lives under the RelBench cache; snapshots, text stores, and
run directories sit beside it. Host names, addresses, and credentials never
belong in this repository: pass tracking URIs and tokens through the
environment.

