# What the proofs say about these schemas

The RelFlow proof suite (`relflow/docs`, `proofs/results.yaml`) records which
modeling behaviors hold and where they stop. This is the cross-reference that
shaped the raw-structure catalog and bounds what a hyperparameter sweep can be
expected to fix. Proof IDs are stable; statuses are as recorded on
October 22, 2026.

## Choices the proofs support

- **Attention reductions everywhere.** P001, P002, and P040 show attention
  learning sums and counts over collections of varying and unseen lengths and
  preserving nested cardinality; P013 and P041 show a mean erasing counts. The
  catalog never sets `reduction: mean`; every root and branch keeps RelFlow's
  default attention reduction, so counts and totals are learnable from raw rows
  without a count column.
- **Hashed identifiers for equality.** P021, P029, and P037 show `Hash`
  carrying unseen-identity equality; P036 shows `Category` transferring values
  by identity only within a small persistent vocabulary, with no evidence for
  unseen identities or large id spaces. Recurring ids (users, posts, products,
  customers, events, ads, searches, studies, facilities, sponsors, races,
  constructors, circuits) are hashed under one namespace per table with
  `n_hashes: 4`.
- **Calendar parts chosen explicitly.** P045 shows weekday cannot be derived
  from day-of-year without year context, P042 shows business windows need
  weekday and hour together, P044 shows month is recoverable from day-of-year.
  Every timestamp therefore carries `month_of_year` and `day_of_week`, and
  timestamps with a time of day (Stack, Avito, event streams) add
  `hour_of_day`. Ages in days relative to the anchor carry recency.
- **Nulls stay nulls.** P038 shows null and zero are distinguished. Columns
  are carried as stored; the only exception is the event word vector, where
  Parquet cannot store a null fixed-size list and zeros stand in for an
  unknown event whose other columns stay null.
- **Sets and enums.** P063 and P064 show Set reconstruction and fallback
  working, so Amazon categories stay a set. Category is used where
  vocabularies are small: P055 and P056 mark unknown-value fallback and
  target coverage as Limited, and P057 shows early rare labels consuming
  vocabulary capacity ahead of frequent values, which is why high-cardinality
  strings are hashed or treated as text instead.
- **Numbers.** P079 shows Number's raw and log lanes within 10 percent of
  the best single lane across additive, moderate-tail, heavy-tail, and
  far-from-zero targets, with the standardized log-scale readout handling
  heavy-tailed and offset targets. No numeric column needs a transform in SQL.
- **Supervised targets.** P051 to P054 and P017 cover masked reconstruction
  and signal-versus-noise calibration; `mask: true` targets with `Boolean`
  or `Number` heads are the supported contract.

## Limits that bound what a schema can expect

- **Cross-collection identity is weak.** P028 (Limited) finds a model near
  chance at noticing the same hashed id in two sibling branches, and P075
  (Partial) finds larger one-hop retrieval seed-sensitive and a two-hop route
  that does not learn identity selection. Nested branches should carry the
  context a row needs on the row itself (which the lookups do) rather than
  rely on the model matching ids across branches.
- **Peer-relative inference needs the group on the row.** P032 to P035 pass
  when a group label is present. The catalog does not add peer contexts; if a
  task needs them, the raw rows of the peers must be nested, not summarized.
- **Vocabulary admission during training is unsettled.** P066 (Partial)
  reports input vocabulary admission shifting predictions; keep categories
  small or hash them.
- **Jitter is unproven.** The extended hypotheses list jitter robustness as
  unmeasured, so no field enables it.

## RelArena constraints the loop respects

From the RelArena README and `docs/adding-a-model.md`:

- A search space may vary by task only through coarse size tiers; the sweep's
  space is keyed by training-anchor tier (small under 20,000, medium under
  500,000, large above) and the default configuration runs first.
- Tuning uses the inner split only: fit on train, select on validation by
  the task's primary metric. The test split is never read; the outer refit is
  a separate step that this loop does not perform.
- Every run reports a default and a tuned number; the sweep's trial 0 is the
  default.
- Runs must be reproducible from a seed; the plan is a function of task,
  trial count, and seed, and each trial records its full configuration,
  metrics, timings, and predictions.
- The ceiling is 24 hours per task including preprocessing and it is "a
  ceiling, not a target"; trials cap at 90 minutes with plateau stopping, and
  prepared splits are materialized once per sweep and reused.
- Cache warmers and preprocessing must be public and must not bypass
  RelArena's data loading; the catalog's queries and generator are the
  warmer.
