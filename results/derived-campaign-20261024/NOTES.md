# Derived-schema campaign, 2026-10-24: changes, findings, TODO

Working notes for the loops launched on 2026-10-24 (Modal sweeps `derived-1024` for
driver-position and user-attendance, `derived-1024b` for the six text tasks). Everything
below is a TODO to consolidate once the other tasks have results; nothing here is final.

## Changes landed today

| where | change | why |
| --- | --- | --- |
| relflow `58b5b92` | loader workers ship each batch packed: consolidated TensorDicts, Arrow source taken into its own buffers (`ship()` in `data/datasets/arrow.py`) | a sliced Arrow table pickles its whole shuffle window (200 MB to 1.1 GB per batch); hundreds of tensor leaves crossed as hundreds of shared-memory segments; workers were slower than no workers on every task |
| `d17bbbb` | runner: `pin_memory=False`, `prefetch_factor=1` | both measured as main-process costs, not overlap; packed medium batches run to hundreds of MB |
| `8803922`, `22f0695` | `--num-workers auto` (default): 4 workers for schemas with Text fields, 0 otherwise; the batch-size finder runs in-process | text schemas are loader-bound at 0 workers (tokenization), others are bound by the training step; 8 workers packing the finder's largest batches exhausted a 32 GB container |
| `modal_sweep.py` `launch()` | deploy once, spawn one call per task (`python -m benchmarks.relarena.modal_sweep ...`) | `modal run --detach` keeps only its last call alive; the first campaign died when the laptop slept |
| `4518ed8` | refit trains at the trial's trained batch size and retries at half on OOM | the lab site-success refit ran at the finder cap (512) and ran out of memory |
| `1889daa` | `--refit-regime checkpoint`: train on train with validation selection on the test-censored store (RelGT / RelGNN-ES regime); runner `--phase outer` | tried as a cure for driver-position; not adopted (see below) |
| `53e408c`, `f089d17` | refit `--stale-validation`, always on in the sweep: validation anchors featurized with the validation-censored store | the only variant that improved driver-position's test score (3.96 to 3.88); a no-op on single-timestamp tasks |
| `derive.py` (10-25) | a child table that never holds more than one row per key folds into the branch row as a lookup (`designs`, `eligibilities` in rel-trial: exactly one row per study) under the branch field budget, placed after siblings and before second-hop vocabularies; the `--nested-text` ablation switch is gone | the site-success ablation (below) showed depth-two nesting adds nothing there at 4.5x the time; one-row children were paying a 16-item collection for one row |
| `sweep.py` (10-25) | `TIER_OVERRIDES` carries the medium (d96 L2 h4 dropout 0.2 lr 4e-4) and large (d128 L2 h8 dropout 0.1 lr 3e-4) trial-0 defaults the Modal campaign launched with | the lab item-churn loop started from the bare `DEFAULT` (d64 L2 h4 dropout 0.1 lr 1e-3), so its trial 0 was not comparable with the Modal large launches |
| `c81ae4a` | prepared-data reuse strips mask/jitter/objective on every field (the comparison was handed the fields mapping, so nothing below it was stripped) | every regularized Optuna proposal in the frozen small sweeps and the lab loop failed the reuse check in seconds; the four `frozen-1024c` sweeps were cleaned (failed rows and folders dropped, default seeds kept) and resumed on the separately deployed app `relflow-relarena-sweep-b` |
| `846a516`, `63f3194` | classification plans pin `objective`; a knob a space fixes to one choice is set by the optimizer, never sampled | TPE spent draws on an inert loss knob on classification tasks; the lab item-churn loop was rebuilt as `optuna-amazon-item-churn-large` with d_model {128, 192, 256}, ffnx 4, no objective knob |
| sweep spaces | narrowed tier spaces and new trial-0 defaults from the 10-23 small sweeps and the lab site-success sweep (`scratchpad/modal/campaign/spaces.json`) | ema 0.999 and dropout 0 never won; the small default (d64 L2 h2 dropout 0.3 lr 2.3e-3 ema 0.99) topped three of four tasks |

## Findings worth keeping

- Loader probe (A10G, 8 CPUs, 256-row batches): text tasks are loader-bound at 0 workers
  (Amazon 2.6 batches/s vs a 10/s step; site-success 1.1 vs 0.9/s); packed batches with
  4 to 8 workers reach the step ceiling (Amazon 12/s). Non-text small tasks are bound by the
  step itself (user-attendance 600 ms per batch, GPU 42% busy): fewer per-field kernels, not
  loaders, is the next lever there. Text token tensors (padded int64) are most of a 300 MB
  medium batch; int32 ids and a bool mask would cut payload and H2D transfer about 4x.
- rel-f1 is the only dataset whose validation and test anchors run years past the split
  boundaries (val +1740 d over 23 timestamps, test +2340 d over 33); every other task has a
  single timestamp at the boundary. With the store censored at the boundary, 43% of F1 test
  anchors have no history and the rest are stale, which every public model suffers from
  (relgt 3.50 val to 4.77 test).
- driver-position grid (test MAE): sweep winner refit 4.98; winner full-length 4.03; winner +
  EMA 4.02; default 3.96; default + stale-validation 3.88; default + stale + history dropout
  3.87; checkpoint regime 4.32 to 4.36; 3-seed ensemble 4.00; absolute-time root feature worse
  on both splits. A regularized default and test-like training rows matter; longer training,
  seeds, dropout alone and absolute time do not.
- Validation selection over 25 trials on a 499-row validation split is a winner's-curse
  draw: the winner's early checkpoint (step 120 of 480) does not survive the test shift.
- Site-success schema ablation (same default d96 L2 dropout 0.2 lr 4e-4, A10G, 3 h cap,
  validation MAE): hand 0.3643 (8 passes, 2.1 h), derived 0.3737 (7.1 passes, time cap),
  derived without depth-two nesting 0.3747 (8 passes, 0.68 h); the nested-text variant ran
  out of memory three ways (DuckDB, loader workers, CUDA) and was abandoned. Same-schema
  seed noise is about 0.01 (the campaign's derived trial 0 scored 0.3827). Depth-two nesting
  costs 4.5x the time for nothing here; `designs` and `eligibilities` are one row per study
  (now folded into the row), `facilities_studies.date` equals `studies.start_date` on every
  row (the dedup is right), and the real collections per study are drop_withdrawals (12),
  outcomes (8), reported_event_totals (7.3), conditions (2.0), interventions (1.9),
  sponsors (1.56).
- Lab site-success sweep (raw hand schema, 0.4.0): 14 trials, best 0.3337 val (under relgt's
  0.3394), refit test 0.3795 (7th of 16).

## TODO after the other tasks report

- [ ] Consolidate per-task validation and refit test scores across `derived-1024`/`1024b`;
      decide whether the selection rule needs a floor (minimum passes) or late-checkpoint
      scoring, with evidence from more than one task.
- [ ] Decide the F1 treatment on evidence from all three F1 tasks (stale-validation is on;
      checkpoint regime, history dropout, absolute time are rejected for now).
- [ ] Text token dtype (int32 ids, bool mask) in RelFlow; re-measure medium-tier throughput.
- [ ] Model-step kernel count on wide small schemas (user-attendance: 459 leaves, 600 ms step).
- [ ] Fold the campaign leaderboards into tier presets for the RelArena wrapper.
- [ ] Re-derive site-success with the folded one-row children and compare against the
      0.3737 / 0.3747 ablation figures before adopting the fold campaign-wide; the running
      loops keep their already-derived bundles.
- [ ] Why the hand schema beats the derived one on site-success by 0.01 (beyond seed noise
      once more seeds exist): field selection under the 40-field budget is the suspect.
