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
