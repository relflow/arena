# RelFlow arena

[RelFlow](https://github.com/relflow/relflow) on the RelArena benchmark, run
the way RelArena's harness runs a model. Every task's structure is derived
from the RelBench database itself: the entity row, the rows that point at it,
the rows they point at, and the tables that align with them, typed from
measured column statistics. No task is described by hand, no feature is
engineered in SQL, and nothing reads the test split.

## Install

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra text
```

`text` installs the frozen text encoders; `tracking` adds the MLflow client.
RelBench downloads datasets into its cache (`RELBENCH_CACHE_DIR`).

## Run on Colab

[Open the notebook in Colab](https://colab.research.google.com/github/relflow/arena/blob/main/notebooks/post_votes_colab.ipynb)
(`notebooks/post_votes_colab.ipynb`). With a GPU runtime it installs this package, downloads the
RelBench task, and runs the Optuna loop on `rel-stack/post-votes` (change `TASK` at the top for
another task): validation-only search, a refit and test score after every trial, state mirrored to
Google Drive so a new session resumes where the last one stopped.

## Derive a task

```bash
uv run python -m benchmarks.relarena.derive \
  --relbench rel-f1/driver-position --cache ~/.cache/relbench --output derived
```

The engine reads keys and time columns from the cached Parquet files, profiles
rows up to the dataset's validation cutoff, and writes
`derived/rel-f1/driver-position/{benchmark.yaml,model.yaml,observations.sql}`
plus a `derivation.json` with the size tier and cutoff. Branch lengths and
field counts are budgets of the tier, not properties of the task.

## Run one trial

```bash
uv run python -m benchmarks.relarena.supervised \
  --benchmark rel-f1/driver-position --spec derived/rel-f1/driver-position/benchmark.yaml \
  --work-dir runs/trial --output runs/trial.json --relbench-cache ~/.cache/relbench \
  --accelerator gpu --precision bf16-mixed --batch-size auto
```

The runner builds a DuckDB snapshot straight from the cached Parquet files,
censored at the validation cutoff, materializes the observations once,
trains with a plateau schedule, and selects the checkpoint by the exact
validation metric. Frozen text vectors persist in a store beside the RelBench
cache by default (`--text-store none` disables it); `--batch-size auto` lets
Lightning's finder pick a batch that fits and keeps half of it.

## Sweep

`benchmarks.relarena.sweep` runs one task's trials under RelArena's rules: the
space is keyed only by the task's training-anchor tier (small below 20,000
anchors, medium below 500,000, large above), the batch cap is a compute knob
fixed per tier, trial 0 is the tier's default configuration, selection reads
the validation split alone, and the winner is refit on train plus validation
with the store censored at the test timestamp.

```bash
uv run python -m benchmarks.relarena.sweep plan \
  --task rel-trial/site-success --cache ~/.cache/relbench --trials 24 --seed 0 \
  --output sweeps/site-success/plan.json
uv run python -m benchmarks.relarena.sweep run \
  --plan sweeps/site-success/plan.json --name site-success --output sweeps/site-success \
  --source . --cache ~/.cache/relbench --gpu-lock /tmp/relflow-gpu.lock
```

`plan` is a pure function of task, trial count, and seed; `--space` narrows
the tier's choices and `--default` sets trial 0 for a follow-up sweep. `run`
derives the structure once per sweep directory, executes pending trials as
`supervised` children, relaunches a trial at half its batch after an
out-of-memory failure, keeps `state.json`, `trial-NNN/`, and the leaderboard,
and resumes when restarted; a `STOP` file ends it after the running trial.
`report` rebuilds the leaderboard from the result files. `queue` runs several
tasks in turn on one GPU.

On Modal, `benchmarks/relarena/modal_sweep.py` fans out one container per
task with everything persisted on a Volume:

```bash
uv run modal run --detach benchmarks/relarena/modal_sweep.py \
  --tasks rel-f1/driver-position,rel-event/user-attendance --trials 24 \
  --trial-seconds 3600 --max-hours 8 --suffix campaign
```

Set `RELFLOW_SRC` to a local `src/relflow` to run an unreleased RelFlow.

## Results and provenance

`results/raw-schema-20261022/` records how the derivation engine was
validated against the earlier hand-written catalog. Trial reports carry the
specification digests, the data lineage, the training configuration, and the
RelFlow version, and never a test score: RelArena's harness owns that number.

## Development

```bash
uv run ruff format --check && uv run ruff check
uv run pytest
```

Tests build their own small databases and specifications; none depend on a
downloaded dataset.
