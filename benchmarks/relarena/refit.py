"""Outer-split refit of a sweep's selected configuration.

Tuning selects on the inner validation split. The refit trains that
configuration on the official train and validation anchors with the store
censored at the test timestamp, for a fixed number of updates carried over
from tuning, then predicts the test anchors and scores them once through
RelBench's evaluator. No validation split exists at this stage, so nothing is
selected or stopped early, and test labels are read only inside RelBench.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as lit
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import relflow as rf
import torch

from benchmarks.relarena import reports
from benchmarks.relarena.catalog import TASKS
from benchmarks.relarena.dataset import materialize
from benchmarks.relarena.resources import resolve
from benchmarks.relarena.spec import BenchmarkSpec
from benchmarks.relarena.store import ParquetSources, prepare_anchors, prepare_store
from benchmarks.relarena.supervised import batches, memory


def scaled(selected_step: int, train_rows: int, validation_rows: int) -> int:
    """Carry the tuned number of updates over to the larger outer training split."""
    if selected_step < 1 or train_rows < 1 or validation_rows < 0:
        raise ValueError("refit needs a positive selected step and training rows and non-negative validation rows")
    return math.ceil(selected_step * (train_rows + validation_rows) / train_rows)


def ordered(output: pa.Table, rows: int) -> np.ndarray:
    """Predictions in RelBench table order, exactly one finite value per anchor."""
    positions = pc.struct_field(output["inputs"], "__row_position").to_numpy()
    content = pc.struct_field(pc.struct_field(output["predictions"], "target"), "content")
    if pa.types.is_struct(content.type):
        content = pc.struct_field(content, "probability")
    values = content.to_numpy().astype(np.float64)
    if len(values) != rows or not np.array_equal(np.sort(positions), np.arange(rows)):
        raise RuntimeError("Predictions must preserve exactly one row per test anchor")
    if not np.isfinite(values).all():
        raise RuntimeError("Test predictions must be finite")
    return values[np.argsort(positions)]


def prepare(
    spec: BenchmarkSpec,
    directory: Path,
    *,
    seed: int,
    duckdb_config: dict,
    training: bool = True,
    stale_validation: bool = False,
) -> dict:
    """Outer split: train plus validation anchors with labels, test anchors without, store censored at test time.

    ``training=False`` materializes only the test anchors; the checkpoint regime
    trains through the supervised runner's outer phase instead.
    ``stale_validation`` featurizes the validation anchors with the store
    censored at the validation timestamp, as tuning saw them: every test anchor
    looks back across a censored boundary, and this gives training rows that do too.
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset = get_dataset(spec.dataset_name, download=False)
    task = get_task(spec.dataset_name, spec.task_name, download=False)
    if spec.target_column in task.get_table("test").df.columns:
        raise RuntimeError(f"RelBench exposes test labels for {spec.benchmark_name}; the refit must not read them")
    from relbench.utils import get_relbench_cache_dir

    sources = ParquetSources(
        Path(get_relbench_cache_dir()) / spec.dataset_name / "db", upto=str(dataset.test_timestamp)
    )
    timings = {}
    started = time.monotonic()
    snapshot = prepare_store(sources, spec, directory / "snapshot", duckdb_config=duckdb_config)
    timings["snapshot"] = time.monotonic() - started
    test = task.get_table("test").df.reset_index(drop=True)
    splits = [("test", test, False, snapshot)]
    if training and stale_validation:
        inner = ParquetSources(sources.directory, upto=str(dataset.val_timestamp))
        stale = prepare_store(inner, spec, directory / "snapshot-validation", duckdb_config=duckdb_config)
        train = task.get_table("train").df.sample(frac=1, random_state=seed).reset_index(drop=True)
        splits.insert(0, ("train-fresh", train, True, snapshot))
        splits.insert(1, ("train-stale", task.get_table("val").df.reset_index(drop=True), True, stale))
    elif training:
        frame = pd.concat([task.get_table("train").df, task.get_table("val").df], ignore_index=True)
        splits.insert(0, ("train", frame.sample(frac=1, random_state=seed).reset_index(drop=True), True, snapshot))
    rows = {}
    for name, frame, labelled, store in splits:
        anchors = prepare_anchors(
            SimpleNamespace(df=frame), spec, directory / f"{name}-anchors.parquet", include_target=labelled
        )
        started = time.monotonic()
        materialize(
            store,
            anchors,
            spec,
            directory / f"{name}.parquet",
            include_target=labelled,
            fetch_size=16384,
            duckdb_config=duckdb_config,
        )
        timings[name] = time.monotonic() - started
        rows[name] = len(frame)
    if training and stale_validation:
        parts = [pq.read_table(directory / f"{name}.parquet") for name in ("train-fresh", "train-stale")]
        pq.write_table(pa.concat_tables(parts, promote_options="default"), directory / "train.parquet")
        rows["train"] = rows.pop("train-fresh") + rows.pop("train-stale")
    return {
        "cutoff": str(dataset.test_timestamp),
        "rows": {
            **rows,
            "train_official": len(task.get_table("train").df),
            "validation_official": len(task.get_table("val").df),
        },
        "seconds": timings,
        "protocol": "outer split: train plus validation anchors, store censored at the test timestamp, test anchors unlabelled"
        if training
        else "outer split: test anchors unlabelled, store censored at the test timestamp",
    }


def loader(model: rf.Model, path: Path, *, seed: int, predict: bool) -> rf.ArrowDataModule:
    split = "predict" if predict else "train"
    return rf.ArrowDataModule(
        model=model,
        **{split: partial(batches, path, predict=predict)},
        seed=seed,
        shuffle={split: not predict},
        shuffle_rows=16384,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        retain=("__row_position",),
    )


SCHEDULE = (
    "patience", "plateau_factor", "plateau_patience", "max_lr_drops", "min_improvement", "val_every",
    "max_train_seconds", "min_full_passes", "max_steps", "max_epochs",
)  # fmt: skip


def selection(args, directory: Path) -> dict:
    """Train on the outer train split with validation monitoring, through the supervised runner's outer phase.

    This is the regime RelGT and RelGNN-ES report under: the test number comes from
    the validation-selected checkpoint of a model that never trained on validation
    rows, with the store censored at the test timestamp.
    """
    output = directory / "selection.json"
    argv = [
        sys.executable, "-m", "benchmarks.relarena.supervised", "--benchmark", args.benchmark, "--spec", str(args.spec),
        "--work-dir", str(directory / "selection"), "--output", str(output), "--phase", "outer",
        "--accelerator", args.accelerator, "--precision", args.precision, "--seed", str(args.seed),
        "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay), "--fused" if args.fused else "--no-fused",
        "--train-rows", "0", "--validation-rows", "0", "--duckdb-memory-limit", args.duckdb_memory_limit,
        "--num-workers", str(args.num_workers), "--text-store", args.text_store,
        "--run-label", f"{args.run_label or 'refit'}-selection",
    ]  # fmt: skip
    if args.relbench_cache:
        argv += ["--relbench-cache", args.relbench_cache]
    if args.ema_decay is not None:
        argv += ["--ema-decay", str(args.ema_decay)]
    for name in SCHEDULE:
        value = getattr(args, name)
        if value is not None:
            argv += [f"--{name.replace('_', '-')}", str(value)]
    code = subprocess.call(argv, cwd=str(Path(__file__).resolve().parents[2]))
    report = json.loads(output.read_text()) if output.exists() else {}
    if code != 0 or report.get("status") != "passed":
        error = (report.get("error") or {}).get("message", f"exit {code}")
        raise RuntimeError(f"outer-phase selection failed: {error}")
    return report


def run(args) -> dict:
    spec = BenchmarkSpec.from_yaml(args.spec)
    if spec.benchmark_name != args.benchmark:
        raise ValueError(f"Spec {args.spec} declares {spec.benchmark_name!r}; expected {args.benchmark!r}")
    cuda = args.accelerator != "cpu" and torch.cuda.is_available()
    directory = args.work_dir / "data"
    directory.mkdir(parents=True, exist_ok=True)
    settings = resolve(args.work_dir / "duckdb-spill", memory_limit=args.duckdb_memory_limit)["settings"]
    data = prepare(
        spec,
        directory,
        seed=args.seed,
        duckdb_config=settings,
        training=args.regime == "full",
        stale_validation=args.stale_validation,
    )

    lit.seed_everything(args.seed, workers=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if args.regime == "checkpoint":
        selected = selection(args, args.work_dir)
        chosen = selected["result"]["selection"]
        model = rf.Model.load(chosen["checkpoint"])
        from benchmarks.relarena.supervised import attach_text_stores

        attach_text_stores(model, args, spec)
        training = {
            "regime": "checkpoint",
            "seconds": time.monotonic() - started,
            "validation": {key: chosen.get(key) for key in ("score", "metric", "step")},
            "selection_report": str(args.work_dir / "selection.json"),
            **memory(cuda=cuda),
        }
        checkpoint = Path(chosen["checkpoint"])
    else:
        model = rf.Model.from_yaml(spec.model_path, batch_size=args.batch_size)
        if model.schema.objectives != [rf.Address("target")]:
            raise ValueError("Refit requires exactly one masked target and no SSL objectives")
        averaging = [] if args.ema_decay is None else [rf.EMA(decay=args.ema_decay)]
        model.fit(
            datamodule=loader(model, directory / "train.parquet", seed=args.seed, predict=False),
            optimizer=rf.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay, fused=args.fused),
            callbacks=averaging,
            early_stopping=False,
            rollback=False,
            default_root_dir=str(args.work_dir),
            max_epochs=-1,
            max_steps=args.steps,
            accelerator=args.accelerator,
            devices=1,
            precision=args.precision,
            gradient_clip_val=0.25,
            deterministic=True,
            logger=False,
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        training = {
            "regime": "full",
            "steps": int(model.trainer.global_step),
            "requested_steps": args.steps,
            "seconds": time.monotonic() - started,
            "equivalent_full_passes": int(model.trainer.global_step) * args.batch_size / data["rows"]["train"],
            **memory(cuda=cuda),
        }
        if training["steps"] != args.steps:
            raise RuntimeError(f"Refit completed {training['steps']} updates; the fixed budget was {args.steps}")
        checkpoint = args.work_dir / "checkpoints/refit.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint)

    model.eval()
    started = time.monotonic()
    output = model.predict(
        datamodule=loader(model, directory / "test.parquet", seed=args.seed, predict=True),
        default_root_dir=str(args.work_dir),
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    values = ordered(output, data["rows"]["test"])
    predictions = args.work_dir / "test-predictions.parquet"
    pq.write_table(pa.table({"__row_position": np.arange(len(values)), "prediction": values}), predictions)
    from relbench.tasks import get_task

    task = get_task(spec.dataset_name, spec.task_name, download=False)
    # The only read of test labels, inside RelBench's own evaluator.
    metrics = {name: float(value) for name, value in task.evaluate(values).items()}
    return {
        "data": data,
        "training": training,
        "prediction_seconds": time.monotonic() - started,
        "test": metrics,
        "artifacts": {"checkpoint": str(checkpoint), "test_predictions": str(predictions)},
        "artifacts_sha256": reports.hashes({"checkpoint": checkpoint, "test_predictions": predictions}),
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=TASKS)
    parser.add_argument("--spec", required=True, type=Path, help="Bundle manifest with the selected architecture")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--regime",
        choices=("full", "checkpoint"),
        default="full",
        help="full: train+val for the carried-over updates; checkpoint: train with validation selection, outer store",
    )
    parser.add_argument("--steps", type=int, help="Fixed number of updates carried over from tuning (full regime)")
    parser.add_argument(
        "--stale-validation",
        action="store_true",
        help="Featurize the validation anchors with the validation-censored store, so training sees censored histories",
    )
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--weight-decay", required=True, type=float)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--duckdb-memory-limit", default="16GB")
    parser.add_argument("--run-label")
    parser.add_argument("--relbench-cache")
    parser.add_argument("--text-store", default="auto")
    parser.add_argument("--num-workers", default="auto")
    schedule = parser.add_argument_group("checkpoint regime: the tuning schedule the selection run repeats")
    schedule.add_argument("--patience", type=int)
    schedule.add_argument("--plateau-factor", type=float)
    schedule.add_argument("--plateau-patience", type=int)
    schedule.add_argument("--max-lr-drops", type=int)
    schedule.add_argument("--min-improvement", type=float)
    schedule.add_argument("--val-every", type=int)
    schedule.add_argument("--max-train-seconds", type=float)
    schedule.add_argument("--min-full-passes", type=int)
    schedule.add_argument("--max-steps", type=int)
    schedule.add_argument("--max-epochs", type=int)
    args = parser.parse_args(argv)
    if args.regime == "full" and (args.steps is None or args.steps < 1):
        parser.error("--steps must be positive in the full regime")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.accelerator == "cpu" and args.precision == "bf16-mixed":
        args.precision = "32-true"
    return args


def main(argv=None) -> int:
    args = arguments(argv)
    report = {
        "benchmark": args.benchmark,
        "stage": "refit",
        "run_label": args.run_label,
        "seed": args.seed,
        "config": {
            key: getattr(args, key)
            for key in (
                "regime",
                "steps",
                "stale_validation",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "ema_decay",
                "fused",
                "precision",
            )
        },
        "spec": str(args.spec),
        "started_utc": datetime.now(UTC).isoformat(),
    }
    try:
        report.update(run(args), status="passed")
    except Exception as error:  # noqa: BLE001 - the report carries the failure
        report.update(
            status="failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-4000:]},
        )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "stage": "refit",
                "status": report["status"],
                "test": report.get("test"),
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
