"""Full-split supervised RelArena trials, selected exclusively on validation."""

from __future__ import annotations

import argparse
import errno
import gc
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import time
import traceback
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import relflow as rf
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks import provenance
from benchmarks.relarena import reports, tracking
from benchmarks.relarena.catalog import TASKS, supervise
from benchmarks.relarena.dataset import materialize
from benchmarks.relarena.evaluation import TEXT_CACHE_POLICY, reset_text_cache
from benchmarks.relarena.resources import hardware, resolve
from benchmarks.relarena.spec import BenchmarkSpec, payload_fingerprint
from benchmarks.relarena.store import ParquetSources, prepare_anchors, prepare_store

OPTIMIZER = {"learning_rate": 3e-4, "weight_decay": 0.01, "fused": False}


def prepare(
    spec: BenchmarkSpec,
    directory: Path,
    *,
    seed: int,
    train_rows: int = 0,
    validation_rows: int = 0,
    duckdb_config: dict | None = None,
    phase: str = "inner",
) -> dict:
    """Use official train and validation splits with a label-free store censored at the phase boundary.

    The inner phase censors the store at the validation timestamp, as tuning
    must. The outer phase censors it at the test timestamp: the checkpoint refit
    regime trains on the same splits there and reports its validation-selected
    checkpoint, exactly as RelGT and RelGNN-ES do.
    """
    if phase not in ("inner", "outer"):
        raise ValueError(f"phase must be 'inner' or 'outer', got {phase!r}")
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.utils import get_relbench_cache_dir

    cache = Path(get_relbench_cache_dir()) / spec.dataset_name
    dataset = get_dataset(spec.dataset_name, download=not (cache / "db").is_dir())
    task = get_task(spec.dataset_name, spec.task_name, download=not (cache / "tasks" / spec.task_name).is_dir())
    for attribute, expected in (
        ("entity_col", spec.anchor_entity_column),
        ("entity_table", spec.entity_table),
        ("time_col", spec.timestamp_column),
        ("target_col", spec.target_column),
    ):
        if getattr(task, attribute) != expected:
            raise ValueError(f"Task {spec.benchmark_name} {attribute} must be {expected!r}")
    # The harness hands a model the database censored at the split boundary; the same censoring
    # happens here in DuckDB over the cached Parquet files, so no table is loaded into pandas.
    cutoff = dataset.val_timestamp if phase == "inner" else dataset.test_timestamp
    sources = ParquetSources(cache / "db", upto=str(cutoff))
    settings = duckdb_config if duckdb_config is not None else resolve(directory / "duckdb-spill")["settings"]
    preparation = {}
    started = time.monotonic()
    snapshot = prepare_store(sources, spec, directory / "snapshot", duckdb_config=settings)
    preparation["snapshot"] = {"seconds": time.monotonic() - started, **memory(cuda=False)}
    artifacts = {"snapshot.duckdb": snapshot}
    splits = {}
    for split, name, limit in (("train", "train", train_rows), ("val", "validate", validation_rows)):
        frame = task.get_table(split).df
        count = len(frame)
        if limit:
            selected = min(limit, count)
            labels = frame[spec.target_column].to_numpy()
            if spec.task_type == "binary_classification" and selected >= 2 and set(np.unique(labels)) == {0, 1}:
                random = np.random.default_rng(seed)
                required = np.array([random.choice(np.flatnonzero(labels == value)) for value in (0, 1)])
                remaining = np.setdiff1d(np.arange(count), required, assume_unique=True)
                indices = np.concatenate((required, random.choice(remaining, size=selected - 2, replace=False)))
                frame = frame.iloc[random.permutation(indices)]
            else:
                frame = frame.sample(n=selected, random_state=seed)
        elif split == "train":
            frame = frame.sample(frac=1, random_state=seed)
        frame = frame.reset_index(drop=True)
        if not len(frame):
            raise ValueError(f"Task {spec.benchmark_name} has an empty {split} split")
        anchors = prepare_anchors(
            SimpleNamespace(df=frame), spec, directory / f"{name}-anchors.parquet", include_target=True
        )
        started = time.monotonic()
        observations = materialize(
            snapshot,
            anchors,
            spec,
            directory / f"{name}.parquet",
            include_target=True,
            fetch_size=16384,
            duckdb_config=settings,
        )
        preparation[name] = {"seconds": time.monotonic() - started, **memory(cuda=False)}
        artifacts[f"{name}-anchors.parquet"] = anchors
        artifacts[f"{name}.parquet"] = observations
        splits[split] = {"source_rows": count, "rows": len(frame), "limit": limit}
    return {
        "duckdb_settings": settings,
        "preparation_memory": preparation,
        "splits": splits,
        "train_rows": splits["train"]["rows"],
        "train_source_rows": splits["train"]["source_rows"],
        "validation_rows": splits["val"]["rows"],
        "validation_source_rows": splits["val"]["source_rows"],
        "full_training_split": train_rows == 0,
        "full_validation_split": validation_rows == 0,
        "sample_seed": seed if train_rows or validation_rows else None,
        "sample_policy": "Class-covered bounded execution fixtures; no performance claim"
        if train_rows or validation_rows
        else "All official rows",
        "phase": phase,
        "cutoff": str(cutoff),
        "test_split_accessed": False,
        "spec_fingerprint": spec.fingerprint,
        "projection_fingerprint": spec.projection_fingerprint,
        "artifacts": {name: str(path.resolve()) for name, path in artifacts.items()},
        "artifacts_sha256": reports.hashes(artifacts),
        "source_parquet_sha256": reports.hashes({name: cache / "db" / f"{name}.parquet" for name in spec.sources}),
        "task_parquet_sha256": reports.hashes(
            {split: cache / "tasks" / spec.task_name / f"{split}.parquet" for split in ("train", "val")}
        ),
    }


def unbounded(model: dict) -> dict:
    """Copy the model with unbounded repeated branches and no overflow policies."""
    result = dict(model)
    if "fields" in result:
        result["fields"] = {name: unbounded(value) for name, value in result["fields"].items()}
    if result.get("type") == "branch":
        result.pop("overflow", None)
        result["length"] = None
    return result


def reuse(spec: BenchmarkSpec, directory: Path, *, report_path: Path, seed: int) -> dict:
    """Verify a complete prior preparation, then link its immutable split files."""
    from relbench.utils import get_relbench_cache_dir

    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    if report.get("complete") is not True or report.get("status") != "passed":
        raise ValueError("Prepared-data reuse requires a completed, passed source report")
    if report.get("benchmark") != spec.benchmark_name:
        raise ValueError("Prepared-data reuse requires the same benchmark")
    prepared = report["data"]
    if (
        prepared.get("full_training_split") is not True
        or prepared.get("full_validation_split") is not True
        or prepared.get("test_split_accessed") is not False
    ):
        raise ValueError("Prepared-data reuse requires full official train/validation splits and no test access")
    for split in ("train", "val"):
        metadata = prepared["splits"][split]
        if metadata["rows"] <= 0 or metadata["rows"] != metadata["source_rows"] or metadata["limit"] != 0:
            raise ValueError(f"Prepared-data reuse requires complete, uncapped {split} rows")
    artifacts = {name: Path(path).resolve() for name, path in prepared["artifacts"].items()}
    original_directory = artifacts["train.parquet"].parent.parent
    original_model, original_query = original_directory / "model.yaml", original_directory / "observations.sql"
    if (
        reports.hashes({"model.yaml": original_model, "observations.sql": original_query})
        != report["specification_sha256"]
    ):
        raise ValueError("Prepared-data source model/query files differ from their recorded hashes")
    original = replace(spec, model_path=original_model, query_path=original_query)
    if (
        original.fingerprint != prepared["spec_fingerprint"]
        or spec.projection_fingerprint != prepared["projection_fingerprint"]
    ):
        raise ValueError("Prepared-data source task/projection contract differs from the recorded specification")
    old_model = yaml.safe_load(original_model.read_text())
    new_model = yaml.safe_load(spec.model_path.read_text())
    old_batch_size, new_batch_size = old_model.get("batch_size"), new_model.get("batch_size")
    # Materialized observations depend on the query and the field contract alone;
    # width, depth, heads, feed-forward, dropout, batch size, and branch caps are training choices.
    if original.query != spec.query or unbounded(old_model.get("fields", {})) != unbounded(new_model.get("fields", {})):
        raise ValueError(
            "Prepared-data reuse allows only architecture, batch-size, and branch length/overflow changes; "
            "SQL and field definitions must match"
        )
    cache = Path(get_relbench_cache_dir()) / spec.dataset_name
    source_hashes = reports.hashes({name: cache / "db" / f"{name}.parquet" for name in spec.sources})
    task_hashes = reports.hashes(
        {split: cache / "tasks" / spec.task_name / f"{split}.parquet" for split in ("train", "val")}
    )
    if source_hashes != prepared["source_parquet_sha256"] or task_hashes != prepared["task_parquet_sha256"]:
        raise ValueError("Prepared-data source/task Parquets differ from their recorded hashes")
    names = ("train.parquet", "validate.parquet", "train-anchors.parquet", "validate-anchors.parquet")
    selected = {name: artifacts[name] for name in names}
    hashes = reports.hashes(selected)
    if hashes != {name: prepared["artifacts_sha256"][name] for name in names}:
        raise ValueError("Prepared-data split/anchor Parquets differ from their recorded hashes")
    for name, split in (("train", "train"), ("validate", "val")):
        expected = prepared["splits"][split]["rows"]
        for suffix in (".parquet", "-anchors.parquet"):
            if pq.ParquetFile(selected[f"{name}{suffix}"]).metadata.num_rows != expected:
                raise ValueError(f"Prepared-data {name}{suffix} does not contain its declared full split")
        next_position = 0
        for batch in pq.ParquetFile(selected[f"{name}.parquet"]).iter_batches(
            columns=["__row_position"], use_threads=False
        ):
            positions = batch.column(0).to_numpy()
            if not np.array_equal(positions, np.arange(next_position, next_position + len(positions))):
                raise ValueError(f"Prepared-data {name} rows must retain physical anchor order")
            next_position += len(positions)
    directory.mkdir(parents=True)
    linked = {}
    for name, source in selected.items():
        destination = directory / name
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                raise
            target = destination.open("xb")
            try:
                with target, source.open("rb") as origin:
                    shutil.copyfileobj(origin, target)
                if reports.hashes({name: destination})[name] != hashes[name]:
                    raise ValueError(f"Copied prepared-data artifact {name} differs from its verified SHA256")
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        linked[name] = str(destination.resolve())
    return {
        **prepared,
        "spec_fingerprint": spec.fingerprint,
        "projection_fingerprint": spec.projection_fingerprint,
        "artifacts": linked,
        "artifacts_sha256": hashes,
        "preparation": "verified_immutable_reuse",
        "lineage": {
            "report": str(report_path),
            "report_sha256": reports.hashes({"report": report_path})["report"],
            "source_spec_fingerprint": prepared["spec_fingerprint"],
            "preparation_seed": report["seed"],
            "training_seed": seed,
            "source_batch_size": old_batch_size,
            "batch_size": new_batch_size,
            "source_artifacts": {name: str(path) for name, path in selected.items()},
            "source_parquet_sha256": source_hashes,
            "task_parquet_sha256": task_hashes,
            "contract": "same SQL, fields, source data, anchor order and full splits; branch length/overflow and execution batch size may differ",
        },
    }


def batches(path: Path, *, predict: bool = False):
    """Open a bounded, restartable physical-order Parquet stream."""
    with pq.ParquetFile(path) as source:
        columns = [name for name in source.schema_arrow.names if not predict or name != "target"]
        yield from source.iter_batches(batch_size=1024, columns=columns, use_threads=False)


def batch_size_argument(value: str) -> int | str:
    """Parse ``--batch-size``: a positive integer, or ``auto`` to let Lightning's finder choose."""
    return value if value == "auto" else int(value)


def workers_argument(value: str) -> int | str:
    """Parse ``--num-workers``: a non-negative integer, or ``auto`` to decide from the schema."""
    return value if value == "auto" else int(value)


def loader_workers(choice: int | str, template: Mapping[str, Any]) -> int:
    """Loader processes for a schema: four when it carries Text fields, none otherwise.

    The loader probe found tokenization makes text schemas loader-bound at zero
    workers, while schemas without text are bound by the training step itself,
    where extra processes only compete for the cores that issue kernels. Four
    workers already outrun the step on every text task measured, and each worker
    holds two packed batches in flight, so more only cost container memory.
    """
    if choice != "auto":
        return int(choice)

    def textual(fields: Mapping[str, Any]) -> bool:
        return any(
            field.get("type") == "text" or (isinstance(field.get("fields"), Mapping) and textual(field["fields"]))
            for field in fields.values()
            if isinstance(field, Mapping)
        )

    return min(4, max(hardware()["cpus"] - 1, 1)) if textual(template.get("fields", {})) else 0


def data(model: rf.Model, directory: Path, *, seed: int, predict: bool = False, workers: int = 0) -> rf.ArrowDataModule:
    """Stream prepared Parquets; labels never enter the prediction encoder.

    ``workers`` tensorize batches in loader processes so the step overlaps with
    the next batch's preparation; the encoding is seed-derived, so worker count
    does not change what a seeded run sees. Pinning stays off and each worker
    prefetches one batch: the loader probe measured pinning as a main-process
    cost, and packed batches of a medium task run to hundreds of megabytes each.
    """
    splits = (
        {"predict": directory / "validate.parquet"}
        if predict
        else {"train": directory / "train.parquet", "validate": directory / "validate.parquet"}
    )
    return rf.ArrowDataModule(
        model=model,
        **{name: partial(batches, path, predict=predict) for name, path in splits.items()},
        seed=seed,
        shuffle={name: name == "train" for name in splits},
        shuffle_rows=16384,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=False,
        prefetch_factor=1,
        retain=("__row_position",),
    )


def text_store(args, spec: BenchmarkSpec) -> Path | None:
    """Where this run keeps frozen text vectors: ``none`` disables, ``auto`` sits beside the RelBench cache."""
    choice = str(getattr(args, "text_store", "auto"))
    if choice == "none":
        return None
    if choice != "auto":
        return Path(choice)
    from relbench.utils import get_relbench_cache_dir

    cache = Path(args.relbench_cache) if getattr(args, "relbench_cache", None) else Path(get_relbench_cache_dir())
    return cache / "text-store" / spec.dataset_name


def attach_text_stores(model: rf.Model, args, spec: BenchmarkSpec) -> dict[str, str]:
    """Point every Text field at the run's persistent store; returns the fields attached.

    The store is a runtime resource rather than part of the schema, so
    checkpoints and fingerprints stay unchanged and a loaded model is attached
    again the same way.
    """
    directory = text_store(args, spec)
    if directory is None:
        return {}
    from relflow.tensorfields.extensions.text import Embedder, EmbeddingStore

    attached = {}
    for address, request in model.schema.requests.items():
        if getattr(request, "type", None) != "text":
            continue
        embedder = model.nodes[address].embedder
        if not isinstance(embedder, Embedder):
            continue
        embedder.store = EmbeddingStore.open(
            directory,
            Embedder.fingerprint(request),
            embedder.hidden_size,
            resident=getattr(args, "text_resident", None),
        )
        attached[str(address)] = str(directory)
    return attached


def memory(*, cuda: bool) -> dict[str, int | None]:
    """Measure CUDA peaks since reset and the process-lifetime host RSS high-water mark."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "process_peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024),
    }


def score(kind: str, labels: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Compute exact whole-split metrics in source units."""
    if labels.ndim != 1 or values.ndim != 1 or len(labels) != len(values) or not len(labels):
        raise RuntimeError("Validation labels and predictions must have the same nonempty row axis")
    if not np.isfinite(labels).all() or not np.isfinite(values).all():
        raise RuntimeError("Validation labels and predictions must be finite")
    if kind == "binary_classification":
        if set(np.unique(labels)) != {0, 1} or ((values < 0) | (values > 1)).any():
            raise RuntimeError("AUROC requires both binary classes and probability predictions")
        return {
            "roc_auc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values)),
            "accuracy": float(np.mean((values >= 0.5) == labels)),
        }
    if kind != "regression":
        raise ValueError(f"Unknown supervised task type {kind!r}")
    residual = values - labels
    return {"mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual**2)))}


class Audit(lit.Callback):
    """Retain actual update/row counts and finite per-step diagnostics."""

    def __init__(self, run=None) -> None:
        self.run = run
        self.trace: list[dict[str, float | int]] = []
        self.examples_seen = 0
        self.parameters: dict[str, int] = {}
        self.started = 0.0
        self.examples_seen_is_lower_bound = False
        self.initial_step = 0
        self.max_train_seconds = math.inf
        self.training_deadline = math.inf
        self.directory: Path | None = None
        self.stop_reason: str | None = None

    def state_dict(self) -> dict:
        return {"examples_seen": self.examples_seen, "examples_seen_is_lower_bound": self.examples_seen_is_lower_bound}

    def load_state_dict(self, state: dict) -> None:
        self.examples_seen = state["examples_seen"]
        self.examples_seen_is_lower_bound = state.get("examples_seen_is_lower_bound", False)

    def on_train_start(self, trainer, pl_module) -> None:
        self.started = time.monotonic()
        self.initial_step = int(trainer.global_step)
        self.parameters = {
            "trainable": sum(value.numel() for value in pl_module.parameters() if value.requires_grad),
            "frozen": sum(value.numel() for value in pl_module.parameters() if not value.requires_grad),
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.examples_seen += len(batch.source)
        record: dict[str, float | int] = {"step": int(trainer.global_step), "examples_seen": self.examples_seen}
        for key, value in trainer.callback_metrics.items():
            if key == "epoch" or key.endswith("_epoch"):
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                number = float(value.detach().cpu())
                if not math.isfinite(number):
                    raise RuntimeError(f"Non-finite metric {key!r} at update {trainer.global_step}")
                record[str(key)] = number
        record["learning_rate"] = float(trainer.optimizers[0].param_groups[0]["lr"])
        record.update(
            {key: value for key, value in memory(cuda=pl_module.device.type == "cuda").items() if value is not None}
        )
        self.trace.append(record)
        if self.run is not None:
            self.run.record("supervised", record)
        if trainer.global_step % 25 == 0:
            print(json.dumps({"training": record}), flush=True)
        exhausted = time.monotonic() - self.started >= self.max_train_seconds or time.time() >= self.training_deadline
        if exhausted:
            self.stop_reason = "training_time_limit"
            trainer.should_stop = True
        if exhausted or trainer.global_step % 250 == 0:
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                pending = self.directory / "last.pending.ckpt"
                trainer.save_checkpoint(pending)
                os.replace(pending, self.directory / "last.ckpt")


class Selection(lit.Callback):
    """Save the exact best full-validation AUROC/MAE, lower the learning rate on plateaus, and stop early.

    A validation improves the record only when the exact metric moves by at
    least ``min_improvement`` relative to the best; smaller gains still save the
    checkpoint but count as stale. After ``plateau_patience`` stale validations
    the learning rate of every parameter group is multiplied by
    ``plateau_factor``, at most ``max_lr_drops`` times and never below
    ``min_lr``. Training stops once ``patience`` stale validations accumulate,
    or once the drops are exhausted and another ``plateau_patience`` stale
    validations follow. Stops respect the minimum update budget; drops do not.
    """

    def __init__(self, spec: BenchmarkSpec, rows: int, directory: Path, args, run=None) -> None:
        self.spec, self.rows, self.directory, self.args, self.run = spec, rows, directory, args, run
        self.metric = "roc_auc" if spec.task_type == "binary_classification" else "mae"
        self.mode = "max" if self.metric == "roc_auc" else "min"
        self.history: list[dict] = []
        self.best: dict | None = None
        self.stale = 0
        self.stale_since_drop = 0
        self.drops: list[dict] = []
        self.plateau_factor = float(getattr(args, "plateau_factor", 0.1))
        self.plateau_patience = int(getattr(args, "plateau_patience", 2))
        self.max_lr_drops = int(getattr(args, "max_lr_drops", 2))
        self.min_improvement = float(getattr(args, "min_improvement", 0.0))
        self.min_lr = float(getattr(args, "min_lr", 1e-6))
        self.active = False
        self.current: np.ndarray | None = None
        self.values: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.started = 0.0
        self.stop_reason: str | None = None
        self.minimum_steps = max(min(400, args.max_steps), getattr(args, "minimum_early_stopping_steps", 0))
        self.hook = None

    def state_dict(self) -> dict:
        return {
            "history": self.history,
            "best": self.best,
            "stale": self.stale,
            "stale_since_drop": self.stale_since_drop,
            "drops": self.drops,
            "examples_seen": getattr(self, "examples_seen", 0),
        }

    def load_state_dict(self, state: dict) -> None:
        self.history = state["history"]
        self.best = state["best"]
        if self.best is not None:
            self.best["checkpoint"] = str((self.directory / "best.ckpt").resolve())
        self.stale = state["stale"]
        self.stale_since_drop = state.get("stale_since_drop", 0)
        self.drops = list(state.get("drops", []))
        if hasattr(self, "examples_seen"):
            self.examples_seen = state["examples_seen"]

    def plateau(self) -> dict:
        """The schedule and every learning-rate drop it made, for the report."""
        return {
            "factor": self.plateau_factor,
            "patience": self.plateau_patience,
            "max_lr_drops": self.max_lr_drops,
            "min_improvement": self.min_improvement,
            "min_lr": self.min_lr,
            "drops": self.drops,
        }

    def drop(self, trainer) -> None:
        """Multiply every parameter group's learning rate by the plateau factor, never below the floor."""
        optimizers = getattr(trainer, "optimizers", None) or []
        before = [group["lr"] for optimizer in optimizers for group in optimizer.param_groups]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * self.plateau_factor, self.min_lr)
        after = [group["lr"] for optimizer in optimizers for group in optimizer.param_groups]
        record = {
            "step": int(trainer.global_step),
            "stale": self.stale_since_drop,
            "from": before[0] if before else None,
            "to": after[0] if after else None,
        }
        self.drops.append(record)
        self.stale_since_drop = 0
        if self.run is not None:
            self.run.record("plateau", record)
        print(json.dumps({"plateau": self.spec.benchmark_name, **record}), flush=True)

    def on_fit_start(self, trainer, pl_module) -> None:
        self.hook = pl_module.nodes[rf.Address("target")].decoder.register_forward_hook(self.capture)

    def on_train_start(self, trainer, pl_module) -> None:
        self.started = time.monotonic()
        if hasattr(self, "examples_seen"):
            for callback in trainer.callbacks:
                if isinstance(callback, Audit):
                    self.examples_seen = callback.examples_seen
                    break

    def capture(self, module, inputs, output) -> None:
        if not self.active:
            return
        if self.current is not None:
            raise RuntimeError("Validation must decode its scalar target exactly once per batch")
        value = output.payload["content"].detach().reshape(-1)
        if self.spec.task_type == "binary_classification":
            value = value.sigmoid()
        # Number.decode and Number.write expose source units. Its normalizer
        # scales the loss residual only; applying an inverse transform is wrong.
        self.current = value.float().cpu().numpy().astype(np.float64)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        reset_text_cache(pl_module)
        self.active = True
        self.values.clear()
        self.labels.clear()
        self.positions.clear()
        self.current = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.current is None or len(self.current) != len(batch.source):
            raise RuntimeError("Validation target predictions must align with Arrow batch rows")
        self.values.append(self.current)
        self.labels.append(batch.source["target"].to_numpy())
        self.positions.append(batch.source["__row_position"].to_numpy())
        self.current = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.active = False
        if not self.positions:
            raise RuntimeError("Checkpoint selection received no validation anchors")
        positions = np.concatenate(self.positions)
        if len(positions) != self.rows or not np.array_equal(np.sort(positions), np.arange(self.rows)):
            raise RuntimeError("Checkpoint selection requires every validation anchor exactly once")
        self.consider(trainer, score(self.spec.task_type, np.concatenate(self.labels), np.concatenate(self.values)))
        exhausted = len(self.drops) >= self.max_lr_drops
        if not exhausted and self.stale_since_drop >= self.plateau_patience:
            self.drop(trainer)
        if self.args.max_train_seconds is not None and time.monotonic() - self.started >= self.args.max_train_seconds:
            self.stop_reason = "training_time_limit"
            trainer.should_stop = True
        elif trainer.global_step >= self.minimum_steps and self.stale >= self.args.patience:
            self.stop_reason = "validation_plateau"
            trainer.should_stop = True
        elif trainer.global_step >= self.minimum_steps and exhausted and self.stale_since_drop >= self.plateau_patience:
            self.stop_reason = "plateau_after_lr_drops"
            trainer.should_stop = True

    def consider(self, trainer, metrics: dict[str, float], *, checkpoint: Path | None = None) -> None:
        if not metrics or not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Checkpoint selection requires finite full-validation metrics")
        record = {"step": int(trainer.global_step), **metrics}
        repeated = bool(self.history and self.history[-1]["step"] == record["step"])
        if repeated:
            if any(
                not np.isclose(value, self.history[-1][key], rtol=1e-5, atol=1e-6) for key, value in metrics.items()
            ):
                raise RuntimeError("Independent predictions disagree with same-step validation metrics")
        else:
            self.history.append(record)
        previous = None if self.best is None else self.best["score"]
        better = previous is None or (
            metrics[self.metric] > previous if self.mode == "max" else metrics[self.metric] < previous
        )
        # Tiny gains keep the best checkpoint but do not reset patience.
        improved = better and (
            previous is None or abs(metrics[self.metric] - previous) >= self.min_improvement * abs(previous)
        )
        record["learning_rate"] = (
            float(trainer.optimizers[0].param_groups[0]["lr"]) if getattr(trainer, "optimizers", None) else None
        )
        if better:
            self.directory.mkdir(parents=True, exist_ok=True)
            pending, path = self.directory / "best.pending.ckpt", self.directory / "best.ckpt"
            self.best = {
                "metric": self.metric,
                "mode": self.mode,
                "step": record["step"],
                "score": metrics[self.metric],
                "checkpoint": str(path.resolve()),
            }
            if improved:
                self.stale = 0
                self.stale_since_drop = 0
            elif not repeated:
                self.stale_since_drop += 1
                if trainer.global_step >= self.minimum_steps:
                    self.stale += 1
            if checkpoint is None:
                trainer.save_checkpoint(pending)
            else:
                import shutil

                shutil.copyfile(checkpoint, pending)
            os.replace(pending, path)
        elif not repeated:
            self.stale_since_drop += 1
            if trainer.global_step >= self.minimum_steps:
                self.stale += 1
        if self.run is not None:
            self.run.record("validation", record)
            self.run.flush()
        print(json.dumps({"validation": self.spec.benchmark_name, **record}), flush=True)

    def teardown(self, trainer, pl_module, stage) -> None:
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def predictions(model: rf.Model, directory: Path, *, args, data_factory=data, root: Path | None = None) -> np.ndarray:
    """Predict the directory's split; ``root`` anchors the trainer so it never consults the process directory."""
    reset_text_cache(model)
    output = model.predict(
        datamodule=data_factory(
            model, directory, seed=args.seed, predict=True, workers=getattr(args, "num_workers", 0)
        ),
        default_root_dir=str(root or directory),
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    positions = pc.struct_field(output["inputs"], "__row_position").to_numpy()
    content = pc.struct_field(pc.struct_field(output["predictions"], "target"), "content")
    if pa.types.is_struct(content.type):
        content = pc.struct_field(content, "probability")
    values = content.to_numpy().astype(np.float64)
    rows = pq.ParquetFile(directory / "validate.parquet").metadata.num_rows
    if len(values) != rows or not np.array_equal(np.sort(positions), np.arange(rows)):
        raise RuntimeError("Predictions must preserve exactly one row per validation anchor")
    if not np.isfinite(values).all():
        raise RuntimeError("Validation predictions must be finite")
    return values[np.argsort(positions)]


def digest(model: rf.Model, *, encoder: bool) -> str:
    result = hashlib.sha256()
    for name, value in sorted(model.nodes.state_dict().items()):
        if not isinstance(value, torch.Tensor) or ((".embedder." in name or ".encoder." in name) != encoder):
            continue
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def find_batch_size(
    model: rf.Model, datamodule: lit.LightningDataModule, args, optimizer_options: dict, *, root: Path
) -> int:
    """Find the largest batch that fits, up to ``--max-batch-size``, and keep only a fraction of it.

    The probe is a short capacity heuristic on the first batches, and online
    vocabularies keep growing afterwards, so the search starts small and the
    trial keeps ``--batch-size-headroom`` of what fit. A later overflow is the
    sweep's relaunch to handle, at half the size chosen here.
    """
    from lightning.pytorch.tuner import Tuner

    # The native Trainer probe needs an explicit optimizer factory on the model.
    model.optimizer = rf.adamw(**optimizer_options)
    try:
        tuning = lit.Trainer(
            accelerator=args.accelerator,
            devices=1,
            precision=args.precision,
            default_root_dir=str(root),
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        Tuner(tuning).scale_batch_size(
            model,
            datamodule=datamodule,
            mode="binsearch",
            init_val=min(8, args.max_batch_size),
            max_val=args.max_batch_size,
            steps_per_trial=3,
            max_trials=12,
            # The headroom below replaces the finder's own trim.
            margin=0.0,
        )
    finally:
        model.optimizer = None
    found = int(model.batch_size)
    model.batch_size = max(1, int(found * args.batch_size_headroom))
    return found


def train(
    spec: BenchmarkSpec,
    directory: Path,
    *,
    args,
    run=None,
    data_factory=data,
    selection_factory=Selection,
    optimizer_factory=None,
) -> dict:
    cuda = args.accelerator != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    lit.seed_everything(args.seed, workers=True)
    model = rf.Model.from_yaml(spec.model_path, batch_size=32 if args.batch_size == "auto" else args.batch_size)
    text_fields = attach_text_stores(model, args, spec)
    if model.schema.objectives != [rf.Address("target")]:
        raise ValueError("Supervised trials require exactly one masked target and no SSL objectives")
    labels = (
        pq.read_table(directory / "data/validate.parquet", columns=["target"])["target"].to_numpy().astype(np.float64)
    )
    training_labels = pq.read_table(directory / "data/train.parquet", columns=["target"])["target"].to_numpy()
    statistic = "training_label_median" if spec.task_type == "regression" else "training_label_mean"
    constant = float(np.median(training_labels) if spec.task_type == "regression" else np.mean(training_labels))
    baseline = score(spec.task_type, labels, np.full(len(labels), constant))
    train_rows = len(training_labels)
    del training_labels
    result = {
        "initial_encoder_digest": digest(model, encoder=True),
        "initial_supervised_head_digest": digest(model, encoder=False),
        "baseline": baseline,
        "baseline_metadata": {"statistic": statistic, "value": constant, "rows": train_rows},
        "text_store": {"fields": text_fields, "resident_mib": getattr(args, "text_resident", None)},
    }
    setup_memory = memory(cuda=cuda)
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    averaging = [] if getattr(args, "ema_decay", None) is None else [rf.EMA(decay=args.ema_decay)]
    optimizer_options = dict(OPTIMIZER)
    if getattr(args, "learning_rate", None) is not None:
        optimizer_options["learning_rate"] = args.learning_rate
    if getattr(args, "fused", None) is not None:
        optimizer_options["fused"] = args.fused
    if getattr(args, "weight_decay", None) is not None:
        optimizer_options["weight_decay"] = args.weight_decay
    if args.batch_size == "auto":
        # The finder probes the largest batches the card holds; several workers packing batches
        # that size exhausted a 32 GB container, and the finder has no throughput to gain.
        found = find_batch_size(
            model,
            data_factory(model, directory / "data", seed=args.seed, workers=0),
            args,
            optimizer_options,
            root=directory,
        )
        args.batch_size = int(model.batch_size)
        args.val_every = max(1, min(args.val_every, math.ceil(train_rows / args.batch_size)))
        result["batch_size_search"] = {
            "max": args.max_batch_size,
            "found": found,
            "headroom": args.batch_size_headroom,
            "selected": args.batch_size,
        }
        if run is not None:
            run.record("batch_size_search", {"step": 0, **result["batch_size_search"]})
    args.minimum_early_stopping_steps = max(
        min(400, args.max_steps),
        math.ceil(train_rows / args.batch_size) * getattr(args, "min_full_passes", 0),
    )
    audit = Audit(run)
    selection = selection_factory(spec, len(labels), directory / "checkpoints", args, run)
    audit.max_train_seconds = math.inf if args.max_train_seconds is None else args.max_train_seconds
    audit.training_deadline = getattr(args, "training_deadline", math.inf)
    audit.directory = directory / "checkpoints"
    if time.time() >= audit.training_deadline:
        raise RuntimeError("Training allocation elapsed during setup")
    started = time.monotonic()
    model.fit(
        datamodule=data_factory(model, directory / "data", seed=args.seed, workers=getattr(args, "num_workers", 0)),
        optimizer=rf.adamw(**optimizer_options) if optimizer_factory is None else optimizer_factory,
        callbacks=[*averaging, audit, selection],
        early_stopping=False,
        rollback=False,
        # Trainers resolve their root here, never through the process's working directory.
        default_root_dir=str(directory),
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
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
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
    )
    trainer = model.trainer
    steps = int(trainer.global_step)
    if steps < 1 or len(audit.trace) != steps - audit.initial_step:
        raise RuntimeError("Completed optimizer updates must match the positive per-step trace")
    result["stages"] = {
        "supervised": {
            "steps": steps,
            "train_batches": steps,
            "new_train_batches": len(audit.trace),
            "resumed_at_step": audit.initial_step,
            "examples_seen_is_lower_bound": audit.examples_seen_is_lower_bound,
            "examples_seen": audit.examples_seen,
            "training_rows": train_rows,
            "equivalent_full_epochs": audit.examples_seen / train_rows,
            "completed_epochs": int(trainer.current_epoch),
            "minimum_early_stopping_steps": args.minimum_early_stopping_steps,
            "seconds": time.monotonic() - started,
            "parameters": audit.parameters,
            "resolved_precision": str(trainer.precision),
            **memory(cuda=cuda),
            "trace": audit.trace,
        }
    }
    result["stopping_reason"] = (
        audit.stop_reason or selection.stop_reason or ("max_steps" if steps >= args.max_steps else "max_epochs")
    )
    result["plateau"] = selection.plateau()
    final = directory / "checkpoints/final.ckpt"
    final.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(final)
    result["memory"] = {"model_setup": setup_memory, "training": memory(cuda=cuda)}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    prediction_started = time.monotonic()
    final_values = predictions(model, directory / "data", args=args, data_factory=data_factory, root=directory)
    result["prediction_seconds"] = {"final": time.monotonic() - prediction_started}
    result["memory"]["final_prediction"] = memory(cuda=cuda)
    final_metrics = score(spec.task_type, labels, final_values)
    selection.consider(trainer, final_metrics, checkpoint=final)
    if selection.best is None:
        raise RuntimeError("Supervised training did not produce a selected validation checkpoint")
    result.update(
        final_validation=final_metrics,
        validation_history=selection.history,
        selection={**selection.best, "sha256": reports.hashes({"best": Path(selection.best["checkpoint"])})["best"]},
        final_checkpoint={
            "checkpoint": str(final.resolve()),
            "step": steps,
            "sha256": reports.hashes({"final": final})["final"],
        },
    )
    del model, trainer, averaging
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    restored = rf.Model.load(result["selection"]["checkpoint"])
    attach_text_stores(restored, args, spec)
    prediction_started = time.monotonic()
    values = predictions(restored, directory / "data", args=args, data_factory=data_factory, root=directory)
    result["prediction_seconds"]["checkpoint_reload"] = time.monotonic() - prediction_started
    result["memory"]["checkpoint_reload_prediction"] = memory(cuda=cuda)
    result["memory"]["overall"] = {
        key: max(phase[key] for phase in result["memory"].values() if phase[key] is not None)
        if any(phase[key] is not None for phase in result["memory"].values())
        else None
        for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "process_peak_rss_bytes")
    }
    result["memory"]["host_scope"] = (
        "process-lifetime peak RSS including preparation; CUDA phases cover model construction, training and inference and include resident allocations"
    )
    metrics = score(spec.task_type, labels, values)
    if not np.isclose(metrics[selection.metric], selection.best["score"], rtol=1e-5, atol=1e-6):
        raise RuntimeError("Reloaded checkpoint does not reproduce its selected validation metric")
    prediction_path = directory / "validation-predictions.parquet"
    pq.write_table(pa.table({"__row_position": np.arange(len(values)), "prediction": values}), prediction_path)
    result.update(
        validation=metrics,
        predicted_validation_rows=len(values),
        checkpoint_prediction_parity=True,
        checkpoint_prediction_parity_scope="full_validation_metric",
        artifacts={"predictions": str(prediction_path.resolve())},
        artifacts_sha256=reports.hashes({"predictions": prediction_path}),
    )
    if run is not None:
        for phase, measurements in result["memory"].items():
            if isinstance(measurements, dict):
                run.record(
                    f"memory/{phase}",
                    {"step": steps, **{key: value for key, value in measurements.items() if value is not None}},
                )
        result["mlflow_run_id"] = run.run_id
        run.complete(result, baseline)
    return result


def execute(args) -> int:
    if args.work_dir.exists() or args.output.exists():
        raise FileExistsError("Supervised work directory and report path must be fresh; choose new paths")
    args.work_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "complete": False,
        "status": "running",
        "benchmark": args.benchmark,
        "run_label": args.run_label,
        "started_utc": datetime.now(UTC).isoformat(),
        "protocol": "Official inner train/validation; supervised only; exact validation selection; no held-out test access",
        "seed": args.seed,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "relflow": rf.__version__},
    }
    reports.publish(report, args.output)
    started = time.monotonic()
    try:
        if args.relbench_cache:
            os.environ["RELBENCH_CACHE_DIR"] = str(args.relbench_cache.resolve())
        torch.set_num_threads(4)
        report["relflow_source_sha256"], report["benchmark_source_sha256"] = provenance.fingerprints(
            Path(__file__).resolve().parents[2], ["benchmarks/provenance.py"]
        )
        # The specification is the structure the derivation engine wrote for this task; the
        # runner adds the task's masked target and the training choices, nothing else.
        original = BenchmarkSpec.from_yaml(args.spec)
        if original.benchmark_name != args.benchmark:
            raise ValueError(f"Specification {args.spec} describes {original.benchmark_name!r}, not {args.benchmark!r}")
        template = yaml.safe_load(original.model_path.read_text())
        args.num_workers = loader_workers(getattr(args, "num_workers", 0), template)
        config: dict[str, Any] = {
            "benchmark": args.benchmark,
            "model": supervise(original, template),
            "original_spec_fingerprint": original.fingerprint,
        }
        config["model"]["batch_size"] = args.batch_size
        config["optimizer"] = dict(OPTIMIZER)
        if getattr(args, "learning_rate", None) is not None:
            config["optimizer"]["learning_rate"] = args.learning_rate
        if getattr(args, "fused", None) is not None:
            config["optimizer"]["fused"] = args.fused
        if getattr(args, "weight_decay", None) is not None:
            config["optimizer"]["weight_decay"] = args.weight_decay
        config["averaging"] = None if getattr(args, "ema_decay", None) is None else {"decay": args.ema_decay}
        config["training"] = {
            name: getattr(args, name)
            for name in (
                "accelerator",
                "precision",
                "batch_size",
                "max_steps",
                "max_epochs",
                "val_every",
                "patience",
                "max_train_seconds",
            )
        }
        config["training"]["minimum_early_stopping_steps"] = min(400, args.max_steps)
        config["training"]["minimum_full_passes"] = getattr(args, "min_full_passes", 0)
        config["training"]["num_workers"] = getattr(args, "num_workers", 0)
        duckdb_resources = resolve(
            args.duckdb_temp_directory or args.work_dir / "duckdb-spill",
            threads=args.duckdb_threads,
            memory_limit=args.duckdb_memory_limit,
            max_temp_directory_size=args.duckdb_max_temp_directory_size,
        )
        duckdb_resources["used_for_preparation"] = args.prepared_from is None
        config["duckdb"] = duckdb_resources
        report["duckdb"] = duckdb_resources
        config["data"] = {
            "train_split": "train",
            "validation_split": "val",
            "full_splits": not (args.train_rows or args.validation_rows),
            "train_rows_limit": args.train_rows,
            "validation_rows_limit": args.validation_rows,
        }
        metric = "roc_auc" if original.task_type == "binary_classification" else "mae"
        config["selection"] = {
            "split": "val",
            "metric": metric,
            "mode": "max" if metric == "roc_auc" else "min",
            "scope": "entire declared validation split with canonical inference features",
            "feature_encoding": "native_validate_matches_predict",
            "text_cache_evaluation_policy": TEXT_CACHE_POLICY,
            "checkpoint_monitor": metric,
        }
        config["config_fingerprint"] = payload_fingerprint(config)
        (args.work_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        model = args.work_dir / "model.yaml"
        model.write_text(yaml.safe_dump(config["model"], sort_keys=False))
        query = args.work_dir / "observations.sql"
        query.write_text(original.query)
        spec = replace(original, model_path=model, query_path=query)
        report.update(
            config=config,
            specification_sha256=reports.hashes({"model.yaml": model, "observations.sql": query}),
            stage="prepare",
            performance_claim="validation search diagnostic"
            if config["data"]["full_splits"]
            else "sampled execution verification; no benchmark performance claim",
        )
        reports.publish(report, args.output, replace_existing=True)
        if args.prepared_from is not None:
            report["data"] = reuse(spec, args.work_dir / "data", report_path=args.prepared_from, seed=args.seed)
        else:
            report["data"] = prepare(
                spec,
                args.work_dir / "data",
                seed=args.seed,
                train_rows=args.train_rows,
                validation_rows=args.validation_rows,
                duckdb_config=duckdb_resources["settings"],
                phase=getattr(args, "phase", "inner"),
            )
        report["stage"] = "train"
        reports.publish(report, args.output, replace_existing=True)
        with tracking.start(
            args.tracking_uri,
            args.experiment,
            f"{args.benchmark.replace('/', '-')}-seed{args.seed}" + (f"-{args.run_label}" if args.run_label else ""),
            parameters={
                "benchmark": args.benchmark,
                "seed": args.seed,
                **config["training"],
                **config["optimizer"],
            },
            tags={
                "benchmark": args.benchmark,
                "protocol": "inner-validation-only",
                "run_label": args.run_label or "unlabeled",
                "full_splits": str(config["data"]["full_splits"]),
            },
            provenance={key: report[key] for key in ("relflow_source_sha256", "benchmark_source_sha256")},
        ) as run:
            report["result"] = train(spec, args.work_dir, args=args, run=run)
        # The finder may have chosen the size; the report's config carries what the run trained with.
        settings = report["config"] if isinstance(report.get("config"), dict) else report
        for section in ("training", "model"):
            if isinstance(settings.get(section), dict):
                settings[section]["batch_size"] = args.batch_size
        report.update(complete=True, status="passed", stage="complete")
        code = 0
    except Exception as error:
        report.update(
            status="failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        code = 1
    report.update(seconds=time.monotonic() - started, finished_utc=datetime.now(UTC).isoformat())
    reports.publish(report, args.output, replace_existing=True)
    print(json.dumps({"benchmark": args.benchmark, "status": report["status"], "output": str(args.output)}), flush=True)
    return code


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=TASKS)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relbench-cache", type=Path)
    parser.add_argument(
        "--text-store",
        default="auto",
        help="Persistent frozen text vectors: 'auto' keeps them beside the RelBench cache per dataset, 'none' disables, "
        "or a directory",
    )
    parser.add_argument(
        "--text-resident", type=int, help="MiB of accelerator memory a text store may occupy; default a quarter of it"
    )
    parser.add_argument(
        "--phase",
        choices=("inner", "outer"),
        default="inner",
        help="Store censored at the validation (inner) or test (outer) timestamp; outer serves the checkpoint refit",
    )
    parser.add_argument(
        "--num-workers",
        type=workers_argument,
        default="auto",
        help="Loader processes that tensorize batches; auto uses four for schemas with Text fields",
    )
    parser.add_argument("--spec", type=Path, required=True, help="The task's derived specification (benchmark.yaml)")
    parser.add_argument("--ema-decay", type=float, help="Optional parameter averaging, independently compared")
    parser.add_argument("--learning-rate", type=float, help="Override the recorded optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, help="Reuse a recorded AdamW weight decay")
    parser.add_argument("--min-full-passes", type=int, default=0, help="Minimum full passes before plateau stopping")
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--prepared-from", type=Path, help="Completed source result.json whose verified full splits are reused"
    )
    parser.add_argument("--duckdb-threads", type=int, help="SQL threads; default uses affinity and CPU quotas")
    parser.add_argument(
        "--duckdb-memory-limit", help="SQL buffer budget, e.g. 80GiB; default reserves host memory headroom"
    )
    parser.add_argument("--duckdb-temp-directory", type=Path, help="Owned spill parent on a fast local filesystem")
    parser.add_argument(
        "--duckdb-max-temp-directory-size", help="Per-connection spill cap, e.g. 800GiB; default 80%% free disk"
    )
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=batch_size_argument,
        default=128,
        help="A positive integer, or 'auto' to let Lightning's finder pick the largest batch up to --max-batch-size",
    )
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument(
        "--batch-size-headroom",
        type=float,
        default=0.5,
        help="Fraction of the largest batch that fit which an automatic batch size keeps, leaving room for growth",
    )
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8, help="Stale validations that end training")
    parser.add_argument("--plateau-factor", type=float, default=0.1, help="Learning-rate multiplier on a plateau")
    parser.add_argument("--plateau-patience", type=int, default=2, help="Stale validations before each drop")
    parser.add_argument("--max-lr-drops", type=int, default=2, help="Drops before a plateau ends training")
    parser.add_argument(
        "--min-improvement", type=float, default=0.0, help="Relative metric gain below which a validation is stale"
    )
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Floor for dropped learning rates")
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        help="Optional elapsed training bound; omitted runs to convergence/update limits",
    )
    parser.add_argument(
        "--train-rows", type=int, default=0, help="Explicit execution-verification cap; zero uses all rows"
    )
    parser.add_argument("--validation-rows", type=int, default=0, help="Explicit verification cap; zero uses all rows")
    parser.add_argument("--run-label", help="Experiment arm label included in reports and MLflow names/tags")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="relflow-relarena-supervised")
    args = parser.parse_args(argv)
    if args.ema_decay is not None and (not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1):
        parser.error("EMA decay must be finite and in [0, 1)")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Learning rate must be finite and positive")
    if args.weight_decay is not None and (not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Weight decay must be finite and nonnegative")
    if args.min_full_passes < 0:
        parser.error("Minimum full passes must be nonnegative")
    if args.duckdb_threads is not None and args.duckdb_threads <= 0:
        parser.error("DuckDB threads must be positive")
    budgets = [getattr(args, name) for name in ("max_steps", "val_every", "patience", "max_batch_size")]
    if args.batch_size != "auto":
        budgets.append(args.batch_size)
    if any(value <= 0 for value in budgets):
        parser.error("Training update, validation, batch, and patience budgets must be positive")
    if not 0 < args.batch_size_headroom <= 1:
        parser.error("--batch-size-headroom must lie in (0, 1]")
    if not 0 < args.plateau_factor < 1 or args.plateau_patience <= 0 or args.max_lr_drops < 0:
        parser.error("Plateau factor must lie in (0, 1), plateau patience be positive, and drops be nonnegative")
    if not math.isfinite(args.min_improvement) or args.min_improvement < 0 or not 0 < args.min_lr:
        parser.error("Minimum improvement must be finite and nonnegative, and the learning-rate floor positive")
    if args.max_epochs != -1 and args.max_epochs <= 0:
        parser.error("Epoch budget must be positive, or -1 to use only the update/time budget")
    if args.prepared_from is not None and (args.train_rows or args.validation_rows):
        parser.error("Prepared-data reuse requires full splits; omit sampled row limits")
    if args.max_train_seconds is not None and (
        not math.isfinite(args.max_train_seconds) or args.max_train_seconds <= 0
    ):
        parser.error("Training time budget must be finite and positive")
    if args.train_rows < 0 or args.validation_rows < 0:
        parser.error("Split row limits must be nonnegative; zero retains the full official split")
    return args


def main(argv=None) -> int:
    return execute(arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
