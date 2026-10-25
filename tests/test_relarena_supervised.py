"""Protect full-split selection, temporal isolation and generic supervised execution."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from benchmarks.relarena import supervised
from benchmarks.relarena.spec import BenchmarkSpec
from benchmarks.relarena.store import read_store_manifest


def synthetic_spec(root: Path, *, task: str = "driver-position", kind: str = "regression", d_model: int = 64) -> Path:
    """A small derived-style bundle: the runner and sweep take only such a specification as input."""
    root.mkdir(parents=True, exist_ok=True)
    target = {"type": "number" if kind == "regression" else "boolean", "mask": True}
    model = {
        "d_model": d_model,
        "n_layers": 2,
        "n_heads": 4,
        "fields": {
            "x": {"type": "number"},
            "results": {"type": "branch", "length": 16, "overflow": "tail", "fields": {"y": {"type": "number"}}},
            "target": target,
        },
    }
    (root / "model.yaml").write_text(yaml.safe_dump(model, sort_keys=False))
    (root / "observations.sql").write_text(
        "SELECT a.__row_position, s.x FROM read_parquet($anchors) a LEFT JOIN entities s ON a.__entity_key = s.entity_id "
        "WHERE a.__row_position >= $start AND a.__row_position < $stop"
    )
    manifest = {
        "spec_version": "3",
        "dataset_name": "rel-f1",
        "task_name": task,
        "task_type": kind,
        "entity_table": "drivers",
        "entity_key": "driverId",
        "anchor_entity_column": "driverId",
        "timestamp_column": "date",
        "target_column": "position",
        "model": "model.yaml",
        "query": "observations.sql",
        "sources": {"drivers": {"columns": ["driverId"], "indexes": [["driverId"]]}},
    }
    (root / "benchmark.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return root / "benchmark.yaml"


def options(tmp_path, **overrides):
    return SimpleNamespace(
        **dict(
            benchmark="rel-f1/driver-position",
            spec=synthetic_spec(tmp_path / "spec"),
            work_dir=tmp_path / "run",
            output=tmp_path / "result.json",
            relbench_cache=None,
            prepared_from=None,
            duckdb_threads=1,
            duckdb_memory_limit="512MiB",
            duckdb_temp_directory=None,
            duckdb_max_temp_directory_size="512MiB",
            run_label=None,
            accelerator="cpu",
            precision="32-true",
            seed=0,
            batch_size=8,
            max_steps=2,
            max_epochs=2,
            val_every=1,
            patience=8,
            max_train_seconds=None,
            train_rows=0,
            validation_rows=0,
            tracking_uri=None,
            experiment="owned-test",
        )
        | overrides
    )


@pytest.fixture(params=("regression", "binary_classification"))
def task(tmp_path, monkeypatch, request):
    kind = request.param
    model = tmp_path / "model.yaml"
    model.write_text(
        "d_model: 16\nn_layers: 1\nn_heads: 2\nbatch_size: 8\nfields:\n"
        "  x: {type: number}\n  entity: {type: hash}\n  target: {type: "
        + ("boolean" if kind == "binary_classification" else "number")
        + ", mask: true}\n"
    )
    query = tmp_path / "observations.sql"
    query.write_text(
        "SELECT a.__row_position, s.x, s.entity_id AS entity FROM read_parquet($anchors) a "
        "LEFT JOIN entities s ON a.__entity_key=s.entity_id "
        "WHERE a.__row_position >= $start AND a.__row_position < $stop"
    )
    spec = BenchmarkSpec(
        dataset_name="owned-db",
        task_name="owned-task",
        task_type=kind,
        entity_table="entities",
        entity_key="entity_id",
        anchor_entity_column="entity_id",
        timestamp_column="timestamp",
        target_column="label",
        model_path=model,
        query_path=query,
        sources={"entities": {"columns": ("entity_id", "x"), "indexes": (("entity_id",),)}},
    )
    # Rows after the validation cutoff must not reach the snapshot; the column carries RelBench's time metadata.
    source = pd.DataFrame(
        {
            "entity_id": np.arange(36),
            "x": np.linspace(-1, 1, 36),
            "seen": pd.to_datetime(["2020-01-15"] * 32 + ["2020-03-01"] * 4),
        }
    )
    frames = {}
    for split, rows, date in (("train", 32, "2020-01-01"), ("val", 13, "2020-02-01")):
        labels = np.arange(rows) % 2 if kind == "binary_classification" else 100 + np.arange(rows) / 10
        frames[split] = pd.DataFrame(
            {"entity_id": np.arange(rows), "timestamp": pd.to_datetime([date] * rows), "label": labels}
        )
    cache = tmp_path / "cache"
    source_path = cache / spec.dataset_name / "db/entities.parquet"
    source_path.parent.mkdir(parents=True)
    entities = pa.Table.from_pandas(source, preserve_index=False)
    entities = entities.replace_schema_metadata({**(entities.schema.metadata or {}), b"time_col": b'"seen"'})
    pq.write_table(entities, source_path)
    task_directory = cache / spec.dataset_name / "tasks" / spec.task_name
    task_directory.mkdir(parents=True)
    for split, frame in frames.items():
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), task_directory / f"{split}.parquet")
    (task_directory / "test.parquet").write_bytes(b"forbidden test data")
    dataset = SimpleNamespace(val_timestamp=pd.Timestamp("2020-02-01"), test_timestamp=pd.Timestamp("2020-03-01"))

    def get_table(split):
        if split not in {"train", "val"}:
            raise AssertionError(f"Supervised runner accessed forbidden split {split}")
        return SimpleNamespace(df=frames[split])

    official = SimpleNamespace(
        entity_col=spec.anchor_entity_column,
        entity_table=spec.entity_table,
        time_col=spec.timestamp_column,
        target_col=spec.target_column,
        get_table=Mock(side_effect=get_table),
    )
    modules = {name: ModuleType(name) for name in ("relbench", "relbench.datasets", "relbench.tasks", "relbench.utils")}
    modules["relbench.datasets"].get_dataset = Mock(return_value=dataset)
    modules["relbench.tasks"].get_task = Mock(return_value=official)
    modules["relbench.utils"].get_relbench_cache_dir = lambda: cache
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(spec=spec, official=official, dataset=dataset)


def test_prepare_preserves_entire_official_splits_and_censors_database(task, tmp_path):
    result = supervised.prepare(task.spec, tmp_path / "data", seed=0)
    manifest = read_store_manifest(result["artifacts"]["snapshot.duckdb"])
    assert manifest["cutoff"] == "2020-02-01 00:00:00" and manifest["source_rows"] == {"entities": 32}
    assert [call.args[0] for call in task.official.get_table.call_args_list] == ["train", "val"]
    assert result["full_training_split"] is result["full_validation_split"] is True
    assert result["splits"] == {
        "train": {"source_rows": 32, "rows": 32, "limit": 0},
        "val": {"source_rows": 13, "rows": 13, "limit": 0},
    }
    assert result["test_split_accessed"] is False
    assert result["duckdb_settings"]["threads"] >= 1
    assert set(result["preparation_memory"]) == {"snapshot", "train", "validate"}
    assert all(phase["process_peak_rss_bytes"] > 0 for phase in result["preparation_memory"].values())
    assert set(result["task_parquet_sha256"]) == {"train", "val"}
    anchors = pq.read_table(tmp_path / "data/train-anchors.parquet")
    expected = pd.Series(np.arange(32)).sample(frac=1, random_state=0).to_numpy()
    np.testing.assert_array_equal(anchors["__entity_key"].to_numpy(), expected)
    np.testing.assert_array_equal(np.sort(anchors["__entity_key"].to_numpy()), np.arange(32))
    validation = pq.read_table(tmp_path / "data/validate-anchors.parquet")
    np.testing.assert_array_equal(validation["__entity_key"].to_numpy(), np.arange(13))
    for name, digest in result["artifacts_sha256"].items():
        assert hashlib.sha256(Path(result["artifacts"][name]).read_bytes()).hexdigest() == digest


def test_outer_phase_censors_the_store_at_the_test_timestamp(task, tmp_path):
    result = supervised.prepare(task.spec, tmp_path / "data", seed=0, phase="outer")
    manifest = read_store_manifest(result["artifacts"]["snapshot.duckdb"])
    assert result["phase"] == "outer" and manifest["cutoff"] == result["cutoff"] != "2020-02-01 00:00:00"
    assert [call.args[0] for call in task.official.get_table.call_args_list] == ["train", "val"]
    assert result["test_split_accessed"] is False
    with pytest.raises(ValueError, match="phase"):
        supervised.prepare(task.spec, tmp_path / "other", seed=0, phase="test")


def test_explicit_smoke_caps_are_recorded_as_incomplete_splits(task, tmp_path):
    result = supervised.prepare(task.spec, tmp_path / "data", seed=5, train_rows=12, validation_rows=6)
    assert result["full_training_split"] is result["full_validation_split"] is False
    assert result["splits"]["train"]["rows"] == 12
    assert result["splits"]["val"]["rows"] == 6


def test_real_model_selects_exact_metric_and_reloads_best_checkpoint(task, tmp_path):
    directory = tmp_path / "run"
    supervised.prepare(task.spec, directory / "data", seed=0)
    args = options(tmp_path)
    result = supervised.train(task.spec, directory, args=args)
    assert result["stages"]["supervised"]["steps"] == 2
    assert result["stages"]["supervised"]["examples_seen"] == 16
    assert result["stages"]["supervised"]["equivalent_full_epochs"] == 0.5
    assert result["predicted_validation_rows"] == 13
    assert result["checkpoint_prediction_parity"] is True
    assert result["checkpoint_prediction_parity_scope"] == "full_validation_metric"
    assert result["memory"]["overall"]["process_peak_rss_bytes"] > 0
    assert result["memory"]["overall"]["peak_cuda_allocated_bytes"] is None
    assert all(
        phase in result["memory"]
        for phase in ("model_setup", "training", "final_prediction", "checkpoint_reload_prediction")
    )
    metric = "roc_auc" if task.spec.task_type == "binary_classification" else "mae"
    assert result["selection"]["metric"] == metric
    assert np.isclose(result["selection"]["score"], result["validation"][metric])
    prediction_path = Path(result["artifacts"]["predictions"])
    values = pq.read_table(prediction_path)["prediction"].to_numpy()
    labels = pq.read_table(directory / "data/validate.parquet")["target"].to_numpy()
    assert supervised.score(task.spec.task_type, labels, values) == result["validation"]
    if task.spec.task_type == "regression":
        # A source-unit offset verifies the hook doesn't inverse-normalize the
        # already source-unit Number output during best-checkpoint selection:
        # the readout starts at the training mean, so predictions sit near the
        # offset rather than near zero.
        assert values.mean() > 50
    assert all(Path(result[name]["checkpoint"]).is_file() for name in ("selection", "final_checkpoint"))


@pytest.mark.parametrize(
    "kind,metric,values",
    [
        ("binary_classification", "roc_auc", [0.6, 0.8, 0.7]),
        ("regression", "mae", [4.0, 2.0, 3.0]),
    ],
)
def test_selection_keeps_best_checkpoint_in_both_directions(tmp_path, kind, metric, values):
    spec = SimpleNamespace(task_type=kind, benchmark_name="owned")
    selection = supervised.Selection(spec, 5, tmp_path, options(tmp_path))
    trainer = SimpleNamespace(global_step=0)
    trainer.save_checkpoint = lambda path: Path(path).write_text(str(trainer.global_step))
    for step, value in enumerate(values, 1):
        trainer.global_step = step
        selection.consider(trainer, {metric: value})
    assert selection.best["step"] == 2
    assert selection.best["score"] == values[1]
    assert (tmp_path / "best.ckpt").read_text() == "2"
    with pytest.raises(RuntimeError, match="same-step"):
        selection.consider(trainer, {metric: values[-1] + 0.1})


def test_selection_scores_full_split_once_and_rejects_duplicate_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(supervised, "reset_text_cache", lambda model: None)
    spec = SimpleNamespace(task_type="regression", benchmark_name="owned")
    selection = supervised.Selection(spec, 5, tmp_path, options(tmp_path))
    trainer = SimpleNamespace(
        global_step=400, should_stop=False, save_checkpoint=lambda path: Path(path).write_text("x")
    )
    selection.on_train_start(trainer, None)
    selection.on_validation_epoch_start(trainer, None)
    selection.positions = [np.arange(4), np.array([4])]
    selection.labels = [np.zeros(4), np.array([10.0])]
    selection.values = [np.zeros(4), np.zeros(1)]
    selection.on_validation_epoch_end(trainer, None)
    assert selection.best["score"] == 2.0  # Averaging batch MAEs would give 5.
    selection.on_validation_epoch_start(trainer, None)
    selection.positions = [np.array([0, 1, 2, 3, 3])]
    with pytest.raises(RuntimeError, match="exactly once"):
        selection.on_validation_epoch_end(trainer, None)


def test_early_stopping_waits_for_minimum_budget_and_time_limit_is_cooperative(tmp_path, monkeypatch):
    monkeypatch.setattr(supervised, "reset_text_cache", lambda model: None)
    spec = SimpleNamespace(task_type="regression", benchmark_name="owned")
    args = options(tmp_path, max_steps=10000, patience=2)
    selection = supervised.Selection(spec, 1, tmp_path, args)
    trainer = SimpleNamespace(global_step=0, should_stop=False, save_checkpoint=lambda path: Path(path).write_text("x"))
    monkeypatch.setattr(supervised.time, "monotonic", lambda: 10.0)
    selection.on_train_start(trainer, None)
    for step in (200, 400, 600):
        trainer.global_step = step
        selection.on_validation_epoch_start(trainer, None)
        selection.positions, selection.labels, selection.values = [np.array([0])], [np.array([0.0])], [np.array([1.0])]
        selection.on_validation_epoch_end(trainer, None)
        assert trainer.should_stop is (step == 600)
    assert selection.stop_reason == "validation_plateau"
    selection = supervised.Selection(spec, 1, tmp_path / "timed", options(tmp_path, max_train_seconds=1))
    selection.on_train_start(trainer, None)
    monkeypatch.setattr(supervised.time, "monotonic", lambda: 12.0)
    selection.on_validation_epoch_start(trainer, None)
    selection.positions, selection.labels, selection.values = [np.array([0])], [np.array([0.0])], [np.array([2.0])]
    selection.on_validation_epoch_end(trainer, None)
    assert selection.stop_reason == "training_time_limit"
    assert selection.best is not None


def test_unlimited_training_preserves_plateau_selection_and_periodic_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(supervised, "reset_text_cache", lambda model: None)
    spec = SimpleNamespace(task_type="regression", benchmark_name="owned")
    args = options(tmp_path, max_steps=10000, patience=1, max_train_seconds=None)
    selection = supervised.Selection(spec, 1, tmp_path / "best", args)
    trainer = SimpleNamespace(
        global_step=250,
        should_stop=False,
        callback_metrics={},
        optimizers=[SimpleNamespace(param_groups=[{"lr": 0.001}])],
        save_checkpoint=lambda path: Path(path).write_text("full state"),
    )
    audit = supervised.Audit()
    audit.directory = tmp_path / "last"
    monkeypatch.setattr(supervised.time, "monotonic", lambda: 10**9)
    audit.on_train_batch_end(trainer, SimpleNamespace(device=torch.device("cpu")), None, SimpleNamespace(source=[1]), 0)
    assert not trainer.should_stop and audit.stop_reason is None
    assert (audit.directory / "last.ckpt").read_text() == "full state"
    for step in (400, 600):
        trainer.global_step = step
        selection.on_validation_epoch_start(trainer, None)
        selection.positions, selection.labels, selection.values = [np.array([0])], [np.array([0.0])], [np.array([1.0])]
        selection.on_validation_epoch_end(trainer, None)
        assert trainer.should_stop is (step == 600)
    assert selection.stop_reason == "validation_plateau"
    assert selection.best["step"] == 400
    assert (tmp_path / "best/best.ckpt").is_file()


def test_prediction_capture_uses_number_source_units_and_boolean_sigmoid(tmp_path):
    for kind, expected in (("regression", 2.0), ("binary_classification", torch.tensor(2.0).sigmoid().item())):
        selection = supervised.Selection(SimpleNamespace(task_type=kind), 1, tmp_path, options(tmp_path))
        selection.active = True
        selection.capture(None, None, SimpleNamespace(payload={"content": torch.tensor([2.0])}))
        np.testing.assert_allclose(selection.current, [expected])


def test_cli_rejects_invalid_budgets_and_defaults_to_full_splits(tmp_path):
    base = [
        "--benchmark",
        "rel-f1/driver-position",
        "--spec",
        str(tmp_path / "spec/benchmark.yaml"),
        "--work-dir",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    args = supervised.arguments(base)
    assert args.train_rows == args.validation_rows == 0
    assert args.max_steps == 100000 and args.max_epochs == 40 and args.val_every == 200
    assert args.batch_size == 128 and args.max_train_seconds is None
    for option, value in (
        ("--max-train-seconds", "nan"),
        ("--max-train-seconds", "inf"),
        ("--max-train-seconds", "0"),
        ("--train-rows", "-1"),
        ("--batch-size", "0"),
    ):
        with pytest.raises(SystemExit):
            supervised.arguments([*base, option, value])


def test_failed_preparation_is_durable_and_cannot_overwrite_history(tmp_path, monkeypatch):
    args = options(tmp_path)
    monkeypatch.setattr(supervised.provenance, "fingerprints", Mock(return_value=("library", "evaluator")))
    monkeypatch.setattr(supervised, "prepare", Mock(side_effect=RuntimeError("owned failure")))
    assert supervised.execute(args) == 1
    report = json.loads(args.output.read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert report["stage"] == "prepare"
    assert report["error"]["message"] == "owned failure"
    assert report["config"]["data"]["full_splits"] is True
    original = args.output.read_bytes()
    with pytest.raises(FileExistsError, match="fresh"):
        supervised.execute(args)
    assert args.output.read_bytes() == original


def test_bfloat16_capture_matches_boolean_writer_probability_rounding(tmp_path):
    from relflow.tensorfields.extensions import boolean

    logits = torch.tensor([-0.01, -0.001, 0.001, 0.01], dtype=torch.bfloat16)
    output = SimpleNamespace(payload={"content": logits})
    selection = supervised.Selection(SimpleNamespace(task_type="binary_classification"), 4, tmp_path, options(tmp_path))
    selection.active = True
    selection.capture(None, None, output)
    written = boolean.write(None, output, boolean.output(None, supervised.rf.Address("target")))
    expected = written.field("content").field("probability").to_numpy().astype(np.float64)
    np.testing.assert_array_equal(selection.current, expected)
    # Promoting before sigmoid changes ties and therefore exact AUROC. Match
    # the pinned public writer, rather than silently changing metric semantics.
    assert not np.array_equal(expected, logits.float().sigmoid().numpy())


def test_validation_uses_native_encoded_features_without_reencoding(tmp_path):
    target = supervised.rf.Address("target")
    model = supervised.rf.Model.xs(
        d_model=8, batch_size=2, entity=supervised.rf.Hash, target=supervised.rf.Boolean(mask=True)
    )
    source = pa.table({"entity": ["one", "two"], "target": [True, False]})
    data = supervised.rf.ArrowDataModule(model, validate=source)
    batch = next(iter(data.val_dataloader()))
    predicted = model.encode(source.drop(["target"]), strata="predict")
    torch.testing.assert_close(batch.tensors["/entity"].content, predicted["/entity"].content)
    assert batch.tensors[target].trainable.all()
    assert "on_validation_batch_start" not in supervised.Selection.__dict__


def prepared_report(task, directory):
    directory.mkdir()
    model = directory / "model.yaml"
    query = directory / "observations.sql"
    model.write_text(task.spec.model_path.read_text())
    query.write_text(task.spec.query)
    original = replace(task.spec, model_path=model, query_path=query)
    prepared = supervised.prepare(original, directory / "data", seed=0)
    report = {
        "complete": True,
        "status": "passed",
        "benchmark": original.benchmark_name,
        "seed": 0,
        "data": prepared,
        "specification_sha256": supervised.reports.hashes({"model.yaml": model, "observations.sql": query}),
    }
    path = directory / "result.json"
    path.write_text(json.dumps(report))
    return original, path, report


def test_unbounded_changes_only_child_branch_capacity_and_preserves_original():
    model = {
        "batch_size": 16,
        "fields": {
            "x": {"type": "number"},
            "history": {
                "type": "branch",
                "length": 4,
                "overflow": "tail",
                "ffnx": 8,
                "fields": {"nested": {"type": "branch", "length": 2, "fields": {"x": {"type": "number"}}}},
            },
        },
    }
    result = supervised.unbounded(model)
    branch = result["fields"]["history"]
    assert branch["length"] is None and "overflow" not in branch
    assert branch["fields"]["nested"]["length"] is None
    assert branch["ffnx"] == 8 and result["batch_size"] == 16
    assert model["fields"]["history"]["length"] == 4
    assert model["fields"]["history"]["overflow"] == "tail"


def test_reuse_verifies_and_links_full_splits_without_materialization(task, tmp_path, monkeypatch):
    spec, path, report = prepared_report(task, tmp_path / "original")
    monkeypatch.setattr(supervised, "materialize", Mock(side_effect=AssertionError("must reuse")))
    result = supervised.reuse(spec, tmp_path / "reused", report_path=path, seed=7301)
    assert result["preparation"] == "verified_immutable_reuse"
    assert result["lineage"]["preparation_seed"] == 0
    assert result["lineage"]["training_seed"] == 7301
    assert result["lineage"]["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, destination in result["artifacts"].items():
        assert Path(destination).stat().st_ino == Path(report["data"]["artifacts"][name]).stat().st_ino
        assert result["artifacts_sha256"][name] == report["data"]["artifacts_sha256"][name]
    assert set(result["task_parquet_sha256"]) == {"train", "val"}


def test_reuse_accepts_regularizers_and_branch_caps_on_every_field(task, tmp_path, monkeypatch):
    """The sweep writes mask rates, jitter, loss objectives and branch caps per field; none change the rows."""
    spec, path, _ = prepared_report(task, tmp_path / "original")
    new_directory = tmp_path / "tuned"
    new_directory.mkdir()
    model, query = new_directory / "model.yaml", new_directory / "observations.sql"
    config = yaml.safe_load(spec.model_path.read_text())

    def tune(fields: dict) -> None:
        for field in fields.values():
            if field.get("type") == "branch":
                field["length"] = 3
                field["overflow"] = "head"
                tune(field["fields"])
            elif field.get("mask") is True:
                if field["type"] == "number":
                    field["objective"] = "huber"
            else:
                field["mask"] = 0.1
                if field["type"] in ("number", "dateparts", "vector", "text"):
                    field["jitter"] = {"add": 0.05}

    tune(config["fields"])
    assert config != yaml.safe_load(spec.model_path.read_text())
    model.write_text(yaml.safe_dump(config))
    query.write_text(spec.query)
    monkeypatch.setattr(supervised, "materialize", Mock(side_effect=AssertionError("must reuse")))
    result = supervised.reuse(
        replace(spec, model_path=model, query_path=query), tmp_path / "reused", report_path=path, seed=1
    )
    assert result["preparation"] == "verified_immutable_reuse"


@pytest.mark.parametrize("change", ["incomplete", "sampled", "test", "query", "model", "data", "source", "order"])
def test_reuse_rejects_changed_data_or_contract(task, tmp_path, change):
    spec, path, report = prepared_report(task, tmp_path / "original")
    new_directory = tmp_path / "new-spec"
    new_directory.mkdir()
    model, query = new_directory / "model.yaml", new_directory / "observations.sql"
    model.write_text(spec.model_path.read_text())
    query.write_text(spec.query)
    current = replace(spec, model_path=model, query_path=query)
    if change == "incomplete":
        report["complete"] = False
    elif change == "sampled":
        report["data"]["full_training_split"] = False
    elif change == "test":
        report["data"]["test_split_accessed"] = True
    elif change == "query":
        query.write_text(spec.query + " -- changed projection")
    elif change == "model":
        config = yaml.safe_load(model.read_text())
        config["fields"]["x"]["type"] = "category"
        model.write_text(yaml.safe_dump(config))
    elif change == "data":
        report["data"]["artifacts_sha256"]["train.parquet"] = "incorrect expected digest"
    elif change == "source":
        report["data"]["source_parquet_sha256"]["entities"] = "incorrect expected digest"
    elif change == "order":
        source = Path(report["data"]["artifacts"]["train.parquet"])
        table = pq.read_table(source)
        # Create a separate synthetic artifact; the original fixture is never changed.
        shuffled = source.with_name("reordered.parquet")
        pq.write_table(table.take(pa.array(np.arange(len(table))[::-1])), shuffled)
        report["data"]["artifacts"]["train.parquet"] = str(shuffled)
        report["data"]["artifacts_sha256"]["train.parquet"] = supervised.reports.hashes({"train": shuffled})["train"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        supervised.reuse(current, tmp_path / "reused", report_path=path, seed=0)
    assert not (tmp_path / "reused").exists()


def test_parquet_stream_is_restartable_bounded_and_drops_prediction_targets(tmp_path):
    path = tmp_path / "observations.parquet"
    pq.write_table(pa.table({"__row_position": np.arange(2500), "x": np.arange(2500), "target": np.arange(2500)}), path)
    for _ in range(2):
        batches = list(supervised.batches(path, predict=True))
        assert [len(batch) for batch in batches] == [1024, 1024, 452]
        assert all(batch.schema.names == ["__row_position", "x"] for batch in batches)
        assert pa.Table.from_batches(batches)["__row_position"].to_pylist() == list(range(2500))


def test_cli_exposes_paired_modes_and_update_only_budget(tmp_path):
    args = supervised.arguments(
        [
            "--benchmark",
            "rel-f1/driver-position",
            "--spec",
            str(tmp_path / "spec/benchmark.yaml"),
            "--work-dir",
            str(tmp_path / "run"),
            "--output",
            str(tmp_path / "result.json"),
            "--run-label",
            "current-variable",
            "--max-epochs",
            "-1",
            "--prepared-from",
            str(tmp_path / "source.json"),
        ]
    )
    assert args.max_epochs == -1
    assert args.run_label == "current-variable"
    assert args.prepared_from == tmp_path / "source.json"


@pytest.mark.parametrize("variable", [False, True])
def test_repeated_context_training_reloads_in_each_length_mode(task, tmp_path, variable):
    directory = tmp_path / "repeated"
    data = directory / "data"
    data.mkdir(parents=True)
    config = yaml.safe_load(task.spec.model_path.read_text())
    config["fields"]["history"] = {
        "type": "branch",
        "length": 3,
        "overflow": "tail",
        "n_layers": 1,
        "fields": {"value": {"type": "number"}},
    }
    if variable:
        config = supervised.unbounded(config)
    model = directory / "model.yaml"
    model.write_text(yaml.safe_dump(config))
    spec = replace(task.spec, model_path=model)
    for split, count in (("train", 16), ("validate", 13)):
        labels = (
            (np.arange(count) % 2).astype(bool)
            if spec.task_type == "binary_classification"
            else 100.0 + np.arange(count) / 10
        )
        table = pa.table(
            {
                "__row_position": np.arange(count),
                "x": np.arange(count, dtype=float),
                "entity": np.arange(count),
                "target": labels,
                "history": pa.array(
                    [[{"value": float(index)} for index in range(row % 4)] for row in range(count)],
                    type=pa.list_(pa.struct([("value", pa.float64())])),
                ),
            }
        )
        pq.write_table(table, data / f"{split}.parquet")
    result = supervised.train(spec, directory, args=options(tmp_path))
    assert result["stages"]["supervised"]["steps"] == 2
    assert result["predicted_validation_rows"] == 13
    assert result["checkpoint_prediction_parity"] is True


def test_parquet_stream_closes_reader_when_consumer_stops_early(tmp_path, monkeypatch):
    reader = Mock()
    reader.schema_arrow = pa.schema([("x", pa.int64()), ("target", pa.bool_())])
    reader.iter_batches.return_value = iter(
        [
            pa.record_batch({"x": [1]}),
            pa.record_batch({"x": [2]}),
        ]
    )
    opened = Mock()
    opened.__enter__ = Mock(return_value=reader)
    opened.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=opened)
    monkeypatch.setattr(supervised.pq, "ParquetFile", factory)
    stream = supervised.batches(tmp_path / "owned.parquet", predict=True)
    assert next(stream).column(0).to_pylist() == [1]
    opened.__exit__.assert_not_called()
    stream.close()
    opened.__exit__.assert_called_once()
    assert opened.__exit__.call_args.args[0] is GeneratorExit
    reader.iter_batches.assert_called_once_with(batch_size=1024, columns=["x"], use_threads=False)


def test_run_label_reaches_report_and_tracking(tmp_path, monkeypatch):
    args = options(tmp_path, run_label="current-variable")
    monkeypatch.setattr(supervised.provenance, "fingerprints", Mock(return_value=("library", "evaluator")))
    monkeypatch.setattr(supervised, "prepare", Mock(return_value={"owned": "data"}))
    monkeypatch.setattr(supervised, "train", Mock(return_value={"owned": "result"}))
    tracking = Mock(return_value=nullcontext(None))
    monkeypatch.setattr(supervised.tracking, "start", tracking)
    assert supervised.execute(args) == 0
    report = json.loads(args.output.read_text())
    assert report["run_label"] == "current-variable"
    assert tracking.call_args.args[2].endswith("-seed0-current-variable")
    assert tracking.call_args.kwargs["tags"]["run_label"] == "current-variable"
    assert report["config"]["model"]["fields"]["results"]["length"] == 16, "the specification's lengths stand"


def test_constant_baseline_uses_the_task_optimal_training_statistic(task, tmp_path):
    kind = task.spec.task_type
    directory = tmp_path / "run"
    supervised.prepare(task.spec, directory / "data", seed=0)
    result = supervised.train(task.spec, directory, args=options(tmp_path))
    expected = "training_label_median" if kind == "regression" else "training_label_mean"
    assert result["baseline_metadata"]["statistic"] == expected


@pytest.mark.parametrize("flag,value", [("--ema-decay", "1"), ("--ema-decay", "nan"), ("--learning-rate", "0")])
def test_experimental_policy_arguments_validate_before_work(flag, value, tmp_path):
    with pytest.raises(SystemExit):
        supervised.arguments(
            [
                "--benchmark",
                "rel-f1/driver-position",
                "--work-dir",
                str(tmp_path / "run"),
                "--output",
                str(tmp_path / "result.json"),
                flag,
                value,
            ]
        )


def test_optional_ema_selection_reloads_the_evaluated_checkpoint(task, tmp_path):
    directory = tmp_path / "run"
    supervised.prepare(task.spec, directory / "data", seed=0)
    result = supervised.train(task.spec, directory, args=options(tmp_path, ema_decay=0.5))
    assert result["checkpoint_prediction_parity"]
    assert result["checkpoint_prediction_parity_scope"] == "full_validation_metric"
    checkpoint = torch.load(result["selection"]["checkpoint"], weights_only=False)
    assert "EMA" in checkpoint["callbacks"]


@pytest.mark.parametrize("changed_identity", [False, True])
def test_specification_identity_and_architecture_reach_the_report(tmp_path, monkeypatch, changed_identity):
    spec = synthetic_spec(tmp_path / "custom", task="driver-dnf" if changed_identity else "driver-position", d_model=24)
    prepare = Mock(side_effect=RuntimeError("stop before data access"))
    monkeypatch.setattr(supervised, "prepare", prepare)
    args = options(tmp_path, spec=spec)
    assert supervised.execute(args) == 1
    report = json.loads(args.output.read_text())
    if changed_identity:
        prepare.assert_not_called()
        assert "describes 'rel-f1/driver-dnf'" in report["error"]["message"]
    else:
        prepare.assert_called_once()
        assert report["config"]["model"]["d_model"] == 24
        assert report["config"]["model"]["fields"]["target"] == {"type": "number", "mask": True}
        assert report["config"]["original_spec_fingerprint"]
        assert report["error"]["message"] == "stop before data access"


def test_bounded_binary_execution_fixture_retains_both_classes(task, tmp_path):
    if task.spec.task_type != "binary_classification":
        return
    prepared = supervised.prepare(task.spec, tmp_path / "data", seed=5, train_rows=2, validation_rows=2)
    for name in ("train", "validate"):
        assert set(pq.read_table(tmp_path / f"data/{name}.parquet", columns=["target"])["target"].to_pylist()) == {0, 1}
    assert prepared["full_training_split"] is prepared["full_validation_split"] is False
    assert "no performance claim" in prepared["sample_policy"]


@pytest.mark.parametrize("bad", ["-0.1", "nan", "inf"])
def test_training_controls_reject_invalid_weight_decay(tmp_path, bad):
    with pytest.raises(SystemExit):
        supervised.arguments(
            [
                "--benchmark",
                "rel-f1/driver-position",
                "--work-dir",
                str(tmp_path / "work"),
                "--output",
                str(tmp_path / "report.json"),
                "--weight-decay",
                bad,
            ]
        )


def test_full_pass_minimum_prevents_premature_plateau_stop(tmp_path):
    spec = SimpleNamespace(task_type="regression", benchmark_name="owned/task")
    args = SimpleNamespace(max_steps=2000, minimum_early_stopping_steps=1000, patience=1, max_train_seconds=7200)
    selection = supervised.Selection(spec, 2, tmp_path / "checkpoints", args)
    selection.best = {"score": 0.5, "step": 0}
    selection.stale = 1
    selection.started = supervised.time.monotonic()
    selection.positions = [np.array([0, 1])]
    selection.values = [np.array([1.0, 1.0])]
    selection.labels = [np.array([0.0, 0.0])]
    trainer = SimpleNamespace(global_step=500, should_stop=False)
    selection.on_validation_epoch_end(trainer, None)
    assert trainer.should_stop is False
    assert selection.stale == 1
    trainer.global_step = 1000
    selection.on_validation_epoch_end(trainer, None)
    assert trainer.should_stop is True
    assert selection.stop_reason == "validation_plateau"


def test_historical_optimizer_and_full_pass_controls_reach_training(task, tmp_path, monkeypatch):
    original = supervised.rf.adamw
    optimizers = []

    def optimizer(**settings):
        optimizers.append(settings)
        return original(**settings)

    monkeypatch.setattr(supervised.rf, "adamw", optimizer)
    args = options(tmp_path, weight_decay=0.007, min_full_passes=2)
    supervised.prepare(task.spec, args.work_dir / "data", seed=0)
    result = supervised.train(task.spec, args.work_dir, args=args)
    assert optimizers[0]["weight_decay"] == 0.007
    assert result["stages"]["supervised"]["minimum_early_stopping_steps"] == 8


def test_plateau_lowers_learning_rate_then_stops_after_exhausting_drops(tmp_path, monkeypatch):
    monkeypatch.setattr(supervised, "reset_text_cache", lambda model: None)
    spec = SimpleNamespace(task_type="regression", benchmark_name="owned")
    args = options(tmp_path, max_steps=10000, patience=10, plateau_patience=2, max_lr_drops=2, plateau_factor=0.1)
    selection = supervised.Selection(spec, 1, tmp_path / "best", args)
    trainer = SimpleNamespace(
        global_step=0,
        should_stop=False,
        optimizers=[SimpleNamespace(param_groups=[{"lr": 1e-3}, {"lr": 1e-3}])],
        save_checkpoint=lambda path: Path(path).write_text("x"),
    )
    monkeypatch.setattr(supervised.time, "monotonic", lambda: 10.0)
    selection.on_train_start(trainer, None)
    rates, stops = [], []
    for step, value in zip((400, 600, 800, 1000, 1200, 1400, 1600), (1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0), strict=True):
        trainer.global_step = step
        selection.on_validation_epoch_start(trainer, None)
        selection.positions, selection.labels, selection.values = (
            [np.array([0])],
            [np.array([0.0])],
            [np.array([value])],
        )
        selection.on_validation_epoch_end(trainer, None)
        rates.append(trainer.optimizers[0].param_groups[0]["lr"])
        stops.append(trainer.should_stop)
    assert rates == pytest.approx([1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])
    assert trainer.optimizers[0].param_groups[1]["lr"] == pytest.approx(1e-5)
    assert stops == [False] * 6 + [True]
    assert selection.stop_reason == "plateau_after_lr_drops"
    assert [drop["step"] for drop in selection.drops] == [800, 1200]
    assert selection.plateau()["drops"] == selection.drops
    assert [record["learning_rate"] for record in selection.history][:3] == pytest.approx([1e-3, 1e-3, 1e-3])
    assert selection.best["step"] == 400


def test_small_gains_keep_the_best_checkpoint_but_count_as_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(supervised, "reset_text_cache", lambda model: None)
    spec = SimpleNamespace(task_type="regression", benchmark_name="owned")
    args = options(tmp_path, max_steps=10000, patience=2, min_improvement=0.01, max_lr_drops=0)
    selection = supervised.Selection(spec, 1, tmp_path / "best", args)
    trainer = SimpleNamespace(global_step=0, should_stop=False, save_checkpoint=lambda path: Path(path).write_text("x"))
    monkeypatch.setattr(supervised.time, "monotonic", lambda: 10.0)
    selection.on_train_start(trainer, None)
    for step, value in ((400, 1.0), (600, 0.995), (800, 0.994)):
        trainer.global_step = step
        selection.on_validation_epoch_start(trainer, None)
        selection.positions, selection.labels, selection.values = (
            [np.array([0])],
            [np.array([0.0])],
            [np.array([value])],
        )
        selection.on_validation_epoch_end(trainer, None)
    assert selection.best["step"] == 800 and selection.best["score"] == pytest.approx(0.994)
    assert selection.stale == 2 and trainer.should_stop is True
    assert selection.stop_reason == "validation_plateau"


def test_cli_accepts_plateau_controls_and_rejects_invalid_values(tmp_path):
    base = [
        "--benchmark",
        "rel-f1/driver-position",
        "--spec",
        "s.yaml",
        "--work-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / "r.json"),
    ]
    args = supervised.arguments(
        [
            *base,
            "--plateau-factor",
            "0.3",
            "--plateau-patience",
            "1",
            "--max-lr-drops",
            "3",
            "--min-improvement",
            "0.002",
        ]
    )
    assert (args.plateau_factor, args.plateau_patience, args.max_lr_drops, args.min_improvement) == (0.3, 1, 3, 0.002)
    for bad in (["--plateau-factor", "1.5"], ["--plateau-patience", "0"], ["--max-lr-drops", "-1"], ["--min-lr", "0"]):
        with pytest.raises(SystemExit):
            supervised.arguments([*base, *bad])


def test_batch_size_argument_accepts_auto_with_a_cap_and_headroom():
    base = ["--benchmark", "rel-f1/driver-position", "--spec", "s.yaml", "--work-dir", "work", "--output", "out.json"]
    args = supervised.arguments([*base, "--batch-size", "auto"])
    assert args.batch_size == "auto" and args.max_batch_size == 512 and args.batch_size_headroom == 0.5
    assert supervised.arguments([*base, "--batch-size", "64"]).batch_size == 64
    with pytest.raises(SystemExit):
        supervised.arguments([*base, "--batch-size", "auto", "--batch-size-headroom", "1.5"])
    with pytest.raises(SystemExit):
        supervised.arguments([*base, "--max-batch-size", "0"])


def test_find_batch_size_keeps_a_fraction_of_the_largest_fitting_batch(tmp_path):
    from types import SimpleNamespace

    import pyarrow as pa

    model = supervised.rf.Model.xs(d_model=8, amount=supervised.rf.Number, target=supervised.rf.Boolean(mask=True))
    model.batch_size = 4
    table = pa.table({"amount": [float(i) for i in range(64)], "target": [i % 2 == 0 for i in range(64)]})
    data = supervised.rf.ArrowDataModule(
        model=model, train=table, validate=table, num_workers=0, persistent_workers=False, pin_memory=False
    )
    args = SimpleNamespace(accelerator="cpu", precision="32-true", max_batch_size=16, batch_size_headroom=0.5)
    found = supervised.find_batch_size(model, data, args, {"learning_rate": 1e-3}, root=tmp_path)
    # Nothing overflows on CPU, so the search climbs to the cap and the model keeps half of it.
    assert found == 16 and model.batch_size == 8 and model.optimizer is None


def test_loader_workers_follow_text_fields():
    from benchmarks.relarena.supervised import loader_workers

    plain = {
        "fields": {"amount": {"type": "number"}, "events": {"type": "branch", "fields": {"code": {"type": "category"}}}}
    }
    textual = {
        "fields": {"amount": {"type": "number"}, "events": {"type": "branch", "fields": {"note": {"type": "text"}}}}
    }

    assert loader_workers("auto", plain) == 0
    assert 1 <= loader_workers("auto", textual) <= 4
    assert loader_workers(3, textual) == 3 and loader_workers("0", textual) == 0


def test_prepared_data_reuse_ignores_field_regularizers():
    base = {"fields": {"amount": {"type": "number"}, "events": {"type": "branch", "length": 8, "overflow": "tail", "fields": {"note": {"type": "text"}}}, "target": {"type": "number", "mask": True}}}  # fmt: skip
    tuned = {"fields": {"amount": {"type": "number", "jitter": {"add": 0.05}, "mask": 0.1}, "events": {"type": "branch", "length": 32, "fields": {"note": {"type": "text", "mask": 0.1}}}, "target": {"type": "number", "mask": True, "objective": "huber"}}}  # fmt: skip
    assert supervised.unbounded(base) == supervised.unbounded(tuned)
    assert supervised.unbounded(tuned)["fields"]["target"] == {"type": "number", "mask": True}, (
        "the target's contract mask stays"
    )
