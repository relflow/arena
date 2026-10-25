"""The sweep plans reproducibly by size tier, bundles trials, runs them durably, and ranks by validation only."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.relarena import sweep
from benchmarks.relarena.spec import BenchmarkSpec


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


REAL_STRUCTURE = sweep.structure


@pytest.fixture(autouse=True)
def synthetic_structure(monkeypatch, tmp_path):
    """Sweeps under test train a synthetic specification instead of deriving one from a RelBench cache."""
    manifest = synthetic_spec(tmp_path / "structure")
    monkeypatch.setattr(sweep, "structure", lambda args, directory: (BenchmarkSpec.from_yaml(manifest), manifest))


def test_tiers_follow_training_rows_and_carry_pass_budgets():
    assert sweep.tier(1)["name"] == "small" and sweep.tier(19_999)["name"] == "small"
    assert sweep.tier(20_000)["name"] == "medium" and sweep.tier(499_999)["name"] == "medium"
    assert sweep.tier(500_000)["name"] == "large" and sweep.tier(10_000_000)["name"] == "large"
    assert sweep.tier(1)["passes"] > sweep.tier(100_000)["passes"] > sweep.tier(10_000_000)["passes"]
    assert all(entry["backstop_seconds"] > 0 for entry in sweep.TIERS)


def test_plan_puts_the_default_first_and_samples_distinct_configs_reproducibly():
    first = sweep.plan("rel-trial/site-success", 151_407, 12, seed=7)
    second = sweep.plan("rel-trial/site-success", 151_407, 12, seed=7)
    assert first == second
    assert first["tier"] == "medium" and len(first["trials"]) == 13
    assert first["budget"] == {"passes": 8, "backstop_seconds": 4 * 3600.0}
    assert first["trials"][0]["config"] == sweep.TIER_DEFAULTS["medium"], "trial 0 is the tier's launch default"
    configs = [json.dumps(t["config"], sort_keys=True) for t in first["trials"]]
    assert len(set(configs)) == len(configs)
    space = sweep.SPACES["medium"]
    for trial in first["trials"][1:]:
        config = trial["config"]
        for name, choices in space.items():
            if name == "learning_rate":
                assert choices[0] <= config[name] <= choices[1]
            else:
                assert config[name] in choices
        assert config["d_model"] % config["n_heads"] == 0
    assert sweep.plan("rel-trial/site-success", 151_407, 12, seed=8) != first


def test_plan_narrows_the_space_and_starts_from_a_chosen_configuration():
    space = {"ema_decay": [None], "learning_rate": [6e-4, 3e-3], "dropout": [0.1, 0.2], "batch_size": [128, 256]}
    default = {"d_model": 64, "ffnx": 4, "learning_rate": 1.46e-3, "weight_decay": 0.0}
    data = sweep.plan("rel-f1/driver-position", 7_453, 12, seed=1, space=space, default=default)
    assert data["trials"][0]["config"] == {**sweep.TIER_DEFAULTS["small"], **default}
    assert data["space"]["ema_decay"] == (None,) and data["space"]["n_layers"] == sweep.SPACES["small"]["n_layers"]
    for trial in data["trials"]:
        if trial["config_id"] == data["trials"][0]["config_id"]:
            continue  # the default's own seeds
        config = trial["config"]
        assert config["ema_decay"] is None and config["dropout"] in (0.1, 0.2) and config["batch_size"] in (128, 256)
        assert 6e-4 <= config["learning_rate"] <= 3e-3
    with pytest.raises(KeyError, match="unknown sweep parameter"):
        sweep.plan("rel-f1/driver-position", 7_453, 2, seed=1, space={"width": [1]})
    with pytest.raises(ValueError, match="learning_rate needs two positive bounds"):
        sweep.plan("rel-f1/driver-position", 7_453, 2, seed=1, space={"learning_rate": [1e-3]})
    with pytest.raises(ValueError, match="non-empty list"):
        sweep.plan("rel-f1/driver-position", 7_453, 2, seed=1, space={"dropout": []})
    with pytest.raises(KeyError, match="unknown default parameter"):
        sweep.plan("rel-f1/driver-position", 7_453, 2, seed=1, default={"heads": 4})
    with pytest.raises(ValueError, match="divisible"):
        sweep.plan("rel-f1/driver-position", 7_453, 2, seed=1, default={"d_model": 30, "n_heads": 4})
    args = sweep.arguments(
        ["plan", "--task", "rel-f1/driver-position", "--cache", "/tmp/cache", "--output", "/tmp/plan.json",
         "--space", json.dumps(space), "--default", json.dumps(default)]
    )  # fmt: skip
    assert args.space == space and args.default == default


def test_bundle_rewrites_architecture_and_keeps_the_field_contract(tmp_path):
    manifest = synthetic_spec(tmp_path / "spec")
    spec = BenchmarkSpec.from_yaml(manifest)
    config = {**sweep.DEFAULT, "d_model": 96, "n_layers": 3, "n_heads": 8, "ffnx": 4, "dropout": 0.0}
    path = sweep.bundle(spec, config, tmp_path / "bundle", manifest)
    model = yaml.safe_load((tmp_path / "bundle/model.yaml").read_text())
    original = yaml.safe_load(spec.model_path.read_text())
    assert {name: model[name] for name in sweep.ARCHITECTURE} == {
        "d_model": 96,
        "n_layers": 3,
        "n_heads": 8,
        "ffnx": 4,
        "dropout": 0.0,
    }
    assert model["fields"] == original["fields"]
    assert (tmp_path / "bundle/observations.sql").read_text() == spec.query
    rebuilt = BenchmarkSpec.from_yaml(path)
    assert rebuilt.benchmark_name == spec.benchmark_name and rebuilt.query == spec.query


def options(**overrides):
    return SimpleNamespace(
        **{
            "python": "python",
            "task": "rel-f1/driver-position",
            "training_rows": 7453,
            "cache": Path("/cache"),
            "seed": 0,
            "min_full_passes": 2,
            "max_steps": 200000,
            "val_every": 1000,
            "patience": 5,
            "plateau_factor": 0.1,
            "plateau_patience": 2,
            "max_lr_drops": 2,
            "min_improvement": 0.001,
            "trial_seconds": 5400,
            "name": "unit",
            "duckdb_memory_limit": "16GB",
            "refit_regime": "full",
            "tracking_uri": None,
            "experiment": "x",
        }
        | overrides
    )


def test_command_carries_the_schedule_full_splits_and_prepared_data(tmp_path):
    config = {**sweep.DEFAULT, "batch_size": 64, "ema_decay": 0.99}
    argv = sweep.command(
        options(), config, tmp_path / "bundle/benchmark.yaml", tmp_path / "trial", tmp_path / "first/result.json"
    )
    text = " ".join(argv)
    assert "--plateau-factor 0.1 --plateau-patience 2 --max-lr-drops 2 --min-improvement 0.001" in text
    assert "--train-rows 0 --validation-rows 0" in text and "--max-train-seconds 5400" in text
    assert "--val-every 117" in text, "validation cadence never exceeds one pass"
    assert "--ema-decay 0.99" in text and "--prepared-from" in text and "--fused" in text
    without = sweep.command(options(), {**config, "ema_decay": None}, tmp_path / "b.yaml", tmp_path / "t", None)
    assert "--ema-decay" not in without and "--prepared-from" not in without


def test_run_records_results_resumes_and_honours_stop(tmp_path, monkeypatch):
    data = sweep.plan("rel-f1/driver-position", 7453, 2, seed=1, seeds=1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    calls = []

    def fake_run(argv, cwd, env, stdout, stderr):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        index = len(calls)
        output.write_text(
            json.dumps(
                {
                    "status": "passed" if index != 2 else "failed",
                    "selection": {"metric": "mae", "score": 4.0 - index, "step": 100 * index},
                    "stages": {"supervised": {"steps": 100 * index, "equivalent_full_epochs": 2.5, "seconds": 60.0}},
                    "stopping_reason": "validation_plateau",
                    "plateau": {"drops": [{"step": 50}]},
                    "baseline": {"mae": 9.0},
                    "validation": {"mae": 4.0 - index},
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = sweep.arguments(
        [
            "run",
            "--plan",
            str(plan_path),
            "--name",
            "unit",
            "--output",
            str(tmp_path / "sweep"),
            "--source",
            str(tmp_path),
            "--cache",
            "/cache",
            "--no-refit",
        ]
    )
    assert sweep.run(args) == 0
    state = json.loads((tmp_path / "sweep/state.json").read_text())
    assert [t["status"] for t in state["trials"]] == ["done", "failed", "done"]
    assert state["trials"][2]["prepared_from"] == state["trials"][0]["result"], (
        "later trials reuse the first prepared split"
    )
    assert "--prepared-from" not in calls[0] and "--prepared-from" in calls[2]
    board = (tmp_path / "sweep/README.md").read_text()
    assert board.index("| 1 | 2 |") < board.index("| 2 | 0 |"), "lower MAE ranks first"
    assert "Failed: trial 1" in board
    # A rerun skips completed trials; a STOP file ends admission.
    calls.clear()
    for trial in state["trials"]:
        if trial["index"] == 1:
            trial["status"] = "pending"
    (tmp_path / "sweep/state.json").write_text(json.dumps(state))
    (tmp_path / "sweep/STOP").write_text("")
    assert sweep.run(args) == 0
    assert calls == []
    (tmp_path / "sweep/STOP").unlink()
    assert sweep.run(args) == 0
    assert len(calls) == 1 and "--prepared-from" in calls[0]


def test_run_rejects_state_from_another_plan(tmp_path):
    data = sweep.plan("rel-f1/driver-position", 7453, 1, seed=1, seeds=1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep/state.json").write_text(json.dumps({"plan_sha256": "other", "trials": []}))
    args = sweep.arguments(
        [
            "run",
            "--plan",
            str(plan_path),
            "--name",
            "unit",
            "--output",
            str(tmp_path / "sweep"),
            "--source",
            str(tmp_path),
            "--cache",
            "/cache",
        ]
    )
    with pytest.raises(ValueError, match="different plan"):
        sweep.run(args)


def test_summarize_reads_the_nested_training_report_and_counted_drops():
    nested = {
        "status": "passed",
        "result": {
            "selection": {"metric": "mae", "mode": "min", "step": 4000, "score": 0.3818},
            "stages": {
                "supervised": {
                    "steps": 4000,
                    "equivalent_full_epochs": 3.38,
                    "seconds": 5412.0,
                    "peak_cuda_allocated_bytes": 4_475_282_944,
                }
            },
            "stopping_reason": "training_time_limit",
            "plateau": {"factor": 0.1, "patience": 2, "drops": 1},
            "baseline": {"mae": 0.4749},
            "validation": {"mae": 0.3818},
        },
    }
    summary = sweep.summarize(nested)
    assert summary["score"] == 0.3818 and summary["selected_step"] == 4000 and summary["steps"] == 4000
    assert summary["stopping_reason"] == "training_time_limit" and summary["lr_drops"] == 1
    assert summary["peak_cuda_gb"] == pytest.approx(4.475, abs=0.01) and summary["baseline"] == {"mae": 0.4749}
    flat = sweep.summarize(
        {"status": "passed", "selection": {"metric": "mae", "score": 1.0}, "plateau": {"drops": [1, 2]}}
    )
    assert flat["score"] == 1.0 and flat["lr_drops"] == 2


def test_report_rebuilds_summaries_from_result_files(tmp_path):
    directory = tmp_path / "sweep"
    directory.mkdir()
    (directory / "trial-000").mkdir()
    result = directory / "trial-000" / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "passed",
                "result": {
                    "selection": {"metric": "mae", "score": 0.5, "step": 10},
                    "stages": {"supervised": {"steps": 10, "equivalent_full_epochs": 1.0, "seconds": 30.0}},
                },
            }
        )
    )
    state = {
        "version": 1,
        "name": "demo",
        "task": "rel-trial/site-success",
        "tier": "medium",
        "seed": 0,
        "plan_sha256": "x",
        "trials": [
            {
                "index": 0,
                "config": sweep.DEFAULT,
                "status": "done",
                "result": str(result),
                "summary": sweep.summarize({"status": "passed"}),
            },
            {"index": 1, "config": sweep.DEFAULT, "status": "pending"},
        ],
    }
    (directory / "state.json").write_text(json.dumps(state))

    board = sweep.report(directory)

    rebuilt = json.loads((directory / "state.json").read_text())["trials"][0]["summary"]
    assert rebuilt["score"] == 0.5 and rebuilt["steps"] == 10
    assert "| 1 | 0 | 0 | 0.50000 |" in board and (directory / "README.md").read_text() == board


def test_command_takes_the_plan_budget_unless_overridden(tmp_path):
    planned = options(trial_seconds=None, passes=None, budget={"passes": 8, "backstop_seconds": 14400.0})
    text = " ".join(sweep.command(planned, sweep.DEFAULT, tmp_path / "b.yaml", tmp_path / "t", None))
    assert "--max-epochs 8" in text and "--max-train-seconds 14400.0" in text
    overridden = options(trial_seconds=600, passes=3, budget={"passes": 8, "backstop_seconds": 14400.0})
    text = " ".join(sweep.command(overridden, sweep.DEFAULT, tmp_path / "b.yaml", tmp_path / "t", None))
    assert "--max-epochs 3" in text and "--max-train-seconds 600" in text


def test_winner_follows_the_metric_direction():
    def trial(index, metric, score, status="done"):
        return {"index": index, "status": status, "summary": {"metric": metric, "score": score}}

    assert (
        sweep.winner({"trials": [trial(0, "mae", 0.5), trial(1, "mae", 0.4), trial(2, "mae", 0.1, "failed")]})["index"]
        == 1
    )
    assert sweep.winner({"trials": [trial(0, "roc_auc", 0.7), trial(1, "roc_auc", 0.8)]})["index"] == 1
    assert sweep.winner({"trials": [trial(0, "mae", None)]}) is None


def test_refit_runs_the_winner_on_the_outer_split_with_scaled_updates(tmp_path, monkeypatch):
    directory = tmp_path / "sweep"
    (directory / "trial-001").mkdir(parents=True)
    result = directory / "trial-001/result.json"
    result.write_text(
        json.dumps({"status": "passed", "data": {"validation_rows": 1000}, "config": {"model": {"batch_size": 128}}})
    )

    def summary(score, step):
        return sweep.summarize(
            {
                "status": "passed",
                "selection": {"metric": "mae", "score": score, "step": step},
                "stages": {"supervised": {"steps": step, "equivalent_full_epochs": 2.0, "seconds": 60.0}},
            }
        )

    state = {
        "version": 1,
        "name": "unit",
        "task": "rel-f1/driver-position",
        "tier": "small",
        "seed": 0,
        "plan_sha256": "x",
        "trials": [
            {"index": 0, "config": sweep.DEFAULT, "status": "done", "result": "missing", "summary": summary(0.9, 50)},
            {
                "index": 1,
                "config": {**sweep.DEFAULT, "batch_size": 256, "ema_decay": 0.99},
                "status": "done",
                "result": str(result),
                "summary": summary(0.4, 300),
            },
        ],
    }
    calls = []

    def fake_run(argv, cwd, env, stdout, stderr):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "passed", "test": {"mae": 0.42}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = options(training_rows=4000, source=str(tmp_path), environment={}, gpu_lock=None)
    record = sweep.refit(args, state, directory)

    text = " ".join(calls[0])
    assert "-m benchmarks.relarena.refit" in text and "--steps 375" in text, (
        "300 updates scaled by (4000 + 1000) / 4000"
    )
    assert "--batch-size 128" in text and "--ema-decay 0.99" in text and "--accelerator gpu" in text, (
        "the refit trains at the size the trial trained with, not the plan's finder cap"
    )
    assert "--regime full --stale-validation" in text
    assert record["status"] == "done" and record["test"] == {"mae": 0.42} and record["trial"] == 1
    assert json.loads((directory / "state.json").read_text())["refit"]["steps"] == 375
    assert (directory / "refit/bundle/model.yaml").exists()


def test_checkpoint_regime_refit_repeats_the_trial_schedule_on_the_outer_store(tmp_path, monkeypatch):
    directory = tmp_path / "sweep"
    (directory / "trial-000").mkdir(parents=True)
    result = directory / "trial-000/result.json"
    result.write_text(
        json.dumps({"status": "passed", "data": {"validation_rows": 100}, "config": {"model": {"batch_size": 64}}})
    )
    summary = sweep.summarize(
        {
            "status": "passed",
            "selection": {"metric": "mae", "score": 0.5, "step": 40},
            "stages": {"supervised": {"steps": 40, "equivalent_full_epochs": 1.0, "seconds": 6.0}},
        }
    )
    state = {
        "version": 1,
        "name": "unit",
        "task": "rel-f1/driver-position",
        "tier": "small",
        "seed": 0,
        "plan_sha256": "x",
        "trials": [{"index": 0, "config": sweep.DEFAULT, "status": "done", "result": str(result), "summary": summary}],
    }
    calls = []

    def fake_run(argv, cwd, env, stdout, stderr):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "passed", "test": {"mae": 0.6}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = options(training_rows=400, source=str(tmp_path), environment={}, gpu_lock=None, refit_regime="checkpoint")
    record = sweep.refit(args, state, directory)

    text = " ".join(calls[0])
    assert "--regime checkpoint" in text and "--batch-size 64" in text
    assert "--patience 5 --plateau-factor 0.1 --plateau-patience 2 --max-lr-drops 2 --min-improvement 0.001" in text
    assert "--max-train-seconds" in text and "--val-every" in text and "--relbench-cache /cache" in text
    assert record["regime"] == "checkpoint" and record["status"] == "done"


def test_refit_of_one_trial_lands_on_that_trial(tmp_path, monkeypatch):
    directory = tmp_path / "sweep"
    (directory / "trial-003").mkdir(parents=True)
    result = directory / "trial-003/result.json"
    result.write_text(
        json.dumps({"status": "passed", "data": {"validation_rows": 100}, "config": {"model": {"batch_size": 64}}})
    )
    summary = sweep.summarize(
        {
            "status": "passed",
            "selection": {"metric": "mae", "score": 0.5, "step": 40},
            "stages": {"supervised": {"steps": 40, "equivalent_full_epochs": 1.0, "seconds": 6.0}},
        }
    )
    trial = {
        "index": 3,
        "config": sweep.DEFAULT,
        "seed": 2,
        "status": "done",
        "result": str(result),
        "summary": summary,
    }
    state = {"version": 1, "name": "unit", "task": "rel-f1/driver-position", "tier": "small", "seed": 0, "plan_sha256": "x", "trials": [trial]}  # fmt: skip

    def fake_run(argv, cwd, env, stdout, stderr):
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "passed", "test": {"mae": 0.7}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = options(training_rows=400, source=str(tmp_path), environment={}, gpu_lock=None)
    record = sweep.refit(args, state, directory, trial=trial)

    assert record["status"] == "done" and record["test"] == {"mae": 0.7}
    assert record["attempts"][0]["command"][record["attempts"][0]["command"].index("--seed") + 1] == "2", (
        "a seed's refit keeps its seed"
    )
    assert trial["refit"] is record and "refit" not in state
    assert (directory / "trial-003/refit/bundle/model.yaml").exists()
    assert "0.70000" in sweep.leaderboard(state), "the leaderboard shows how the trial converted"


def test_refit_retries_at_half_the_batch_after_running_out_of_memory(tmp_path, monkeypatch):
    directory = tmp_path / "sweep"
    (directory / "trial-000").mkdir(parents=True)
    result = directory / "trial-000/result.json"
    result.write_text(
        json.dumps({"status": "passed", "data": {"validation_rows": 100}, "config": {"model": {"batch_size": 64}}})
    )
    summary = sweep.summarize(
        {
            "status": "passed",
            "selection": {"metric": "mae", "score": 0.5, "step": 40},
            "stages": {"supervised": {"steps": 40, "equivalent_full_epochs": 1.0, "seconds": 6.0}},
        }
    )
    state = {
        "version": 1,
        "name": "unit",
        "task": "rel-f1/driver-position",
        "tier": "small",
        "seed": 0,
        "plan_sha256": "x",
        "trials": [{"index": 0, "config": sweep.DEFAULT, "status": "done", "result": str(result), "summary": summary}],
    }
    batches = []

    def fake_run(argv, cwd, env, stdout, stderr):
        batches.append(int(argv[argv.index("--batch-size") + 1]))
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if len(batches) == 1:
            output.write_text(
                json.dumps({"status": "failed", "error": {"type": "OutOfMemoryError", "message": "CUDA out of memory"}})
            )
            return SimpleNamespace(returncode=1)
        output.write_text(json.dumps({"status": "passed", "test": {"mae": 0.6}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = options(training_rows=400, source=str(tmp_path), environment={}, gpu_lock=None)
    record = sweep.refit(args, state, directory)

    assert batches == [64, 32] and record["status"] == "done" and record["batch_size"] == 32
    assert len(record["attempts"]) == 2


def test_queue_runs_tasks_smallest_first_and_skips_finished_ones(tmp_path, monkeypatch):
    sizes = {"rel-f1/driver-position": 7453, "rel-trial/study-adverse": 43335, "rel-avito/user-visits": 86619}
    monkeypatch.setattr(sweep, "rows", lambda cache, task: sizes[task])
    order = []

    def fake_run(task_args):
        order.append(
            task_args.task if hasattr(task_args, "task") else json.loads(Path(task_args.plan).read_text())["task"]
        )
        return 0

    monkeypatch.setattr(sweep, "run", fake_run)
    runs = tmp_path / "runs"
    (runs / "raw-sweep-rel-trial-study-adverse").mkdir(parents=True)
    (runs / "raw-sweep-rel-trial-study-adverse/DONE").write_text("earlier")
    args = sweep.arguments(
        [
            "queue",
            "--runs",
            str(runs),
            "--tasks",
            *sizes,
            "--trials",
            "3",
            "--source",
            str(tmp_path),
            "--cache",
            "/cache",
        ]
    )
    assert sweep.queue(args) == 0
    assert order == ["rel-f1/driver-position", "rel-avito/user-visits"]
    plan_data = json.loads((runs / "raw-sweep-rel-f1-driver-position/plan.json").read_text())
    assert plan_data["task"] == "rel-f1/driver-position" and len(plan_data["trials"]) == 4 * sweep.SEEDS["small"]
    assert (runs / "raw-sweep-rel-avito-user-visits/DONE").exists()
    (runs / "STOP").write_text("")
    order.clear()
    for name in ("raw-sweep-rel-f1-driver-position", "raw-sweep-rel-avito-user-visits"):
        (runs / name / "DONE").unlink()
    assert sweep.queue(args) == 0 and order == []


def test_refit_subcommand_refits_without_admitting_pending_trials(tmp_path, monkeypatch):
    data = sweep.plan("rel-f1/driver-position", 7453, 2, seed=1, seeds=1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    directory = tmp_path / "sweep"
    (directory / "trial-000").mkdir(parents=True)
    result = directory / "trial-000/result.json"
    result.write_text(json.dumps({"status": "passed", "data": {"validation_rows": 100}}))
    import hashlib

    state = {
        "version": 1,
        "name": "unit",
        "task": data["task"],
        "tier": data["tier"],
        "seed": 1,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "trials": [
            {
                "index": 0,
                "config": sweep.DEFAULT,
                "status": "done",
                "result": str(result),
                "summary": sweep.summarize(
                    {
                        "status": "passed",
                        "selection": {"metric": "mae", "score": 0.5, "step": 200},
                        "stages": {"supervised": {"steps": 200, "equivalent_full_epochs": 2.0, "seconds": 60.0}},
                    }
                ),
            },
            {"index": 1, "config": data["trials"][1]["config"], "status": "pending"},
            {"index": 2, "config": data["trials"][2]["config"], "status": "pending"},
        ],
    }
    (directory / "state.json").write_text(json.dumps(state))
    calls = []

    def fake_run(argv, cwd, env, stdout, stderr):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "passed", "test": {"mae": 0.44}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = sweep.arguments(
        [
            "refit",
            "--plan",
            str(plan_path),
            "--name",
            "unit",
            "--output",
            str(directory),
            "--source",
            str(tmp_path),
            "--cache",
            "/cache",
        ]
    )
    assert sweep.refit_only(args) is True
    assert (
        len(calls) == 1 and "-m benchmarks.relarena.refit" in " ".join(calls[0]) and "--steps 203" in " ".join(calls[0])
    )
    saved = json.loads((directory / "state.json").read_text())
    assert saved["refit"]["status"] == "done" and saved["refit"]["test"] == {"mae": 0.44}
    assert [t["status"] for t in saved["trials"]] == ["done", "pending", "pending"], "pending trials stay pending"


def test_execute_runs_the_child_inside_its_own_directory(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, cwd, env, stdout, stderr):
        seen["cwd"] = Path(cwd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    trial_dir = tmp_path / "trial-000"
    trial_dir.mkdir()
    args = SimpleNamespace(gpu_lock=None, source=str(tmp_path / "gone"), environment={})

    assert sweep.execute(args, ["child"], tmp_path, trial_dir / "run.log") == 0
    # A rebuilt candidate source must not take the child's working directory with it.
    assert seen["cwd"] == trial_dir


def test_spaces_cap_the_batch_size_finder_instead_of_sampling_sizes():
    caps = [sweep.SPACES[level]["batch_size"] for level in ("small", "medium", "large")]
    assert all(len(cap) == 1 for cap in caps), "the cap is a compute knob fixed per tier, not a searched size"
    assert [cap[0] for cap in caps] == [256, 512, 1024]
    assert all(
        sweep.TIER_DEFAULTS[level]["batch_size"] == sweep.SPACES[level]["batch_size"][0]
        for level in caps and sweep.SPACES
    )


def test_failure_names_the_child_error_or_its_exit():
    report = {"error": "{'type': 'OutOfMemoryError', 'message': 'CUDA out of memory. Tried to allocate 272.00 MiB.'}"}
    assert sweep.failure(report, 1).startswith("exit 1, OutOfMemoryError: CUDA out of memory.")
    assert (
        sweep.failure({"error": {"type": "FileNotFoundError", "message": "gone"}}, 1)
        == "exit 1, FileNotFoundError: gone"
    )
    assert sweep.failure(None, 137) == "exit 137, no result"


def test_run_relaunches_an_out_of_memory_trial_at_half_the_batch_size(tmp_path, monkeypatch):
    data = sweep.plan("rel-f1/driver-position", 7453, 1, seed=1, seeds=1)
    data["trials"] = data["trials"][:1]
    data["trials"][0]["config"]["batch_size"] = 64
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    calls = []

    def fake_run(argv, cwd, env, stdout, stderr):
        assert argv[argv.index("--batch-size") + 1] == "auto"
        calls.append(int(argv[argv.index("--max-batch-size") + 1]))
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) < 3:
            # The first attempt records the size the finder chose below its cap; the second dies before choosing.
            output.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "config": {"training": {"batch_size": 48 if len(calls) == 1 else "auto"}},
                        "error": {"type": "OutOfMemoryError", "message": "CUDA out of memory. Tried to allocate 2 GiB"},
                    }
                )
            )
            return SimpleNamespace(returncode=1)
        output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "selection": {"metric": "mae", "score": 1.0, "step": 300},
                    "stages": {"supervised": {"steps": 300, "equivalent_full_epochs": 2.5, "seconds": 60.0}},
                    "stopping_reason": "validation_plateau",
                    "plateau": {"drops": []},
                    "baseline": {"mae": 9.0},
                    "validation": {"mae": 1.0},
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = sweep.arguments(
        [
            "run",
            "--plan",
            str(plan_path),
            "--name",
            "unit",
            "--output",
            str(tmp_path / "sweep"),
            "--source",
            str(tmp_path),
            "--cache",
            "/cache",
            "--no-refit",
        ]
    )
    assert sweep.run(args) == 0
    # Caps follow the size each attempt trained with: 48 chosen under 64, then 24 as its own cap.
    assert calls == [64, 24, 12]
    state = json.loads((tmp_path / "sweep/state.json").read_text())
    trial = state["trials"][0]
    assert trial["status"] == "done" and trial["config"]["batch_size"] == 12
    assert [(attempt["cap"], attempt["batch_size"]) for attempt in trial["attempts"]] == [(64, 48), (24, 24)]
    assert "OutOfMemoryError" in trial["attempts"][0]["error"]
    # Each failed attempt keeps its log beside the trial; the leaderboard shows the batch size that fit.
    assert (tmp_path / "sweep/trial-000-oom-48/run.log").exists()
    assert (tmp_path / "sweep/trial-000-oom-24/run.log").exists()
    assert "| 12 |" in (tmp_path / "sweep/README.md").read_text()


def test_run_does_not_relaunch_other_failures_or_below_the_batch_floor(tmp_path, monkeypatch):
    data = sweep.plan("rel-f1/driver-position", 7453, 2, seed=1, seeds=1)
    data["trials"] = data["trials"][:2]
    data["trials"][0]["config"]["batch_size"] = 64
    data["trials"][1]["config"]["batch_size"] = sweep.BATCH_FLOOR
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    calls = []

    def fake_run(argv, cwd, env, stdout, stderr):
        assert argv[argv.index("--batch-size") + 1] == "auto"
        calls.append(int(argv[argv.index("--max-batch-size") + 1]))
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            output.write_text(json.dumps({"status": "failed", "error": {"type": "ValueError", "message": "bad spec"}}))
        else:
            # No report at all: the log tail is the only evidence.
            Path(cwd, "run.log").write_text("RuntimeError: MPS backend out of memory (MPS allocated: 9 GB)\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = sweep.arguments(
        [
            "run",
            "--plan",
            str(plan_path),
            "--name",
            "unit",
            "--output",
            str(tmp_path / "sweep"),
            "--source",
            str(tmp_path),
            "--cache",
            "/cache",
            "--no-refit",
        ]
    )
    assert sweep.run(args) == 0
    assert calls == [64, sweep.BATCH_FLOOR]
    state = json.loads((tmp_path / "sweep/state.json").read_text())
    assert [t["status"] for t in state["trials"]] == ["failed", "failed"]
    assert state["trials"][0]["attempts"] == [] and state["trials"][1]["attempts"] == []
    assert "out of memory" in state["trials"][1]["error"] or "no result" in state["trials"][1]["error"]


def test_run_trains_a_derived_structure_built_once_per_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "structure", REAL_STRUCTURE)
    data = sweep.plan("rel-f1/driver-position", 7453, 1, seed=1, seeds=1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    derivations, trials = [], []

    def fake_run(argv, cwd=None, env=None, stdout=None, stderr=None, check=False):
        if "benchmarks.relarena.derive" in argv:
            derivations.append(argv)
            synthetic_spec(Path(argv[argv.index("--output") + 1]) / "rel-f1" / "driver-position", d_model=24)
            return SimpleNamespace(returncode=0)
        trials.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "selection": {"metric": "mae", "score": 3.0, "step": 100},
                    "stages": {"supervised": {"steps": 100, "equivalent_full_epochs": 2.5, "seconds": 60.0}},
                    "stopping_reason": "validation_plateau",
                    "plateau": {"drops": []},
                    "baseline": {"mae": 9.0},
                    "validation": {"mae": 3.0},
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = sweep.arguments(
        ["run", "--plan", str(plan_path), "--name", "unit", "--output", str(tmp_path / "sweep"), "--source", str(tmp_path),
         "--cache", "/cache", "--no-refit"]
    )  # fmt: skip
    assert sweep.run(args) == 0
    assert len(derivations) == 1 and len(trials) == 2, "one derivation serves every trial"
    assert (
        "--relbench" in derivations[0]
        and derivations[0][derivations[0].index("--relbench") + 1] == "rel-f1/driver-position"
    )
    for index in range(2):
        model = yaml.safe_load((tmp_path / f"sweep/trial-{index:03d}/bundle/model.yaml").read_text())
        assert "results" in model["fields"], "trials train the derived structure"
        assert model["d_model"] == data["trials"][index]["config"]["d_model"]
    assert all("--num-workers" in argv and "--text-store" not in argv for argv in trials), (
        "the runner's store default stands"
    )


def test_children_keep_the_inherited_python_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/root/patched")
    args = sweep.arguments(
        ["run", "--plan", "p.json", "--name", "unit", "--output", str(tmp_path), "--source", str(tmp_path), "--cache", "/c"]
    )  # fmt: skip
    assert args.environment["PYTHONPATH"].split(os.pathsep) == [str(tmp_path.resolve()), "/root/patched"]
    monkeypatch.delenv("PYTHONPATH")
    args = sweep.arguments(
        ["run", "--plan", "p.json", "--name", "unit", "--output", str(tmp_path), "--source", str(tmp_path), "--cache", "/c"]
    )  # fmt: skip
    assert args.environment["PYTHONPATH"] == str(tmp_path.resolve())


def test_bundle_writes_field_regularizers(tmp_path):
    model = {"d_model": 32, "fields": {"amount": {"type": "number"}, "code": {"type": "category"}, "events": {"type": "branch", "fields": {"note": {"type": "text"}}}, "target": {"type": "number", "mask": True}}}  # fmt: skip
    sweep.regularize(model["fields"], jitter=0.05, mask=0.1, objective="huber")
    assert model["fields"]["amount"] == {"type": "number", "mask": 0.1, "jitter": {"add": 0.05}}
    assert model["fields"]["code"] == {"type": "category", "mask": 0.1}, "jitter only touches continuous fields"
    assert model["fields"]["events"]["fields"]["note"] == {"type": "text", "mask": 0.1, "jitter": {"add": 0.05}}
    assert model["fields"]["target"] == {"type": "number", "mask": True, "objective": "huber"}
    untouched = {"fields": {"amount": {"type": "number"}, "target": {"type": "boolean", "mask": True}}}
    sweep.regularize(untouched["fields"], jitter=0.0, mask=0.0, objective="mse")
    assert untouched == {"fields": {"amount": {"type": "number"}, "target": {"type": "boolean", "mask": True}}}


def test_smoothed_objective_averages_the_best_validations():
    summary = {"metric": "mae", "score": 0.30, "validation": [{"step": 1, "mae": 0.5}, {"step": 2, "mae": 0.3}, {"step": 3, "mae": 0.4}, {"step": 4, "mae": 0.6}]}  # fmt: skip
    assert abs(sweep.smoothed(summary) - 0.4) < 1e-9
    assert (
        abs(
            sweep.smoothed({"metric": "roc_auc", "score": 0.7, "validation": [{"roc_auc": 0.6}, {"roc_auc": 0.7}]})
            - 0.65
        )
        < 1e-9
    )
    assert sweep.smoothed({"metric": "mae", "score": 0.2, "validation": []}) == 0.2
    assert sweep.within(sweep.TIER_DEFAULTS["small"], sweep.SPACES["small"])
    assert not sweep.within({**sweep.TIER_DEFAULTS["small"], "dropout": 0.05}, sweep.SPACES["small"])


def test_optimize_proposes_trials_from_a_warm_started_study(tmp_path, monkeypatch, synthetic_structure):
    plan_data = sweep.plan("rel-f1/driver-position", 7453, 0, 0, seeds=1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_data))
    earlier = tmp_path / "earlier-state.json"
    earlier.write_text(
        json.dumps(
            {
                "name": "earlier",
                "trials": [
                    {"index": 0, "config": sweep.TIER_DEFAULTS["small"], "status": "done", "summary": {"metric": "mae", "score": 3.8, "validation": [{"mae": 3.8}, {"mae": 3.9}]}},
                    {"index": 1, "config": {**sweep.TIER_DEFAULTS["small"], "dropout": 0.05}, "status": "done", "summary": {"metric": "mae", "score": 3.5, "validation": [{"mae": 3.5}]}},
                ],
            }
        )
    )  # fmt: skip
    scores = iter([3.7, 3.6, 3.65])

    def fake_run(argv, cwd, env, stdout, stderr):
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"status": "passed", "result": {"selection": {"metric": "mae", "score": next(scores), "step": 10}, "stages": {"supervised": {"steps": 10, "equivalent_full_epochs": 1.0, "seconds": 1.0}}, "validation": [{"step": 10, "mae": 3.7}]}})
        )  # fmt: skip
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = options(plan=str(plan_path), output=str(tmp_path / "sweep"), trials=2, startup=1, seeds=1, warm_start=[str(earlier)], direction="minimize", source=str(tmp_path), environment={}, gpu_lock=None, no_refit=True, refit_every_trial=False, max_hours=0)  # fmt: skip
    assert sweep.optimize(args) == 0

    state = json.loads((tmp_path / "sweep/state.json").read_text())
    assert [t["index"] for t in state["trials"]] == [0, 1, 2] and all(t["status"] == "done" for t in state["trials"])
    assert state["trials"][1]["optuna"] is not None and sweep.within(
        state["trials"][1]["config"], sweep.SPACES["small"]
    )
    import optuna

    study = optuna.load_study(study_name="unit", storage=f"sqlite:///{tmp_path / 'sweep/optuna.db'}")
    origins = {t.user_attrs.get("origin") for t in study.trials}
    assert "earlier:0" in origins and "earlier:1" not in origins, "warm start admits only configs inside the space"
    assert len(study.trials) == 4 and (tmp_path / "sweep/README.md").exists()


def test_small_plans_repeat_each_configuration_over_seeds_and_select_on_the_mean():
    data = sweep.plan("rel-f1/driver-position", 7453, 2, 5)
    assert data["seeds"] == 3 and len(data["trials"]) == 9
    first = [t for t in data["trials"] if t["config_id"] == data["trials"][0]["config_id"]]
    assert [t["seed"] for t in first] == [5, 6, 7] and all(t["config"] == first[0]["config"] for t in first)
    assert sweep.within(first[0]["config"], sweep.SPACES["small"]) and first[0]["config"]["ema_decay"] == 0.99
    assert "mask" in sweep.SPACES["small"] and "jitter" in sweep.SPACES["small"]

    def done(index, config_id, seed, score, step):
        summary = sweep.summarize(
            {
                "status": "passed",
                "selection": {"metric": "mae", "score": score, "step": step},
                "stages": {"supervised": {"steps": step * 2, "equivalent_full_epochs": 2.0, "seconds": 30.0}},
            }
        )
        return {"index": index, "config": sweep.TIER_DEFAULTS["small"], "config_id": config_id, "seed": seed, "status": "done", "summary": summary}  # fmt: skip

    lucky = {**sweep.TIER_DEFAULTS["small"], "dropout": 0.4}
    state = {"name": "unit", "task": "rel-f1/driver-position", "tier": "small", "seed": 0, "plan_sha256": "x",
             "trials": [done(0, "a", 0, 3.6, 100), done(1, "a", 1, 3.7, 120), done(2, "a", 2, 3.65, 110), done(3, "b", 0, 3.5, 50), done(4, "b", 1, 3.9, 60), done(5, "b", 2, 3.85, 70)]}  # fmt: skip
    for t in state["trials"][3:]:
        t["config"] = lucky
    best = sweep.winner(state)
    assert best["config_id"] == "a" and best["index"] == 0 and best["seeds"] == 3
    assert abs(best["summary"]["score"] - 3.65) < 1e-9 and best["summary"]["selected_step"] == 110, (
        "a single lucky seed (3.5) does not beat a configuration whose three seeds average better"
    )
    assert "| 1 | a | 3 | 3.65000 |" in sweep.leaderboard(state)


def test_trial_commands_carry_their_own_seed(tmp_path):
    args = options(training_rows=400, source=str(tmp_path), environment={}, gpu_lock=None)
    argv = sweep.command(args, sweep.TIER_DEFAULTS["small"], tmp_path / "b.yaml", tmp_path / "t", None, seed=7)
    assert argv[argv.index("--seed") + 1] == "7"


def test_exhausted_only_believes_accelerator_memory(tmp_path):
    cuda = {"error": {"type": "OutOfMemoryError", "message": "CUDA out of memory. Tried to allocate 2.00 MiB"}}
    duck = {
        "error": {
            "type": "OutOfMemoryException",
            "message": "duckdb.duckdb.OutOfMemoryException: Out of Memory Error: failed to allocate data of size 64.0 KiB",
        }
    }
    assert sweep.exhausted(cuda, tmp_path / "none.log") and not sweep.exhausted(duck, tmp_path / "none.log")
