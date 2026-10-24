"""Protect optional tracking, per-step histories, and honest failure status."""

import re
import sys
from collections import namedtuple
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.relarena import tracking


@pytest.fixture
def client(monkeypatch):
    result = Mock()
    result.get_experiment_by_name.return_value = None
    result.create_experiment.return_value = "experiment-1"
    result.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
    modules = {name: ModuleType(name) for name in ("mlflow", "mlflow.entities")}
    modules["mlflow"].MlflowClient = Mock(return_value=result)
    modules["mlflow.entities"].Metric = namedtuple("Metric", "key value timestamp step")
    modules["mlflow.entities"].Param = namedtuple("Param", "key value")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return result


def test_disabled_tracking_never_imports_or_contacts_mlflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with tracking.start(None, "pilot", "scratch", parameters={}, tags={}, provenance={}) as run:
        assert run is None


def test_stage_histories_retain_every_step_and_flush_live_and_at_completion(client):
    with tracking.start(
        "http://lab:5000",
        "pilot",
        "jepa",
        parameters={"seed": 7301},
        tags={"arm": "jepa"},
        provenance={"relflow_source_sha256": "source-1"},
    ) as run:
        for step in range(1, 26):
            run.record("pretrain", {"step": step, "loss/train": 1 / step})
        assert not run.pending
        run.record("supervised", {"step": 1, "loss/train": 0.5})
        assert len(run.pending) == 1

    logged = [metric for call in client.log_batch.call_args_list for metric in call.kwargs.get("metrics", [])]
    assert [(metric.key, metric.step) for metric in logged] == [
        *(("pretrain/loss/train", step) for step in range(1, 26)),
        ("supervised/loss/train", 1),
    ]
    client.create_experiment.assert_called_once_with("pilot")
    client.set_terminated.assert_called_once_with("run-1", status="FINISHED")
    client.log_dict.assert_called_once_with("run-1", {"relflow_source_sha256": "source-1"}, "provenance.json")


def test_existing_experiment_is_reused_and_result_has_validation_and_budget(client):
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="existing")
    result = {
        "initial_encoder_digest": "encoder-1",
        "initial_supervised_head_digest": "head-1",
        "jepa_eligible_fraction": 0.9,
        "validation": {"mae": 3.2},
        "stages": {
            "pretrain": {
                "steps": 500,
                "seconds": 30.0,
                "teacher_updates": 500,
                "parameters": {"trainable": 123, "frozen": 456},
            }
        },
    }
    with tracking.start("http://lab:5000", "pilot", "jepa", parameters={}, tags={}, provenance={}) as run:
        run.complete(result, {"mae": 4.4})
    client.create_experiment.assert_not_called()
    assert client.create_run.call_args.args == ("existing",)
    logged = {
        metric.key: metric.value
        for call in client.log_batch.call_args_list
        for metric in call.kwargs.get("metrics", [])
    }
    assert logged == {
        "validation/mae": 3.2,
        "baseline/mae": 4.4,
        "pretrain/steps": 500.0,
        "pretrain/seconds": 30.0,
        "pretrain/teacher_updates": 500.0,
        "pretrain/parameters/trainable": 123,
        "pretrain/parameters/frozen": 456,
        "pretrain/eligible_fraction": 0.9,
    }
    client.log_dict.assert_any_call("run-1", {"result": result, "baseline": {"mae": 4.4}}, "result.json")


def test_internal_epoch_aggregates_do_not_become_step_charts(client):
    with tracking.start("http://lab:5000", "pilot", "scratch", parameters={}, tags={}, provenance={}) as run:
        run.record("supervised", {"step": 1, "epoch": 0, "loss/train_epoch": 0.7, "loss/train": 0.8})
    logged = [metric.key for call in client.log_batch.call_args_list for metric in call.kwargs.get("metrics", [])]
    assert logged == ["supervised/loss/train"]


def test_large_metric_buffer_respects_server_batch_size(client):
    with tracking.start("http://lab:5000", "pilot", "jepa", parameters={}, tags={}, provenance={}) as run:
        for step in range(1, 24):
            run.record("pretrain", {"step": step, **{f"field-{n}/loss": 0.5 for n in range(60)}})
    sizes = [len(call.kwargs["metrics"]) for call in client.log_batch.call_args_list if "metrics" in call.kwargs]
    assert sizes == [1000, 380]


@pytest.mark.parametrize("error,status", [(ValueError("bad training"), "FAILED"), (KeyboardInterrupt(), "KILLED")])
def test_training_failure_flushes_partial_history_and_keeps_original_exception(client, error, status):
    with pytest.raises(type(error)) as captured:
        with tracking.start("http://lab:5000", "pilot", "jepa", parameters={}, tags={}, provenance={}) as run:
            run.record("pretrain", {"step": 1, "loss/train": 0.8})
            raise error
    assert captured.value is error
    client.set_terminated.assert_called_once_with("run-1", status=status)
    assert client.log_batch.call_args.kwargs["metrics"][0].step == 1


def test_tracking_failure_does_not_replace_training_error(client):
    original = RuntimeError("training failed")
    client.set_tag.side_effect = ConnectionError("tracking unavailable")
    with pytest.warns(UserWarning, match="MLflow could not record"):
        with pytest.raises(RuntimeError) as captured:
            with tracking.start("http://lab:5000", "pilot", "jepa", parameters={}, tags={}, provenance={}):
                raise original
    assert captured.value is original
    client.set_terminated.assert_called_once_with("run-1", status="FAILED")


def test_threshold_metric_names_are_safe_distinct_and_archived_while_throughput_is_unchanged(client):
    keys = [".target/train.accuracy@0.5.content", ".target/train.accuracy@0.7.content", "throughput/train"]
    with tracking.start("http://lab:5000", "pilot", "scratch", parameters={}, tags={}, provenance={}) as run:
        run.record("supervised", {"step": 1, **{key: 0.5 for key in keys}})
    logged = [metric.key for call in client.log_batch.call_args_list for metric in call.kwargs.get("metrics", [])]
    assert len(set(logged)) == 3
    assert all(re.fullmatch(r"[\w .:/-]+", key) for key in logged)
    assert "supervised/throughput/train" in logged
    mapping = next(call.args[1] for call in client.log_dict.call_args_list if call.args[2] == "metric-names.json")
    assert set(mapping.values()) == {f"supervised/{key}" for key in keys}


def test_encoded_name_cannot_silently_merge_with_an_existing_valid_name(client):
    raw = "supervised/.target/train.accuracy@0.5.content"
    encoded = tracking.name(raw)
    with pytest.raises(ValueError, match="collide"):
        with tracking.start("http://lab:5000", "pilot", "scratch", parameters={}, tags={}, provenance={}) as run:
            run.record("supervised", {"step": 1, encoded.removeprefix("supervised/"): 0.5})
            run.record("supervised", {"step": 2, raw.removeprefix("supervised/"): 0.7})
    client.set_terminated.assert_called_once_with("run-1", status="FAILED")


def test_long_metric_name_remains_backend_safe_and_identifiable():
    raw = "supervised/" + "a" * 300
    assert len(tracking.name(raw)) == 250
    assert tracking.name(raw) != tracking.name(raw + "b")


def test_metric_upload_failure_still_marks_run_failed_and_preserves_original_error(client):
    original = RuntimeError("metric upload rejected")

    def reject(run_id, **values):
        if values.get("metrics"):
            raise original

    client.log_batch.side_effect = reject
    with pytest.warns(UserWarning, match="MLflow could not record"):
        with pytest.raises(RuntimeError) as captured:
            with tracking.start("http://lab:5000", "pilot", "scratch", parameters={}, tags={}, provenance={}) as run:
                run.record("supervised", {"step": 25, "loss/train": 0.5})
    assert captured.value is original
    client.set_tag.assert_called_once()
    client.set_terminated.assert_called_once_with("run-1", status="FAILED")
