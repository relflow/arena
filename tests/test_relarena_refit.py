import numpy as np
import pyarrow as pa
import pytest

from benchmarks.relarena import refit


def test_scaled_updates_grow_with_the_outer_split():
    assert refit.scaled(4000, 151_407, 19_740) == 4522
    assert refit.scaled(10, 100, 0) == 10
    with pytest.raises(ValueError):
        refit.scaled(0, 100, 10)


def test_ordered_predictions_follow_row_positions_and_unwrap_probabilities():
    inputs = pa.array([{"__row_position": 2}, {"__row_position": 0}, {"__row_position": 1}])
    predictions = pa.array([{"target": {"content": 0.3}}, {"target": {"content": 0.1}}, {"target": {"content": 0.2}}])
    table = pa.table({"inputs": inputs, "predictions": predictions})
    assert refit.ordered(table, 3).tolist() == [0.1, 0.2, 0.3]

    probabilities = pa.array(
        [
            {"target": {"content": {"probability": 0.9, "label": True}}},
            {"target": {"content": {"probability": 0.2, "label": False}}},
        ]
    )
    table = pa.table({"inputs": pa.array([{"__row_position": 1}, {"__row_position": 0}]), "predictions": probabilities})
    assert refit.ordered(table, 2).tolist() == [0.2, 0.9]
    with pytest.raises(RuntimeError, match="exactly one row"):
        refit.ordered(table, 3)
    nan = pa.table(
        {"inputs": pa.array([{"__row_position": 0}]), "predictions": pa.array([{"target": {"content": float("nan")}}])}
    )
    with pytest.raises(RuntimeError, match="finite"):
        refit.ordered(nan, 1)


def test_refit_arguments_fall_back_to_full_precision_on_cpu():
    args = refit.arguments(
        [
            "--benchmark",
            "rel-f1/driver-position",
            "--spec",
            "b.yaml",
            "--work-dir",
            "w",
            "--output",
            "o.json",
            "--steps",
            "5",
            "--batch-size",
            "8",
            "--learning-rate",
            "1e-3",
            "--weight-decay",
            "0.0",
            "--accelerator",
            "cpu",
        ]
    )
    assert args.precision == "32-true" and args.fused is True and args.ema_decay is None
    assert np.isfinite(args.learning_rate)


def test_checkpoint_regime_needs_no_update_budget():
    base = ["--benchmark", "rel-f1/driver-position", "--spec", "b.yaml", "--work-dir", "w", "--output", "o.json"]
    base += ["--batch-size", "8", "--learning-rate", "0.001", "--weight-decay", "0.0"]
    args = refit.arguments([*base, "--regime", "checkpoint", "--patience", "3", "--max-train-seconds", "60"])
    assert args.regime == "checkpoint" and args.steps is None and args.patience == 3 and args.max_train_seconds == 60.0
    with pytest.raises(SystemExit):
        refit.arguments(base)
