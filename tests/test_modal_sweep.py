from pathlib import Path

from benchmarks.relarena import modal_sweep


def test_modal_commands_keep_every_artifact_on_the_volume():
    plan, run = modal_sweep.commands(
        "rel-f1/driver-position",
        output=Path("/vol/runs/modal-rel-f1-driver-position"),
        cache=Path("/vol/cache/relbench"),
        python="/usr/local/bin/python",
        source="/root",
        trials=2,
        seed=0,
        trial_seconds=600,
        max_hours=1.5,
        tracking=Path("/vol/mlflow"),
    )
    assert plan[:4] == ["/usr/local/bin/python", "-m", "benchmarks.relarena.sweep", "plan"]
    assert plan[plan.index("--output") + 1] == "/vol/runs/modal-rel-f1-driver-position/plan.json"
    assert run[run.index("--name") + 1] == "modal-rel-f1-driver-position"
    assert run[run.index("--tracking-uri") + 1] == "file:///vol/mlflow"
    assert run[run.index("--source") + 1] == "/root" and "--gpu-lock" not in run
    assert all(not value.startswith("/") or value.startswith(("/vol/", "/root", "/usr")) for value in plan + run)


def test_modal_commands_pass_the_narrowing_and_the_schedule_through():
    plan, run = modal_sweep.commands(
        "rel-f1/driver-position",
        output=Path("/vol/runs/modal-rel-f1-driver-position-2"),
        cache=Path("/vol/cache/relbench"),
        python="/usr/local/bin/python",
        source="/root",
        trials=24,
        seed=0,
        trial_seconds=3600,
        max_hours=22.0,
        tracking=Path("/vol/mlflow"),
        space='{"ema_decay": [null]}',
        default='{"ffnx": 4}',
        patience=12,
        plateau_patience=4,
    )
    assert plan[plan.index("--space") + 1] == '{"ema_decay": [null]}'
    assert plan[plan.index("--default") + 1] == '{"ffnx": 4}'
    assert run[run.index("--patience") + 1] == "12" and run[run.index("--plateau-patience") + 1] == "4"
    bare, _ = modal_sweep.commands(
        "rel-f1/driver-position",
        output=Path("/vol/runs/x"),
        cache=Path("/vol/cache/relbench"),
        python="python",
        source="/root",
        trials=2,
        seed=0,
        trial_seconds=600,
        max_hours=1.0,
        tracking=Path("/vol/mlflow"),
    )
    assert "--space" not in bare and "--default" not in bare
    plan, run = modal_sweep.commands(
        "rel-avito/ad-ctr",
        output=Path("/vol/runs/x"),
        cache=Path("/vol/cache/relbench"),
        python="python",
        source="/root",
        trials=0,
        seed=0,
        trial_seconds=600,
        max_hours=1.0,
        tracking=Path("/vol/mlflow"),
        refit_every_trial=True,
        grid='[{}, {"learning_rate": 0.00076}]',
        refit_windows="1,3,5",
    )
    assert plan[plan.index("--grid") + 1] == '[{}, {"learning_rate": 0.00076}]' and "--trials" in plan
    assert run[run.index("--refit-windows") + 1] == "1,3,5" and "--refit-every-trial" in run
    assert "--grid" not in bare and "--refit-windows" not in bare
