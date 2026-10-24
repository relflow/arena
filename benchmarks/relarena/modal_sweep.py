"""Run RelArena sweeps on Modal: one GPU container per task, everything persisted on a Volume.

Fan out across tasks, not across trials: each container owns one sweep
directory under ``/vol/runs`` and the loop's file-based state, logs, trial
reports, leaderboard, and MLflow file store all live on the Volume, so a
run survives the container and can be inspected or resumed later.

    uv run modal deploy benchmarks/relarena/modal_sweep.py
    uv run python -m benchmarks.relarena.modal_sweep --tasks rel-f1/driver-position,rel-f1/driver-dnf --trials 24

Deploy, then spawn: calls spawned on the deployed app run to completion with
no client attached. ``modal run`` (the ``main`` entrypoint) still serves short
attended tests, but its calls die with the client; a laptop going to sleep
cancelled a whole campaign that way.

Set ``RELFLOW_SRC`` to a local ``src/relflow`` checkout to run an unreleased
RelFlow: the directory is mounted over the installed package for every command.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

VOLUME = "relflow-sweeps"
ROOT = Path("/vol")
SOURCE = "/root"
RELFLOW = "relflow[text] @ git+https://github.com/relflow/relflow@58b5b92"
RELFLOW_SRC = os.environ.get("RELFLOW_SRC")  # a local src/relflow to run ahead of the pinned release
PATCHED = "/root/patched"
COMMIT_SECONDS = 600

app = modal.App("relflow-relarena-sweep")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        RELFLOW,
        "duckdb==1.5.0",
        "polars>=2,<3",
        "relbench==2.1.2",
        "datasets==5.0.1",
        "huggingface-hub==0.36.2",
        "mlflow-skinny==3.17.0",
        "optuna>=4,<5",
    )
    # The catalog reads task specifications beside the code, so mount the whole package, not only its Python.
    .add_local_dir(
        "benchmarks", remote_path=f"{SOURCE}/benchmarks", ignore=["**/__pycache__", "**/*.pyc", "**/.DS_Store"]
    )
)
if RELFLOW_SRC:
    image = image.add_local_dir(RELFLOW_SRC, remote_path=f"{PATCHED}/relflow", ignore=["**/__pycache__", "**/*.pyc"])


def commands(
    task: str,
    *,
    output: Path,
    cache: Path,
    python: str,
    source: str,
    trials: int,
    seed: int,
    trial_seconds: int,
    max_hours: float,
    tracking: Path,
    space: str = "",
    default: str = "",
    patience: int = 12,
    plateau_patience: int = 4,
    workers: str = "auto",
    refit_every_trial: bool = False,
    optimize: int = 0,
    warm_start: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """The plan and run commands for one task's sweep, all paths on the Volume.

    ``space`` and ``default`` are JSON for the planner's narrowing options; the
    patience values set the run's early-stop and plateau schedule.
    """
    name = output.name
    plan = [
        python, "-m", "benchmarks.relarena.sweep", "plan",
        "--task", task, "--cache", str(cache), "--trials", str(0 if optimize else trials), "--seed", str(seed),
        "--output", str(output / "plan.json"),
    ]  # fmt: skip
    if space:
        plan += ["--space", space]
    if default:
        plan += ["--default", default]
    run = [
        python, "-m", "benchmarks.relarena.sweep", "optimize" if optimize else "run",
        "--plan", str(output / "plan.json"), "--name", name, "--output", str(output),
        *(["--trials", str(optimize), *(["--warm-start", *warm_start] if warm_start else [])] if optimize else []),
        "--python", python, "--source", source, "--cache", str(cache),
        "--seed", str(seed), "--trial-seconds", str(trial_seconds), "--max-hours", str(max_hours),
        "--patience", str(patience), "--plateau-patience", str(plateau_patience),
        "--num-workers", str(workers),
        *(["--refit-every-trial"] if refit_every_trial else []),
        "--tracking-uri", f"file://{tracking}", "--experiment", "relflow-relarena-modal",
    ]  # fmt: skip
    return plan, run


def execute(argv: list[str], log: Path, *, source: str) -> int:
    """Run one loop command, streaming its output to the console and to a log on the Volume."""
    python_path = f"{PATCHED}:{source}" if os.path.isdir(f"{PATCHED}/relflow") else source
    environment = {**os.environ, "PYTHONPATH": python_path, "PYTHONUNBUFFERED": "1"}
    with log.open("a") as stream:
        process = subprocess.Popen(
            argv, cwd=source, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return process.wait()


def prefetch(task: str, cache: Path) -> None:
    """Download the dataset and task into the RelBench cache, which planning reads to count anchors."""
    os.environ["RELBENCH_CACHE_DIR"] = str(cache)
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset, name = task.split("/")
    if not (cache / dataset / "tasks" / name / "train.parquet").exists():
        get_dataset(dataset, download=True)
        get_task(dataset, name, download=True)


def committing(stop: threading.Event) -> None:
    """Persist the Volume every few minutes so a long sweep is visible while it runs."""
    while not stop.wait(COMMIT_SECONDS):
        volume.commit()


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, volumes={str(ROOT): volume}, timeout=24 * 60 * 60)
def sweep(
    task: str,
    trials: int,
    trial_seconds: int,
    max_hours: float,
    seed: int,
    name: str,
    space: str = "",
    default: str = "",
    patience: int = 12,
    plateau_patience: int = 4,
    workers: str = "auto",
    suffix: str = "",
    refit_every_trial: bool = False,
    optimize: int = 0,
    warm_start: str = "",
) -> str:
    """Plan and run one task's sweep on this container's GPU; return the leaderboard."""
    output = ROOT / "runs" / (name or f"modal-{task.replace('/', '-')}" + (f"-{suffix}" if suffix else ""))
    cache = ROOT / "cache/relbench"
    for directory in (output, cache, ROOT / "cache/huggingface", ROOT / "mlflow"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    # One writer per task on a plain file store: MLflow 3 refuses it unless told the maintenance mode is fine.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    prefetch(task, cache)
    volume.commit()
    plan, run = commands(
        task,
        output=output,
        cache=cache,
        python=sys.executable,
        source=SOURCE,
        trials=trials,
        seed=seed,
        trial_seconds=trial_seconds,
        max_hours=max_hours,
        tracking=ROOT / "mlflow",
        space=space,
        default=default,
        patience=patience,
        plateau_patience=plateau_patience,
        workers=workers,
        refit_every_trial=refit_every_trial,
        optimize=optimize,
        warm_start=tuple(item for item in warm_start.split(",") if item),
    )
    log = output / "sweep.log"
    stop = threading.Event()
    keeper = threading.Thread(target=committing, args=(stop,), daemon=True)
    keeper.start()
    try:
        if not (output / "plan.json").exists() and execute(plan, log, source=SOURCE):
            raise RuntimeError(f"planning {task} failed; see {log}")
        volume.commit()
        started = time.monotonic()
        code = execute(run, log, source=SOURCE)
    finally:
        stop.set()
        volume.commit()
    leaderboard = (output / "README.md").read_text() if (output / "README.md").exists() else ""
    return f"{task}: exit {code} after {(time.monotonic() - started) / 60:.0f} min\n\n{leaderboard}"


@app.local_entrypoint()
def main(
    task: str = "rel-f1/driver-position",
    tasks: str = "",
    trials: int = 2,
    trial_seconds: int = 600,
    max_hours: float = 2.0,
    seed: int = 0,
    name: str = "",
    space: str = "",
    default: str = "",
    patience: int = 12,
    plateau_patience: int = 4,
    workers: str = "auto",
    suffix: str = "",
    refit_every_trial: bool = False,
    optimize: int = 0,
    warm_start: str = "",
) -> None:
    """Sweep one task, or fan out across a comma-separated list of tasks, one container each.

    ``--space`` and ``--default`` take the planner's JSON narrowing options, for
    a follow-up sweep around an earlier sweep's winners.
    """
    selected = [item for item in tasks.split(",") if item] or [task]
    for result in sweep.map(
        selected,
        kwargs={
            "trials": trials,
            "trial_seconds": trial_seconds,
            "max_hours": max_hours,
            "seed": seed,
            "name": name if len(selected) == 1 else "",
            "space": space,
            "default": default,
            "patience": patience,
            "plateau_patience": plateau_patience,
            "workers": workers,
            "suffix": suffix,
            "refit_every_trial": refit_every_trial,
            "optimize": optimize,
            "warm_start": warm_start,
        },
    ):
        print(result, flush=True)


def launch(argv: list[str] | None = None) -> list[str]:
    """Spawn one sweep per task on the deployed app and return the function call ids."""
    import argparse

    parser = argparse.ArgumentParser(description="spawn sweeps on the deployed app")
    parser.add_argument("--tasks", required=True, help="comma-separated dataset/task names")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--trial-seconds", type=int, default=600)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--space", default="")
    parser.add_argument("--default", default="")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--plateau-patience", type=int, default=4)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--refit-every-trial", action="store_true")
    parser.add_argument(
        "--optimize", type=int, default=0, help="Optuna proposals per task; 0 runs the planned random trials"
    )
    parser.add_argument(
        "--warm-start", default="", help="comma-separated state.json paths on the Volume to seed the study"
    )
    args = parser.parse_args(argv)
    function = modal.Function.from_name(app.name, "sweep")
    calls = []
    for task in [item for item in args.tasks.split(",") if item]:
        call = function.spawn(
            task,
            args.trials,
            args.trial_seconds,
            args.max_hours,
            args.seed,
            "",
            space=args.space,
            default=args.default,
            patience=args.patience,
            plateau_patience=args.plateau_patience,
            workers=args.workers,
            suffix=args.suffix,
            refit_every_trial=args.refit_every_trial,
            optimize=args.optimize,
            warm_start=args.warm_start,
        )
        print(f"{task} {call.object_id}", flush=True)
        calls.append(call.object_id)
    return calls


if __name__ == "__main__":
    launch()
