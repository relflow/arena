"""Sweep training and architecture hyperparameters for one task on a frozen catalog, one trial at a time.

The search space depends on the task only through a coarse size tier, as
RelArena requires. The default configuration runs first; the remaining trials
are sampled reproducibly from the tier's space. Every trial trains with the
supervised runner's plateau schedule and early stops, selects its checkpoint by
the exact full-validation metric, and never reads the test split. State is a
JSON file that survives restarts; a ``STOP`` file ends the sweep after the
running trial.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import yaml

from benchmarks.relarena.catalog import TASKS
from benchmarks.relarena.spec import BenchmarkSpec

# Each tier bounds a trial by full passes over the training anchors, with a
# wall-clock backstop, so the plateau schedule decides most endings and heavy
# tasks are not tuned on truncated training.
TIERS = (
    {"name": "small", "limit": 20_000, "passes": 40, "backstop_seconds": 3600.0},
    {"name": "medium", "limit": 500_000, "passes": 8, "backstop_seconds": 4 * 3600.0},
    {"name": "large", "limit": None, "passes": 4, "backstop_seconds": 6 * 3600.0},
)
DEFAULT = {
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "ffnx": 8,
    "dropout": 0.1,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "ema_decay": None,
    "jitter": 0.0,
    "mask": 0.0,
    "objective": "mae",
}
ARCHITECTURE = ("d_model", "n_layers", "n_heads", "ffnx", "dropout")
# Field-level regularization written into every leaf: additive input jitter in standard deviations,
# a training-only input mask rate, and the regression target's loss (ignored on classification).
REGULARIZERS = {"jitter": (0.0, 0.05, 0.1), "mask": (0.0, 0.1, 0.2), "objective": ("mae", "mse", "huber")}
# batch_size is the finder's cap, not a sampled size: each trial lets Lightning's
# finder pick the largest batch that fits on its first steps and keeps half of it.
# Online vocabularies grow their embedding tables while training, so a batch that
# fits early can exceed the card later; the loop then relaunches the trial at half
# the size the finder chose. The cap is a compute knob fixed per tier, as the
# harness requires, so the trained batch is 128, 256, and 512 by tier. The
# October 23 sweeps shaped the rest: smaller batches and learning rates above
# 1e-3 won on driver-position, EMA 0.99 with learning rates near 2e-4 won on
# user-attendance, so both ends stay in the space and EMA stays available.
SPACES = {
    "small": {
        # Frozen 2026-10-24 from the conversion dataset: within a sweep the validation order did not
        # transfer to test, while regularization did (EMA, dropout 0.3+, weight decay 0.1+), so the tier
        # searches only regularization and the input perturbations, with seeds averaged per configuration.
        "d_model": (64, 96),
        "n_layers": (1, 2),
        "n_heads": (4,),
        "ffnx": (4, 8),
        "dropout": (0.3, 0.4),
        "batch_size": (256,),
        "learning_rate": (1e-3, 3e-3),
        "weight_decay": (0.1, 0.3),
        "ema_decay": (0.99, 0.995),
        "jitter": (0.0, 0.05, 0.1),
        "mask": (0.0, 0.1, 0.2),
        "objective": ("mae", "huber"),
    },
    "medium": {
        "d_model": (64, 96, 128),
        "n_layers": (1, 2, 3),
        "n_heads": (4, 8),
        "ffnx": (4, 8),
        "dropout": (0.0, 0.1, 0.2),
        "batch_size": (512,),
        "learning_rate": (1e-4, 3e-3),
        "weight_decay": (0.0, 0.01, 0.1),
        "ema_decay": (None, 0.99, 0.999),
        **REGULARIZERS,
    },
    "large": {
        "d_model": (96, 128, 192),
        "n_layers": (2, 3),
        "n_heads": (4, 8),
        "ffnx": (4, 8),
        "dropout": (0.0, 0.1),
        "batch_size": (1024,),
        "learning_rate": (1e-4, 1e-3),
        "weight_decay": (0.0, 0.01, 0.1),
        "ema_decay": (None, 0.99, 0.999),
        **REGULARIZERS,
    },
}
# The untuned regime takes its tier's batch cap and the shared architecture defaults; a tier may
# override the rest, within the coarse size tiers the harness allows.
TIER_OVERRIDES = {"small": {"dropout": 0.3, "learning_rate": 2e-3, "weight_decay": 0.1, "ema_decay": 0.99}}
# Trials that share a configuration and differ only by seed are selected on their mean validation score.
SEEDS = {"small": 3, "medium": 1, "large": 1}
TIER_DEFAULTS = {
    level: {**DEFAULT, **TIER_OVERRIDES.get(level, {}), "batch_size": space["batch_size"][0]}
    for level, space in SPACES.items()
}


def tier(training_rows: int) -> dict:
    """The size tier of a task, by training anchors."""
    for entry in TIERS:
        if entry["limit"] is None or training_rows < entry["limit"]:
            return entry
    raise AssertionError("the last tier is unbounded")


def sample(space: dict, rng: random.Random) -> dict:
    """One configuration: categorical choices, and a log-uniform learning rate between its bounds."""
    config = {}
    for name, choices in space.items():
        if name == "learning_rate":
            low, high = choices
            config[name] = float(f"{math.exp(rng.uniform(math.log(low), math.log(high))):.3g}")
        else:
            config[name] = rng.choice(choices)
    if config["d_model"] % config["n_heads"]:
        config["n_heads"] = max(head for head in space["n_heads"] if config["d_model"] % head == 0)
    return config


def narrow(level: str, overrides: dict | None) -> dict:
    """The tier's space with per-parameter overrides: non-empty choices, or two positive learning-rate bounds."""
    space = dict(SPACES[level])
    for name, choices in (overrides or {}).items():
        if name not in space:
            raise KeyError(f"unknown sweep parameter {name!r}; expected one of {sorted(space)}")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ValueError(f"sweep parameter {name!r} needs a non-empty list of choices, got {choices!r}")
        if name == "learning_rate" and (
            len(choices) != 2
            or not all(isinstance(bound, (int, float)) and bound > 0 for bound in choices)
            or choices[0] > choices[1]
        ):
            raise ValueError(f"learning_rate needs two positive bounds [low, high], got {choices!r}")
        space[name] = tuple(choices)
    return space


def plan(
    task: str,
    training_rows: int,
    trials: int,
    seed: int,
    *,
    space: dict | None = None,
    default: dict | None = None,
    seeds: int | None = None,
) -> dict:
    """The default configuration first, then ``trials`` distinct samples from the task's tier.

    Each configuration becomes ``seeds`` trials (the tier's count by default) that differ only
    by seed; selection averages them.

    ``space`` narrows the tier's choices per parameter and ``default`` overrides
    the first trial, so a second sweep can follow the evidence of an earlier one.
    """
    if task not in TASKS:
        raise KeyError(f"Unknown benchmark {task!r}")
    if trials < 0 or training_rows < 1:
        raise ValueError("trials must be nonnegative and training rows positive")
    entry = tier(training_rows)
    level = entry["name"]
    chosen = narrow(level, space)
    first = {**TIER_DEFAULTS[level], **(default or {})}
    if unknown := sorted(set(first) - set(DEFAULT)):
        raise KeyError(f"unknown default parameter(s) {unknown}; expected one of {sorted(DEFAULT)}")
    if first["d_model"] % first["n_heads"]:
        raise ValueError(f"default d_model {first['d_model']} must be divisible by n_heads {first['n_heads']}")
    rng = random.Random(seed)
    configs, seen = [first], {json.dumps(first, sort_keys=True)}
    attempts = 0
    while len(configs) < trials + 1 and attempts < 100 * (trials + 1):
        candidate = sample(chosen, rng)
        attempts += 1
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(candidate)
    return {
        "version": 1,
        "task": task,
        "training_rows": training_rows,
        "tier": level,
        "budget": {"passes": entry["passes"], "backstop_seconds": entry["backstop_seconds"]},
        "seed": seed,
        "space": chosen,
        "seeds": seeds or SEEDS[level],
        "trials": [
            {"index": index, "config": config, "config_id": identity(config), "seed": seed + repeat}
            for index, (config, repeat) in enumerate((c, r) for c in configs for r in range(seeds or SEEDS[level]))
        ],
    }


def identity(config: dict) -> str:
    """A stable name for one configuration, shared by its seeds."""
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


JITTERED = {"number", "vector", "dateparts", "text"}


def regularize(fields: dict, *, jitter: float, mask: float, objective: str) -> None:
    """Write field-level regularization into a schema: jitter and masks on input leaves, the loss on a numeric target."""
    for name, field in fields.items():
        if not isinstance(field, dict):
            continue
        if field.get("type") == "branch":
            regularize(field.get("fields", {}), jitter=jitter, mask=mask, objective=objective)
            continue
        if name == "target":
            if field.get("type") == "number" and objective != DEFAULT["objective"]:
                field["objective"] = objective
            continue
        if mask:
            field["mask"] = mask
        if jitter and field.get("type") in JITTERED:
            field["jitter"] = {"add": jitter}


def bundle(spec: BenchmarkSpec, config: dict, directory: Path, manifest_path: Path) -> Path:
    """Copy the task's query and manifest, and write its schema with this trial's architecture."""
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec.query_path, directory / "observations.sql")
    model = yaml.safe_load(spec.model_path.read_text())
    model.update({name: config[name] for name in ARCHITECTURE})
    regularize(model["fields"], **{name: config.get(name, DEFAULT[name]) for name in REGULARIZERS})
    (directory / "model.yaml").write_text(yaml.safe_dump(model, sort_keys=False, width=1000))
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest.update(model="model.yaml", query="observations.sql")
    (directory / "benchmark.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=1000))
    return directory / "benchmark.yaml"


def structure(args, directory: Path) -> tuple[BenchmarkSpec, Path]:
    """The task's specification, derived once per sweep directory from the RelBench cache.

    The derivation engine reads the database censored at the validation cutoff,
    so every trial and the refit train the structure the harness's inner split
    would present to a model.
    """
    dataset, task = args.task.split("/")
    manifest = directory / "derived" / dataset / task / "benchmark.yaml"
    if not manifest.exists():
        argv = [
            args.python, "-m", "benchmarks.relarena.derive",
            "--relbench", args.task, "--cache", str(args.cache), "--output", str(directory / "derived"),
        ]  # fmt: skip
        subprocess.run(argv, check=True, cwd=args.source, env=args.environment)
    return BenchmarkSpec.from_yaml(manifest), manifest


def budget(args) -> dict:
    """Full passes and the wall-clock backstop for one trial: explicit options first, then the plan's tier."""
    planned = getattr(args, "budget", None) or {}
    passes = getattr(args, "passes", None) or planned.get("passes") or -1
    seconds = getattr(args, "trial_seconds", None) or planned.get("backstop_seconds") or 6 * 3600.0
    return {"passes": passes, "backstop_seconds": seconds}


def command(
    args, config: dict, bundle_spec: Path, directory: Path, prepared: Path | None, *, seed: int | None = None
) -> list[str]:
    """The supervised runner's command for one trial: full splits, plateau schedule, early stops, no test access."""
    result = [
        args.python,
        "-m",
        "benchmarks.relarena.supervised",
        "--benchmark",
        args.task,
        "--spec",
        str(bundle_spec),
        "--work-dir",
        str(directory / "work"),
        "--output",
        str(directory / "result.json"),
        "--relbench-cache",
        str(args.cache),
        "--accelerator",
        "gpu",
        "--precision",
        "bf16-mixed",
        "--seed",
        str(args.seed if seed is None else seed),
        "--batch-size",
        "auto",
        "--max-batch-size",
        str(config["batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--fused",
        "--min-full-passes",
        str(args.min_full_passes),
        "--max-epochs",
        str(budget(args)["passes"]),
        "--max-steps",
        str(args.max_steps),
        "--val-every",
        str(max(1, min(args.val_every, math.ceil(args.training_rows / config["batch_size"])))),
        "--patience",
        str(args.patience),
        "--plateau-factor",
        str(args.plateau_factor),
        "--plateau-patience",
        str(args.plateau_patience),
        "--max-lr-drops",
        str(args.max_lr_drops),
        "--min-improvement",
        str(args.min_improvement),
        "--max-train-seconds",
        str(budget(args)["backstop_seconds"]),
        "--train-rows",
        "0",
        "--validation-rows",
        "0",
        "--run-label",
        f"sweep-{args.name}",
        "--duckdb-memory-limit",
        args.duckdb_memory_limit,
        "--num-workers",
        str(getattr(args, "num_workers", 0)),
    ]
    if config.get("ema_decay") is not None:
        result.extend(["--ema-decay", str(config["ema_decay"])])
    if prepared is not None:
        result.extend(["--prepared-from", str(prepared)])
    if args.tracking_uri:
        result.extend(["--tracking-uri", args.tracking_uri, "--experiment", args.experiment])
    return result


def summarize(report: dict) -> dict:
    """The figures a trial contributes to the leaderboard.

    The supervised runner nests its training report under ``result``; a flat
    report is read as is.
    """
    result = report["result"] if isinstance(report.get("result"), dict) else report
    stage = result.get("stages", {}).get("supervised", {})
    selection = result.get("selection", {})
    drops = (result.get("plateau") or {}).get("drops", 0)
    return {
        "status": report.get("status"),
        "batch_size": chosen(report),
        "metric": selection.get("metric"),
        "score": selection.get("score"),
        "selected_step": selection.get("step"),
        "steps": stage.get("steps"),
        "equivalent_full_epochs": stage.get("equivalent_full_epochs"),
        "seconds": stage.get("seconds"),
        "stopping_reason": result.get("stopping_reason"),
        "lr_drops": drops if isinstance(drops, int) else len(drops),
        "peak_cuda_gb": (stage.get("peak_cuda_allocated_bytes") or 0) / 1e9,
        "baseline": result.get("baseline"),
        "validation": result.get("validation"),
    }


def leaderboard(state: dict) -> str:
    """Markdown: every trial ordered by score in the metric's direction, failures last."""
    trials = state["trials"]
    mode = next((t["summary"]["metric"] for t in trials if t.get("summary", {}).get("metric")), None)
    maximize = mode == "roc_auc"
    done = [t for t in trials if t["status"] == "done" and t.get("summary", {}).get("score") is not None]
    done.sort(key=lambda t: t["summary"]["score"], reverse=maximize)
    lines = [
        f"# Sweep {state['name']}: {state['task']} ({state['tier']} tier, seed {state['seed']})",
        "",
        f"{len(done)} scored of {len(trials)} trials; metric {mode or '?'} "
        f"({'higher' if maximize else 'lower'} is better). Validation-only selection; the test split is never read.",
        "",
        "| rank | trial | seed | score | test | d_model | layers | heads | ffnx | dropout | batch | lr | wd | ema | jitter | mask | loss | steps | passes | drops | stop | minutes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    ranked = groups(state)
    if any(len(g["members"]) > 1 for g in ranked):
        lines = (
            lines[:4]
            + [
                "| rank | configuration | seeds | mean score | spread | mean test | d_model | layers | dropout | lr | wd | ema | jitter | mask | loss |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                *(
                    f"| {rank} | {g['config_id']} | {len(g['members'])} | {g['score']:.5f} | {g['spread']:.5f} | "
                    f"{'' if g['test'] is None else f'{g["test"]:.5f}'} | {g['config']['d_model']} | {g['config']['n_layers']} | "
                    f"{g['config']['dropout']} | {g['config']['learning_rate']:.2e} | {g['config']['weight_decay']} | {g['config']['ema_decay'] or '-'} | "
                    f"{g['config'].get('jitter', 0.0)} | {g['config'].get('mask', 0.0)} | {g['config'].get('objective', 'mae')} |"
                    for rank, g in enumerate(ranked, start=1)
                ),
                "",
            ]
            + lines[4:]
        )
    for rank, trial in enumerate(done, start=1):
        c, s = trial["config"], trial["summary"]
        test = ((trial.get("refit") or {}).get("test") or {}).get(s["metric"])
        lines.append(
            f"| {rank} | {trial['index']} | {trial.get('seed', state['seed'])} | {s['score']:.5f} | {'' if test is None else f'{test:.5f}'} | "
            f"{c['d_model']} | {c['n_layers']} | {c['n_heads']} | {c['ffnx']} | "
            f"{c['dropout']} | {s.get('batch_size') or c['batch_size']} | {c['learning_rate']:.2e} | {c['weight_decay']} | {c['ema_decay'] or '-'} | "
            f"{c.get('jitter', 0.0)} | {c.get('mask', 0.0)} | {c.get('objective', 'mae')} | "
            f"{s['steps'] or 0} | {(s['equivalent_full_epochs'] or 0):.1f} | {s['lr_drops']} | {s['stopping_reason'] or ''} | "
            f"{(s['seconds'] or 0) / 60:.0f} |"
        )
    pending = [t for t in trials if t["status"] in ("pending", "running")]
    failed = [t for t in trials if t["status"] == "failed"]
    if pending:
        lines.append(f"\n{len(pending)} trials pending.")
    if failed:
        lines.append("\nFailed: " + ", ".join(f"trial {t['index']} ({t.get('error', '')[:80]})" for t in failed))
    return "\n".join(lines) + "\n"


def load_state(directory: Path, plan_data: dict, name: str) -> dict:
    path = directory / "state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["plan_sha256"] != plan_data["sha256"]:
            raise ValueError("Existing sweep state belongs to a different plan")
        for trial in state["trials"]:
            if trial["status"] == "running":
                trial["status"] = "pending"
        return state
    return {
        "version": 1,
        "name": name,
        "task": plan_data["task"],
        "tier": plan_data["tier"],
        "seed": plan_data["seed"],
        "plan_sha256": plan_data["sha256"],
        "trials": [
            {
                "index": t["index"],
                "config": t["config"],
                "status": "pending",
                **{k: t[k] for k in ("config_id", "seed") if k in t},
            }
            for t in plan_data["trials"]
        ],
    }


def save(directory: Path, state: dict) -> None:
    pending = directory / "state.pending.json"
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(directory / "state.json")
    (directory / "README.md").write_text(leaderboard(state))


def report(directory: Path) -> str:
    """Recompute every finished trial's summary from its result file and rewrite the leaderboard."""
    state = json.loads((directory / "state.json").read_text())
    for trial in state["trials"]:
        path = Path(trial.get("result") or "")
        if trial["status"] in ("done", "failed") and path.exists():
            trial["summary"] = summarize(json.loads(path.read_text()))
            trial["status"] = "done" if trial["summary"]["status"] == "passed" else "failed"
    save(directory, state)
    return leaderboard(state)


def prune(directory: Path) -> None:
    """Keep each trial's best checkpoint, result, and logs; drop the rest."""
    for name in ("last.ckpt", "final.ckpt", "last.pending.ckpt"):
        path = directory / "work/checkpoints" / name
        if path.exists():
            path.unlink()


BATCH_FLOOR = 8


def child_error(report: dict | None) -> dict | str | None:
    """A failed child's reported error, parsed from the string form older reports carry."""
    error = (report or {}).get("error")
    if isinstance(error, str):
        try:
            return ast.literal_eval(error)
        except (ValueError, SyntaxError):
            return error
    return error


def chosen(report: dict | None) -> int | None:
    """The batch size an attempt trained with, when its report's config recorded the finder's pick."""
    settings = (report or {}).get("config")
    settings = settings if isinstance(settings, dict) else (report or {})
    size = (settings.get("training") or {}).get("batch_size")
    return size if isinstance(size, int) else None


def failure(report: dict | None, code: int) -> str:
    """Name a failed child's error from its report, or from its exit alone."""
    error = child_error(report)
    if isinstance(error, dict):
        return f"exit {code}, {error.get('type', 'error')}: {str(error.get('message', ''))[:80]}"
    return f"exit {code}, {str(error)[:80]}" if error else f"exit {code}, no result"


def exhausted(report: dict | None, log_path: Path) -> bool:
    """Whether a failed child ran out of device memory, judged from its report or else its log tail.

    Online vocabularies and observed collection widths grow memory while a
    trial trains, so a batch that fit at the first step can exceed the device
    later; such a trial is worth another attempt at half the batch size.
    """
    error = child_error(report)
    text = f"{error.get('type', '')} {error.get('message', '')}" if isinstance(error, dict) else str(error or "")
    if not text.strip() and log_path.exists():
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 65536))
            text = log.read().decode("utf-8", "replace")
    return "OutOfMemoryError" in text or "out of memory" in text.lower()


def execute(args, argv: list[str], directory: Path, log_path: Path) -> int:
    """Run one child under the host's GPU lock, logging to ``log_path``.

    The child works inside its own directory, which the sweep owns for the
    child's whole life. A candidate's source tree may be rebuilt meanwhile, and
    a process whose working directory has vanished fails the next time anything
    asks for it, as Lightning does whenever it builds a trainer.
    """
    with (directory / "gpu.lock").open("a") as lock, log_path.open("w") as log:
        if args.gpu_lock:
            lock = open(args.gpu_lock, "a")  # noqa: SIM115 - held for the child's duration
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return subprocess.run(
                argv, cwd=log_path.parent, env=args.environment, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        finally:
            if args.gpu_lock:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()


def groups(state: dict) -> list[dict]:
    """Scored trials grouped by configuration: mean score and selected step over seeds, ranked best first."""
    done = [t for t in state["trials"] if t["status"] == "done" and t.get("summary", {}).get("score") is not None]
    if not done:
        return []
    maximize = done[0]["summary"]["metric"] == "roc_auc"
    by: dict[str, list[dict]] = {}
    for trial in done:
        by.setdefault(trial.get("config_id") or identity(trial["config"]), []).append(trial)
    out = []
    for key, members in by.items():
        members.sort(key=lambda t: t.get("seed", 0))
        scores = [t["summary"]["score"] for t in members]
        tests = [((t.get("refit") or {}).get("test") or {}).get(t["summary"]["metric"]) for t in members]
        tests = [v for v in tests if v is not None]
        out.append(
            {
                "config_id": key,
                "config": members[0]["config"],
                "members": members,
                "score": sum(scores) / len(scores),
                "spread": max(scores) - min(scores),
                "selected_step": round(sum(t["summary"]["selected_step"] or 0 for t in members) / len(members)),
                "test": sum(tests) / len(tests) if tests else None,
            }
        )
    out.sort(key=lambda g: g["score"], reverse=maximize)
    return out


def winner(state: dict) -> dict | None:
    """The trial that represents the best configuration: its lowest seed, carrying the seeds' mean score and step."""
    ranked = groups(state)
    if not ranked:
        return None
    best = ranked[0]
    representative = best["members"][0]
    return {
        **representative,
        "summary": {**representative["summary"], "score": best["score"], "selected_step": best["selected_step"]},
        "seeds": len(best["members"]),
    }


def refit(args, state: dict, directory: Path, trial: dict | None = None) -> dict | None:
    """Train a trial's configuration on the outer split for its tuned number of updates and score the test anchors.

    Without ``trial`` this refits the leaderboard winner into ``refit/`` and records it on the
    state; with one it refits that trial into its own directory and records it on the trial, so
    a sweep can show how every configuration converts from validation to test.
    """
    from benchmarks.relarena.refit import scaled

    best = trial or winner(state)
    if best is None:
        print(
            json.dumps({"sweep": args.name, "event": "refit", "status": "skipped", "reason": "no scored trial"}),
            flush=True,
        )
        return None
    report = json.loads(Path(best["result"]).read_text())
    validation_rows = int((report.get("data") or {}).get("validation_rows") or 0)
    steps = scaled(int(best["summary"]["selected_step"]), int(args.training_rows), validation_rows)
    config = best["config"]
    refit_dir = directory / "refit" if trial is None else directory / f"trial-{best['index']:03d}" / "refit"
    holder = state if trial is None else best
    if refit_dir.exists():
        shutil.rmtree(refit_dir)
    spec, manifest = structure(args, directory)
    bundle_spec = bundle(spec, config, refit_dir / "bundle", manifest)
    # The trial's report carries the size it trained with; the plan only carries the finder's cap.
    trained = ((report.get("config") or {}).get("model") or {}).get("batch_size")
    batch = int(best["summary"].get("batch_size") or trained or config["batch_size"])
    record = {
        "trial": best["index"],
        "config": config,
        "steps": steps,
        "regime": args.refit_regime,
        "started_at": time.time(),
        "attempts": [],
    }
    print(json.dumps({"sweep": args.name, "event": "refit", "trial": best["index"], "steps": steps}), flush=True)
    while True:
        argv = [
            args.python,
            "-m",
            "benchmarks.relarena.refit",
            "--benchmark",
            args.task,
            "--spec",
            str(bundle_spec),
            "--work-dir",
            str(refit_dir / "work"),
            "--output",
            str(refit_dir / "result.json"),
            "--steps",
            str(steps),
            "--batch-size",
            str(batch),
            "--learning-rate",
            str(config["learning_rate"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--seed",
            str(args.seed),
            "--accelerator",
            "gpu",
            "--precision",
            "bf16-mixed",
            "--duckdb-memory-limit",
            args.duckdb_memory_limit,
            "--run-label",
            f"refit-{args.name}" if trial is None else f"refit-{args.name}-{best['index']}",
            "--regime",
            args.refit_regime,
        ]
        if args.refit_regime == "full":
            # Validation rows enter the refit as tuning saw them. On single-timestamp tasks this changes
            # nothing; on rel-f1, whose anchors run years past each boundary, it is worth 0.1 MAE on test.
            argv.append("--stale-validation")
        if config.get("ema_decay") is not None:
            argv.extend(["--ema-decay", str(config["ema_decay"])])
        if args.refit_regime == "checkpoint":
            # The selection run repeats the trial's schedule on the outer store.
            argv.extend(
                [
                    "--relbench-cache", str(args.cache), "--num-workers", str(getattr(args, "num_workers", "auto")),
                    "--min-full-passes", str(args.min_full_passes), "--max-epochs", str(budget(args)["passes"]),
                    "--max-steps", str(args.max_steps),
                    "--val-every", str(max(1, min(args.val_every, math.ceil(args.training_rows / config["batch_size"])))),
                    "--patience", str(args.patience), "--plateau-factor", str(args.plateau_factor),
                    "--plateau-patience", str(args.plateau_patience), "--max-lr-drops", str(args.max_lr_drops),
                    "--min-improvement", str(args.min_improvement),
                    "--max-train-seconds", str(budget(args)["backstop_seconds"]),
                ]
            )  # fmt: skip
        record["attempts"].append({"batch_size": batch, "command": argv})
        holder["refit"] = {**record, "status": "running"}
        save(directory, state)
        code = execute(args, argv, directory, refit_dir / "run.log")
        result_path = refit_dir / "result.json"
        outcome = json.loads(result_path.read_text()) if result_path.exists() else {}
        # The outer split holds more rows and wider collections than any trial saw, so the
        # trained batch can still exceed the card; retry at half, as the trials do.
        if code != 0 and outcome.get("status") != "passed" and batch > 1 and exhausted(outcome, refit_dir / "run.log"):
            print(json.dumps({"sweep": args.name, "event": "refit", "relaunch": batch // 2}), flush=True)
            batch //= 2
            shutil.rmtree(refit_dir / "work", ignore_errors=True)
            result_path.unlink(missing_ok=True)
            continue
        break
    holder["refit"] = {
        **record,
        "batch_size": batch,
        "finished_at": time.time(),
        "exit": code,
        "result": str(result_path),
        "status": "done" if code == 0 and outcome.get("status") == "passed" else "failed",
        "test": outcome.get("test"),
        "error": (outcome.get("error") or {}).get("message") if outcome.get("status") != "passed" else None,
    }
    save(directory, state)
    print(
        json.dumps(
            {
                "sweep": args.name,
                "event": "refit",
                "trial": best["index"],
                **{k: holder["refit"][k] for k in ("status", "test", "error")},
            }
        ),
        flush=True,
    )
    return holder["refit"]


def execute_trial(args, spec: BenchmarkSpec, manifest: Path, state: dict, directory: Path, trial: dict) -> None:
    """Run one pending trial to completion: bundle, train with the finder and OOM relaunch, summarize, refit if asked."""
    prepared = next(
        (
            Path(t["result"])
            for t in state["trials"]
            if t["status"] == "done" and Path(t["result"]).exists() and t.get("summary", {}).get("status") == "passed"
        ),
        None,
    )
    trial_dir = directory / f"trial-{trial['index']:03d}"
    config = dict(trial["config"])
    attempts: list[dict] = []
    trial.update(status="running", started_at=time.time(), prepared_from=str(prepared) if prepared else None)
    while True:
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        bundle_spec = bundle(spec, config, trial_dir / "bundle", manifest)
        argv = command(args, config, bundle_spec, trial_dir, prepared, seed=trial.get("seed"))
        trial.update(config=config, command=argv)
        save(directory, state)
        print(
            json.dumps({"sweep": args.name, "event": "trial", "index": trial["index"], "config": config}),
            flush=True,
        )
        code = execute(args, argv, directory, trial_dir / "run.log")
        result_path = trial_dir / "result.json"
        report = json.loads(result_path.read_text()) if result_path.exists() else None
        trained = chosen(report) or config["batch_size"]
        if code == 0 or not exhausted(report, trial_dir / "run.log") or trained // 2 < BATCH_FLOOR:
            break
        # The device ran out of memory: keep this attempt's log and start over, capping the finder
        # at half the size this attempt trained with.
        kept = directory / f"trial-{trial['index']:03d}-oom-{trained}"
        shutil.rmtree(kept, ignore_errors=True)
        prune(trial_dir)
        trial_dir.rename(kept)
        attempts.append(
            {
                "cap": config["batch_size"],
                "batch_size": trained,
                "exit": code,
                "error": failure(report, code),
                "log": str(kept / "run.log"),
            }
        )
        config["batch_size"] = trained // 2
        print(
            json.dumps(
                {
                    "sweep": args.name,
                    "event": "relaunch",
                    "index": trial["index"],
                    "batch_size": config["batch_size"],
                }
            ),
            flush=True,
        )
    trial.update(finished_at=time.time(), exit=code, result=str(result_path), attempts=attempts)
    if code == 0 and report is not None:
        trial["summary"] = summarize(report)
        trial["status"] = "done" if trial["summary"]["status"] == "passed" else "failed"
        if trial["status"] == "failed":
            error = report.get("error")
            trial["error"] = (
                error.get("message", str(error)) if isinstance(error, dict) else str(error or report.get("status"))
            )
    else:
        trial["status"] = "failed"
        trial["error"] = f"{failure(report, code)}; see {trial_dir / 'run.log'}"
    prune(trial_dir)
    save(directory, state)
    print(
        json.dumps(
            {
                "sweep": args.name,
                "event": "result",
                "index": trial["index"],
                "status": trial["status"],
                "summary": trial.get("summary"),
            }
        ),
        flush=True,
    )
    if trial["status"] == "done" and getattr(args, "refit_every_trial", False):
        refit(args, state, directory, trial=trial)
        save(directory, state)


def run(args) -> int:
    import hashlib

    plan_path = Path(args.plan)
    plan_data = json.loads(plan_path.read_text())
    plan_data["sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    args.task, args.training_rows = plan_data["task"], plan_data["training_rows"]
    args.budget = plan_data.get("budget")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    spec, manifest = structure(args, directory)
    state = load_state(directory, plan_data, args.name)
    save(directory, state)
    started = time.monotonic()
    stopped = False
    for trial in state["trials"]:
        if trial["status"] != "pending":
            continue
        if (directory / "STOP").exists():
            print(json.dumps({"sweep": args.name, "event": "stopped", "reason": "STOP file"}), flush=True)
            stopped = True
            break
        if args.max_hours and time.monotonic() - started > args.max_hours * 3600:
            print(json.dumps({"sweep": args.name, "event": "stopped", "reason": "max hours"}), flush=True)
            break
        execute_trial(args, spec, manifest, state, directory, trial)
    state["finished_at"] = time.time()
    save(directory, state)
    if not stopped and not getattr(args, "no_refit", False):
        best = winner(state)
        if best is not None and (best.get("refit") or {}).get("status") == "done":
            state["refit"] = best["refit"]  # already converted trial by trial
            save(directory, state)
        else:
            refit(args, state, directory)
    return 0


def smoothed(summary: dict, k: int = 3) -> float | None:
    """A trial's objective: the mean of its k best validation evaluations, not its single best checkpoint."""
    history = summary.get("validation") or []
    metric = summary.get("metric")
    values = [float(entry[metric]) for entry in history if isinstance(entry, dict) and entry.get(metric) is not None]
    if not values:
        return summary.get("score")
    values.sort(reverse=metric == "roc_auc")
    top = values[: max(1, min(k, len(values)))]
    return sum(top) / len(top)


def distributions(space: dict):
    """Optuna distributions for a tier space: categorical choices, a log-uniform learning rate."""
    import optuna

    out = {}
    for name, choices in space.items():
        if name == "learning_rate":
            out[name] = optuna.distributions.FloatDistribution(float(choices[0]), float(choices[1]), log=True)
        else:
            out[name] = optuna.distributions.CategoricalDistribution(list(choices))
    return out


def within(config: dict, space: dict) -> bool:
    for name, choices in space.items():
        value = config.get(name, DEFAULT.get(name))
        if name == "learning_rate":
            if not (float(choices[0]) <= float(value) <= float(choices[1])):
                return False
        elif value not in choices:
            return False
    return True


def optimize(args) -> int:
    """Optuna over the task's tier space: validation-only objective, every trial refit and tested when asked.

    The study lives in ``optuna.db`` beside the sweep state and resumes with it. Earlier
    sweeps of the same task warm-start the sampler through ``--warm-start``; only their
    configurations inside the current space are admitted.
    """
    import hashlib

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    plan_path = Path(args.plan)
    plan_data = json.loads(plan_path.read_text())
    plan_data["sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    args.task, args.training_rows = plan_data["task"], plan_data["training_rows"]
    args.budget = plan_data.get("budget")
    level = plan_data["tier"]
    # The plan carries the space its trials were sampled from; knobs it predates come from the tier.
    space = {**SPACES[level], **{name: tuple(choices) for name, choices in plan_data.get("space", {}).items()}}
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    spec, manifest = structure(args, directory)
    state = load_state(directory, plan_data, args.name)
    save(directory, state)
    first = state["trials"][0]["config"] if state["trials"] else None
    dists = distributions(space)
    direction = None
    for trial in state["trials"]:
        metric = (trial.get("summary") or {}).get("metric")
        if metric:
            direction = "maximize" if metric == "roc_auc" else "minimize"
            break
    if direction is None:
        direction = "maximize" if plan_data.get("metric") == "roc_auc" else args.direction
    study = optuna.create_study(
        study_name=args.name,
        storage=f"sqlite:///{directory / 'optuna.db'}",
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=plan_data["seed"], n_startup_trials=args.startup, multivariate=True),
        load_if_exists=True,
    )
    known = {t.user_attrs.get("origin") for t in study.trials}

    def admit(config: dict, summary: dict, origin: str) -> None:
        if origin in known or not within(config, space):
            return
        value = smoothed(summary)
        if value is None:
            return
        params = {name: config.get(name, DEFAULT[name]) for name in space}
        study.add_trial(
            optuna.trial.create_trial(
                params=params, distributions=dists, value=float(value), user_attrs={"origin": origin}
            )
        )
        known.add(origin)

    for path in args.warm_start or []:
        other = json.loads(Path(path).read_text())
        for trial in other["trials"]:
            if trial.get("status") == "done" and (trial.get("summary") or {}).get("score") is not None:
                admit(trial["config"], trial["summary"], f"{other['name']}:{trial['index']}")
    for trial in state["trials"]:
        if trial.get("status") == "done" and (trial.get("summary") or {}).get("score") is not None:
            admit(trial["config"], trial["summary"], f"{state['name']}:{trial['index']}")

    started = time.monotonic()
    for trial in state["trials"]:
        if trial["status"] == "pending":
            execute_trial(args, spec, manifest, state, directory, trial)
            if trial["status"] == "done":
                admit(trial["config"], trial["summary"], f"{state['name']}:{trial['index']}")
    proposed = sum(1 for t in state["trials"] if t.get("optuna") is not None)
    while proposed < args.trials:
        if (directory / "STOP").exists():
            print(json.dumps({"sweep": args.name, "event": "stopped", "reason": "STOP file"}), flush=True)
            break
        if args.max_hours and time.monotonic() - started > args.max_hours * 3600:
            print(json.dumps({"sweep": args.name, "event": "stopped", "reason": "max hours"}), flush=True)
            break
        asked = study.ask(dists)
        config = {**DEFAULT, **(first or {}), **asked.params}
        config["batch_size"] = space["batch_size"][0]
        if config["d_model"] % config["n_heads"]:
            config["n_heads"] = max(head for head in space["n_heads"] if config["d_model"] % head == 0)
        seeds = args.seeds or plan_data.get("seeds") or SEEDS[level]
        members = [
            {"index": len(state["trials"]) + repeat, "config": config, "config_id": identity(config), "seed": plan_data["seed"] + repeat, "status": "pending", "optuna": asked.number}
            for repeat in range(seeds)
        ]  # fmt: skip
        state["trials"].extend(members)
        save(directory, state)
        proposed += 1
        values = []
        for trial in members:
            execute_trial(args, spec, manifest, state, directory, trial)
            if trial["status"] == "done" and smoothed(trial["summary"]) is not None:
                values.append(float(smoothed(trial["summary"])))
        trial = members[0]
        if values:
            study.tell(asked, sum(values) / len(values))
        else:
            study.tell(asked, state=optuna.trial.TrialState.FAIL)
        best = study.best_trial if any(t.state.is_finished() and t.values for t in study.trials) else None
        print(
            json.dumps(
                {
                    "sweep": args.name,
                    "event": "optuna",
                    "trial": trial["index"],
                    "objective": sum(values) / len(values) if values else None,
                    "seeds": len(values),
                    "best": None if best is None else {"value": best.value, "params": best.params},
                    "study_trials": len(study.trials),
                }
            ),
            flush=True,
        )
    state["finished_at"] = time.time()
    save(directory, state)
    if not getattr(args, "no_refit", False):
        best = winner(state)
        if best is not None and (best.get("refit") or {}).get("status") == "done":
            state["refit"] = best["refit"]
            save(directory, state)
        else:
            refit(args, state, directory)
    (directory / "README.md").write_text(leaderboard(state))
    return 0


def refit_only(args) -> bool:
    """Refit a sweep that has already run, without admitting any pending trial."""
    import hashlib

    plan_path = Path(args.plan)
    plan_data = json.loads(plan_path.read_text())
    plan_data["sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    args.task, args.training_rows = plan_data["task"], plan_data["training_rows"]
    directory = Path(args.output)
    state = load_state(directory, plan_data, args.name)
    record = refit(args, state, directory)
    return record is not None and record["status"] == "done"


def rows(cache: Path, task: str) -> int:
    """Training anchors of a task, from the RelBench cache."""
    dataset, name = task.split("/")
    return pq.ParquetFile(cache / dataset / "tasks" / name / "train.parquet").metadata.num_rows


def queue(args) -> int:
    """Sweep tasks one after another, smallest training split first, skipping finished ones."""
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    tasks = sorted(args.tasks or TASKS, key=lambda task: rows(args.cache, task))
    for task in tasks:
        slug = task.replace("/", "-")
        directory = root / f"raw-sweep-{slug}"
        if (root / "STOP").exists():
            print(json.dumps({"queue": str(root), "event": "stopped", "reason": "STOP file"}), flush=True)
            return 0
        if (directory / "DONE").exists():
            continue
        plan_path = directory / "plan.json"
        if not plan_path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(plan(task, rows(args.cache, task), args.trials, args.seed), indent=2) + "\n"
            )
        print(json.dumps({"queue": str(root), "event": "task", "task": task}), flush=True)
        task_args = SimpleNamespace(**vars(args), plan=str(plan_path), name=f"raw-sweep-{slug}", output=str(directory))
        run(task_args)
        if (directory / "STOP").exists():
            return 0
        (directory / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    return 0


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="Write a reproducible trial plan for one task")
    planner.add_argument("--task", required=True, choices=TASKS)
    planner.add_argument("--cache", type=Path, required=True, help="RelBench cache holding the task's train split")
    planner.add_argument("--trials", type=int, default=24)
    planner.add_argument("--seed", type=int, default=0)
    planner.add_argument("--seeds", type=int, help="Seeds per configuration; default is the tier's count")
    planner.add_argument(
        "--space",
        type=json.loads,
        help="JSON object narrowing the tier's choices per parameter, such as '{\"ema_decay\": [null]}'",
    )
    planner.add_argument("--default", type=json.loads, help="JSON object overriding the first trial's configuration")
    planner.add_argument("--output", type=Path, required=True)
    reporter = commands.add_parser("report", help="Rebuild summaries and the leaderboard from finished trials")
    reporter.add_argument("--output", required=True, help="Sweep directory holding state.json and trial-NNN/")
    execution = argparse.ArgumentParser(add_help=False)
    execution.add_argument("--python", default=sys.executable)
    execution.add_argument("--source", required=True, help="Benchmark source tree whose catalog the trials run")
    execution.add_argument("--cache", type=Path, required=True)
    execution.add_argument("--seed", type=int, default=0)
    execution.add_argument("--passes", type=int, help="Full passes per trial; default is the plan's tier budget")
    execution.add_argument("--trial-seconds", type=float, help="Wall-clock backstop per trial; default is the tier's")
    execution.add_argument(
        "--max-hours", type=float, default=0, help="Stop admitting trials after this many hours; 0 runs the plan"
    )
    execution.add_argument("--min-full-passes", type=int, default=2)
    execution.add_argument("--max-steps", type=int, default=200000)
    execution.add_argument("--val-every", type=int, default=1000)
    execution.add_argument("--patience", type=int, default=12)
    execution.add_argument("--plateau-factor", type=float, default=0.1)
    execution.add_argument("--plateau-patience", type=int, default=4)
    execution.add_argument("--num-workers", default="auto", help="Loader processes per trial, or auto by schema")
    execution.add_argument(
        "--refit-every-trial",
        action="store_true",
        help="Refit and test every scored trial as it finishes, to see how validation converts to test",
    )
    execution.add_argument(
        "--refit-regime",
        choices=("full", "checkpoint"),
        default="full",
        help="full: train+val for the carried-over updates; checkpoint: the validation-selected checkpoint on the outer store",
    )
    execution.add_argument("--max-lr-drops", type=int, default=2)
    execution.add_argument("--min-improvement", type=float, default=0.001)
    execution.add_argument("--duckdb-memory-limit", default="16GB")
    execution.add_argument("--gpu-lock", help="flock file serializing GPU work on the host")
    execution.add_argument("--tracking-uri")
    execution.add_argument("--experiment", default="relflow-relarena-sweep")
    execution.add_argument("--no-refit", action="store_true", help="Skip the outer-split refit after the sweep")

    runner = commands.add_parser("run", parents=[execution], help="Execute a plan's pending trials in order")
    runner.add_argument("--plan", required=True)
    runner.add_argument("--name", required=True)
    runner.add_argument("--output", required=True, help="Sweep directory holding state, trials, and the leaderboard")
    optimizer = commands.add_parser(
        "optimize", parents=[execution], help="Optuna over the tier space, validation objective"
    )
    optimizer.add_argument(
        "--plan", required=True, help="A plan whose first trial is the default; Optuna adds the rest"
    )
    optimizer.add_argument("--name", required=True)
    optimizer.add_argument("--output", required=True)
    optimizer.add_argument("--trials", type=int, default=24, help="Optuna proposals to run beyond the plan's trials")
    optimizer.add_argument("--startup", type=int, default=8, help="Random proposals before the TPE sampler takes over")
    optimizer.add_argument("--seeds", type=int, help="Seeds per proposal; default is the plan's count")
    optimizer.add_argument("--warm-start", nargs="*", help="state.json files of earlier sweeps of this task")
    optimizer.add_argument(
        "--direction",
        choices=("maximize", "minimize"),
        default="minimize",
        help="Used before any scored trial reveals the metric",
    )
    refitter = commands.add_parser(
        "refit", parents=[execution], help="Refit a finished sweep's winner on the outer split"
    )
    refitter.add_argument("--plan", required=True)
    refitter.add_argument("--name", required=True)
    refitter.add_argument("--output", required=True, help="Sweep directory holding state.json and trial-NNN/")
    queued = commands.add_parser("queue", parents=[execution], help="Sweep tasks in sequence, smallest first")
    queued.add_argument("--runs", required=True, help="Directory receiving one sweep directory per task")
    queued.add_argument("--tasks", nargs="*", choices=TASKS, help="Default: every RelBench task")
    queued.add_argument("--trials", type=int, default=24)
    args = parser.parse_args(argv)
    if args.command in ("run", "queue", "refit", "optimize"):
        import os

        # Children import the benchmark source first and keep whatever the caller already put on the path,
        # such as a RelFlow checkout mounted ahead of the installed release.
        inherited = os.environ.get("PYTHONPATH", "")
        path = os.pathsep.join(part for part in (str(Path(args.source).resolve()), inherited) if part)
        args.environment = {**os.environ, "PYTHONPATH": path, "PYTHONUNBUFFERED": "1"}
        if (args.trial_seconds is not None and args.trial_seconds <= 0) or (
            args.passes is not None and args.passes <= 0
        ):
            parser.error("--trial-seconds and --passes must be positive when given")
        if args.min_full_passes < 0 or args.val_every <= 0 or args.patience <= 0:
            parser.error("Trial budgets must be positive")
    return args


def main(argv=None) -> int:
    args = arguments(argv)
    if args.command == "plan":
        count = rows(args.cache, args.task)
        data = plan(args.task, count, args.trials, args.seed, space=args.space, default=args.default, seeds=args.seeds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(
            json.dumps({"task": args.task, "tier": data["tier"], "training_rows": count, "trials": len(data["trials"])})
        )
        return 0
    if args.command == "report":
        print(report(Path(args.output)), end="")
        return 0
    if args.command == "queue":
        return queue(args)
    if args.command == "refit":
        return 0 if refit_only(args) else 1
    if args.command == "optimize":
        return optimize(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
