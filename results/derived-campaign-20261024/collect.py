"""Pull every run's durable records into results/derived-campaign-20261024/data/.

Sources: the Modal Volume (sweep state, trial and refit reports, leaderboards, probe reports),
the lab (sweep state and refit reports over SSH), and optionally a scratch directory holding
derived tables (conversion dataset, loop-table history, schema diffs). Re-run to refresh.

    uv run python results/derived-campaign-20261024/collect.py [--scratch DIR] [--lab HOST]
"""

import argparse
import json
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
VOLUME = "relflow-sweeps"


def modal(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", "run", "--no-sync", "modal", *args], capture_output=True, text=True, cwd=str(HERE.parents[1]))


def listing(path: str) -> list[str]:
    out = modal("volume", "ls", VOLUME, path).stdout
    return [line.strip() for line in out.splitlines() if line.strip().startswith(path.split("/")[0])]


def fetch(remote: str, local: pathlib.Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    return modal("volume", "get", "--force", VOLUME, remote, str(local)).returncode == 0


def pull_runs() -> int:
    copied = 0
    for entry in listing("runs"):
        run = entry.split("/")[-1]
        if not run.startswith("modal-"):
            continue
        target = DATA / "modal" / run
        for name in ("state.json", "plan.json", "README.md", "sweep.log"):
            copied += fetch(f"runs/{run}/{name}", target / name)
        for trial in listing(f"runs/{run}"):
            leaf = trial.split("/")[-1]
            if leaf.startswith("trial-") or leaf == "refit":
                copied += fetch(f"runs/{run}/{leaf}/result.json", target / leaf / "result.json")
                copied += fetch(f"runs/{run}/{leaf}/refit/result.json", target / leaf / "refit" / "result.json")
    for probe in ("loader-probe",):
        for entry in listing(f"runs/{probe}"):
            label = entry.split("/")[-1]
            for item in listing(f"runs/{probe}/{label}"):
                leaf = item.split("/")[-1]
                for name in ("report.json", "result.json", "micro.json"):
                    copied += fetch(f"runs/{probe}/{label}/{leaf}/{name}", DATA / probe / label / leaf / name)
    return copied


def pull_lab(host: str) -> int:
    copied = 0
    runs = subprocess.run(["ssh", "-o", "ConnectTimeout=15", host, "ls -d ~/relflow-research/runs/raw-sweep-* ~/relflow-research/runs/optuna-* 2>/dev/null"], capture_output=True, text=True).stdout.split()
    for run in runs:
        name = run.rstrip("/").split("/")[-1]
        target = DATA / "lab" / name
        target.mkdir(parents=True, exist_ok=True)
        for pattern in ("state.json", "plan.json", "README.md", "refit/result.json", "refit2/result.json", "trial-*/result.json", "trial-*/refit/result.json"):
            subprocess.run(["rsync", "-a", "--relative", "--ignore-missing-args", f"{host}:{run}/./{pattern}", str(target) + "/"], capture_output=True)
        copied += sum(1 for _ in target.rglob("*.json"))
    return copied


def pull_scratch(scratch: pathlib.Path) -> int:
    copied = 0
    for src, dst in (
        ("modal/conversion.json", "tables/conversion.json"),
        ("modal/campaign/earlier.json", "tables/earlier-best-kapso.json"),
        ("modal/campaign/spaces.json", "tables/narrowed-spaces-20261024.json"),
        ("modal/campaign/history", "tables/loop-table-history"),
        ("modal/loader-probe", "probes/loader"),
        ("modal/refit-check", "probes/refit-check"),
        ("schema-diff", "schema-diff/site-success"),
        ("modal/grid.log", "logs/grid.log"),
        ("modal/regime-check.log", "logs/regime-check.log"),
        ("modal/adctr-refits.log", "logs/adctr-refits.log"),
        ("modal/convert.log", "logs/convert.log"),
        ("modal/ablation.log", "logs/ablation.log"),
        ("relarena-docs/repo/baseline_results/results.csv", "public/relarena-results.csv"),
    ):
        source = scratch / src
        if not source.exists():
            continue
        target = DATA / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.parquet", "*.log", "*.duckdb"))
        else:
            shutil.copyfile(source, target)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=pathlib.Path)
    parser.add_argument("--lab", default=None, help="SSH host of the lab; omit to skip")
    args = parser.parse_args()
    counts = {"modal": pull_runs()}
    if args.lab:
        counts["lab"] = pull_lab(args.lab)
    if args.scratch:
        counts["scratch"] = pull_scratch(args.scratch)
    (DATA / "collected.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
