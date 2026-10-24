"""The RelBench v1 entity tasks RelArena scores, and the supervised target each one binds."""

from __future__ import annotations

from benchmarks.relarena.spec import BenchmarkSpec

# The 21 entity-level tasks of RelBench v1, as RelArena's leaderboard lists them.
TASKS: tuple[str, ...] = (
    "rel-amazon/item-churn",
    "rel-amazon/item-ltv",
    "rel-amazon/user-churn",
    "rel-amazon/user-ltv",
    "rel-avito/ad-ctr",
    "rel-avito/user-clicks",
    "rel-avito/user-visits",
    "rel-event/user-attendance",
    "rel-event/user-ignore",
    "rel-event/user-repeat",
    "rel-f1/driver-dnf",
    "rel-f1/driver-position",
    "rel-f1/driver-top3",
    "rel-hm/item-sales",
    "rel-hm/user-churn",
    "rel-stack/post-votes",
    "rel-stack/user-badge",
    "rel-stack/user-engagement",
    "rel-trial/site-success",
    "rel-trial/study-adverse",
    "rel-trial/study-outcome",
)


def supervise(spec: BenchmarkSpec, model: dict) -> dict:
    """Bind the task's masked scalar label without changing its feature schema."""
    target = {"type": "boolean" if spec.task_type == "binary_classification" else "number", "mask": True}
    existing = model["fields"].get("target") or {}
    if existing and {key: existing.get(key) for key in target} != target:
        raise ValueError(f"Benchmark {spec.benchmark_name} target must be the supervised {target!r}; got {existing!r}")
    # Other declared target options ride along; type and mask stay the task's contract.
    return {**model, "fields": {**model["fields"], "target": {**existing, **target}}}
