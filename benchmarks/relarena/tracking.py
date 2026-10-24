"""Optional benchmark tracking without changing RelFlow's training loggers."""

from __future__ import annotations

import hashlib
import re
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


def name(value: str) -> str:
    """Keep supported names; distinguish transformed or shortened names by digest."""
    supported = re.sub(r"[^\w .:/-]", "_", value)
    if supported == value and len(value) <= 250:
        return value
    return f"{supported[:216]}__{hashlib.sha256(value.encode()).hexdigest()[:32]}"


@dataclass
class Run:
    """Own one benchmark arm's metric buffer and completed result."""

    client: Any
    run_id: str
    pending: list[Any] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)
    published_names: int = 0

    def metric(self, key: str, value: float, timestamp: int, step: int) -> Any:
        from mlflow.entities import Metric

        encoded = name(key)
        previous = self.names.get(encoded)
        if previous is not None and previous != key:
            raise ValueError(f"MLflow metric names {previous!r} and {key!r} collide at {encoded!r}")
        self.names[encoded] = key
        return Metric(encoded, float(value), timestamp, step)

    def record(self, stage: str, record: dict[str, Any]) -> None:
        timestamp = int(time.time() * 1000)
        step = int(record["step"])
        self.pending.extend(
            self.metric(f"{stage}/{key}", float(value), timestamp, step)
            for key, value in record.items()
            if key not in {"step", "epoch"} and not key.endswith("_epoch")
        )
        if step % 25 == 0:
            self.flush()

    def flush(self) -> None:
        if len(self.names) != self.published_names and any(key != raw for key, raw in self.names.items()):
            self.client.log_dict(self.run_id, self.names.copy(), "metric-names.json")
            self.published_names = len(self.names)
        # The REST batch endpoint permits at most 1,000 metrics per request.
        while self.pending:
            batch = self.pending[:1000]
            self.client.log_batch(self.run_id, metrics=batch, synchronous=True)
            del self.pending[: len(batch)]

    def complete(self, result: dict[str, Any], baseline: dict[str, float]) -> None:
        timestamp = int(time.time() * 1000)
        metrics = [self.metric(f"validation/{key}", value, timestamp, 0) for key, value in result["validation"].items()]
        metrics.extend(self.metric(f"baseline/{key}", value, timestamp, 0) for key, value in baseline.items())
        for stage, values in result["stages"].items():
            metrics.extend(
                self.metric(f"{stage}/{key}", float(values[key]), timestamp, values["steps"])
                for key in (
                    "steps",
                    "seconds",
                    "teacher_updates",
                    "peak_cuda_allocated_bytes",
                    "peak_cuda_reserved_bytes",
                )
                if values.get(key) is not None
            )
            metrics.extend(
                self.metric(f"{stage}/parameters/{key}", value, timestamp, 0)
                for key, value in values.get("parameters", {}).items()
            )
        if result.get("jepa_eligible_fraction") is not None:
            metrics.append(self.metric("pretrain/eligible_fraction", result["jepa_eligible_fraction"], timestamp, 0))
        self.pending.extend(metrics)
        self.flush()
        for key in ("initial_encoder_digest", "initial_supervised_head_digest"):
            self.client.set_tag(self.run_id, key, result[key])
        self.client.log_dict(self.run_id, {"result": result, "baseline": baseline}, "result.json")


@contextmanager
def start(
    uri: str | None,
    experiment: str,
    name: str,
    *,
    parameters: dict[str, Any],
    tags: dict[str, Any],
    provenance: dict[str, Any],
) -> Iterator[Run | None]:
    """Track only explicitly requested runs and preserve the original failure."""
    if uri is None:
        yield None
        return
    try:
        from mlflow import MlflowClient
        from mlflow.entities import Param
    except ImportError as error:
        raise ImportError(
            "benchmark tracking requires mlflow-skinny; install benchmarks/relarena/tracking-requirements.txt"
        ) from error

    client = MlflowClient(tracking_uri=uri)
    selected = client.get_experiment_by_name(experiment)
    experiment_id = selected.experiment_id if selected is not None else client.create_experiment(experiment)
    run = Run(client, client.create_run(experiment_id, tags=tags, run_name=name).info.run_id)
    try:
        client.log_batch(
            run.run_id,
            params=[Param(key, str(value)) for key, value in parameters.items()],
            synchronous=True,
        )
        client.log_dict(run.run_id, provenance, "provenance.json")
        yield run
        run.flush()
        client.set_terminated(run.run_id, status="FINISHED")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"[:5000]
        status = "KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        for operation in (
            run.flush,
            lambda: client.set_tag(run.run_id, "failure", failure),
            lambda: client.set_terminated(run.run_id, status=status),
        ):
            try:
                operation()
            except Exception as tracking_error:
                warnings.warn(f"MLflow could not record the failed benchmark: {tracking_error}", stacklevel=2)
        raise
