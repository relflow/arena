"""Declarative benchmark manifests, model templates, and feature queries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

TaskType = Literal["binary_classification", "regression"]


class Source(TypedDict):
    """The label-free columns and indexes persisted for a source table."""

    columns: tuple[str, ...]
    indexes: tuple[tuple[str, ...], ...]


class Loader(yaml.SafeLoader):
    """Preserve manifest intent by rejecting duplicate YAML keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConstructorError(None, None, "manifest keys must be strings", key_node.start_mark)
            if key in result:
                raise ConstructorError(None, None, f"duplicate manifest key {key!r}", key_node.start_mark)
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One benchmark's protocol metadata and file-backed observation contract.

    Model construction belongs to the runner. Loading a manifest never creates
    a model or downloads text encoder weights. The complete fingerprint includes
    model and SQL contents; projection fingerprints share compatible snapshots.
    """

    dataset_name: str
    task_name: str
    task_type: TaskType
    entity_table: str
    entity_key: str
    anchor_entity_column: str
    timestamp_column: str
    target_column: str
    model_path: Path
    query_path: Path
    sources: Mapping[str, Source]
    spec_version: str = "3"

    def __post_init__(self) -> None:
        for name in (
            "dataset_name",
            "task_name",
            "entity_table",
            "entity_key",
            "anchor_entity_column",
            "timestamp_column",
            "target_column",
        ):
            validate_name(getattr(self, name), name)
        if self.task_type not in ("binary_classification", "regression"):
            raise ValueError(f"unsupported task_type {self.task_type!r}")
        if self.spec_version != "3":
            raise ValueError(f"spec_version must be '3'; got {self.spec_version!r}")
        for name in ("model_path", "query_path"):
            path = Path(getattr(self, name)).resolve()
            if not path.is_file():
                raise ValueError(f"{name} does not name an existing file: {path}")
            if not path.read_text(encoding="utf-8").strip():
                raise ValueError(f"{name} is empty: {path}")
            object.__setattr__(self, name, path)

        if not isinstance(self.sources, Mapping) or not self.sources:
            raise ValueError("sources must be a nonempty mapping of table names to projections")
        normalized: dict[str, Source] = {}
        for table, source in self.sources.items():
            validate_name(table, "source table")
            if not isinstance(source, Mapping) or set(source) != {"columns", "indexes"}:
                raise ValueError(f"sources.{table} must contain only 'columns' and 'indexes'")
            columns = names(source["columns"], f"sources.{table}.columns")
            if not columns:
                raise ValueError(f"sources.{table}.columns must not be empty")
            if "__source_row_id" in columns:
                raise ValueError(f"sources.{table}: __source_row_id is supplied by the store")
            indexes_value = source["indexes"]
            if not isinstance(indexes_value, Sequence) or isinstance(indexes_value, str | bytes):
                raise ValueError(f"sources.{table}.indexes must be a sequence of column sequences")
            indexes = tuple(
                names(value, f"sources.{table}.indexes[{index}]") for index, value in enumerate(indexes_value)
            )
            for index in indexes:
                if not index or not set(index) <= set(columns):
                    raise ValueError(f"sources.{table}: index {index!r} must name projected columns {columns!r}")
            if len(indexes) != len(set(indexes)):
                raise ValueError(f"sources.{table}.indexes contains duplicate indexes")
            normalized[table] = {"columns": columns, "indexes": indexes}
        if self.entity_table not in normalized:
            raise ValueError(f"entity_table {self.entity_table!r} must be declared in sources")
        if self.entity_key not in normalized[self.entity_table]["columns"]:
            raise ValueError(f"entity_key {self.entity_key!r} must be projected by {self.entity_table}")
        object.__setattr__(self, "sources", normalized)

    @classmethod
    def from_yaml(cls, pathname: str | Path) -> BenchmarkSpec:
        """Read a strict manifest and resolve its two companion files locally."""
        path = Path(pathname).resolve()
        try:
            with path.open(encoding="utf-8") as source:
                payload = yaml.load(source, Loader=Loader)
            if not isinstance(payload, Mapping):
                raise ValueError("expected a mapping")
            required = {
                "dataset_name",
                "task_name",
                "task_type",
                "entity_table",
                "entity_key",
                "anchor_entity_column",
                "timestamp_column",
                "target_column",
                "model",
                "query",
                "sources",
                "spec_version",
            }
            missing, unknown = required - payload.keys(), payload.keys() - required
            if missing or unknown:
                raise ValueError(f"manifest keys: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}")
            options = dict(payload)
            for name in ("model", "query"):
                value = options.pop(name)
                validate_name(value, name)
                relative = Path(value)
                resolved = (path.parent / relative).resolve()
                if relative.is_absolute() or not resolved.is_relative_to(path.parent):
                    raise ValueError(f"{name} must reference a file within {path.parent}: {value!r}")
                options[f"{name}_path"] = resolved
            return cls(**options)
        except (ValueError, TypeError, yaml.YAMLError) as error:
            raise ValueError(f"benchmark manifest {path}: {error}") from error

    @property
    def benchmark_name(self) -> str:
        return f"{self.dataset_name}/{self.task_name}"

    @property
    def query(self) -> str:
        """The source query, with bound $anchors, $start, and $stop parameters."""
        return self.query_path.read_text(encoding="utf-8")

    def required_columns(self) -> dict[str, tuple[str, ...]]:
        return {table: source["columns"] for table, source in self.sources.items()}

    def projection_payload(self) -> dict[str, Any]:
        """Identify only the reusable, label-free database projection."""
        return {
            "dataset_name": self.dataset_name,
            "spec_version": self.spec_version,
            "sources": dict(self.sources),
        }

    def materialization_payload(self) -> dict[str, Any]:
        return {
            **self.projection_payload(),
            "task_name": self.task_name,
            "task_type": self.task_type,
            "entity_table": self.entity_table,
            "entity_key": self.entity_key,
            "anchor_entity_column": self.anchor_entity_column,
            "timestamp_column": self.timestamp_column,
            "target_column": self.target_column,
            "model": self.model_path.name,
            "query": self.query_path.name,
            "model_yaml": self.model_path.read_text(encoding="utf-8"),
            "observations_sql": self.query,
        }

    @property
    def projection_fingerprint(self) -> str:
        return payload_fingerprint(self.projection_payload())

    @property
    def fingerprint(self) -> str:
        return payload_fingerprint(self.materialization_payload())


def payload_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_name(value: Any, description: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonempty string; got {value!r}")


def names(value: Any, description: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{description} must be a sequence of column names")
    result = tuple(value)
    for column in result:
        validate_name(column, description)
    if len(result) != len(set(result)):
        raise ValueError(f"{description} contains duplicate columns")
    return result


__all__ = ["BenchmarkSpec", "Source", "TaskType"]
