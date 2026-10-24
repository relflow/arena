"""Disk-backed, label-free snapshots for RelBench databases.

Only columns named by a :class:`~benchmarks.relarena.spec.BenchmarkSpec` are copied
to DuckDB, read straight from the Parquet files RelBench caches and censored
at the split's cutoff the way the harness censors a database, so no table is
ever loaded into pandas.  Task labels live separately in small anchor parquet
files, so a snapshot can be shared by training and prediction workers.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .resources import resolve, spill
from .spec import BenchmarkSpec

STORE_FILENAME = "snapshot.duckdb"
ANCHOR_FILENAME = "anchors.parquet"
STORE_FORMAT_VERSION = "3"
MANIFEST_TABLE = "__relflow_snapshot_manifest"
SOURCE_ROW_ID = "__source_row_id"


class TableLike(Protocol):
    """The small part of ``relbench.base.Table`` used for anchors."""

    df: Any


@dataclass(frozen=True)
class ParquetSources:
    """A RelBench database as DuckDB relations over its cached Parquet files, censored at a cutoff.

    ``directory`` is ``<cache>/<dataset>/db``. Tables with a time column keep
    only rows at or before ``upto``, which is what the harness's split does;
    ``None`` keeps every row. Row ids are the physical row numbers of the
    files, so ties resolve the same way on every build.
    """

    directory: Path
    upto: str | None = None

    def path(self, table: str) -> Path:
        return Path(self.directory) / f"{table}.parquet"

    def tables(self) -> set[str]:
        return {path.stem for path in Path(self.directory).glob("*.parquet")}

    def columns(self, table: str) -> list[str]:
        return list(pq.read_schema(self.path(table)).names)

    def time_column(self, table: str) -> str | None:
        metadata = pq.read_schema(self.path(table)).metadata or {}
        return json.loads(metadata.get(b"time_col", b"null").decode())

    def relation(self, table: str, columns: tuple[str, ...] | None = None, *, row_ids: bool = False) -> str:
        """A SQL relation of the table's selected columns, censored, optionally carrying ``__source_row_id``."""
        selected = ", ".join(quote(column) for column in columns) if columns is not None else "*"
        if row_ids:
            selected = (selected + ", " if selected else "") + f"file_row_number AS {quote(SOURCE_ROW_ID)}"
        source = f"read_parquet('{self.path(table)}'{', file_row_number=true' if row_ids else ''})"
        time = self.time_column(table)
        censor = f" WHERE {quote(time)} <= TIMESTAMP '{self.upto}'" if self.upto and time else ""
        return f"(SELECT {selected} FROM {source}{censor})"


def prepare_store(
    sources: ParquetSources,
    spec: BenchmarkSpec,
    directory: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    duckdb_config: dict | None = None,
) -> Path:
    """Build (or validate) the path-backed DuckDB relational snapshot.

    ``directory`` normally names an artifact directory, in which case the
    database is written to ``snapshot.duckdb``.  An explicit ``.duckdb`` file is
    accepted as well.  Creation happens in a sibling temporary directory and is
    atomically published only after DuckDB has closed the database.
    """

    target = store_path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    required = spec.required_columns()
    source_rows = validate_sources(sources, required)

    if target.exists() and not overwrite:
        check_store(target, spec, source_rows=source_rows)
        return target

    settings = duckdb_config if duckdb_config is not None else resolve(target.parent / "duckdb-spill")["settings"]
    with (
        tempfile.TemporaryDirectory(prefix="relflow-store-", dir=target.parent) as temporary,
        spill(settings) as config,
    ):
        temporary_path = Path(temporary) / STORE_FILENAME
        connection = duckdb.connect(str(temporary_path), config=config)
        try:
            for table_name, columns in required.items():
                connection.execute(
                    f"CREATE TABLE {quote(table_name)} AS SELECT * FROM {sources.relation(table_name, columns, row_ids=True)}"
                )

            create_indexes(connection, spec)
            manifest = {
                "format_version": STORE_FORMAT_VERSION,
                "projection_fingerprint": spec.projection_fingerprint,
                "dataset_name": spec.dataset_name,
                "cutoff": sources.upto,
                "source_rows": source_rows,
                "projected_columns": {table: list(columns) for table, columns in required.items()},
            }
            connection.execute(f"CREATE TABLE {quote(MANIFEST_TABLE)} (manifest_json VARCHAR NOT NULL)")
            connection.execute(
                f"INSERT INTO {quote(MANIFEST_TABLE)} VALUES (?)",
                [json.dumps(manifest, sort_keys=True, separators=(",", ":"))],
            )
            connection.execute("CHECKPOINT")
        finally:
            connection.close()

        # Another cache warmer may have won the race while this one was building.
        if target.exists() and not overwrite:
            check_store(target, spec, source_rows=source_rows)
            return target
        os.replace(temporary_path, target)

    check_store(target, spec, source_rows=source_rows)
    return target


def prepare_anchors(
    table: TableLike,
    spec: BenchmarkSpec,
    path: str | os.PathLike[str],
    *,
    include_target: bool,
    row_group_size: int = 1_024,
    overwrite: bool = False,
) -> Path:
    """Write a task table as a narrow, row-grouped anchor parquet file.

    The generated columns have stable internal names, independent of the source
    benchmark.  ``__row_position`` preserves the exact RelArena table ordering;
    prediction code can therefore restore it after parallel loading.
    """

    if row_group_size < 1:
        raise ValueError("row_group_size must be at least one")

    target = anchor_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"anchor parquet already exists: {target}")

    frame = table.df
    required = [spec.anchor_entity_column, spec.timestamp_column]
    if include_target:
        required.append(spec.target_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"task table for {spec.benchmark_name!r} is missing columns: {missing!r}")
    if frame[spec.anchor_entity_column].isna().any():
        raise ValueError("anchor entity column contains null values")
    if frame[spec.timestamp_column].isna().any():
        raise ValueError("anchor timestamp column contains null values")
    if include_target and frame[spec.target_column].isna().any():
        raise ValueError("training target column contains null values")

    # Arrow conversion turns pandas extension-dtype missing values into real nulls
    # and copies only the two or three narrow task columns, not the relational DB.
    arrays: dict[str, Any] = {
        "__row_position": np.arange(len(frame), dtype=np.int64),
        "__entity_key": frame[spec.anchor_entity_column],
        "__as_of": frame[spec.timestamp_column],
    }
    if include_target:
        arrays["__target"] = frame[spec.target_column]
    arrow_table = pa.Table.from_pydict(arrays)
    metadata = dict(arrow_table.schema.metadata or {})
    metadata.update(
        {
            b"relflow.format_version": STORE_FORMAT_VERSION.encode("utf-8"),
            b"relflow.spec_fingerprint": spec.fingerprint.encode("utf-8"),
            b"relflow.include_target": str(include_target).lower().encode("ascii"),
            b"relflow.row_count": str(len(frame)).encode("ascii"),
        }
    )
    arrow_table = arrow_table.replace_schema_metadata(metadata)

    with tempfile.TemporaryDirectory(prefix="relflow-anchors-", dir=target.parent) as temporary:
        temporary_path = Path(temporary) / ANCHOR_FILENAME
        pq.write_table(
            arrow_table,
            temporary_path,
            compression="zstd",
            row_group_size=row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
        os.replace(temporary_path, target)
    return target


def read_store_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the self-contained manifest from a snapshot database."""

    resolved = store_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"relational snapshot does not exist: {resolved}")
    connection = duckdb.connect(str(resolved), read_only=True)
    try:
        row = connection.execute(f"SELECT manifest_json FROM {quote(MANIFEST_TABLE)}").fetchone()
    except duckdb.Error as error:
        raise ValueError(f"invalid relational snapshot {resolved}: {error}") from error
    finally:
        connection.close()
    if row is None:
        raise ValueError(f"snapshot manifest is empty: {resolved}")
    return dict(json.loads(row[0]))


def validate_store(path: str | os.PathLike[str], spec: BenchmarkSpec) -> Path:
    """Validate a snapshot against a benchmark spec and return its file path."""

    resolved = store_path(path)
    check_store(resolved, spec)
    return resolved


def check_store(
    path: Path,
    spec: BenchmarkSpec,
    *,
    source_rows: Mapping[str, int] | None = None,
) -> None:
    manifest = read_store_manifest(path)
    if manifest.get("format_version") != STORE_FORMAT_VERSION:
        raise ValueError(
            f"snapshot format mismatch at {path}: "
            f"expected {STORE_FORMAT_VERSION!r}, got {manifest.get('format_version')!r}"
        )
    if manifest.get("projection_fingerprint") != spec.projection_fingerprint:
        raise ValueError(f"snapshot projection mismatch at {path}; use a distinct cache key or overwrite it")
    if source_rows is not None and manifest.get("source_rows") != dict(source_rows):
        raise ValueError(f"snapshot source-row counts differ at {path}; use a phase-specific cache key or overwrite it")


def validate_sources(sources: ParquetSources, required: Mapping[str, tuple[str, ...]]) -> dict[str, int]:
    """Check that every required table and column exists; return the censored row count per table."""
    available = sources.tables()
    missing_tables = [table for table in required if table not in available]
    if missing_tables:
        raise ValueError(f"database is missing required tables: {missing_tables!r}")
    source_rows: dict[str, int] = {}
    connection = duckdb.connect()
    try:
        for table_name, columns in required.items():
            present = sources.columns(table_name)
            missing_columns = [column for column in columns if column not in present]
            if missing_columns:
                raise ValueError(f"database table {table_name!r} is missing configured columns: {missing_columns!r}")
            if SOURCE_ROW_ID in present:
                raise ValueError(f"database table {table_name!r} contains reserved column {SOURCE_ROW_ID!r}")
            source_rows[table_name] = connection.execute(
                f"SELECT count(*) FROM {sources.relation(table_name)}"
            ).fetchone()[0]
    finally:
        connection.close()
    return source_rows


def create_indexes(connection: duckdb.DuckDBPyConnection, spec: BenchmarkSpec) -> None:
    definitions = [
        (table, tuple(columns)) for table, source in spec.sources.items() for columns in source.get("indexes", ())
    ]

    seen: set[tuple[str, tuple[str, ...]]] = set()
    for index_number, definition in enumerate(definitions):
        if definition in seen:
            continue
        seen.add(definition)
        table, columns = definition
        column_sql = ", ".join(quote(column) for column in columns)
        connection.execute(f"CREATE INDEX {quote(f'__relflow_index_{index_number}')} ON {quote(table)} ({column_sql})")


def store_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    return resolved if resolved.suffix == ".duckdb" else resolved / STORE_FILENAME


def anchor_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    return resolved if resolved.suffix == ".parquet" else resolved / ANCHOR_FILENAME


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


__all__ = [
    "ANCHOR_FILENAME",
    "STORE_FILENAME",
    "prepare_anchors",
    "prepare_store",
    "read_store_manifest",
    "validate_store",
]
