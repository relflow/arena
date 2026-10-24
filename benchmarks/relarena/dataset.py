"""Bounded SQL materialization into canonical, nested Arrow observations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .resources import resolve, spill
from .spec import BenchmarkSpec
from .store import validate_store


def materialize(
    store_path: str | os.PathLike[str],
    anchor_path: str | os.PathLike[str],
    spec: BenchmarkSpec,
    destination: str | os.PathLike[str],
    *,
    include_target: bool,
    fetch_size: int,
    duckdb_config: dict | None = None,
) -> Path:
    """Write exactly one ordered observation per anchor using bounded SQL joins.

    The benchmark query receives ``$anchors`` (a Parquet path) and the inclusive
    ``$start`` / exclusive ``$stop`` row-position bounds. It returns label-free
    nested Arrow columns, including ``__row_position``. Labels are attached only
    for supervised splits. All query and writer resources close before the
    completed Parquet file atomically replaces ``destination``.
    """

    if isinstance(fetch_size, bool) or not isinstance(fetch_size, int) or fetch_size < 1:
        raise ValueError(f"fetch_size must be a positive integer; got {fetch_size!r}")
    if not isinstance(include_target, bool):
        raise TypeError("include_target must be a boolean")

    store = validate_store(store_path, spec)
    anchors = Path(anchor_path).resolve()
    if anchors.suffix != ".parquet":
        anchors /= "anchors.parquet"
    row_count = validate_anchor(anchors, spec.fingerprint, include_target, fetch_size)
    target = Path(destination).resolve()
    if target.suffix != ".parquet":
        target /= "observations.parquet"
    if target in (anchors, store.resolve()):
        raise ValueError("materialized observations must not replace their source files")
    target.parent.mkdir(parents=True, exist_ok=True)

    query = spec.query.strip().removesuffix(";").strip()
    feature_projection = "features.*"
    label_join = ""
    if include_target:
        target_type = "BOOLEAN" if spec.task_type == "binary_classification" else "DOUBLE"
        feature_projection += f", CAST(labels.__target AS {target_type}) AS target"
        label_join = (
            " LEFT JOIN (SELECT __row_position, __target FROM read_parquet($anchors)"
            " WHERE __row_position >= $start AND __row_position < $stop) AS labels"
            " ON features.__row_position = labels.__row_position"
        )
    statement = f"SELECT {feature_projection} FROM ({query}) AS features{label_join} ORDER BY features.__row_position"

    metadata = {
        b"relflow.spec_fingerprint": spec.fingerprint.encode("utf-8"),
        b"relflow.projection_fingerprint": spec.projection_fingerprint.encode("utf-8"),
        b"relflow.include_target": str(include_target).lower().encode("ascii"),
        b"relflow.row_count": str(row_count).encode("ascii"),
    }

    settings = duckdb_config if duckdb_config is not None else resolve(target.parent / "duckdb-spill")["settings"]
    with (
        tempfile.TemporaryDirectory(prefix="relflow-observations-", dir=target.parent) as temporary,
        spill(settings) as config,
    ):
        directory = Path(temporary)
        output = directory / "observations.parquet"
        schema: pa.Schema | None = None
        writer: pq.ParquetWriter | None = None
        try:
            with duckdb.connect(
                str(store),
                read_only=True,
                config=config,
            ) as connection:
                statements = connection.extract_statements(query)
                if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                    raise ValueError("benchmark observations SQL must contain exactly one SELECT")
                for start in range(0, max(row_count, 1), fetch_size):
                    stop = min(start + fetch_size, row_count)
                    parameters = {"anchors": str(anchors), "start": start, "stop": stop}
                    with connection.execute(statement, parameters).to_arrow_reader(fetch_size) as reader:
                        current = reader.schema
                        expected_target_count = int(include_target)
                        if (
                            current.names.count("__row_position") != 1
                            or current.names.count("target") != expected_target_count
                            or "__target" in current.names
                        ):
                            raise ValueError(
                                "benchmark observations SQL must return __row_position and "
                                "label-free features; the runner alone attaches target"
                            )
                        if current.field("__row_position").type != pa.int64():
                            raise ValueError("observation __row_position must have Arrow int64 type")
                        if schema is None:
                            schema = current.with_metadata(metadata)
                            writer = pq.ParquetWriter(output, schema, compression="zstd")
                        elif not schema.equals(current, check_metadata=False):
                            raise ValueError(f"observation schema changed for anchor chunk [{start}, {stop})")
                        position = start
                        for batch in reader:
                            positions = batch.column(batch.schema.get_field_index("__row_position"))
                            end = position + batch.num_rows
                            if end > stop or not positions_equal(positions, position, end):
                                raise ValueError(
                                    "observations must preserve exactly one row per anchor in "
                                    f"[{start}, {stop}); "
                                    "missing, duplicate, or unexpected positions"
                                )
                            if include_target and batch.column("target").null_count:
                                raise ValueError(f"observation targets contain nulls in [{start}, {stop})")
                            assert writer is not None
                            writer.write_batch(batch, row_group_size=fetch_size)
                            position = end
                        if position != stop:
                            raise ValueError(
                                "observations must preserve exactly one row per anchor in "
                                f"[{start}, {stop}); expected {stop - start}, "
                                f"got {position - start}"
                            )
        finally:
            if writer is not None:
                writer.close()
        os.replace(output, target)
    return target


def positions_equal(positions: pa.Array, start: int, stop: int) -> bool:
    """Check a contiguous position interval without materializing Python rows."""

    expected = pa.array(np.arange(start, stop, dtype=np.int64))
    return positions.null_count == 0 and pc.all(pc.equal(positions, expected)).as_py() is not False


def validate_anchor(path: Path, fingerprint: str, include_target: bool, fetch_size: int) -> int:
    """Validate anchor identity, label isolation and the complete row-position axis."""

    if not path.is_file():
        raise FileNotFoundError(f"anchor Parquet does not exist: {path}")
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        names = set(schema.names)
        required = {"__row_position", "__entity_key", "__as_of"}
        if include_target:
            required.add("__target")
        missing = required - names
        if missing:
            raise ValueError(f"anchor Parquet is missing columns: {sorted(missing)!r}")
        if not include_target and "__target" in names:
            raise ValueError("prediction anchor Parquet unexpectedly contains target labels")
        if schema.field("__row_position").type != pa.int64():
            raise ValueError("anchor __row_position must have Arrow int64 type")
        metadata = schema.metadata or {}
        if metadata.get(b"relflow.spec_fingerprint") != fingerprint.encode("utf-8"):
            raise ValueError(f"anchor spec mismatch at {path}; rebuild it for the selected benchmark")
        if metadata.get(b"relflow.include_target") != str(include_target).lower().encode("ascii"):
            raise ValueError(f"anchor target mode mismatch at {path}: expected include_target={include_target}")
        row_count = parquet.metadata.num_rows
        if metadata.get(b"relflow.row_count") != str(row_count).encode("ascii"):
            raise ValueError(f"anchor row-count metadata differs from the Parquet row count at {path}")
        position = 0
        for batch in parquet.iter_batches(batch_size=fetch_size, columns=["__row_position"]):
            end = position + batch.num_rows
            if not positions_equal(batch.column(0), position, end):
                raise ValueError(f"anchor positions must be contiguous and ordered from zero at {path}")
            position = end
    return row_count


__all__ = ["materialize"]
