"""Derive raw-structure RelArena tasks from a database's keys and measured column statistics.

The hand-written structures in ``raw.py`` are the oracle. This module builds the
same ``Task``, ``Branch``, ``Lookup``, and ``Col`` objects from the foreign-key
graph and per-column measurements, so ``raw.query`` and ``raw.model`` render the
query and schema without any task-specific knowledge:

- the entity row carries its typed columns and its forward foreign keys as
  lookups, two hops deep;
- every table with a foreign key into the entity becomes a branch, ordered and
  censored by its time column and capped by its size tier's length budget, a
  resource guard that compaction makes free for short histories;
- a table sharing two or more foreign-key targets with a branch, whose shared
  key is unique and mostly covered by the branch, attaches to it as an as-of
  lookup; a table already kept as a branch never doubles as a lookup;
- children of a branch, or of a dimension the branch or the root points at,
  nest under it when their fan-out is small;
- each column's tensorfield follows its dtype, cardinality, null pattern,
  string shape, and identifier naming, measured on a sample of the table; a
  row's own key is carried only when another table references it, and a
  dimension timestamp that repeats the row's time is left out;
- every node keeps at most its tier's field budget, filled with own columns
  first, then first-hop lookups, aligned siblings, and second-hop lookups,
  which reach only vocabulary tables;
- prose is carried as text only by rows at depth one: the root, its lookups,
  branch rows, and their first-hop and aligned lookups. Deeper rows keep it as
  a category when its vocabulary allows and leave it out otherwise.

A branch nests the children of a dimension only when the branch is a link
table (at most two columns of its own) and the child belongs solely to that
dimension, meaning its other foreign keys point at vocabulary tables with no
time column and no keys of their own.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import duckdb
import yaml

from benchmarks.relarena import raw
from benchmarks.relarena.raw import Branch, Col, Lookup, Task
from benchmarks.relarena.store import ParquetSources

SAMPLE = 200_000
# Size tiers by training anchors, as the sweep defines them; every budget below is a tier property.
TIERS = ((20_000, "small"), (500_000, "medium"), (None, "large"))
BUDGETS = {
    "small": {"branch": 128, "nested": 16, "fields": 48, "root": 64},
    "medium": {"branch": 64, "nested": 16, "fields": 40, "root": 48},
    "large": {"branch": 32, "nested": 8, "fields": 24, "root": 32},
}
NESTED_FANOUT = 32  # p90 rows per key above which a child table is not nested
LINK_COLUMNS = 2  # own columns at most for a branch to count as a link table
ALIGNED = (0.99, 0.5)  # unique shared key, and a majority of the sibling's keys covered by the branch
SOURCE_ROW_ID = "__source_row_id"
RESERVED = {"as_of", "target", "occurred_at", "age_days"}
# Table aliases the rendered query already uses; a lookup alias must not shadow them.
RESERVED_ALIASES = {"anchor", "anchors", "root", "parent", "event", "linked", "labels", "features"}
IDENTIFIER = re.compile(r"(?i:(^|[^a-z])(id|ids|code|type|status|kind|class)([^a-z]|$))|[A-Za-z](Id|ID)$")
INTEGERS = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}
VECTOR_BLOCK = re.compile(r"^(.*?)_?(\d+)$")


@dataclass(frozen=True)
class Column:
    """Measurements of one column on a sample of its table."""

    name: str
    dtype: str
    n: int
    nonnull: int
    distinct: int
    mean_len: float | None = None
    wordy: float | None = None
    spaced: float | None = None
    fixed: bool | None = None
    integral: float | None = None
    daytime: float | None = None  # share of timestamps carrying a time of day


@dataclass(frozen=True)
class Table:
    name: str
    source: str  # SQL relation the profiler reads, such as read_parquet('...') or a view name
    pkey: str | None
    time: str | None
    fkeys: dict[str, str]
    rows: int
    columns: dict[str, Column]


@dataclass(frozen=True)
class Meta:
    dataset: str
    task: str
    task_type: str
    entity_table: str
    entity_col: str  # the task table's entity column
    time_col: str  # the task table's timestamp column
    target_col: str
    training_rows: int | None = None  # anchors in the training split; selects the size tier


def tier(training_rows: int | None) -> str:
    """The size tier of a task; unknown sizes take the smallest tier's budgets."""
    for limit, name in TIERS:
        if limit is None or training_rows is None or training_rows < limit:
            return name
    raise AssertionError("the last tier is unbounded")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def profile(con: duckdb.DuckDBPyConnection, sources: Mapping[str, str], schema: Mapping[str, dict]) -> dict[str, Table]:
    """Measure every column of every table on one sample per table."""
    tables = {}
    for index, (name, source) in enumerate(sources.items()):
        spec = schema[name]
        sample = f"sample_{index}"
        con.execute(f"CREATE OR REPLACE TEMP TABLE {sample} AS SELECT * FROM {source} USING SAMPLE {SAMPLE} ROWS")
        rows = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
        dtypes = con.execute(f"DESCRIBE {sample}").fetchall()
        columns = {}
        for column, dtype, *_ in dtypes:
            if column == SOURCE_ROW_ID:
                continue
            v = quote(column)
            n, nonnull, distinct = con.execute(
                f"SELECT count(*), count({v}), count(DISTINCT {v}) FROM {sample}"
            ).fetchone()
            measured = Column(column, dtype, n, nonnull, distinct)
            if dtype.startswith("VARCHAR") and not dtype.endswith("[]"):
                row = con.execute(
                    f"SELECT avg(length({v})), avg(CASE WHEN regexp_matches({v}, '\\p{{L}}{{2,}}') THEN 1 ELSE 0 END), "
                    f"avg(CASE WHEN {v} LIKE '% %' THEN 1 ELSE 0 END), min(length({v})) = max(length({v})) "
                    f"FROM {sample} WHERE {v} IS NOT NULL"
                ).fetchone()
                measured = Column(
                    column, dtype, n, nonnull, distinct, mean_len=row[0], wordy=row[1], spaced=row[2], fixed=row[3]
                )
            elif dtype in ("DOUBLE", "FLOAT") or dtype.startswith("DECIMAL"):
                integral = con.execute(
                    f"SELECT avg(CASE WHEN {v} = floor({v}) THEN 1 ELSE 0 END) FROM {sample} WHERE {v} IS NOT NULL AND isfinite({v})"
                ).fetchone()[0]
                measured = Column(column, dtype, n, nonnull, distinct, integral=integral)
            elif dtype.startswith("TIMESTAMP"):
                daytime = con.execute(
                    f"SELECT avg(CASE WHEN date_trunc('day', {v}) <> {v} THEN 1 ELSE 0 END) FROM {sample} WHERE {v} IS NOT NULL"
                ).fetchone()[0]
                measured = Column(column, dtype, n, nonnull, distinct, daytime=daytime)
            columns[column] = measured
        tables[name] = Table(
            name, source, spec.get("pkey"), spec.get("time_col"), dict(spec.get("fkeys") or {}), rows, columns
        )
    return tables


def fanout(con: duckdb.DuckDBPyConnection, table: Table, keys: tuple[str, ...]) -> float:
    """The ninetieth percentile of rows per key tuple."""
    group = ", ".join(quote(key) for key in keys)
    present = " AND ".join(f"{quote(key)} IS NOT NULL" for key in keys)
    value = con.execute(
        f"SELECT quantile_cont(n, 0.9) FROM (SELECT count(*) AS n FROM {table.source} WHERE {present} GROUP BY {group})"
    ).fetchone()[0]
    return float(value or 0)


def alignment(
    con: duckdb.DuckDBPyConnection, sibling: Table, branch: Table, pairs: tuple[tuple[str, str], ...]
) -> tuple[float, float]:
    """How well a sibling's rows align one-to-one with a branch's rows on their shared keys.

    Returns the share of sibling rows with a unique key tuple and the share of the
    sibling's distinct key tuples that occur in the branch.
    """
    own = ", ".join(quote(s) for s, _ in pairs)
    other = ", ".join(f"{quote(b)} AS {quote(s)}" for s, b in pairs)
    own_present = " AND ".join(f"{quote(s)} IS NOT NULL" for s, _ in pairs)
    other_present = " AND ".join(f"{quote(b)} IS NOT NULL" for _, b in pairs)
    unique = con.execute(
        f"SELECT count(DISTINCT ({own})) / greatest(count(*), 1) FROM {sibling.source} WHERE {own_present}"
    ).fetchone()[0]
    coverage = con.execute(
        f"WITH s AS (SELECT DISTINCT {own} FROM {sibling.source} WHERE {own_present}), "
        f"b AS (SELECT DISTINCT {other} FROM {branch.source} WHERE {other_present}) "
        f"SELECT (SELECT count(*) FROM s JOIN b USING ({own})) / greatest((SELECT count(*) FROM s), 1)"
    ).fetchone()[0]
    return float(unique or 0), float(coverage or 0)


def references(tables: Mapping[str, Table], name: str) -> int:
    """Rows in other tables that point at this table: how often one of its rows recurs."""
    return sum(t.rows for t in tables.values() for target in t.fkeys.values() if target == name)


def kind(column: Column, table: Table, tables: Mapping[str, Table], *, root: bool) -> str | None:
    """The tensorfield for one measured column, or None to leave it out."""
    d = column.dtype
    if column.nonnull == 0 or (column.distinct <= 1 and column.nonnull == column.n):
        return None  # constant with no null signal
    if column.distinct <= 1:
        return "category"  # one value or null: a flag
    if column.name == table.pkey:
        return "hash" if not root and references(tables, table.name) else None
    if column.name in table.fkeys:
        return "hash"
    if d.startswith(("TIMESTAMP", "DATE")):
        return "dateparts"
    if d == "BOOLEAN":
        return "boolean"
    if d.endswith("[]"):
        return "set" if d.startswith("VARCHAR") else "vector"
    if d.startswith(("STRUCT", "MAP")):
        return None
    numeric = d in INTEGERS or d in ("DOUBLE", "FLOAT") or d.startswith("DECIMAL")
    if numeric:
        integral = d in INTEGERS or (column.integral or 0) >= 0.999
        if integral and column.distinct <= 3:
            return "category"  # binary and ternary flags
        if integral and IDENTIFIER.search(column.name):
            return "category" if column.distinct <= 20_000 else "hash"
        return "number"
    if d.startswith("VARCHAR"):
        weight = max(1.0, references(tables, table.name) / max(table.rows, 1))
        ratio = column.distinct / max(column.nonnull, 1)
        wordy = column.wordy or 0
        if column.fixed and not column.spaced and ratio >= 0.5 and (column.mean_len or 0) >= 8:
            return "hash"  # opaque fixed-width identifiers
        if (column.mean_len or 0) >= 40 and wordy >= 0.3:
            return "text"  # prose, however often it recurs
        if column.distinct <= 25_000 and ratio / weight <= 0.25:
            return "category"
        if wordy >= 0.3:
            return "text"
        return "hash"
    return None


def budget(mean_len: float | None) -> int:
    length = mean_len or 0
    return 16 if length < 40 else 32 if length < 160 else 64 if length < 700 else 96


def same_time(con: duckdb.DuckDBPyConnection, table: Table, column: str, row_time: str, dimension: Table) -> bool:
    """Whether a dimension's timestamp repeats the row's own time column on the joined rows."""
    rate = con.execute(
        f"SELECT avg(CASE WHEN r.{quote(row_time)} = d.{quote(dimension.time)} THEN 1 ELSE 0 END) "
        f"FROM (SELECT {quote(column)}, {quote(row_time)} FROM {table.source} USING SAMPLE {SAMPLE} ROWS) r "
        f"JOIN {dimension.source} d ON r.{quote(column)} = d.{quote(dimension.pkey)} "
        f"WHERE r.{quote(row_time)} IS NOT NULL AND d.{quote(dimension.time)} IS NOT NULL"
    ).fetchone()[0]
    return (rate or 0) >= 0.99


def trim(columns: tuple[Col, ...], groups: list[list[Lookup]], budget: int) -> tuple[tuple[Col, ...], list[Lookup]]:
    """Keep at most ``budget`` fields on one node: own columns first, then each lookup group in order."""
    own = columns[:budget]
    remaining = budget - len(own)
    kept: list[Lookup] = []
    for group in groups:
        for lookup in group:
            if remaining <= 0:
                break
            take = lookup.columns[:remaining]
            if take:
                kept.append(replace(lookup, columns=take))
                remaining -= len(take)
    return own, kept


def snake(name: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text or text[0].isdigit():
        text = "c_" + text
    return text


def stem(column: str) -> str:
    """A lookup alias from its foreign-key column: ``raceId`` and ``race_id`` become ``race``."""
    name = re.sub(r"(?i)[_ ]?id$", "", column) or column
    return snake(name)


class Names:
    """Unique field names within one node; reserved and internal names get a prefix."""

    def __init__(self, reserved: frozenset[str] = frozenset(RESERVED)) -> None:
        self.used: set[str] = set()
        self.reserved = reserved

    def take(self, name: str) -> str:
        name = snake(name)
        if name in self.reserved or name.startswith("__"):
            name = "col_" + name.lstrip("_")
        candidate, index = name, 2
        while candidate in self.used:
            candidate = f"{name}_{index}"
            index += 1
        self.used.add(candidate)
        return candidate


def vector_blocks(table: Table) -> dict[str, list[str]]:
    """Runs of eight or more numeric columns that differ only by a trailing number form one vector."""
    groups: dict[str, list[str]] = {}
    for column in table.columns.values():
        match = VECTOR_BLOCK.match(column.name)
        if match and match.group(1) and (column.dtype in INTEGERS or column.dtype in ("DOUBLE", "FLOAT")):
            groups.setdefault(match.group(1), []).append(column.name)
    return {prefix: members for prefix, members in groups.items() if len(members) >= 8}


def typed(
    table: Table,
    tables: Mapping[str, Table],
    dataset: str,
    names: Names,
    *,
    prefix: str = "",
    exclude: set[str] = frozenset(),
    root: bool = False,
    text: bool = True,
) -> tuple[Col, ...]:
    """Every column of a table as tensorfield columns, in table order; ``text`` allows prose fields."""
    blocks = vector_blocks(table)
    member_of = {member: block for block, members in blocks.items() for member in members}
    emitted: set[str] = set()
    result: list[Col] = []
    for column in table.columns.values():
        if column.name in exclude:
            continue
        if column.name in member_of:
            block = member_of[column.name]
            if block in emitted:
                continue
            emitted.add(block)
            members = blocks[block]
            values = ", ".join(f"coalesce({{t}}.{quote(member)}, 0)" for member in members)
            field = names.take(prefix + block)
            result.append(
                Col(
                    field,
                    f"CAST(list_value({values}) AS FLOAT[{len(members)}])",
                    "vector",
                    {"n_dim": len(members)},
                    source=tuple(members),
                )
            )
            continue
        k = kind(column, table, tables, root=root)
        if k == "text" and not text:
            k = "category" if column.distinct <= 25_000 else None
        if k is None:
            continue
        field = names.take(prefix + column.name)
        if k == "dateparts":
            parts = raw.when(column.name, field)
            if (column.daytime or 0) >= 0.01:
                parts[0] = replace(parts[0], options={**parts[0].options, "intraday": True})
            result.extend(parts)
        elif k == "hash":
            namespace = f"{dataset}:{table.fkeys.get(column.name, table.name)}"
            result.append(raw.hashed(column.name, namespace, field))
        elif k == "text":
            result.append(raw.text(column.name, field, budget(column.mean_len)))
        elif k == "number":
            result.append(raw.number(column.name, field))
        elif k == "category":
            result.append(raw.category(column.name, field))
        elif k == "boolean":
            result.append(raw.boolean(column.name, field))
        elif k == "set":
            result.append(raw.setvalued(column.name, field))
    return tuple(result)


def lookups_for(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    tables: Mapping[str, Table],
    dataset: str,
    names: Names,
    *,
    exclude: set[str] = frozenset(),
    hops: int = 2,
    text: bool = True,
    row_time: str | None = None,
) -> tuple[list[Lookup], list[Lookup]]:
    """Forward foreign keys as lookups: a first hop into any dimension, a second hop only into vocabulary tables.

    A timed dimension is read as of the anchor. A dimension timestamp that repeats
    the row's own time column is left out, since the row already carries it.
    """
    aliases = Names(frozenset(RESERVED_ALIASES))
    first: list[Lookup] = []
    second: list[Lookup] = []
    for column, target in table.fkeys.items():
        if column in exclude or target not in tables or tables[target].pkey is None:
            continue
        dimension = tables[target]
        alias = aliases.take(stem(column))
        skip = {dimension.pkey}
        if row_time and dimension.time and same_time(con, table, column, row_time, dimension):
            skip.add(dimension.time)
        columns = typed(dimension, tables, dataset, names, prefix=alias + "_", exclude=skip, text=text)
        onward = [
            (column2, target2)
            for column2, target2 in dimension.fkeys.items()
            if hops >= 2 and target2 in tables and tables[target2].pkey is not None and vocabulary(tables[target2])
        ]
        # The second hop joins through the first hop's columns, so the snapshot keeps them whatever the budget keeps.
        # They are not part of the key: a composite index over them is useless and aborts DuckDB on wide tables.
        first.append(
            Lookup(
                alias,
                target,
                f"{{t}}.{quote(dimension.pkey)} = {{row}}.{quote(column)}",
                columns,
                time=dimension.time,
                key=(dimension.pkey,),
                via=(column,),
                reads=tuple(column2 for column2, _ in onward),
            )
        )
        for column2, target2 in onward:
            dimension2 = tables[target2]
            alias2 = aliases.take(f"{alias}_{stem(column2)}")
            columns2 = typed(
                dimension2, tables, dataset, names, prefix=alias2 + "_", exclude={dimension2.pkey}, text=False
            )
            second.append(
                Lookup(
                    alias2,
                    target2,
                    f"{{t}}.{quote(dimension2.pkey)} = {alias}.{quote(column2)}",
                    columns2,
                    time=dimension2.time,
                    key=(dimension2.pkey,),
                )
            )
    return first, second


def shared_keys(sibling: Table, branch: Table) -> tuple[tuple[str, str], ...]:
    """One (sibling column, branch column) pair per foreign-key target the two tables share."""
    pairs = []
    seen = set()
    for s_column, target in sibling.fkeys.items():
        for b_column, b_target in branch.fkeys.items():
            if target == b_target and target not in seen:
                pairs.append((s_column, b_column))
                seen.add(target)
    return tuple(pairs)


def vocabulary(table: Table) -> bool:
    """A dimension with no time column and no keys of its own, such as a list of conditions or circuits."""
    return table.time is None and not table.fkeys


def belongs_solely(child: Table, dimension: str, tables: Mapping[str, Table]) -> bool:
    """Whether a child's other foreign keys all point at vocabulary tables, so its rows are details of the dimension."""
    return all(
        target == dimension or (target in tables and vocabulary(tables[target])) for target in child.fkeys.values()
    )


def intraday(table: Table) -> bool:
    """Whether a table's time column carries a time of day worth an hour part."""
    column = table.columns.get(table.time) if table.time else None
    return bool(column is not None and (column.daytime or 0) >= 0.01)


def own_columns(table: Table) -> int:
    """Columns that are not keys or the time column."""
    return sum(1 for name in table.columns if name != table.pkey and name != table.time and name not in table.fkeys)


def nested_for(
    con: duckdb.DuckDBPyConnection,
    branch: Table,
    tables: Mapping[str, Table],
    dataset: str,
    *,
    entity: str,
    absorbed: set[str],
    budgets: Mapping[str, int],
) -> tuple[str | None, tuple[Branch, ...]]:
    """Children of the branch's rows or of one dimension it points at, keyed by the branch column that reaches them."""
    options: dict[str, list[tuple[Table, str]]] = {}
    link = own_columns(branch) <= LINK_COLUMNS
    for name, candidate in tables.items():
        if name in absorbed:
            continue
        for column, target in candidate.fkeys.items():
            if target == branch.name and branch.pkey is not None:
                options.setdefault(branch.pkey, []).append((candidate, column))
            elif link and target != entity and target in set(branch.fkeys.values()):
                if candidate.name == branch.name or not belongs_solely(candidate, target, tables):
                    continue
                for via, via_target in branch.fkeys.items():
                    if via_target == target:
                        options.setdefault(via, []).append((candidate, column))
    best: tuple[str | None, list[Branch]] = (None, [])
    for child_key, candidates in options.items():
        branches: list[Branch] = []
        taken = Names()
        for candidate, column in candidates:
            if fanout(con, candidate, (column,)) > NESTED_FANOUT:
                continue
            names = Names()
            columns = typed(
                candidate,
                tables,
                dataset,
                names,
                exclude={column, *((candidate.time,) if candidate.time else ())},
                text=False,
            )
            hop1, _ = lookups_for(
                con, candidate, tables, dataset, names, exclude={column}, hops=1, text=False, row_time=candidate.time
            )
            columns, lookups = trim(columns, [hop1], budgets["fields"])
            branches.append(
                Branch(
                    taken.take(
                        f"{candidate.name}_via_{column}"
                        if sum(1 for c, _ in candidates if c is candidate) > 1
                        else candidate.name
                    ),
                    candidate.name,
                    f'{{t}}.{quote(column)} = {{parent}}."__child_key"',
                    candidate.time,
                    budgets["nested"],
                    columns,
                    lookups=tuple(lookups),
                    key=(column,),
                    intraday=intraday(candidate),
                )
            )
        if len(branches) > len(best[1]):
            best = (child_key, branches)
    return best[0], tuple(best[1])


def derive(con: duckdb.DuckDBPyConnection, tables: Mapping[str, Table], meta: Meta) -> Task:
    """Build the raw-structure task for one entity table from measured tables."""
    entity = tables[meta.entity_table]
    if entity.pkey is None:
        raise ValueError(f"entity table {meta.entity_table!r} has no primary key")
    budgets = BUDGETS[tier(meta.training_rows)]
    root_names = Names()
    root_columns = typed(entity, tables, meta.dataset, root_names, exclude={entity.pkey}, root=True)
    hop1, hop2 = lookups_for(con, entity, tables, meta.dataset, root_names, row_time=entity.time)
    root_columns, root_lookups = trim(root_columns, [hop1, hop2], budgets["root"])

    candidates = [
        (table, column) for table in tables.values() for column, target in table.fkeys.items() if target == entity.name
    ]
    candidates.sort(key=lambda item: (-len(item[0].columns), -item[0].rows, item[0].name, item[1]))
    absorbed: set[str] = set()
    kept: set[str] = set()
    branch_names = Names()
    branches: list[Branch] = []
    for table, column in candidates:
        if table.name in absorbed:
            continue
        names = Names()
        columns = typed(table, tables, meta.dataset, names, exclude={column, *((table.time,) if table.time else ())})
        hop1, hop2 = lookups_for(con, table, tables, meta.dataset, names, exclude={column}, row_time=table.time)
        aliases = Names(frozenset(RESERVED_ALIASES))
        aliases.used.update(lookup.alias for lookup in hop1 + hop2)
        siblings: list[Lookup] = []
        for sibling in tables.values():
            if sibling.name == table.name or sibling.name in absorbed or sibling.name in kept:
                continue
            pairs = shared_keys(sibling, table)
            if len(pairs) < 2:
                continue
            unique, coverage = alignment(con, sibling, table, pairs)
            if unique < ALIGNED[0] or coverage < ALIGNED[1]:
                continue
            alias = aliases.take(sibling.name)
            own = {s for s, _ in pairs}
            sibling_columns = typed(
                sibling,
                tables,
                meta.dataset,
                names,
                prefix=alias + "_",
                exclude=own | ({sibling.time} if sibling.time else set()),
            )
            predicate = " AND ".join(f"{{t}}.{quote(s)} = {{row}}.{quote(b)}" for s, b in pairs)
            siblings.append(
                Lookup(
                    alias,
                    sibling.name,
                    predicate,
                    sibling_columns,
                    time=sibling.time,
                    key=tuple(s for s, _ in pairs),
                    via=tuple(b for _, b in pairs),
                )
            )
            absorbed.add(sibling.name)
        columns, lookups = trim(columns, [hop1, siblings, hop2], budgets["fields"])
        child_key, nested = nested_for(
            con, table, tables, meta.dataset, entity=entity.name, absorbed=absorbed, budgets=budgets
        )
        multiple = sum(1 for t, _ in candidates if t is table) > 1
        branches.append(
            Branch(
                branch_names.take(f"{table.name}_via_{column}" if multiple else table.name),
                table.name,
                f'{{t}}.{quote(column)} = {{parent}}."__entity_key"',
                table.time,
                budgets["branch"],
                columns,
                lookups=tuple(lookups),
                nested=nested,
                child_key=child_key,
                key=(column,),
                intraday=intraday(table),
            )
        )
        kept.add(table.name)

    keys: list[tuple[str, str]] = []
    for via, target in entity.fkeys.items():
        for candidate in tables.values():
            if (
                candidate.name == entity.name
                or candidate.name in absorbed
                or not belongs_solely(candidate, target, tables)
            ):
                continue
            reaching = [column for column, child_target in candidate.fkeys.items() if child_target == target]
            for column in reaching:
                if fanout(con, candidate, (column,)) > budgets["branch"]:
                    continue
                key = f"__{snake(via)}"
                if (key, via) not in keys:
                    keys.append((key, via))
                names = Names()
                columns = typed(
                    candidate,
                    tables,
                    meta.dataset,
                    names,
                    exclude={column, *((candidate.time,) if candidate.time else ())},
                )
                hop1, _ = lookups_for(
                    con, candidate, tables, meta.dataset, names, exclude={column}, hops=1, row_time=candidate.time
                )
                columns, lookups = trim(columns, [hop1], budgets["fields"])
                branches.append(
                    Branch(
                        branch_names.take(
                            f"{stem(via)}_{candidate.name}_via_{column}"
                            if len(reaching) > 1
                            else f"{stem(via)}_{candidate.name}"
                        ),
                        candidate.name,
                        f"{{t}}.{quote(column)} = {{parent}}.{quote(key)}",
                        candidate.time,
                        budgets["branch"],
                        columns,
                        lookups=tuple(lookups),
                        key=(column,),
                        intraday=intraday(candidate),
                    )
                )

    return Task(
        dataset=meta.dataset,
        task=meta.task,
        task_type=meta.task_type,
        entity_table=entity.name,
        entity_key=entity.pkey,
        anchor_entity_column=meta.entity_col,
        target_column=meta.target_col,
        root_time=entity.time,
        columns=root_columns,
        lookups=tuple(root_lookups),
        branches=tuple(branches),
        keys=tuple(keys),
    )


def write(task: Task, meta: Meta, root: pathlib.Path) -> pathlib.Path:
    """Render the task under ``root/<dataset>/<task>`` with the same files as ``raw.write``."""
    directory = root / task.dataset / task.task
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "observations.sql").write_text(raw.query(task))
    (directory / "model.yaml").write_text(yaml.safe_dump(raw.model(task), sort_keys=False, width=1000))
    benchmark = {
        "spec_version": "3",
        "dataset_name": task.dataset,
        "task_name": task.task,
        "task_type": task.task_type,
        "entity_table": task.entity_table,
        "entity_key": task.entity_key,
        "anchor_entity_column": task.anchor_entity_column,
        "timestamp_column": meta.time_col,
        "target_column": task.target_column,
        "model": "model.yaml",
        "query": "observations.sql",
        "sources": raw.sources(task),
    }
    (directory / "benchmark.yaml").write_text(yaml.safe_dump(benchmark, sort_keys=False, width=1000))
    return directory


# --- RelBench cache: schema from Parquet metadata, rows up to the inner-split cutoff ------------------


def relbench_schema(cache: pathlib.Path, dataset: str) -> dict[str, dict]:
    """Keys and time columns of every table, read from the metadata RelBench writes into its Parquet files."""
    import pyarrow.parquet as pq

    schema: dict[str, dict] = {}
    for path in sorted((cache / dataset / "db").glob("*.parquet")):
        metadata = {k.decode(): v.decode() for k, v in (pq.read_schema(path).metadata or {}).items() if k != b"pandas"}
        schema[path.stem] = {
            "pkey": json.loads(metadata.get("pkey_col", "null")),
            "time_col": json.loads(metadata.get("time_col", "null")),
            "fkeys": json.loads(metadata.get("fkey_col_to_pkey_table", "{}")),
        }
    if not schema:
        raise FileNotFoundError(f"no RelBench tables under {cache / dataset / 'db'}")
    return schema


def relbench_sources(cache: pathlib.Path, dataset: str, schema: Mapping[str, dict], upto: str | None) -> dict[str, str]:
    """SQL relations over the cached Parquet tables, censored at ``upto`` the way the harness censors a split."""
    sources = ParquetSources(cache / dataset / "db", upto=upto)
    return {table: sources.relation(table) for table in schema}


def relbench_meta(cache: pathlib.Path, dataset: str, task: str) -> tuple[Meta, str]:
    """Task metadata and the validation cutoff from RelBench, without loading the database."""
    import pyarrow.parquet as pq
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    os.environ.setdefault("RELBENCH_CACHE_DIR", str(cache))
    source = get_dataset(dataset, download=False)
    entity = get_task(dataset, task, download=False)
    train = cache / dataset / "tasks" / task / "train.parquet"
    rows = pq.read_metadata(train).num_rows if train.exists() else None
    kind = str(getattr(entity.task_type, "value", entity.task_type))
    meta = Meta(dataset, task, kind, entity.entity_table, entity.entity_col, entity.time_col, entity.target_col, rows)
    return meta, str(source.val_timestamp)


def derive_relbench(cache: pathlib.Path, name: str, output: pathlib.Path, *, threads: int = 8) -> pathlib.Path:
    """Derive one RelBench task from the cache as the harness would present it, writing its bundle under ``output``."""
    dataset, task = name.split("/")
    meta, cutoff = relbench_meta(cache, dataset, task)
    schema = relbench_schema(cache, dataset)
    con = duckdb.connect()
    con.execute(f"SET threads TO {threads}")
    tables = profile(con, relbench_sources(cache, dataset, schema, cutoff), schema)
    directory = write(derive(con, tables, meta), meta, output)
    record = {
        "dataset": dataset,
        "task": task,
        "training_rows": meta.training_rows,
        "tier": tier(meta.training_rows),
        "censored_upto": cutoff,
    }
    (directory / "derivation.json").write_text(json.dumps(record, indent=2) + "\n")
    return directory


# --- offline driver over a RelBench cache ---------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relbench", required=True, help="dataset/task to derive from the RelBench cache")
    parser.add_argument("--cache", type=pathlib.Path, required=True, help="RelBench cache root")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Root receiving <dataset>/<task>/")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    directory = derive_relbench(args.cache, args.relbench, args.output, threads=args.threads)
    print(json.dumps({"task": args.relbench, "bundle": str(directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
