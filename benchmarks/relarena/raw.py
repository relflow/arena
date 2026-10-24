"""Raw-structure tasks: the structures a derivation produces and the query and schema they render to.

A task is one entity row plus its related rows, nested as RelFlow branches.
Columns are carried as they are stored; the only derived values are the ages of
timestamps relative to the anchor, which express the temporal relation the
anchor defines. No counts, windows, rates, or summaries are computed in SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DAY = 86400000000.0
TEXT = {
    "type": "text",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "encoder_batch_size": 320,
    "encoder_pooling": "mean",
}
DATEPARTS = ["month_of_year", "day_of_week"]


@dataclass(frozen=True)
class Col:
    """One output column: ``name`` in the model, ``expr`` in SQL (``{t}`` is the table alias), and its field."""

    name: str
    expr: str
    kind: str
    options: dict = field(default_factory=dict)
    source: tuple[str, ...] = ()

    def spec(self) -> dict:
        if self.kind == "text":
            return {**TEXT, "max_length": self.options.get("max_length", 64)}
        if self.kind == "dateparts":
            return {
                "type": "dateparts",
                "dateparts": list(DATEPARTS) + (["hour_of_day"] if self.options.get("intraday") else []),
            }
        if self.kind == "hash":
            return {"type": "hash", "n_hashes": 4}
        if self.kind == "vector":
            return {"type": "vector", "n_dim": self.options["n_dim"]}
        return {"type": self.kind}


def number(column: str, name: str | None = None) -> Col:
    return Col(name or column.lower(), f'{{t}}."{column}"', "number", source=(column,))


def category(column: str, name: str | None = None) -> Col:
    return Col(name or column.lower(), f'{{t}}."{column}"', "category", source=(column,))


def boolean(column: str, name: str | None = None) -> Col:
    return Col(name or column.lower(), f'{{t}}."{column}"', "boolean", source=(column,))


def text(column: str, name: str | None = None, max_length: int = 64) -> Col:
    """A text column; budgets never exceed 96 tokens, the suite's per-field ceiling."""
    return Col(
        name or column.lower(), f'{{t}}."{column}"', "text", {"max_length": min(max_length, 96)}, source=(column,)
    )


def hashed(column: str, namespace: str, name: str | None = None) -> Col:
    expr = (
        f"""CASE WHEN {{t}}."{column}" IS NULL THEN NULL ELSE '{namespace}:' || CAST({{t}}."{column}" AS VARCHAR) END"""
    )
    return Col(name or column.lower(), expr, "hash", source=(column,))


def setvalued(column: str, name: str | None = None) -> Col:
    return Col(name or column.lower(), f'{{t}}."{column}"', "set", source=(column,))


def when(column: str, name: str) -> list[Col]:
    """A timestamp as calendar parts plus its age in days before the anchor."""
    return [
        Col(name, f'{{t}}."{column}"', "dateparts", source=(column,)),
        Col(
            f"{name}_age_days",
            f"""date_diff('microseconds', {{t}}."{column}", {{anchor}}."__as_of") / {DAY}""",
            "number",
            source=(column,),
        ),
    ]


@dataclass(frozen=True)
class Lookup:
    """A many-to-one join from a row to one related row, censored by ``time`` when the related table has one."""

    alias: str
    table: str
    on: str  # SQL predicate with {t} (related row) and {row} (the row being enriched)
    columns: tuple[Col, ...]
    time: str | None = None
    key: tuple[str, ...] = ()  # related-table columns the predicate uses; the snapshot indexes them
    via: tuple[str, ...] = ()  # row-side columns the predicate reads
    reads: tuple[str, ...] = ()  # related-table columns that onward lookups join through; persisted, not indexed

    def sql(self, row: str, anchor: str) -> str:
        """A timed lookup takes the latest row before the anchor; a static one joins its table on its unique key."""
        predicate = self.on.format(t=self.alias if not self.time else "linked", row=row)
        if not self.time:
            return f'    LEFT JOIN "{self.table}" AS {self.alias} ON {predicate}'
        return (
            f'    LEFT JOIN LATERAL (SELECT linked.* FROM "{self.table}" AS linked WHERE {predicate} '
            f'AND linked."{self.time}" <= {anchor}."__as_of" '
            f'ORDER BY linked."{self.time}" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS {self.alias} ON TRUE'
        )


@dataclass(frozen=True)
class Branch:
    """Rows of ``table`` related to the parent key, censored by ``time`` and capped by recency."""

    name: str
    table: str
    on: str  # predicate with {t} (child row) and {parent} (the parent key expression)
    time: str | None
    length: int
    columns: tuple[Col, ...]
    lookups: tuple[Lookup, ...] = ()
    nested: tuple[Branch, ...] = ()
    child_key: str | None = None  # child column that nested branches join to
    key: tuple[str, ...] = ()
    overflow: str = "tail"
    intraday: bool = False  # the row timestamp has a time of day worth an hour part

    def __post_init__(self) -> None:
        if isinstance(self.columns, Col):
            object.__setattr__(self, "columns", (self.columns,))

    def own(self) -> tuple[Col, ...]:
        """The row's own timestamp as calendar parts and its age before the anchor."""
        if not self.time:
            return ()
        return (
            Col(
                "occurred_at",
                f'{{t}}."{self.time}"',
                "dateparts",
                {"intraday": True} if self.intraday else {},
                source=(self.time,),
            ),
            Col(
                "age_days",
                f"""date_diff('microseconds', {{t}}."{self.time}", {{anchor}}."__as_of") / {DAY}""",
                "number",
            ),
        )

    def spec(self) -> dict:
        fields = {column.name: column.spec() for column in (*self.own(), *self.columns)}
        for lookup in self.lookups:
            fields.update({column.name: column.spec() for column in lookup.columns})
        for inner in self.nested:
            fields[inner.name] = inner.spec()
        return {"type": "branch", "length": self.length, "overflow": self.overflow, "fields": fields}


@dataclass(frozen=True)
class Task:
    dataset: str
    task: str
    task_type: str
    entity_table: str
    entity_key: str
    anchor_entity_column: str
    target_column: str
    root_time: str | None
    columns: tuple[Col, ...]
    lookups: tuple[Lookup, ...]
    branches: tuple[Branch, ...]
    keys: tuple[
        tuple[str, str], ...
    ] = ()  # internal root columns (``__name``, source column) that branches join through
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4


def selections(columns: tuple[Col, ...], table: str, anchor: str, indent: str = "        ") -> list[str]:
    return [f'{indent}{column.expr.format(t=table, anchor=anchor)} AS "{column.name}"' for column in columns]


def branch_ctes(branch: Branch, parent: str, prefix: str, nested: bool) -> tuple[list[str], str]:
    """Emit the CTEs for one branch and return their text plus the grouped CTE name.

    ``parent`` is a CTE carrying ``__row_position``, ``__as_of``, and the key
    columns the branch predicate reads; nested branches read ``__child_key``.
    """
    events, ranked, grouped = f"{prefix}_events", f"{prefix}_ranked", f"{prefix}_grouped"
    keys = ["__row_position", "__child_key"] if nested else ["__row_position"]
    joins = [f'    INNER JOIN "{branch.table}" AS event ON {branch.on.format(t="event", parent="parent")}']
    if branch.time:
        joins[0] += f' AND event."{branch.time}" <= parent."__as_of"'
    joins.extend(lookup.sql("event", "parent") for lookup in branch.lookups)
    fields = selections((*branch.own(), *branch.columns), "event", "parent")
    for lookup in branch.lookups:
        fields.extend(selections(lookup.columns, lookup.alias, "parent"))
    time_expr = f'event."{branch.time}"' if branch.time else "NULL"
    carried = ", ".join(f'parent."{key}"' for key in keys)
    lines = [
        f"{events} AS (",
        f'    SELECT {carried}, parent."__as_of",',
        f'        {time_expr} AS "__event_time", event."__source_row_id" AS "__source_order",',
    ]
    if branch.nested:
        lines.append(f'        event."{branch.child_key}" AS "__nested_key",')
    lines.append(",\n".join(fields))
    lines.append(f"    FROM {parent} AS parent")
    lines.extend(joins)
    lines.append("),")
    text = ["\n".join(lines)]
    partition = ", ".join(f'"{key}"' for key in keys)
    text.append(
        f"{ranked} AS (\n    SELECT *, row_number() OVER (PARTITION BY {partition} "
        f'ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"\n    FROM {events}\n),'
    )
    inner_groups = []
    if branch.nested:
        text.append(
            f'{events}_keys AS (\n    SELECT "__row_position", "__as_of", "__nested_key" AS "__child_key" FROM {ranked} '
            f'WHERE "__recency" <= {branch.length}\n),'
        )
        for inner in branch.nested:
            inner_text, inner_grouped = branch_ctes(inner, f"{events}_keys", f"{prefix}_{inner.name}", nested=True)
            text.extend(inner_text)
            inner_groups.append((inner.name, inner_grouped))
    packed = [f'"{column.name}" := {ranked}."{column.name}"' for column in (*branch.own(), *branch.columns)]
    for lookup in branch.lookups:
        packed.extend(f'"{column.name}" := {ranked}."{column.name}"' for column in lookup.columns)
    inner_joins = []
    for name, inner_grouped in inner_groups:
        packed.append(f'"{name}" := coalesce({inner_grouped}."{name}", [])')
        inner_joins.append(
            f'    LEFT JOIN {inner_grouped} ON {inner_grouped}."__row_position" = {ranked}."__row_position" '
            f'AND {inner_grouped}."__child_key" = {ranked}."__nested_key"'
        )
    group = ", ".join(f'{ranked}."{key}"' for key in keys)
    text.append(
        f"{grouped} AS (\n    SELECT {group},\n        list(struct_pack(\n            "
        + ",\n            ".join(packed)
        + f'\n        ) ORDER BY {ranked}."__event_time" ASC NULLS LAST, {ranked}."__source_order" ASC) AS "{branch.name}"\n'
        + f"    FROM {ranked}\n"
        + ("\n".join(inner_joins) + "\n" if inner_joins else "")
        + f'    WHERE {ranked}."__recency" <= {branch.length}\n    GROUP BY {group}\n),'
    )
    return text, grouped


def query(task: Task) -> str:
    lines = [
        "-- Raw structure: the entity row, its related rows nested by recency, and their looked-up context.",
        "-- No counts, windows, or summaries are computed here; ages measure each timestamp against the anchor.",
        "WITH",
        "anchors AS (",
        '    SELECT "__row_position", "__entity_key", "__as_of"',
        "    FROM read_parquet($anchors)",
        '    WHERE "__row_position" >= $start AND "__row_position" < $stop',
        "),",
    ]
    root_join = f'    LEFT JOIN "{task.entity_table}" AS root ON anchor."__entity_key" = root."{task.entity_key}"'
    order = 'root."__source_row_id" DESC NULLS LAST'
    if task.root_time:
        root_join += f' AND root."{task.root_time}" <= anchor."__as_of"'
        order = f'root."{task.root_time}" DESC NULLS LAST, root."__source_row_id" DESC NULLS LAST'
    fields = [f'        root."{column}" AS "{name}"' for name, column in task.keys]
    fields.extend(selections(task.columns, "root", "anchor"))
    for lookup in task.lookups:
        fields.extend(selections(lookup.columns, lookup.alias, "anchor"))
    lines.append("root AS (")
    lines.append('    SELECT anchor."__row_position", anchor."__entity_key", anchor."__as_of",')
    lines.append('        anchor."__as_of" AS "as_of",')
    lines.append(",\n".join(fields))
    lines.append("    FROM anchors AS anchor")
    lines.append(root_join)
    lines.extend(lookup.sql("root", "anchor") for lookup in task.lookups)
    lines.append(f'    QUALIFY row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY {order}) = 1')
    lines.append("),")
    grouped_names = []
    for branch in task.branches:
        text, grouped = branch_ctes(branch, "root", branch.name, nested=False)
        lines.extend(text)
        grouped_names.append((branch.name, grouped))
    lines[-1] = lines[-1].rstrip(",")
    output = ['root."__row_position"', 'root."as_of"']
    output.extend(f'root."{column.name}"' for column in task.columns)
    for lookup in task.lookups:
        output.extend(f'root."{column.name}"' for column in lookup.columns)
    for name, grouped in grouped_names:
        output.append(f'coalesce({grouped}."{name}", []) AS "{name}"')
    lines.append("SELECT\n    " + ",\n    ".join(output))
    lines.append("FROM root")
    for name, grouped in grouped_names:
        lines.append(f'LEFT JOIN {grouped} USING ("__row_position")')
    lines.append('ORDER BY root."__row_position"')
    return "\n".join(lines) + "\n"


def model(task: Task) -> dict:
    fields: dict = {"as_of": {"type": "dateparts", "dateparts": list(DATEPARTS)}}
    fields.update({column.name: column.spec() for column in task.columns})
    for lookup in task.lookups:
        fields.update({column.name: column.spec() for column in lookup.columns})
    for branch in task.branches:
        fields[branch.name] = branch.spec()
    fields["target"] = {"type": "boolean" if task.task_type == "binary_classification" else "number", "mask": True}
    return {"d_model": task.d_model, "n_layers": task.n_layers, "n_heads": task.n_heads, "fields": fields}


def sources(task: Task) -> dict:
    """Every table and column the query reads, with indexes on the keys and times it joins through."""
    tables: dict[str, dict] = {}

    def add(table: str, columns: tuple[str, ...], index: tuple[str, ...] | None = None) -> None:
        entry = tables.setdefault(table, {"columns": [], "indexes": []})
        for column in columns:
            if column not in entry["columns"]:
                entry["columns"].append(column)
        if index and list(index) not in entry["indexes"]:
            entry["indexes"].append(list(index))

    add(task.entity_table, (task.entity_key, *((task.root_time,) if task.root_time else ())), (task.entity_key,))
    add(task.entity_table, tuple(column for _, column in task.keys))
    add(task.entity_table, tuple(c for column in task.columns for c in column.source))
    for lookup in task.lookups:
        add(task.entity_table, lookup.via)
        add(lookup.table, (*lookup.key, *lookup.reads, *((lookup.time,) if lookup.time else ())), lookup.key)
        add(lookup.table, tuple(c for column in lookup.columns for c in column.source))

    def walk(branch: Branch) -> None:
        add(
            branch.table,
            (*branch.key, *((branch.time,) if branch.time else ()), *((branch.child_key,) if branch.child_key else ())),
            (*branch.key, *((branch.time,) if branch.time else ())),
        )
        add(branch.table, tuple(c for column in branch.columns for c in column.source))
        for lookup in branch.lookups:
            add(branch.table, lookup.via)
            add(lookup.table, (*lookup.key, *lookup.reads, *((lookup.time,) if lookup.time else ())), lookup.key)
            add(lookup.table, tuple(c for column in lookup.columns for c in column.source))
        for inner in branch.nested:
            walk(inner)

    for branch in task.branches:
        walk(branch)
    return tables
