"""The derivation engine builds a raw-structure task from keys and measurements alone, and its renderings run."""

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from benchmarks.relarena import derive, raw

SCHEMA = {
    "teams": {"pkey": "team_id"},
    "users": {"pkey": "user_id", "time_col": "joined", "fkeys": {"team_id": "teams"}},
    "regions": {"pkey": "region_id"},
    "shops": {"pkey": "shop_id", "fkeys": {"region_id": "regions"}},
    "orders": {"pkey": "order_id", "time_col": "placed", "fkeys": {"user_id": "users", "shop_id": "shops"}},
    "ratings": {"time_col": "rated", "fkeys": {"user_id": "users", "shop_id": "shops"}},
    "items": {"pkey": "item_id", "time_col": None, "fkeys": {"order_id": "orders"}},
    "team_events": {"time_col": "happened", "fkeys": {"team_id": "teams"}},
}


@pytest.fixture(scope="module")
def database():
    rng = np.random.default_rng(0)
    con = duckdb.connect()
    users, shops, orders = 200, 12, 2000
    day = np.timedelta64(1, "D")
    start = np.datetime64("2020-01-01T00:00:00")
    frames = {
        "teams": pa.table({"team_id": np.arange(4), "name": ["red", "green", "blue", "gold"]}),
        "users": pa.table(
            {
                "user_id": np.arange(users),
                "team_id": rng.integers(0, 4, users),
                "joined": start + rng.integers(0, 300, users) * day,
                "country": rng.choice(["US", "DE", "FR", "JP"], users),
                "bio": [
                    " ".join(rng.choice(["likes", "coffee", "and", "long", "walks", "by", "the", "sea"], 12))
                    for _ in range(users)
                ],
                "active": np.where(rng.random(users) < 0.5, 1.0, np.nan),
                **{f"e_{i}": rng.normal(size=users) for i in range(1, 9)},
            }
        ),
        "shops": pa.table(
            {
                "shop_id": np.arange(shops),
                "name": [f"Shop Number {i} Market" for i in range(shops)],
                "city": rng.choice(["Berlin", "Paris"], shops),
                "region_id": rng.integers(0, 3, shops),
            }
        ),
    }
    order_users, order_shops = rng.integers(0, users, orders), rng.integers(0, shops, orders)
    frames["orders"] = pa.table(
        {
            "order_id": np.arange(orders),
            "user_id": order_users,
            "shop_id": order_shops,
            "placed": start + rng.integers(0, 700, orders) * day,
            "amount": rng.gamma(2.0, 20.0, orders),
            "status_code": rng.integers(0, 5, orders),
            "gift": rng.integers(0, 2, orders),
        }
    )
    pairs = sorted({(int(u), int(s)) for u, s in zip(order_users, order_shops, strict=True)})
    frames["ratings"] = pa.table(
        {
            "user_id": np.array([u for u, _ in pairs]),
            "shop_id": np.array([s for _, s in pairs]),
            "rated": start + rng.integers(0, 700, len(pairs)) * day,
            "stars": rng.integers(1, 6, len(pairs)),
        }
    )
    frames["items"] = pa.table(
        {
            "item_id": np.arange(orders * 3),
            "order_id": np.repeat(np.arange(orders), 3),
            "sku": ["".join(rng.choice(list("ABCDEFGHJK0123456789"), 10)) for _ in range(orders * 3)],
            "qty": rng.integers(1, 9, orders * 3),
        }
    )
    frames["team_events"] = pa.table(
        {
            "team_id": np.repeat(np.arange(4), 5),
            "happened": start + rng.integers(0, 700, 20) * day,
            "note": rng.choice(["launch", "offsite", "review"], 20),
        }
    )
    frames["regions"] = pa.table({"region_id": np.arange(3), "label": ["north", "south", "east"]})
    for name, table in frames.items():
        con.register(f"{name}_arrow", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM {name}_arrow')
        con.execute(
            f'CREATE VIEW "{name}_rows" AS SELECT *, row_number() OVER () - 1 AS "__source_row_id" FROM "{name}"'
        )
    return con


@pytest.fixture(scope="module")
def task(database):
    profiles = derive.profile(database, {name: f'"{name}"' for name in SCHEMA}, SCHEMA)
    meta = derive.Meta("toy", "churn", "binary_classification", "users", "user_id", "timestamp", "target")
    return derive.derive(database, profiles, meta)


def test_relations_follow_the_foreign_key_graph(task):
    assert task.entity_key == "user_id" and task.root_time == "joined"
    assert [lookup.table for lookup in task.lookups] == ["teams"]
    branches = {branch.table: branch for branch in task.branches}
    assert set(branches) == {"orders", "team_events"}, (
        "ratings is absorbed as an aligned sibling, team events reach through the root's team"
    )
    orders = branches["orders"]
    assert [lookup.table for lookup in orders.lookups] == ["shops", "ratings", "regions"], (
        "the second hop reaches the vocabulary behind the shop"
    )
    shops, ratings = orders.lookups[0], orders.lookups[1]
    assert shops.key == ("shop_id",) and shops.reads == ("region_id",), "only the joined key is indexed"
    sources = raw.sources(task)
    assert sources["shops"]["indexes"] == [["shop_id"]] and "region_id" in sources["shops"]["columns"]
    assert ratings.time == "rated" and ratings.key == ("user_id", "shop_id") and ratings.via == ("user_id", "shop_id")
    assert orders.child_key == "order_id" and [nested.table for nested in orders.nested] == ["items"]
    assert orders.time == "placed" and orders.length == derive.BUDGETS["small"]["branch"] and orders.overflow == "tail"
    assert orders.nested[0].length == derive.BUDGETS["small"]["nested"]
    assert task.keys == (("__team_id", "team_id"),)
    assert branches["team_events"].on == '{t}."team_id" = {parent}."__team_id"'


def test_columns_follow_the_measured_rules(task):
    root = {column.name: column.kind for column in task.columns}
    assert root["team_id"] == "hash" and root["country"] == "category" and root["bio"] == "text"
    assert root["joined"] == "dateparts" and root["joined_age_days"] == "number"
    assert root["active"] == "category", "a one-value-or-null column is a flag"
    assert root["e"] == "vector" and [c for c in task.columns if c.name == "e"][0].options == {"n_dim": 8}
    orders = {column.name: column.kind for column in task.branches[0].columns}
    assert orders == {
        "order_id": "hash",
        "shop_id": "hash",
        "amount": "number",
        "status_code": "category",
        "gift": "category",
    }, "a branch row keeps its own key as a hash; identifier-named codes and 0/1 flags are categories"
    nested = {column.name: column.kind for column in task.branches[0].nested[0].columns}
    assert nested == {"sku": "hash", "qty": "number"}, (
        "an unreferenced row id is dropped; opaque fixed-width strings hash"
    )
    lookups = {column.name: column.kind for lookup in task.branches[0].lookups for column in lookup.columns}
    assert lookups["shop_name"] == "category", "twelve shop names recurring over thousands of orders form a vocabulary"
    assert lookups["shop_city"] == "category" and lookups["ratings_stars"] == "number"


def test_rendered_query_returns_one_row_per_anchor_and_the_schema_loads(task, database, tmp_path):
    from relflow.architecture.template import read

    anchors = pa.table(
        {
            "__row_position": pa.array(np.arange(50, dtype=np.int64)),
            "__entity_key": pa.array(np.arange(50, dtype=np.int64)),
            "__as_of": pa.array([np.datetime64("2021-06-01T00:00:00")] * 50),
        }
    )
    pq.write_table(anchors, tmp_path / "anchors.parquet")
    sql = raw.query(task)
    for name in SCHEMA:
        sql = sql.replace(f'"{name}" AS', f'"{name}_rows" AS')
    result = (
        database.execute(sql, {"anchors": str(tmp_path / "anchors.parquet"), "start": 0, "stop": 50}).arrow().read_all()
    )
    assert result.num_rows == 50 and sorted(result["__row_position"].to_pylist()) == list(range(50))
    orders = result["orders"]
    assert pa.types.is_list(orders.type) and orders.type.value_type.get_field_index("items") >= 0
    assert max(pa.compute.list_value_length(orders).to_pylist()) <= task.branches[0].length
    meta = derive.Meta("toy", "churn", "binary_classification", "users", "user_id", "timestamp", "target")
    directory = derive.write(task, meta, tmp_path / "derived")
    options, _ = read(directory / "model.yaml", {})
    fields = yaml.safe_load((directory / "model.yaml").read_text())["fields"]
    assert fields["target"] == {"type": "boolean", "mask": True} and fields["orders"]["type"] == "branch"
    assert options["d_model"] == task.d_model


def test_lookup_aliases_never_shadow_the_query_aliases(task):
    aliases = [lookup.alias for branch in task.branches for lookup in branch.lookups] + [
        lookup.alias for lookup in task.lookups
    ]
    assert not set(aliases) & derive.RESERVED_ALIASES
    taken = derive.Names(frozenset(derive.RESERVED_ALIASES))
    assert taken.take("event") == "col_event" and taken.take("event") == "col_event_2"
