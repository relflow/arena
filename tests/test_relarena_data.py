"""Protect the portable benchmark's temporal and label-isolation boundaries."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.relarena import resources
from benchmarks.relarena.spec import BenchmarkSpec

duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")
dataset = pytest.importorskip("benchmarks.relarena.dataset")
store = pytest.importorskip("benchmarks.relarena.store")

ROOT = Path(__file__).resolve().parents[1]
QUERY = """
WITH anchors AS (
    SELECT __row_position, __entity_key, __as_of FROM read_parquet($anchors)
    WHERE __row_position >= $start AND __row_position < $stop
), roots AS (
    SELECT anchor.__row_position, root.entity_id, root.root_value
    FROM anchors AS anchor LEFT JOIN entities AS root
    ON anchor.__entity_key = root.entity_id AND root.created_at <= anchor.__as_of
    QUALIFY row_number() OVER (
        PARTITION BY anchor.__row_position
        ORDER BY root.created_at DESC NULLS LAST, root.__source_row_id DESC NULLS LAST
    ) = 1
), ranked AS (
    SELECT anchor.__row_position, event.event_time, event.__source_row_id, event.value,
        date_diff('microseconds', event.event_time, anchor.__as_of) / 86400000000.0 AS age_days,
        row_number() OVER (
            PARTITION BY anchor.__row_position
            ORDER BY event.event_time DESC, event.__source_row_id DESC
        ) AS recency
    FROM anchors AS anchor INNER JOIN events AS event
    ON anchor.__entity_key = event.entity_id AND event.event_time <= anchor.__as_of
), histories AS (
    SELECT __row_position,
        list(struct_pack(value := value, age_days := age_days)
             ORDER BY event_time, __source_row_id) AS events
    FROM ranked WHERE recency <= 2 GROUP BY __row_position
)
SELECT anchor.__row_position, root.entity_id, root.root_value,
    coalesce(history.events, []::STRUCT(value DOUBLE, age_days DOUBLE)[]) AS events
FROM anchors AS anchor LEFT JOIN roots AS root USING (__row_position)
LEFT JOIN histories AS history USING (__row_position)
"""


def specification(directory, query=QUERY):
    model = directory / "model.yaml"
    model.write_text("fields:\n  root_value: {type: number}\n")
    sql = directory / "observations.sql"
    sql.write_text(query)
    return BenchmarkSpec(
        dataset_name="unit-db",
        task_name="unit-regression",
        task_type="regression",
        entity_table="entities",
        entity_key="entity_id",
        anchor_entity_column="seed_id",
        timestamp_column="timestamp",
        target_column="label",
        model_path=model,
        query_path=sql,
        sources={
            "entities": {
                "columns": ("entity_id", "created_at", "root_value"),
                "indexes": (("entity_id", "created_at"),),
            },
            "events": {"columns": ("entity_id", "event_time", "value"), "indexes": (("entity_id", "event_time"),)},
        },
    )


def prepared(directory, *, include_target=True, empty=False, query=QUERY, task_type="regression"):
    spec = replace(specification(directory, query), task_type=task_type)
    entities = pd.DataFrame(
        {
            "entity_id": [10, 20],
            "created_at": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "root_value": [1.5, 2.5],
            "ignored": ["not", "stored"],
        }
    )
    events = pd.DataFrame(
        {
            "entity_id": [10, 10, 10, 20, 10],
            "event_time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-02", "2020-01-02"]),
            "value": [1.0, 2.0, 4.0, 20.0, 2.5],
        }
    )
    anchors = pd.DataFrame(
        {
            "seed_id": [10, 10, 20, 20, 10],
            "timestamp": pd.to_datetime(
                ["2020-01-01 12:00", "2020-01-03", "2020-01-01", "2020-01-04", "2020-01-05"], format="mixed"
            ),
            "label": [True, False, True, False, True] if task_type == "binary_classification" else [1, 2, 3, 4, 5],
        }
    )
    if not include_target:
        anchors = anchors.drop(columns="label")
    if empty:
        anchors = anchors.iloc[:0]
    db = directory / "cache" / spec.dataset_name / "db"
    db.mkdir(parents=True, exist_ok=True)
    for name, frame in (("entities", entities), ("events", events)):
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), db / f"{name}.parquet")
    snapshot = store.prepare_store(store.ParquetSources(db), spec, directory / "store")
    path = store.prepare_anchors(
        SimpleNamespace(df=anchors),
        spec,
        directory / "anchors.parquet",
        include_target=include_target,
        row_group_size=2,
    )
    return spec, snapshot, path


def test_projection_temporal_cutoffs_and_tied_event_order(tmp_path):
    spec, snapshot, anchors = prepared(tmp_path)
    manifest = store.read_store_manifest(snapshot)
    assert manifest["projected_columns"] == {
        "entities": ["entity_id", "created_at", "root_value"],
        "events": ["entity_id", "event_time", "value"],
    }
    output = dataset.materialize(snapshot, anchors, spec, tmp_path / "output", include_target=True, fetch_size=2)
    table = pq.read_table(output)
    rows = table.to_pylist()
    assert table["__row_position"].to_pylist() == list(range(5))
    assert table["target"].type == pa.float64() and table["target"].to_pylist() == [1, 2, 3, 4, 5]
    assert rows[0]["events"] == [{"value": 1.0, "age_days": 0.5}]
    assert [event["value"] for event in rows[1]["events"]] == [2.0, 2.5]
    assert rows[2]["root_value"] is None and rows[2]["events"] == []
    assert [event["value"] for event in rows[4]["events"]] == [2.5, 4.0]


@pytest.mark.parametrize("empty", [False, True])
def test_unlabeled_pretraining_preserves_typed_empty_histories(tmp_path, empty):
    spec, snapshot, anchors = prepared(tmp_path, include_target=False, empty=empty)
    assert "__target" not in pq.read_schema(anchors).names
    output = dataset.materialize(snapshot, anchors, spec, tmp_path / "output", include_target=False, fetch_size=2)
    table = pq.read_table(output)
    assert not {"target", "__target", "label"} & set(table.column_names)
    assert pa.types.is_list(table["events"].type) and pa.types.is_struct(table["events"].type.value_type)
    assert table["events"].type.value_type == pa.struct([("value", pa.float64()), ("age_days", pa.float64())])
    if empty:
        assert table.num_rows == 0
    else:
        assert table["events"].to_pylist()[2] == []


def test_supervised_boolean_labels_remain_hidden_from_the_feature_snapshot(tmp_path):
    spec, snapshot, anchors = prepared(tmp_path, task_type="binary_classification")
    output = dataset.materialize(snapshot, anchors, spec, tmp_path / "output", include_target=True, fetch_size=3)
    table = pq.read_table(output)
    assert table["target"].type == pa.bool_()
    assert table["target"].to_pylist() == [True, False, True, False, True]
    with duckdb.connect(str(snapshot), read_only=True) as connection:
        assert connection.sql("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall() == [
            ("__relflow_snapshot_manifest",),
            ("entities",),
            ("events",),
        ]


@pytest.mark.parametrize(
    "query",
    [
        f"SELECT * FROM ({QUERY}) WHERE __row_position <> 1",
        f"SELECT * FROM ({QUERY}) UNION ALL SELECT * FROM ({QUERY})",
        f"SELECT * REPLACE (__row_position + 1 AS __row_position) FROM ({QUERY})",
    ],
    ids=["missing", "duplicate", "unexpected"],
)
def test_invalid_query_positions_keep_existing_output_and_close_resources(tmp_path, query):
    spec, snapshot, anchors = prepared(tmp_path, query=query)
    destination = tmp_path / "output.parquet"
    destination.write_bytes(b"previous result")
    with pytest.raises(ValueError, match="exactly one row per anchor"):
        dataset.materialize(snapshot, anchors, spec, destination, include_target=True, fetch_size=2)
    assert destination.read_bytes() == b"previous result"
    assert not list(tmp_path.glob("relflow-observations-*"))
    with duckdb.connect(str(snapshot)) as connection:
        connection.execute("CREATE TABLE cleanup_probe (id INTEGER)")


def test_labeled_anchors_cannot_be_used_for_unlabeled_pretraining(tmp_path):
    spec, snapshot, anchors = prepared(tmp_path)
    with pytest.raises(ValueError, match="unexpectedly contains target"):
        dataset.materialize(snapshot, anchors, spec, tmp_path / "output", include_target=False, fetch_size=2)


def test_feature_query_cannot_supply_the_supervised_target(tmp_path):
    spec, snapshot, anchors = prepared(tmp_path, query=f"SELECT *, 42.0 AS target FROM ({QUERY})")
    with pytest.raises(ValueError, match="runner alone attaches target"):
        dataset.materialize(snapshot, anchors, spec, tmp_path / "output", include_target=True, fetch_size=2)


def test_parallel_resource_budgets_preserve_temporal_values_positions_and_ties(tmp_path):
    spec, snapshot, anchors = prepared(tmp_path)
    settings = resources.resolve(
        tmp_path / "owned-spill", threads=1, memory_limit="128MiB", max_temp_directory_size="256MiB"
    )["settings"]
    sequential = dataset.materialize(
        snapshot,
        anchors,
        spec,
        tmp_path / "sequential.parquet",
        include_target=True,
        fetch_size=2,
        duckdb_config=settings,
    )
    parallel_settings = {**settings, "threads": 4, "memory_limit": "512MiB"}
    parallel = dataset.materialize(
        snapshot,
        anchors,
        spec,
        tmp_path / "parallel.parquet",
        include_target=True,
        fetch_size=3,
        duckdb_config=parallel_settings,
    )
    assert pq.read_table(sequential).equals(pq.read_table(parallel), check_metadata=True)
    assert list((tmp_path / "owned-spill").iterdir()) == []
