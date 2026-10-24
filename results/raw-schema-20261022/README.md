# Raw-structure schemas (October 22, 2026)

> **Superseded on October 23, 2026.** The hand-written catalog this page
> describes, the 21 `benchmarks/relarena/tasks/*` bundles, the task
> definitions in `raw.py`, the PRQL renderings, the profiles, and the older
> campaign scripts were removed from the package in favor of structures the
> derivation engine builds from the RelBench cache at the validation cutoff,
> which is what RelArena's model contract permits. The text below is kept as
> the record of that catalog and of how the engine was validated against it;
> the validation outputs live in this directory, not in the package.

Every RelArena task was regenerated from `benchmarks/relarena/raw.py` so that the observation
query only joins tables into nested contexts. Hand-crafted features are a
non-starter: no counts, windows, rates, distinct counts, first or last ages,
peer contexts, or rolling summaries are computed in SQL. RelFlow reads the
raw rows and learns what it needs from them.

## What a task looks like now

- The root is the entity row with its stored columns, typed by what they
  are: numbers, categories, hashed identifiers, text, calendar parts.
- Timestamps carry two raw relations to the anchor: their calendar parts and
  their age in days before the anchor. Branch rows name theirs `occurred_at`
  and `age_days`; root and lookup timestamps carry a prefix.
- Every related table becomes a branch of its rows before the anchor,
  ordered by time and capped by recency (`length`, `overflow: tail`).
- Many-to-one relations are looked up into the row that refers to them: the
  reviewer behind a review, the race, circuit, constructor, qualifying, and
  standings behind a result, the post behind a vote, the study behind a
  facility link. Timed lookups take the latest row before the anchor; static
  dimension tables join through a deduplicated copy.
- One-to-many relations below a child row become nested branches: a user's
  posts carry their own votes and comments, a search carries its results, a
  facility's studies carry their conditions, interventions, sponsors,
  reported events, and outcomes.
- Event records pack the 100 word-count columns into one 100-dimensional
  vector; Amazon categories stay a set.
- Identifiers that recur across fields are hashed under one namespace per
  table so equality survives across branches.

Branch caps follow the previous catalog (64 rows; 16 F1 results for the dnf
and position tasks; 4 for designs and eligibilities; smaller nested caps).

## Validation

Local, against relflow `stage/0.3.6` at commit 819cc71 (run with
`uv run --with-editable ../relflow --with "polars>=2,<3" --with transformers python -m pytest`):

- `tests/test_relarena_catalog.py`: 112 tests over all 21 tasks (metadata
  contracts, fixture tensorization, label hygiene, censoring, history bounds,
  multiplicity).
- `tests/test_raw_structure_coverage.py`: five physical-fixture tests that
  check looked-up context and nested branches on F1, event, Avito, trial, and
  Stack tasks, with rows after the anchor censored.
- `benchmarks/relarena/suite_profiles.py` carries a regenerated per-field
  text policy: labels 16 tokens, titles and summaries 32, bodies and
  descriptions 64, long documents 96.

Retired with the engineered schemas: the information-coverage tests for
trial/F1/event, Avito/Stack, and Amazon/H&M, the source-type tests for Stack
account ids and trial duration parsing, the event location-history test, and
the two old context-coverage files. Their censoring and join semantics live
on in the catalog and raw-structure tests.

Real-data smokes on the lab (`~/relflow-raw-20261022/smoke3`, `smoke4`,
`smoke5`; the suite's `smoke` entrypoint on the GPU with the 9de4787 wheel):
all 21 tasks pass, 6 to 90 seconds each. Two DuckDB 1.5 limits shaped the
queries: static dimension tables join directly rather than through a
deduplicated lateral, and nested branches hang off the capped parent rows.
Parquet cannot store a null fixed-size list inside a nested record, so an
unknown event's word vector is zeros while its null city, country, and host
mark it unknown.

Two failures remain in the benchmarks suite and predate this rewrite:
`tests/test_relarena_throughput_search.py` expects two spawn workers to
reproduce the single-process row order, which relflow 0.3.5's worker sharding
changed, and `tests/test_dnf_benchmark.py` plus `tests/test_full_benchmark.py`
import `SwiGLU` from a relflow module that no longer exports it.

## PRQL renderings

`benchmarks/relarena/prql.py` renders the same 21 task structures as PRQL, next to each
`observations.sql` as `observations.prql`. The renderings are a readable
clone of the DuckDB queries, not a second execution path: the benchmark runner
reads only the SQL. `tests/test_prql_clone.py` compiles every rendering with
`prqlc` for the DuckDB target and checks that it reads the SQL's tables.

PRQL has no lateral join, so each timed lookup becomes its own ranked
relation (`group ... (sort ... | take 1)`, which prqlc compiles to
`DISTINCT ON`) joined back by row identity, and lookup aliases carry a `lk_`
prefix so a lookup named like a column (Avito's `category`) still resolves.
On October 23 the compiled renderings for the three F1 and three event tasks
returned results identical to their SQL on the Modal sweeps' DuckDB snapshots
(499 and 2,013 validation anchors).

## Derived structures

`benchmarks/relarena/derive.py` builds the same `Task`, `Branch`, `Lookup`, and
`Col` objects as the hand-written functions in `benchmarks/relarena/raw.py`, from a database's keys and measured
column statistics alone, so the renderers in `raw.py` produce a schema and a
query without task-specific knowledge. This is the path a RelArena submission
needs: the harness hands a model the censored RelBench database and the task,
and every baseline derives its inputs from the foreign-key graph.

The rules, in order: the entity row carries its typed columns and its forward
foreign keys as lookups, a first hop into any dimension and a second hop only
into vocabulary tables (no time column, no keys of their own); tables with a
foreign key into the entity become branches, ordered and censored by their time
column; a table sharing two or more foreign-key targets with a branch attaches
as an as-of lookup when its shared key is unique and the branch covers a
majority of its distinct keys, and a table already kept as a branch never
doubles as a lookup; children of a branch nest under it when their fan-out
stays under 32 rows, and children of a dimension nest only under link tables
and only when they belong solely to that dimension; children of a root
dimension become branches joined through the root key. Columns type by dtype,
cardinality, null pattern, string length and shape, and identifier naming; a
row's own key is carried only when another table references it; a dimension
timestamp that repeats the row's time is left out; prose is carried as text
only by rows at depth one.

Lengths and field counts are budgets of the task's size tier (training anchors
below 20,000, below 500,000, and above), not percentiles of the data: branch
lengths 128, 64, and 32 with tail overflow, nested lengths 16, 16, and 8, and
at most 48, 40, and 24 fields per node, filled with own columns first, then
first-hop lookups, aligned siblings, and second-hop lookups. Compaction makes a
generous length free for short histories, so the length is a resource guard
rather than a recency prior.

Checked on October 23 against the RelBench tables in the lab cache: the
derivation reaches every hand-written relation of all 21 tasks and types
hand-written columns alike except for Avito's hashed category and location
keys, two H&M one-or-null flags, nine short name-like strings where the hand
choices themselves disagree between category and text, and the nested outcome
texts of site-success that the depth policy keeps out. Driver-position comes
out at 43 columns against the hand-written 40, post-votes at 183 against 25.
The derived queries for F1, event, Stack, trial, Amazon, and Avito tasks return
one row per anchor on 256 real training anchors. `tests/test_raw_derive.py`
exercises the rules on a synthetic database and runs the rendered query.

The engine also derives straight from a RelBench cache, the way the harness
presents a task: `python -m benchmarks.relarena.derive --relbench rel-f1/driver-position
--cache ~/.cache/relbench --output DIR` reads keys and time columns from the
Parquet metadata, profiles rows up to the dataset's validation cutoff, and
writes a bundle with a `derivation.json` recording the tier and cutoff. The
sweep's `--structure derived` builds that bundle once per sweep and trains it
in every trial and the refit; the Modal entrypoint defaults to it.

## Relationship to the lean-schema loop

The v4 lean-schema loop was stopped on October 22 before any qualified
trial; see `results/lean-schema-20261021/README.md`. Its catalog still
carried dozens of windowed summaries per task. The next loop candidate
should freeze this catalog with a wheel built from relflow 819cc71 or later,
where Number embeds raw and log lanes and trains MAE targets on the signed
log scale without options.
