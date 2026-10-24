"""Keep evaluation independent of frozen Text cache history without changing the SDK."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import relflow as rf
import torch
from relflow.structs.enums import Tokens
from relflow.tensorfields.extensions.text import ATTENTION_MASK, INPUT_IDS, CachedModel, Embedder
from tensordict import TensorDict

from benchmarks.relarena import supervised
from benchmarks.relarena.evaluation import reset_text_cache
from benchmarks.relarena.supervised import Selection


@pytest.fixture
def model(monkeypatch):
    if importlib.util.find_spec("transformers") is None:
        module = ModuleType("transformers")
        module.__spec__ = ModuleSpec("transformers", loader=None)
        monkeypatch.setitem(sys.modules, "transformers", module)
    frozen = CachedModel(key="owned-evaluation", model=torch.nn.Identity(), hidden_size=4)
    monkeypatch.setattr(CachedModel, "get_model", classmethod(lambda cls, key: frozen))
    monkeypatch.setattr(
        CachedModel, "get_tokenizer", classmethod(lambda cls, key: SimpleNamespace(padding_side="right"))
    )
    calls = []

    def pool(self, ids, mask):
        # Simulate miss-cohort-dependent rounding without CUDA or external weights.
        calls.append(len(ids))
        return ids[:, :1].float().repeat(1, 4) + len(ids) / 100

    monkeypatch.setattr(Embedder, "pool", pool)
    result = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        body=rf.Text(model="owned-evaluation", cache=True, max_length=2),
        brief=rf.Text(model="owned-evaluation", cache=3, max_length=2),
        uncached=rf.Text(model="owned-evaluation", cache=False, max_length=2),
        target=rf.Number(mask=True),
    )
    return result, calls


def encode(embedder, values):
    ids = torch.tensor(values).reshape(-1, 1)
    content = TensorDict({INPUT_IDS: ids, ATTENTION_MASK: torch.ones_like(ids)}, batch_size=[len(values)])
    return embedder.encode(content, torch.full((len(values),), Tokens.valued.value))


def test_cold_phase_removes_warm_history_and_retains_within_phase_reuse(model):
    module, calls = model
    embedder = module.nodes[rf.Address("body")].embedder
    encode(embedder, [7, 8, 9])
    warm = encode(embedder, [7, 10])
    assert calls == [3, 1]
    assert reset_text_cache(module)["/body"] == 4
    cold = encode(embedder, [7, 10])
    assert not torch.equal(warm, cold)
    count = len(calls)
    torch.testing.assert_close(encode(embedder, [7, 10]), cold, rtol=0, atol=0)
    assert len(calls) == count  # Caching stays enabled within the evaluation phase.
    encode(embedder, [7, 8, 9])
    reset_text_cache(module)
    torch.testing.assert_close(encode(embedder, [7, 10]), cold, rtol=0, atol=0)


def test_reset_preserves_capacity_configuration_weights_and_nontext_state(model):
    module, _ = model
    body = module.nodes[rf.Address("body")].embedder
    brief = module.nodes[rf.Address("brief")].embedder
    uncached = module.nodes[rf.Address("uncached")].embedder
    other = module.nodes[rf.Address("target")].embedder
    sentinel = object()
    other.cache = sentinel
    encode(body, [1, 2])
    encode(brief, [3])
    requests = module.schema.model_dump(mode="json")
    weights = {name: value.clone() for name, value in module.state_dict().items() if isinstance(value, torch.Tensor)}
    original_frozen = body.__dict__["_cached_model"]
    assert reset_text_cache(module) == {"/body": 2, "/brief": 1}
    assert body.cache.capacity is None and brief.cache.capacity == 3
    assert len(body.cache) == len(brief.cache) == 0
    assert uncached.cache is None and other.cache is sentinel
    assert body.__dict__["_cached_model"] is original_frozen
    assert module.schema.model_dump(mode="json") == requests
    for name, value in weights.items():
        torch.testing.assert_close(module.state_dict()[name], value, rtol=0, atol=0)


def test_full_split_selection_begins_each_validation_epoch_cold(model, tmp_path):
    module, _ = model
    embedder = module.nodes[rf.Address("body")].embedder
    callback = Selection(SimpleNamespace(task_type="regression"), 2, tmp_path, SimpleNamespace(max_steps=2))
    for _ in range(2):
        encode(embedder, [1, 2])
        assert len(embedder.cache) == 2
        callback.on_validation_epoch_start(None, module)
        assert callback.active and len(embedder.cache) == 0


def test_predictions_begin_cold_and_preserve_validation_row_identity(model, tmp_path, monkeypatch):
    module, _ = model
    embedder = module.nodes[rf.Address("body")].embedder
    pq.write_table(pa.table({"target": [0.0, 0.0]}), tmp_path / "validate.parquet")
    phases = []

    def predict(**kwargs):
        phases.append(len(embedder.cache))
        encode(embedder, [10, 11])
        return pa.table(
            {
                "inputs": pa.StructArray.from_arrays([pa.array([1, 0])], names=["__row_position"]),
                "predictions": pa.StructArray.from_arrays(
                    [pa.StructArray.from_arrays([pa.array([3.0, 2.0])], names=["content"])], names=["target"]
                ),
            }
        )

    monkeypatch.setattr(module, "predict", predict)
    args = SimpleNamespace(seed=0, accelerator="cpu", precision="32-true")
    for _ in range(2):
        encode(embedder, [7, 8, 9])
        values = supervised.predictions(module, tmp_path, args=args, data_factory=lambda *args, **kwargs: None)
        np.testing.assert_array_equal(values, [2.0, 3.0])
    assert phases == [0, 0]
