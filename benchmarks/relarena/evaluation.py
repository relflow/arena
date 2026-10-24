"""Checkpoint-comparable evaluation policies for the pinned RelFlow SDK."""

from __future__ import annotations

import relflow as rf
from relflow.tensorfields.extensions.text import Embedder, EmbeddingCache

TEXT_CACHE_POLICY = "cold_per_validation_epoch_and_prediction_phase"


def reset_text_cache(model: rf.Model) -> dict[str, int]:
    """Start an evaluation phase without training-history-dependent Text vectors.

    BF16 frozen-encoder results can depend on the miss-batch shape. Checkpoints
    omit these transient vectors, so validation selection and prediction must
    start with the same empty per-field cache. Capacity, enabled caching, frozen
    encoder resources, and learned weights remain unchanged. This deliberately
    targets the pinned SDK's Text extension rather than other extensions' state.
    """
    cleared = {}
    for address in model.schema.requests:
        embedder = model.nodes[address].embedder
        if isinstance(embedder, Embedder) and embedder.cache is not None:
            cache = embedder.cache
            cleared[str(address)] = len(cache)
            embedder.cache = EmbeddingCache(cache.capacity)
    return cleared
