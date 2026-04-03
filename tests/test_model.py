# tests/test_model.py

import pytest

from toy_gpt_train.c_model import EmbeddingNextTokenModel


def test_forward_output_sums_to_one():
    model = EmbeddingNextTokenModel(vocab_size=10, embedding_dim=4, context_size=2)
    probs = model.forward([0, 1])
    assert abs(sum(probs) - 1.0) < 1e-6


def test_forward_context_size_mismatch_raises():
    model = EmbeddingNextTokenModel(vocab_size=10, embedding_dim=4, context_size=2)
    with pytest.raises(ValueError):
        model.forward([0])
