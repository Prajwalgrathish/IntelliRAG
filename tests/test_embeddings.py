import os

import numpy as np
import pytest
import torch

from src.embeddings.transformer_encoder import TextEncoder, pool_embeddings


def test_fake_encoder_satisfies_protocol_and_dimensions(encoder):
    assert isinstance(encoder, TextEncoder)
    vectors = encoder.encode_documents(["alpha beta", "gamma delta", "epsilon"])
    assert vectors.shape == (3, encoder.dimension)
    assert vectors.dtype == np.float32
    assert encoder.encode_queries([]).shape == (0, encoder.dimension)


def test_embeddings_are_l2_normalised_and_similarity_reflects_overlap(encoder):
    a, b, c = encoder.encode_documents(["solar panel energy output", "solar energy panel efficiency", "medieval castle architecture"])
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)
    assert float(a @ b) > float(a @ c)


def test_mean_pooling_ignores_padding_tokens():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(pool_embeddings(hidden, mask, "mean"), torch.tensor([[2.0, 2.0]]))


def test_cls_pooling_takes_first_token():
    hidden = torch.tensor([[[5.0, 6.0], [1.0, 1.0]]])
    assert torch.allclose(pool_embeddings(hidden, torch.ones(1, 2), "cls"), torch.tensor([[5.0, 6.0]]))


def test_unknown_pooling_mode_raises():
    with pytest.raises(ValueError):
        pool_embeddings(torch.zeros(1, 2, 2), torch.ones(1, 2), "max")


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to download and test the real model")
def test_real_transformer_encoder_dimension_and_normalisation():
    from src.embeddings.transformer_encoder import TransformerEncoder

    enc = TransformerEncoder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vectors = enc.encode_documents(["A short sentence.", "Another sentence about retrieval."])
    assert vectors.shape == (2, enc.dimension)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
