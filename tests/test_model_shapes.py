# tests/test_model_shapes.py

import torch
import pytest
from plm.model.embeddings import ProteinEmbeddings


VOCAB   = 24
D_MODEL = 128
MAX_LEN = 512
B, L    = 4, 64   # batch size, sequence length for tests


@pytest.fixture
def emb():
    return ProteinEmbeddings(vocab_size=VOCAB, d_model=D_MODEL, max_len=MAX_LEN)


def test_output_shape(emb):
    ids = torch.randint(0, VOCAB, (B, L))
    out = emb(ids)
    assert out.shape == (B, L, D_MODEL)


def test_pad_positions_are_zero_contribution(emb):
    """PAD token (id=0) should produce the zero vector from token_emb."""
    pad_emb = emb.token_emb(torch.tensor([0]))
    assert torch.allclose(pad_emb, torch.zeros(D_MODEL), atol=1e-6)


def test_different_lengths_work(emb):
    for length in [1, 10, 128, 512]:
        ids = torch.randint(1, VOCAB, (1, length))
        out = emb(ids)
        assert out.shape == (1, length, D_MODEL)