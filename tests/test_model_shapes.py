
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


from plm.model.attention import MultiHeadSelfAttention

N_HEADS = 4

@pytest.fixture
def attn():
    return MultiHeadSelfAttention(d_model=D_MODEL, n_heads=N_HEADS)


def test_attention_output_shape(attn):
    x = torch.randn(B, L, D_MODEL)
    out, weights = attn(x, return_weights=True)
    assert out.shape == (B, L, D_MODEL)
    assert weights.shape == (B, N_HEADS, L, L)


def test_attention_no_weights(attn):
    x = torch.randn(B, L, D_MODEL)
    out, weights = attn(x, return_weights=False)
    assert out.shape == (B, L, D_MODEL)
    assert weights is None


def test_attention_padding_mask(attn):
    x = torch.randn(B, L, D_MODEL)
    # Mark last 10 positions as padding
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -10:] = True
    out, weights = attn(x, padding_mask=pad_mask, return_weights=True)
    # PAD key positions should have zero attention weight
    assert weights[:, :, :, -10:].sum().item() == 0.0


def test_attention_invalid_d_model():
    with pytest.raises(AssertionError):
        MultiHeadSelfAttention(d_model=130, n_heads=4)  # not divisible