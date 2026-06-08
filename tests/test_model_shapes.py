
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


from plm.model.transformer import TransformerBlock


@pytest.fixture
def block():
    return TransformerBlock(d_model=D_MODEL, n_heads=N_HEADS)


def test_block_output_shape(block):
    x = torch.randn(B, L, D_MODEL)
    out, weights = block(x, return_weights=True)
    assert out.shape == (B, L, D_MODEL)
    assert weights.shape == (B, N_HEADS, L, L)

import torch.nn as nn
def test_block_residual(block):
    """
    With all weights zeroed, the block should return input unchanged.
    The residual connection guarantees this — if attention and FFN
    produce zero output, x = x + 0 = x.
    """
    for p in block.parameters():
         nn.init.zeros_(p)
   
    x = torch.randn(B, L, D_MODEL)
    out, _ = block(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_block_no_weights(block):
    x = torch.randn(B, L, D_MODEL)
    out, weights = block(x, return_weights=False)
    assert out.shape == (B, L, D_MODEL)
    assert weights is None


def test_block_padding_mask(block):
    x = torch.randn(B, L, D_MODEL)
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -10:] = True
    out, weights = block(x, padding_mask=pad_mask, return_weights=True)
    assert out.shape == (B, L, D_MODEL)
    assert weights[:, :, :, -10:].sum().item() == 0.0
    
from plm.model.mlm import ProteinMLM

VOCAB    = 24
N_LAYERS = 4


@pytest.fixture
def model():
    return ProteinMLM(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )


def test_mlm_output_shapes(model):
    ids    = torch.randint(1, VOCAB, (B, L))
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[:, 5] = ids[:, 5]   # pretend position 5 is masked

    out = model(ids, labels=labels)
    assert out["logits"].shape == (B, L, VOCAB)
    assert out["loss"] is not None
    assert out["loss"].shape == ()   # scalar


def test_mlm_no_labels(model):
    ids = torch.randint(1, VOCAB, (B, L))
    out = model(ids)
    assert out["logits"].shape == (B, L, VOCAB)
    assert out["loss"] is None


def test_mlm_attentions(model):
    ids = torch.randint(1, VOCAB, (B, L))
    out = model(ids, return_attentions=True)
    assert len(out["attentions"]) == N_LAYERS
    assert out["attentions"][0].shape == (B, N_HEADS, L, L)


def test_mlm_parameter_count(model):
    n = model.count_parameters()
    # 1M model — allow generous range
    assert 500_000 < n < 2_000_000, f"Unexpected parameter count: {n}"


def test_mlm_pad_gets_zero_attention(model):
    ids = torch.randint(1, VOCAB, (B, L))
    ids[:, -10:] = 0   # last 10 positions are PAD
    out = model(ids, return_attentions=True)
    for layer_weights in out["attentions"]:
        assert layer_weights[:, :, :, -10:].sum().item() == 0.0