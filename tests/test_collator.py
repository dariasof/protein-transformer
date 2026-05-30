

import pytest
import torch

from plm.data.collator import MLMCollator
from plm.data.tokenizer import AMINO_ACIDS, ProteinTokenizer


@pytest.fixture
def tokenizer() -> ProteinTokenizer:
    return ProteinTokenizer()


@pytest.fixture
def collator() -> MLMCollator:
    return MLMCollator()


def _make_batch(
    tokenizer: ProteinTokenizer,
    sequences: list[str],
) -> list[torch.Tensor]:
    """Helper: tokenize a list of strings into a batch."""
    return [
        torch.tensor(tokenizer.encode(seq, add_cls=True), dtype=torch.long)
        for seq in sequences
    ]


# -- Output shapes


def test_output_keys(collator, tokenizer):
    """Collator must return exactly the three expected keys."""
    batch = _make_batch(tokenizer, ["MKAAVL", "ACDEF"])
    out = collator(batch)
    assert set(out.keys()) == {"input_ids", "labels", "attention_mask"}


def test_output_shapes_match(collator, tokenizer):
    """All three tensors must have identical shape [batch, max_len]."""
    seqs = ["MKAAVLALLM", "ACDEF", "GHIKLM"]
    batch = _make_batch(tokenizer, seqs)
    out = collator(batch)
    expected_shape = (3, max(len(s) + 1 for s in seqs))  # +1 for CLS
    assert out["input_ids"].shape == expected_shape
    assert out["labels"].shape == expected_shape
    assert out["attention_mask"].shape == expected_shape


# -- Padding and attention mask


def test_padding_fills_short_sequences(collator, tokenizer):
    """Shorter sequences must be padded to the length of the longest."""
    batch = _make_batch(tokenizer, ["MKAAVL", "AC"])  # lengths 7 and 3
    out = collator(batch)
    # Row 1 (shorter): positions 3 and 4 should be PAD (id=0)
    assert out["input_ids"][1, 3].item() == 0
    assert out["input_ids"][1, 4].item() == 0


def test_attention_mask_ones_for_real_tokens(collator, tokenizer):
    """attention_mask must be 1 for real tokens, 0 for padding."""
    batch = _make_batch(tokenizer, ["MKAAVL", "AC"])
    out = collator(batch)
    # Row 0 (length 7): all 7 positions are real
    assert out["attention_mask"][0, :7].sum().item() == 7
    # Row 1 (length 3): first 3 positions real, rest padding
    assert out["attention_mask"][1, :3].sum().item() == 3
    assert out["attention_mask"][1, 3:].sum().item() == 0


# -- Labels


def test_labels_minus100_at_padding(collator, tokenizer):
    """Padding positions must always have label -100."""
    batch = _make_batch(tokenizer, ["MKAAVL", "AC"])
    out = collator(batch)
    # Row 1 padding positions
    assert (out["labels"][1, 3:] == -100).all()


def test_labels_minus100_at_cls(collator, tokenizer):
    """CLS (position 0) must never be a prediction target."""
    batch = _make_batch(tokenizer, ["MKAAVLALLM"] * 8)
    out = collator(batch)
    assert (out["labels"][:, 0] == -100).all()


def test_labels_preserve_original_ids(collator, tokenizer):
    """Where labels != -100, they must equal the ORIGINAL token IDs."""
    # Use a deterministic collator with 100% masking to force selections.
    greedy = MLMCollator(mask_prob=1.0, mask_token_prob=1.0)
    batch = _make_batch(tokenizer, ["MKAAVL"])
    original = batch[0].clone()
    out = greedy(batch)
    predicted_positions = (out["labels"][0] != -100).nonzero(as_tuple=False)
    for pos in predicted_positions.squeeze(1).tolist():
        assert out["labels"][0, pos].item() == original[pos].item()


# -- Masking rate


def test_masking_rate_approximately_15_percent(collator, tokenizer):
    """
    Over a large batch, the fraction of masked positions should be
    close to 15% of eligible (non-CLS, non-PAD) positions.
    """
    sequences = ["ACDEFGHIKLMNPQRSTVWY" * 5] * 64   # 64 sequences of 100 AA
    batch = _make_batch(tokenizer, sequences)
    out = collator(batch)
    # Count eligible positions (non-CLS, non-PAD real tokens)
    # Each sequence has 100 AAs + 1 CLS = 101 tokens. Eligible = 100.
    total_eligible = 100 * 64
    total_selected = (out["labels"] != -100).sum().item()
    rate = total_selected / total_eligible
    # Allow ±5% tolerance around 15%
    assert 0.10 < rate < 0.20, f"Masking rate {rate:.3f} outside expected range"


# -- Input ids after masking


def test_masked_positions_get_mask_token(collator, tokenizer):
    """
    With mask_token_prob=1.0, every selected position must become [MASK].
    """
    always_mask = MLMCollator(mask_prob=1.0, mask_token_prob=1.0)
    batch = _make_batch(tokenizer, ["MKAAVLALLM"])
    out = always_mask(batch)
    selected = (out["labels"][0] != -100).nonzero(as_tuple=False).squeeze(1)
    for pos in selected.tolist():
        assert out["input_ids"][0, pos].item() == 3  # [MASK] id


def test_random_replacement_uses_valid_aa(collator, tokenizer):
    """
    With random_token_prob=1.0, every selected position must be a valid AA.
    """
    always_random = MLMCollator(
        mask_prob=1.0, mask_token_prob=0.0, random_token_prob=1.0
    )
    aa_ids = set(tokenizer.token_to_id[aa] for aa in AMINO_ACIDS)
    batch = _make_batch(tokenizer, ["MKAAVLALLM"] * 16)
    out = always_random(batch)
    selected = (out["labels"] != -100).nonzero(as_tuple=False)
    for row, pos in selected.tolist():
        assert out["input_ids"][row, pos].item() in aa_ids