"""Tests for the protein tokenizer."""

import pytest

from plm.data.tokenizer import (
    AMINO_ACIDS,
    SPECIAL_TOKENS,
    ProteinTokenizer,
)


@pytest.fixture
def tokenizer() -> ProteinTokenizer:
    """A tokenizer instance for each test."""
    return ProteinTokenizer()


#Vocab structure


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 24


def test_pad_is_zero(tokenizer):
    assert tokenizer.pad_id == 0


def test_special_token_ids_are_contiguous(tokenizer):
    for expected_id, tok in enumerate(SPECIAL_TOKENS):
        assert tokenizer.token_to_id[tok] == expected_id


def test_amino_acid_ids_are_contiguous(tokenizer):
    """AAs occupy ids 4-23, in alphabetical order."""
    for offset, aa in enumerate(AMINO_ACIDS):
        assert tokenizer.token_to_id[aa] == 4 + offset


#Encoding


def test_encode_prepends_cls_by_default(tokenizer):
    ids = tokenizer.encode("MKA")
    assert ids[0] == tokenizer.cls_id


def test_encode_without_cls(tokenizer):
    ids = tokenizer.encode("MKA", add_cls=False)
    assert ids[0] != tokenizer.cls_id
    assert len(ids) == 3


def test_encode_is_case_insensitive(tokenizer):
    assert tokenizer.encode("mka") == tokenizer.encode("MKA")


def test_encode_unknown_chars_map_to_unk(tokenizer):
    """B, J, O, U, X, Z and other non-standard chars become [UNK]."""
    ids = tokenizer.encode("BJOUXZ", add_cls=False)
    assert all(i == tokenizer.unk_id for i in ids)


def test_encode_does_not_crash_on_unknown(tokenizer):
    """Non-standard chars must be never raise an error."""
    tokenizer.encode("ACXZ$%^@!", add_cls=False)  #should not raise


#Round trip


def test_round_trip_standard_sequence(tokenizer):
    """encode -> decode -> original (for sequences of only standard AAs)."""
    seq = "MKAAVLALLMAGLALQPGTALAPSPSPSDS"
    ids = tokenizer.encode(seq, add_cls=False)
    assert tokenizer.decode(ids) == seq


def test_round_trip_with_cls(tokenizer):
    """With add_cls=True, decode(skip_special=True) recovers the original."""
    seq = "MKAAVLALLM"
    ids = tokenizer.encode(seq, add_cls=True)
    assert tokenizer.decode(ids, skip_special=True) == seq


@pytest.mark.parametrize(
    "seq",
    [
        "M",                          # single residue
        "MK",                         # very short
        "A" * 512,                    # max length
        "MKAAVLALLMAGLALQPGTALAP",    # representative snippet
    ],
)
def test_round_trip_various_lengths(tokenizer, seq):
    ids = tokenizer.encode(seq, add_cls=False)
    assert tokenizer.decode(ids) == seq
    assert len(ids) == len(seq)