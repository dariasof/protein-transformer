"""
extract.py — pull per-head attention matrices out of a trained model.

The analysis pipeline scores each (layer, head) against ground-truth contact
maps, so it needs attention indexed by *residue*, not by token. This module
handles that translation: it runs a single sequence through the model, collects
the attention weights from every block, and strips the special-token row and
column so the result aligns positionally with the [L, L] matrices produced by
plm.analysis.contacts.

No transformation of the weights happens here. Symmetrization and APC belong in
the scoring layer, so that a single cached extraction can be scored both with
and without APC.
"""

from __future__ import annotations

import numpy as np
import torch

from plm.data.tokenizer import ProteinTokenizer
from plm.model.mlm import ProteinMLM


def extract_attention(
    model: ProteinMLM,
    tokenizer: ProteinTokenizer,
    sequence: str,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Return per-head attention matrices for one protein sequence.

    The sequence is run alone (batch size 1) rather than batched with others.
    Batching would require padding, and padded positions would then have to be
    stripped alongside the special tokens -- an extra alignment step with no
    upside here, since extraction is a one-off whose results are cached.

    Args:
        model: A trained ProteinMLM. Set to eval mode by this function.
        tokenizer: The tokenizer the model was trained with. Must be the same
            one, or the token IDs will be meaningless.
        sequence: Raw amino acid string, e.g. 'MKTAYIAKQRQ'. This is the full
            sequence including residues with no resolved coordinates -- the
            model attends to them normally, and they are excluded only later,
            by the eligibility mask.
        device: Where to run the forward pass.

    Returns:
        Float array of shape [n_layers, n_heads, L, L] where L == len(sequence).
        Entry [layer, head, i, j] is the attention weight from residue i to
        residue j. Rows do not sum to 1, because the [CLS] column has been
        removed and it typically absorbs a substantial share of the mass.
    """
    model.eval()
    model.to(device)

    token_ids = tokenizer.encode(sequence, add_cls=True)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids, return_attentions=True)

    # One [B, H, L+1, L+1] tensor per block -> [n_layers, B, H, L+1, L+1].
    attention = torch.stack(outputs["attentions"])
    attention = attention.squeeze(1)  # B == 1

    # Every row is a softmax over the key dimension, so it must sum to 1. This
    # catches three failures that would otherwise produce a plausible-looking
    # heatmap: a forgotten eval() (dropout rescales the surviving weights),
    # accidentally capturing pre-softmax scores, and a transposed tensor.
    # It must run before stripping -- afterwards the rows legitimately sum to
    # less than 1.
    row_sums = attention.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"attention rows do not sum to 1 (min {row_sums.min():.4f}, "
        f"max {row_sums.max():.4f}) -- not a post-softmax tensor, or dropout is active"
    )

    # Drop the [CLS] row and column. The tokenizer prepends exactly one special
    # token and appends none, so residue i sits at token index i + 1.
    attention = attention[..., 1:, 1:]

    # Guards the assumption above. If the tokenizer ever gains a trailing token,
    # this fires here rather than surfacing later as a quietly misaligned
    # contact score.
    n_layers, n_heads, rows, cols = attention.shape
    assert rows == cols == len(sequence), (
        f"expected [{n_layers}, {n_heads}, {len(sequence)}, {len(sequence)}], "
        f"got {tuple(attention.shape)} -- token layout does not match the "
        f"single-leading-[CLS] assumption"
    )

    return attention.cpu().numpy()
def extract_attention_esm2(model, tokenizer, sequence, device="cpu"):
    """Return per-head attention matrices from an ESM-2 model.

    Serves as the positive control for the contact-scoring pipeline: ESM-2 is
    known (Rao et al. 2021) to recover long-range contacts well above chance,
    so running it through this same function and the same scorer answers
    whether a null result on the in-house model reflects the model or a bug
    in contacts.py / contact_score.py.

    The tokenizer's cls and eos tokens sit at both ends, unlike the in-house
    tokenizer which only prepends cls -- hence stripping index 0 and -1
    rather than 0 and none.

    Args:
        model: A loaded EsmModel or EsmForMaskedLM, in eval mode.
        tokenizer: The matching EsmTokenizer.
        sequence: Raw amino acid string. Fed as-is, including any expression
            tag -- see build_eligibility_mask, which already excludes
            positions with no resolved coordinates.
        device: Where to run the forward pass.

    Returns:
        Float array [n_layers, n_heads, L, L], L == len(sequence).
    """
    model.eval()
    model.to(device)

    encoded = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]

    assert input_ids[0, 0].item() == tokenizer.cls_token_id, "expected leading [CLS]"
    assert input_ids[0, -1].item() == tokenizer.eos_token_id, "expected trailing [EOS]"
    assert input_ids.shape[1] == len(sequence) + 2, (
        f"expected {len(sequence) + 2} tokens (cls + residues + eos), "
        f"got {input_ids.shape[1]} -- tokenizer may be inserting more than expected"
    )

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    attention = torch.stack(outputs.attentions).squeeze(1)  # [n_layers, H, L+2, L+2]

    row_sums = attention.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        f"attention rows do not sum to 1 (min {row_sums.min():.4f}, "
        f"max {row_sums.max():.4f})"
    )

    attention = attention[..., 1:-1, 1:-1]  # drop cls AND eos, both ends

    n_layers, n_heads, rows, cols = attention.shape
    assert rows == cols == len(sequence), (
        f"expected [{n_layers}, {n_heads}, {len(sequence)}, {len(sequence)}], "
        f"got {tuple(attention.shape)}"
    )

    return attention.cpu().numpy()