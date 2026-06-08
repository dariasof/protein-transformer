# src/plm/model/embeddings.py

"""
Input embeddings for the protein transformer.

Combines token embeddings (one vector per vocabulary token) with learned
positional embeddings (one vector per sequence position). The two are summed
and then layer-normalised before being passed into the transformer stack.

    x = dropout(layernorm(token_emb(input_ids) + pos_emb(positions)))

Shape convention used throughout the model:
    B  = batch size
    L  = sequence length (≤ 512)
    D  = model dimension (d_model, e.g. 128)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ProteinEmbeddings(nn.Module):
    """
    Token + positional embeddings with layer norm and dropout.

    Args:
        vocab_size:  Number of tokens in the vocabulary (24).
        d_model:     Model hidden dimension. 
        max_len:     Maximum sequence length the model will ever see (512).
        pad_id:      Token ID used for padding
        dropout:     Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: LongTensor [B, L] — token IDs (including CLS and PAD).
        Returns:
            Tensor [B, L, D] — combined embeddings, normed and dropped out.
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [0, 1, 2, ..., L-1], shape [1, L]
        x = self.token_emb(input_ids) + self.pos_emb(positions)            # [B, L, D]
        return self.drop(self.norm(x))