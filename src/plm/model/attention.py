"""
Multi-head self-attention for the protein transformer.

Implements scaled dot-product attention across h independent heads,
each operating in a head_dim = d_model/n_heads dimensional subspace.

    B  = batch size
    L  = sequence length
    D  = d_model (e.g. 128)
    H  = number of heads (e.g. 4)
    hd = head_dim = D // H (e.g. 32)
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with padding mask support.

    Args:
        d_model:   Total model dimension. Must be divisible by n_heads.
        n_heads:   Number of parallel attention heads.
        dropout:   Dropout applied to attention weights during training.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads  # 32 for d_model=128, n_heads=4

        # Single projection matrices for all heads at once.
        # Instead of n_heads separate [D, hd] matrices, we use one [D, D] matrix per role. The output is then reshaped to split heads apart.
        # Equivalent mathematically, faster in practice.
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x:             [B, L, D] — token representations from embeddings
                           or previous transformer block.
            padding_mask:  [B, L] boolean tensor, True at [PAD] positions.
                           If None, no masking is applied.
            return_weights: If True, also return attention weight tensor.
                           Set True during analysis, False
                           during training (saves memory).
        Returns:
            output:   [B, L, D] — context-informed representations.
            weights:  [B, H, L, L] or None — attention weights per head.
        """
        B, L, D = x.shape

        # Project input into Q, K, V for all heads simultaneously
        # Each Linear maps [B, L, D] → [B, L, D].
        Q = self.W_Q(x)  # [B, L, D]
        K = self.W_K(x)  # [B, L, D]
        V = self.W_V(x)  # [B, L, D]

        # Reshape: [B, L, D] → [B, L, H, hd] → [B, H, L, hd]
        # The transpose moves heads before sequence length so that the
        # [L, L] attention matrix is computed per head cleanly.
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, hd]
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, hd]
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, hd]

        #Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = Q @ K.transpose(-2, -1) / scale  # [B, H, L, L]

        # Apply padding mask
        if padding_mask is not None:
            # padding_mask: [B, L], True where PAD
            # Expand to [B, 1, 1, L] so it broadcasts over H and query L
            # We mask the KEY dimension (last L) — we don't want any query to attend to a PAD position.
            mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over key dimension
        weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
        weights = self.attn_drop(weights)    # zero some weights during training

        output = weights @ V  # [B, H, L, hd]

        # Reassemble heads and project
        # Reverse the transpose and reshape to merge heads back
        # [B, H, L, hd] → [B, L, H, hd] → [B, L, D]
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.W_O(output)  # [B, L, D]

        return output, (weights if return_weights else None)