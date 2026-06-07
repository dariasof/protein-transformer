"""
Transformer encoder block

One block = attention sub-layer + feedforward sub-layer, each wrapped
in a pre-norm residual connection.

    x = x + attention(layernorm(x))
    x = x + feedforward(layernorm(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from plm.model.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.

    Two linear projections with GELU in between.
    Expands to 4*d_model internally then contracts back.

    Args:
        d_model:  Input and output dimension.
        dropout:  Dropout applied after the second projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, 4 * d_model)
        self.linear_2 = nn.Linear(4 * d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D]
        """
        x = self.linear_1(x)           # [B, L, D] → [B, L, 4D]
        x = F.gelu(x)                  # [B, L, 4D] nonlinearity
        x = self.drop(self.linear_2(x))# [B, L, 4D] → [B, L, D]
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer encoder block.

    Pre-norm convention: layernorm is applied to the input of each
    sub-layer, not the output. The residual path is never normalized.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads.
        dropout:  Dropout for both attention weights and FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_1  = nn.LayerNorm(d_model)
        self.norm_2  = nn.LayerNorm(d_model)
        self.attn    = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn     = FeedForward(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x:             [B, L, D]
            padding_mask:  [B, L] bool, True at PAD positions.
            return_weights: If True, return attention weights from this block.
        Returns:
            x:       [B, L, D]
            weights: [B, H, L, L] or None
        """
        #attention sub-layer (pre-norm)
        residual = x
        x, weights = self.attn(
            self.norm_1(x),
            padding_mask=padding_mask,
            return_weights=return_weights,
        )
        x = residual + x   # residual connection

        # feedforward sub-layer (pre-norm)
        x = x + self.ffn(self.norm_2(x))   # residual connection

        return x, weights