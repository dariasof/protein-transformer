"""
Full masked language model for protein sequences.
ProteinEmbeddings  →  N × TransformerBlock  →  MLM head

The MLM head projects each token's final hidden state to vocab_size
logits. Loss is computed only at masked positions (labels != -100).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from plm.model.embeddings import ProteinEmbeddings
from plm.model.transformer import TransformerBlock


class ProteinMLM(nn.Module):
    """
    Encoder-only protein language model trained with masked language modeling.

    Args:
        vocab_size:  Number of tokens (24).
        d_model:     Hidden dimension throughout the model.
        n_heads:     Number of attention heads per block.
        n_layers:    Number of transformer blocks to stack.
        max_len:     Maximum sequence length (for positional embeddings).
        pad_id:      Token ID for padding.
        dropout:     Dropout probability applied throughout.
    """

    def __init__(
        self,
        vocab_size: int = 24,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 512,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.pad_id = pad_id

        # Input embeddings
        self.embeddings = ProteinEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            pad_id=pad_id,
            dropout=dropout,
        )

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm before MLM head 
        # Stabilizes the input to the head after the last residual addition.
        self.norm = nn.LayerNorm(d_model)

        # MLM prediction head 
        # Projects each token's hidden state to vocab logits.
        # Bias included, each token has a prior probability of appearing
        self.mlm_head = nn.Linear(d_model, vocab_size, bias=True)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights with small normal values.
        Embeddings and linear layers get std=0.02
        LayerNorm gets weight=1, bias=0 by default
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        return_attentions: bool = False,
    ) -> dict[str, Tensor | None]:
        """
        Args:
            input_ids:        [B, L] — token IDs including CLS and PAD.
            labels:           [B, L] — original token IDs at masked positions,
                              -100 elsewhere. If None, loss is not computed.
            return_attentions: If True, collect attention weights from all
                              blocks. Set True during analysis, False during
                              training.
        Returns:
            dict with keys:
                logits:      [B, L, vocab_size]
                loss:        scalar or None
                attentions:  list of [B, H, L, L] per block, or None
        """
        # Build padding mask from input_ids
        # True at PAD positions, passed to every attention block.
        padding_mask = (input_ids == self.pad_id)  # [B, L]

        # Forward pass through the model
        x = self.embeddings(input_ids)  # [B, L, D]

        attentions = [] if return_attentions else None

        for block in self.blocks:
            x, weights = block(
                x,
                padding_mask=padding_mask,
                return_weights=return_attentions,
            )
            if return_attentions:
                attentions.append(weights)  # [B, H, L, L]

        x = self.norm(x)               # [B, L, D]
        logits = self.mlm_head(x)      # [B, L, vocab_size]

        #  Compute loss if labels provided 
        loss = None
        if labels is not None:
            # Flatten to [B*L, vocab_size] and [B*L] for cross_entropy.
            # ignore_index=-100 means unmasked positions don't contribute.
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits":     logits,
            "loss":       loss,
            "attentions": attentions,
        }

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)