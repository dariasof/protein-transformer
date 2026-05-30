"""
MLM collator for protein sequences.

Takes a list of variable-length token ID tensors (one per sequence in the batch) and returns a dictionary of three padded tensors ready for the model:

    input_ids      [batch_size, max_len]  corrupted token IDs (model input)
    labels         [batch_size, max_len]  original IDs at masked positions, -100 everywhere else (loss target)
    attention_mask [batch_size, max_len]  1 for real tokens, 0 for padding
"""

import torch
from torch import Tensor

from plm.data.tokenizer import AMINO_ACIDS, ProteinTokenizer


# IDs of the 20 standard amino acids — the only tokens we ever mask.
_AA_IDS = tuple(
    ProteinTokenizer().token_to_id[aa] for aa in AMINO_ACIDS
)


class MLMCollator:
    """
    Collate a batch of tokenized protein sequences for masked language modeling.

    Args:
        mask_prob:      Fraction of eligible positions to select for
                        prediction. Default 0.15 (15%)
        mask_token_prob: Of selected positions, fraction replaced with
                        [MASK]. Default 0.80
        random_token_prob: Of selected positions, fraction replaced with a
                        random amino acid. Default 0.10
                        The remaining fraction (default 0.10) are left
                        unchanged but still included in the loss
        pad_id:         Token ID used for padding. Default 0 ([PAD])
        mask_id:        Token ID used for [MASK]. Default 3
    """

    def __init__(
        self,
        *,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.80,
        random_token_prob: float = 0.10,
        pad_id: int = 0,
        mask_id: int = 3,
    ) -> None:
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.pad_id = pad_id
        self.mask_id = mask_id

    def __call__(self, sequences: list[Tensor]) -> dict[str, Tensor]:
        """
        Args:
            sequences: list of 1-D LongTensors, variable length, each
                       starting with [CLS].
        Returns:
            dict with keys 'input_ids', 'labels', 'attention_mask',
            each a LongTensor of shape [batch_size, max_len].
        """
        batch_size = len(sequences)
        max_len = max(s.size(0) for s in sequences)

        # input_ids starts as a copy of the originals
        # labels starts as all -100 
        # attention_mask starts as all 0 
        input_ids = torch.full(
            (batch_size, max_len), self.pad_id, dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_len), -100, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long
        )

        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)

            input_ids[i, :seq_len] = seq

            # Mark real token positions as 1 in the attention mask
            attention_mask[i, :seq_len] = 1

            # Build a boolean mask of positions eligible for masking
            aa_id_set = set(_AA_IDS)
            eligible = torch.tensor(
                [tok.item() in aa_id_set for tok in seq],
                dtype=torch.bool,
            )

            # Sample 15% of eligible positions
            # torch.rand gives uniform [0, 1) for each position
            selected = eligible & (
                torch.rand(seq_len) < self.mask_prob
            )
            # selected: bool tensor, True at positions chosen for prediction.

            # For each selected position, apply the 80/10/10 decision.
            for pos in selected.nonzero(as_tuple=False).squeeze(1).tolist():
                original_id = seq[pos].item()

                # Record the original token in labels so we can compute loss.
                labels[i, pos] = original_id

                # Draw a uniform random number to decide which bucket.
                r = torch.rand(1).item()

                if r < self.mask_token_prob:
                    # 80%: replace with [MASK]
                    input_ids[i, pos] = self.mask_id

                elif r < self.mask_token_prob + self.random_token_prob:
                    # 10%: replace with a random amino acid
                    random_aa_id = _AA_IDS[
                        torch.randint(len(_AA_IDS), (1,)).item()
                    ]
                    input_ids[i, pos] = random_aa_id

                # else: remaining 10% — leave input_ids[i, pos] unchanged.
                # labels[i, pos] is already set, so loss is still computed.

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }