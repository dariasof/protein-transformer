"""
Data inspection script

Loads the prepared SwissProt dataset, runs the MLM collator, and prints
5 random batches with shapes and human-readable decoded sequences.

Run from the project root:

    python scripts/inspect_data.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plm.data.collator import MLMCollator
from plm.data.dataset import SwissProtDataset
from plm.data.tokenizer import ProteinTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PT_PATH = PROJECT_ROOT / "data" / "processed" / "swissprot_10k.pt"

BATCH_SIZE = 4
N_BATCHES = 5


def main() -> None:
    tokenizer = ProteinTokenizer()
    dataset = SwissProtDataset(PT_PATH)
    collator = MLMCollator()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    print(f"Dataset size : {len(dataset):,} sequences")
    print(f"Vocab size   : {tokenizer.vocab_size} tokens")
    print(f"Batch size   : {BATCH_SIZE}")
    print("=" * 60)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= N_BATCHES:
            break

        input_ids = batch["input_ids"]        # [B, L]
        labels = batch["labels"]              # [B, L]
        attention_mask = batch["attention_mask"]  # [B, L]

        print(f"\nBatch {batch_idx + 1}")
        print(f"  input_ids shape      : {tuple(input_ids.shape)}")
        print(f"  labels shape         : {tuple(labels.shape)}")
        print(f"  attention_mask shape : {tuple(attention_mask.shape)}")

        # Count how many positions are masked in this batch.
        n_real = attention_mask.sum().item()
        n_predicted = (labels != -100).sum().item()
        mask_rate = n_predicted / (n_real - BATCH_SIZE)  # subtract CLS tokens
        print(f"  real tokens          : {int(n_real)} (incl. {BATCH_SIZE} CLS)")
        print(f"  predicted positions  : {n_predicted} ({mask_rate:.1%} of AA tokens)")

        # Decode and print each sequence in the batch.
        for seq_idx in range(BATCH_SIZE):
            ids = input_ids[seq_idx]
            lbs = labels[seq_idx]
            mask = attention_mask[seq_idx]

            # Trim to real length (drop padding).
            real_len = mask.sum().item()
            ids = ids[:real_len]
            lbs = lbs[:real_len]

            # Build a human-readable string.
            # Real tokens: show AA letter (or [MASK]).
            # Predicted positions: show as lowercase to highlight them.
            chars = []
            for pos, (tok_id, label_id) in enumerate(
                zip(ids.tolist(), lbs.tolist())
            ):
                tok = tokenizer.id_to_token[tok_id]
                if pos == 0:
                    chars.append("[CLS]")
                elif tok == "[MASK]":
                    # Show what was originally here (from labels).
                    original = tokenizer.id_to_token[label_id]
                    chars.append(f"[{original}]")   # bracketed = was masked
                elif label_id != -100:
                    # Unchanged or random-replaced but still predicted.
                    chars.append(tok.lower())        # lowercase = predicted
                else:
                    chars.append(tok)                # uppercase = not selected

            sequence_str = " ".join(chars)
            print(f"\n  Seq {seq_idx + 1} (len={int(real_len) - 1} AA):")
            print(f"    {sequence_str}")

        print("-" * 60)



if __name__ == "__main__":
    main()