"""
Training entry point for the protein language model.

Run from the project root:

    python scripts/train.py

To resume from a checkpoint:

    python scripts/train.py --resume data/checkpoints/resume.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plm.data.dataset import SwissProtDataset
from plm.data.collator import MLMCollator
from plm.model.mlm import ProteinMLM
from plm.training.trainer import train


#  Hardcoded config for now 

DATA_PATH      = Path("data/processed/swissprot_10k.pt")
CHECKPOINT_DIR = Path("data/checkpoints")

VOCAB_SIZE  = 24
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 4
MAX_LEN     = 512
PAD_ID      = 0
DROPOUT     = 0.1

BATCH_SIZE     = 32
N_EPOCHS       = 10
LEARNING_RATE  = 3e-4
WARMUP_RATIO   = 0.05
MAX_GRAD_NORM  = 1.0

CHECKPOINT_EVERY = 500
RETAIN_EVERY     = 1000

WANDB_PROJECT = "protein-mlm"


def main(
    resume_from: Path | None = None,
    data_path: Path = Path("data/processed/swissprot_10k.pt"),
    checkpoint_dir: Path = Path("data/checkpoints"),
) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #  Data 
    dataset = SwissProtDataset(data_path)
    collator = MLMCollator()

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    print(f"Dataset: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(train_loader)}")

    #  Model 
    model = ProteinMLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        pad_id=PAD_ID,
        dropout=DROPOUT,
    )

    print(f"Model parameters: {model.count_parameters():,}")

    #  Train 
    train(
        model=model,
        train_loader=train_loader,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=CHECKPOINT_EVERY,
        retain_every=RETAIN_EVERY,
        device=device,
        wandb_project=WANDB_PROJECT,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/swissprot_10k.pt"),
        help="Path to tokenized dataset",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints"),
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()
    main(
        resume_from=args.resume,
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
    )