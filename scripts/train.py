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
from plm.data.tokenizer import ProteinTokenizer
from plm.config import load_config




def main(
    config_path: Path,
    resume_from: Path | None = None,
    data_dir: Path = Path("data/processed"),
    checkpoint_dir: Path = Path("data/checkpoints"),
) -> None:
    
    config = load_config(config_path)
    tokenizer = ProteinTokenizer()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #  Data 
    dataset = SwissProtDataset(data_dir / "train.pt")
    collator = MLMCollator(
        mask_prob=config.data.masking.mask_prob,
        mask_token_prob=config.data.masking.mask_token_prob,
        random_token_prob=config.data.masking.random_token_prob,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    print(f"Dataset: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(train_loader)}")
    
    #  Model 
    model = ProteinMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        max_len=config.data.max_len,
        pad_id=tokenizer.pad_id,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
    )

    print(f"Model parameters: {model.count_parameters():,}")

    #  Train 
    train(
        model=model,
        train_loader=train_loader,
        precision=config.training.precision,
        n_epochs=config.training.n_epochs,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=config.training.checkpoint_every,
        retain_every=config.training.retain_every,
        device=device,
        wandb_project=config.training.wandb_project,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data-dir", type=Path, 
        default=Path("data/processed"),
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
        config_path=args.config,
        resume_from=args.resume,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )