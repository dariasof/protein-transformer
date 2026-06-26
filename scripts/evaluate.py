"""
Evaluation entry point for the protein language model.

Computes masked-language-modeling perplexity on the held-out validation set.

Run from the project root:

    python scripts/evaluate.py --config configs/1M.yaml --checkpoint data/checkpoints/resume.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plm.config import load_config
from plm.data.collator import MLMCollator
from plm.data.dataset import SwissProtDataset
from plm.data.tokenizer import ProteinTokenizer
from plm.eval.perplexity import evaluate
from plm.model.mlm import ProteinMLM
from plm.eval.knn_probe import embed_sequences, score_knn
from plm.eval.tape_data import load_tape_lmdb


def main(
    config_path: Path,
    checkpoint_path: Path,
    data_dir: Path = Path("data/processed"),
    tape_path: Path = None,
) -> None:

    config = load_config(config_path)
    tokenizer = ProteinTokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    dataset = SwissProtDataset(data_dir / "val.pt")
    collator = MLMCollator(
        mask_prob=config.data.masking.mask_prob,
        mask_token_prob=config.data.masking.mask_token_prob,
        random_token_prob=config.data.masking.random_token_prob,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    print(f"Val dataset: {len(dataset)} sequences")

    # Model
    model = ProteinMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        max_len=config.data.max_len,
        pad_id=tokenizer.pad_id,
        dropout=config.model.dropout,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Evaluate
    perplexity = evaluate(model, val_loader, device)
    print(f"Validation perplexity: {perplexity:.2f}")
    
    if tape_path is not None:
        records = load_tape_lmdb(tape_path, label_field="fold_label")
        sequences = [seq for seq, _ in records]
        labels = [label for _, label in records]
        embeddings = embed_sequences(model, sequences, tokenizer, device, batch_size = config.training.batch_size, max_len=config.data.max_len-1)
        hit_rate, baseline, lift = score_knn(embeddings, labels, k=10, min_fold_size=2)
        print(f"kNN hit rate:  {hit_rate:.3f}")
        print(f"kNN baseline:  {baseline:.3f}")
        print(f"kNN lift:      {lift:.3f}")

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--tape-path",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        tape_path=args.tape_path,
    )