"""
export_for_protspace.py — export model embeddings + fold labels for ProtSpace.

Produces two files that ProtSpace consumes:
    1. an HDF5 file: protein_id -> embedding vector
    2. a features CSV: identifier, fold_label

These feed the ProtSpace CLI:
    protspace-json -i embeddings.h5 -m features.csv -o output.json --methods pca2 umap2
    protspace output.json

Run from the project root:

    python scripts/export_for_protspace.py \
        --config configs/5M.yaml \
        --checkpoint data/checkpoints/resume.pt \
        --tape-path data/remote_homology/remote_homology_train.lmdb \
        --out-dir data/protspace

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import torch

from plm.config import load_config
from plm.data.tokenizer import ProteinTokenizer
from plm.eval.knn_probe import embed_sequences
from plm.eval.tape_data import load_tape_lmdb
from plm.model.mlm import ProteinMLM


def main(
    config_path: Path,
    checkpoint_path: Path,
    tape_path: Path,
    out_dir: Path,
) -> None:

    config = load_config(config_path)
    tokenizer = ProteinTokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load TAPE data with ids — three aligned lists, order preserved
    records = load_tape_lmdb(tape_path, label_field="fold_label", include_id=True)
    ids = [r[0].decode() if isinstance(r[0], bytes) else r[0] for r in records]
    sequences = [r[1] for r in records]
    labels = [r[2] for r in records]
    print(f"Loaded {len(records)} proteins")

    # Model
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Embed — row i aligned to ids[i] and labels[i]
    embeddings = embed_sequences(
        model, sequences, tokenizer, device,
        batch_size=config.training.batch_size,
        max_len=config.data.max_len - 1,  # -1 for CLS
    )
    print(f"Embedded: {tuple(embeddings.shape)}")

    # Write HDF5: one dataset per protein, keyed by id
    h5_path = out_dir / "embeddings.h5"
    with h5py.File(h5_path, "w") as f:
        for i, pid in enumerate(ids):
            f[pid] = embeddings[i].numpy()
    print(f"Wrote {h5_path}")

    # Write features CSV: identifier + fold label
    csv_path = out_dir / "features.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "fold_label"])
        for pid, label in zip(ids, labels):
            writer.writerow([pid, label])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tape-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/protspace"))
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tape_path=args.tape_path,
        out_dir=args.out_dir,
    )