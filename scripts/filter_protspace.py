"""
filter_protspace.py — keep only the N largest folds for a legible ProtSpace plot.

The full TAPE set has 1,195 folds, most with 2-3 members — visually noisy.
This filters both the HDF5 embeddings and the features CSV (by the same protein
set, so they stay aligned) down to the top-N most populated folds.

Run from the project root:

    python scripts/filter_protspace.py \
        --in-dir data/protspace \
        --out-dir data/protspace_top \
        --top-n 12
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import h5py


def main(in_dir: Path, out_dir: Path, top_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the features CSV — identifier -> fold_label
    id_to_fold = {}
    with open(in_dir / "features.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_fold[row["identifier"]] = row["fold_label"]

    # Find the top-N largest folds
    fold_counts = Counter(id_to_fold.values())
    top_folds = {fold for fold, _ in fold_counts.most_common(top_n)}
    print(f"Top {top_n} folds: {sorted(top_folds, key=int)}")

    # Which protein ids survive
    keep_ids = {pid for pid, fold in id_to_fold.items() if fold in top_folds}
    print(f"Keeping {len(keep_ids)} of {len(id_to_fold)} proteins")

    # Write filtered CSV
    with open(out_dir / "features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "fold_label"])
        for pid, fold in id_to_fold.items():
            if pid in keep_ids:
                writer.writerow([pid, fold])

    # Write filtered HDF5 — copy only surviving datasets
    with h5py.File(in_dir / "embeddings.h5", "r") as src, \
         h5py.File(out_dir / "embeddings.h5", "w") as dst:
        for pid in keep_ids:
            dst[pid] = src[pid][:]

    print(f"Wrote filtered files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("data/protspace"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/protspace_top"))
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.top_n)