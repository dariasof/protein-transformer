from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

import torch

from plm.config import load_config
from plm.data.fasta import iter_fasta
from plm.data.tokenizer import ProteinTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_mmseqs(
    filtered_fasta: Path,
    out_dir: Path,
    tmp_dir: Path,
    min_seq_id: float = 0.3,
) -> Path:
    """
    Run MMseqs2 easy-cluster on filtered_fasta.
    Returns the path to the cluster TSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = out_dir / "clusters"

    subprocess.run(
        [
            "mmseqs", "easy-cluster",
            str(filtered_fasta),
            str(out_prefix),
            str(tmp_dir),
            "--min-seq-id", str(min_seq_id),
            "--cov-mode", "0",
            "-c", "0.8",
        ],
        check=True,
    )

    cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"MMseqs2 finished but {cluster_tsv} not found. "
            "Check MMseqs2 output above for errors."
        )

    return cluster_tsv

def load_sequences(filtered_fasta: Path) -> dict[str, str]:
    id_to_seq: dict[str, str] = {}
    for header, seq in iter_fasta(filtered_fasta, gzipped=False):
        id_to_seq[header] = seq
    return id_to_seq

def parse_clusters(cluster_tsv: Path) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    with open(cluster_tsv, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rep_id, member_id = line.split("\t")
            if rep_id not in clusters:
                clusters[rep_id] = []
            clusters[rep_id].append(member_id)
    return clusters



def assign_splits(
    clusters: dict[str, list[str]],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Assign whole clusters to train/val/test by protein count.
    """
    total = sum(len(members) for members in clusters.values())
    train_target = int(0.8 * total)
    val_target   = int(0.9 * total)

    # shuffle cluster order
    cluster_list = list(clusters.values())
    rng = random.Random(seed)
    rng.shuffle(cluster_list)

    train_ids: list[str] = []
    val_ids:   list[str] = []
    test_ids:  list[str] = []

    accumulated = 0
    for members in cluster_list:
        if accumulated < train_target:
            train_ids.extend(members)
        elif accumulated < val_target:
            val_ids.extend(members)
        else:
            test_ids.extend(members)
        accumulated += len(members)

    return train_ids, val_ids, test_ids

def tokenize_and_save(member_ids: list[str],
    id_to_seq: dict[str, str],
    tokenizer: ProteinTokenizer,
    out_path: Path,) -> None:
    kept = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for member_id in member_ids:
        seq = id_to_seq.get(member_id)
        if seq is None:
            raise KeyError(
            f"Member ID {member_id!r} from cluster.tsv not found in "
            f"filtered.fasta. Re-run build_filtered_fasta.py to regenerate.")
        ids = tokenizer.encode(seq, add_cls=True)
        kept.append(torch.tensor(ids, dtype=torch.long)) 
    torch.save(kept, out_path)
    print(f"Saved {len(kept)} tokenized sequences to {out_path}")
                         

def main(config_path: Path) -> None:
    config = load_config(config_path)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    filtered_fasta = processed_dir / "filtered.fasta"
    out_dir        = processed_dir / "mmseqs"
    tmp_dir        = processed_dir / "mmseqs_tmp"

    cluster_tsv = run_mmseqs(
        filtered_fasta=filtered_fasta,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
        min_seq_id=0.3,
    )

    clusters  = parse_clusters(cluster_tsv)
    id_to_seq = load_sequences(filtered_fasta)

    train_ids, val_ids, test_ids = assign_splits(
        clusters=clusters,
        seed=config.seed,
    )

    print(f"Split sizes — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    tokenizer = ProteinTokenizer()
    tokenize_and_save(train_ids, id_to_seq, tokenizer, processed_dir / "train.pt")
    tokenize_and_save(val_ids,   id_to_seq, tokenizer, processed_dir / "val.pt")
    tokenize_and_save(test_ids,  id_to_seq, tokenizer, processed_dir / "test.pt")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)