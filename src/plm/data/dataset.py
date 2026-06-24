"""
SwissProt dataset for protein masked language modeling.

Two-stage flow:
    1. build_swissprot_pt(fasta_gz, out_path, ...) — one-time preparation:
       reads a gzipped FASTA, filters sequences, tokenizes them, and saves the
       result as a torch `.pt` file (a list of 1-D LongTensors).
    2. SwissProtDataset(pt_path) — runtime: loads the prepared tensors and
       exposes them via PyTorch's Dataset interface.
"""

from __future__ import annotations


from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from plm.data.fasta import iter_fasta, is_standard_sequence
from plm.data.tokenizer import ProteinTokenizer


def build_swissprot_pt(
    fasta_gz_path: str | Path,
    out_path: str | Path,
    *,
    n_proteins: int = 10_000,
    max_length: int = 512,
    min_length: int = 30,
    standard_aa_only: bool = True,
) -> Path:
    """
    Parse SwissProt FASTA, filter, tokenize, and save to a `.pt` file.

    Filtering criteria:
        - min_length <= len(sequence) <= max_length
        - if standard_aa_only=True, only sequences using the 20 standard AAs

    The output `.pt` file is a Python list of 1-D LongTensors (token IDs),
    each starting with [CLS]. Sequence lengths vary; the collator pads at
    batch time.

    Returns the output path.
    """
    fasta_gz_path = Path(fasta_gz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = ProteinTokenizer()
    kept: list[torch.Tensor] = []
 
    pbar = tqdm(iter_fasta(fasta_gz_path), desc="Parsing SwissProt", unit="seq")
    for _header, seq in pbar:
        if not (min_length <= len(seq) <= max_length):
            continue
        if standard_aa_only and not is_standard_sequence(seq):
            continue

        ids = tokenizer.encode(seq, add_cls=True)
        kept.append(torch.tensor(ids, dtype=torch.long))

        if len(kept) >= n_proteins:
            break

    if len(kept) < n_proteins:
        print(
            f"WARNING: requested {n_proteins} sequences, got only {len(kept)}. ")

    torch.save(kept, out_path)
    print(f"Saved {len(kept)} tokenized sequences to {out_path}")
    return out_path

class SwissProtDataset(Dataset):
    """
    PyTorch Dataset over tokenized SwissProt sequences.

    Each item is a 1-D LongTensor of token IDs (starting with [CLS]), of
    variable length. Batching and padding are the collator's job.
    """

    def __init__(self, pt_path: str | Path) -> None:
        pt_path = Path(pt_path)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found at {pt_path}. "
                "Run `scripts/build_dataset.py` first."
            )

        self.sequences: list[torch.Tensor] = torch.load(
            pt_path, weights_only=False
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]
