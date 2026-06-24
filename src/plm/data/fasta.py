"""
FASTA parsing utilities.

Shared by the dataset builder and the split pipeline so the parsing
logic lives in exactly one place.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

from plm.data.tokenizer import AMINO_ACIDS


_VALID_AA_SET = set(AMINO_ACIDS)


def iter_fasta(fasta_gz_path: Path) -> Iterator[tuple[str, str]]:
    """
    Stream (header, sequence) pairs from a gzipped FASTA file.

    This is a generator so the file is never fully loaded into memory —
    safe for files larger than available RAM.

    Args:
        fasta_gz_path: Path to a gzipped FASTA file.
    Yields:
        (header, sequence) pairs. Header has the leading '>' stripped.
    """
    with gzip.open(fasta_gz_path, "rt") as f:
        header: str | None = None
        seq_chunks: list[str] = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        # yield the final record — no trailing '>' to trigger it
        if header is not None:
            yield header, "".join(seq_chunks)


def is_standard_sequence(seq: str) -> bool:
    """Return True if every character is one of the 20 standard amino acids."""
    return all(c in _VALID_AA_SET for c in seq)