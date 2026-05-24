"""
One-time data preparation: parse SwissProt FASTA and save a tokenized .pt file.

"""

from pathlib import Path

from plm.data.dataset import build_swissprot_pt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FASTA = PROJECT_ROOT / "data" / "raw" / "uniprot_sprot.fasta.gz"
OUT = PROJECT_ROOT / "data" / "processed" / "swissprot_10k.pt"


def main() -> None:
    build_swissprot_pt(
        fasta_gz_path=FASTA,
        out_path=OUT,
        n_proteins=10_000,
        max_length=512,
        min_length=30,
        standard_aa_only=True,
    )


if __name__ == "__main__":
    main()