"""
One-time data preparation: parse SwissProt FASTA and save a tokenized .pt file.

"""

from pathlib import Path

from plm.data.dataset import build_swissprot_pt
from plm.config import load_config
import argparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FASTA = PROJECT_ROOT / "data" / "raw" / "uniprot_sprot.fasta.gz"


def main(config_path: Path) -> None:
    config = load_config(config_path)
    OUT = PROJECT_ROOT / "data" / "processed" / f"swissprot_{config.data.n_proteins // 1000}k.pt"
    build_swissprot_pt(
        fasta_gz_path=FASTA,
        out_path=OUT,
        n_proteins=config.data.n_proteins,
        max_length=config.data.max_len-1,  # reserve one slot for [CLS]
        min_length=config.data.min_length,
        standard_aa_only=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)