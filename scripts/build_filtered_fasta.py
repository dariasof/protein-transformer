
from __future__ import annotations


from pathlib import Path

from tqdm import tqdm

from plm.data.fasta import iter_fasta, is_standard_sequence


from plm.config import load_config
import argparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(config_path: Path) -> None:
    config = load_config(config_path)
    out_path = PROJECT_ROOT / "data" / "processed" / "filtered.fasta"    
    fasta_gz_path=PROJECT_ROOT / "data" / "raw" / "uniprot_sprot.fasta.gz"
    n_proteins=config.data.n_proteins
    max_length=config.data.max_len-1  # reserve one slot for [CLS]
    min_length=config.data.min_length
    standard_aa_only=True

    fasta_gz_path = Path(fasta_gz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept: list[str] = []
 
    pbar = tqdm(iter_fasta(fasta_gz_path), desc="Parsing SwissProt", unit="seq")
    for _header, seq in pbar:
        if not (min_length <= len(seq) <= max_length):
            continue
        if standard_aa_only and not is_standard_sequence(seq):
            continue

        kept.append(seq)

        if len(kept) >= n_proteins:
            break

    if len(kept) < n_proteins:
        print(
            f"WARNING: requested {n_proteins} sequences, got only {len(kept)}. ")
        
    with open(out_path, 'w') as f:
        for i, seq in enumerate(kept):
            f.write(f">seq_{i}\n")
            f.write(f"{seq}\n")
   
    return out_path
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)

