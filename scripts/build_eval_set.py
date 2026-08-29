"""
build_eval_set.py — construct the contact-map evaluation set from TAPE/ProteinNet.

Reads the ProteinNet validation split, applies structural quality filters, and
writes two artifacts:

    eval_manifest.csv  — one row per candidate, with the filter decision recorded
    eval_seqs.fasta    — sequences of the surviving candidates, for MMseqs2

The manifest deliberately records *every* candidate, not just the survivors, with
a boolean `kept` column and the individual filter outcomes. This makes the
selection auditable: a reader can see how many chains each filter removed without
re-running anything.

Contamination filtering against the training set is a separate step that consumes eval_seqs.fasta and narrows the set further.
"""

import csv
import pickle
from pathlib import Path

import lmdb
import numpy as np
from plm.analysis.contacts import build_contact_map, build_eligibility_mask

# Filter thresholds. Kept as module-level constants so they appear in one place
# and can be cited directly in the writeup's methods section.
MIN_LENGTH = 80      # below this, too few pairs at |i-j| >= 24 for a stable estimate
MAX_LENGTH = 300     # attention cache is O(L^2) per layer per head
MIN_COVERAGE = 0.90  # fraction of residues with resolved coordinates
REPO_ROOT = Path(__file__).resolve().parents[1]

def scan_split(lmdb_path: Path) -> list[dict]:
    """Read every record from a TAPE LMDB split.

    Returns one dict per record with the fields needed for filtering. Coordinates
    are not retained here — this pass is only about deciding which chains to use.
    """
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    records = []
    try:
        with env.begin() as txn:
            n_expected = pickle.loads(txn.get(b"num_examples"))
            for i in range(n_expected):
                item = pickle.loads(txn.get(str(i).encode()))
                valid_mask = np.asarray(item["valid_mask"], dtype=bool)
                records.append({
                    "id": item["id"].decode(),      # stored as bytes; decode once, here
                    "length": int(item["protein_length"]),
                    "coverage": float(valid_mask.mean()),
                    "sequence": item["primary"],
                    "identity_bin": int(item["id"].decode().split("#", 1)[0]),
                    "tertiary": np.asarray(item["tertiary"], dtype=float),  # retained for downstream contact-map construction
                    "valid_mask": np.asarray(item["valid_mask"], dtype=bool),        # retained for downstream contact-map construction
                })
    finally:
        env.close()

    # Guard against silent truncation: if unpickling ever fails partway through,
    # we want a loud error rather than a quietly shorter eval set.
    assert len(records) == n_expected, f"expected {n_expected} records, got {len(records)}"
    return records


def apply_filters(records: list[dict]) -> list[dict]:
    """Annotate each record with per-filter outcomes and an overall `kept` flag."""
    for r in records:
        r["pass_length"] = MIN_LENGTH <= r["length"] <= MAX_LENGTH
        r["pass_coverage"] = r["coverage"] >= MIN_COVERAGE
        eligible = build_eligibility_mask(r["valid_mask"])
        contacts = build_contact_map(r["tertiary"], r["valid_mask"])
        r["n_eligible"] = int(eligible.sum())
        r["n_contacts"] = int((contacts & eligible).sum())
        r["k"]= r["length"] // 5
        r["pass_contacts"] = r["n_contacts"] >= r["k"]
        r["kept"] = r["pass_length"] and r["pass_coverage"] and r["pass_contacts"]
    return records


def write_manifest(records: list[dict], path: Path) -> None:
    """Write the full candidate list, including rejects, as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "length", "coverage", "pass_length", "pass_coverage", "kept", "identity_bin", "n_eligible", "n_contacts", "k", "pass_contacts"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_fasta(records: list[dict], path: Path) -> None:
    """Write surviving sequences. Headers use the same ids as the manifest so
    MMseqs2 output can be joined back on them without a lookup table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            if r["kept"]:
                f.write(f">{r['id']}\n{r['sequence']}\n")


def main() -> None:
    lmdb_path = REPO_ROOT / "data/raw/proteinnet/proteinnet_valid.lmdb"
    out_dir = REPO_ROOT / "data/processed"

    records = apply_filters(scan_split(lmdb_path))
    write_manifest(records, out_dir / "eval_manifest.csv")
    write_fasta(records, out_dir / "eval_seqs.fasta")

    n_kept = sum(r["kept"] for r in records)
    print(f"candidates:        {len(records)}")
    print(f"  fail length:     {sum(not r['pass_length'] for r in records)}")
    print(f"  fail coverage:   {sum(not r['pass_coverage'] for r in records)}")
    print(f"  fail contacts:   {sum(not r['pass_contacts'] for r in records)}")
    print(f"kept:              {n_kept}")


if __name__ == "__main__":
    main()