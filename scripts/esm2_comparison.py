"""
esm2_comparison.py — run the contact-scoring pipeline on ESM-2 as a positive
control.

Rao et al. (2021) established that ESM-family attention recovers long-range
contacts well above chance. Running the exact same eval set, geometry, and
scorer against ESM-2 answers a question the in-house model's numbers alone
cannot: whether a null result there reflects the model or a bug in the
scoring pipeline.

Usage:
    python scripts/esm2_comparison.py --model facebook/esm2_t6_8M_UR50D
"""
import argparse
import csv
import lmdb
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from plm.analysis.extract import extract_attention_esm2
from plm.analysis.contacts import build_contact_map, build_eligibility_mask
from plm.analysis.contact_score import score_head

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_kept_records(manifest_path, lmdb_path):
    with open(manifest_path) as f:
        kept_ids = {r["id"] for r in csv.DictReader(f) if r["kept"] == "True"}

    records = {}
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        n = pickle.loads(txn.get(b"num_examples"))
        for i in range(n):
            item = pickle.loads(txn.get(str(i).encode()))
            pid = item["id"].decode()
            if pid in kept_ids:
                records[pid] = item
    env.close()

    assert len(records) == len(kept_ids), (
        f"expected {len(kept_ids)} kept ids, found {len(records)} in the LMDB"
    )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--min-separation", type=int, default=24)
    parser.add_argument("--use-apc", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    records = load_kept_records(
        REPO_ROOT / "data/processed/eval_manifest.csv",
        REPO_ROOT / "data/raw/proteinnet/proteinnet_valid.lmdb",
    )

    precision = np.full((len(records), n_layers, n_heads), np.nan)
    baseline = np.full(len(records), np.nan)
    ids = []

    for idx, (pid, item) in enumerate(records.items()):
        seq = item["primary"]
        coords = np.asarray(item["tertiary"], dtype=float)
        valid = np.asarray(item["valid_mask"], dtype=bool)
        k = len(seq) // 5

        attn = extract_attention_esm2(model, tokenizer, seq, device=args.device)
        cmap = build_contact_map(coords, valid)
        elig = build_eligibility_mask(valid, min_separation=args.min_separation)
        baseline[idx] = (cmap & elig).sum() / elig.sum()

        for l in range(n_layers):
            for h in range(n_heads):
                p, _ = score_head(attn[l, h], cmap, elig, k, use_apc=args.use_apc)
                precision[idx, l, h] = p

        ids.append(pid)
        if idx % 15 == 0:
            print(f"{idx:>3}/{len(records)}  {pid}  L={len(seq)}")

    suffix = f"_{args.tag}" if args.tag else ""
    model_name = args.model.split("/")[-1]
    out_path = REPO_ROOT / "data/processed" / f"head_scores_{model_name}{suffix}.npz"
    np.savez(out_path, precision=precision, baseline=baseline, ids=np.array(ids))

    mean_lift = precision.mean(axis=0) - baseline.mean()
    print(f"wrote {out_path}")
    print(f"mean precision={precision.mean():.4f}  mean baseline={baseline.mean():.4f}")
    print(f"best head lift={mean_lift.max():.4f} at layer/head "
          f"{np.unravel_index(mean_lift.argmax(), mean_lift.shape)}")


if __name__ == "__main__":
    main()