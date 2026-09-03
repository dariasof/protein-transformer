"""
run_head_scoring.py — score every (layer, head) of a checkpoint against the
contact evaluation set.

Usage:
    python scripts/run_head_scoring.py --config configs/20M.yaml \
        --repo-id dariasof/protein-transformer-20M \
        --checkpoint ckpt_step_025140.pt \
        --min-separation 24 \
        --use-apc

Writes data/processed/head_scores_<checkpoint-stem>.npz containing:
    precision  [n_proteins, n_layers, n_heads]
    baseline   [n_proteins]
    ids        list of protein ids, matching precision's first axis
"""
import argparse, csv, lmdb, pickle
from pathlib import Path

import numpy as np
import torch

from plm.analysis.load import load_from_hub
from plm.analysis.extract import extract_attention
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--min-separation", type=int, default=24)
    parser.add_argument("--use-apc", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer, step = load_from_hub(args.config, args.repo_id, args.checkpoint, args.device)
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.n_heads

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

        attn = extract_attention(model, tokenizer, seq, device=args.device)
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

    out_path = REPO_ROOT / "data/processed" / f"head_scores_{Path(args.checkpoint).stem}.npz"
    np.savez(out_path, precision=precision, baseline=baseline, ids=np.array(ids))
    print(f"wrote {out_path}")
    print(f"step={step}  mean precision={precision.mean():.4f}  mean baseline={baseline.mean():.4f}")


if __name__ == "__main__":
    main()