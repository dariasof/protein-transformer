"""
run_behavioural_scoring.py — compute label-free per-head scores for one model.
 
Produces a single .npz holding the offset profile and the same-amino-acid lift
across a sweep of min_sep values. One extraction pass over the eval set feeds
every score, since extraction dominates the cost.
 
Separate entry point from run_head_scoring.py: both write to data/analysis, and
sharing a script would eventually mean sharing an output filename. --tag is
mandatory for the same reason.
 
Run once per model, and once more with --untrained. The untrained baseline is
not optional: both scores are maxima over 64 heads, so they sit above 1.0 even
on a random-init model, and that offset has to be measured rather than assumed.
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import numpy as np
 
from plm.analysis.aa_score import same_aa_lift
from plm.analysis.extract import extract_attention
from plm.analysis.load import load_from_hub
from plm.analysis.offset_score import offset_profile
from plm.config import load_config
from plm.data.tokenizer import ProteinTokenizer
from plm.model.mlm import ProteinMLM
 
 
def read_fasta(path: Path) -> dict[str, str]:
    records, name, chunks = {}, None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name, chunks = line[1:], []
        else:
            chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks)
    return records
 
 
def build_untrained(config_path: Path, tokenizer: ProteinTokenizer) -> ProteinMLM:
    """Same architecture, fresh init.
 
    Re-instantiated rather than perturbed, so the control differs from the
    trained model in exactly one respect: the weights were never updated.
    """
    config = load_config(config_path)
    model = ProteinMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        max_len=config.data.max_len,
        pad_id=tokenizer.pad_id,
        d_ff=config.model.d_ff,
        dropout=0.0,
    )
    model.eval()
    return model
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--tag", required=True, help="output suffix; must be unique")
    parser.add_argument("--max-offset", type=int, default=8)
    parser.add_argument("--min-sep-sweep", type=int, nargs="+",
                        default=[1, 2, 3, 4, 6, 8, 12, 16, 20])
    parser.add_argument("--out-dir", default="data/analysis")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
 
    offsets = list(range(-args.max_offset, args.max_offset + 1))
    sequences = read_fasta(Path(args.fasta))
 
    # The trained model is loaded either way: it supplies the tokenizer, and
    # loading it confirms the checkpoint the control is being compared against.
    model, tokenizer, step = load_from_hub(
        Path(args.config), args.repo_id, args.filename, device=args.device
    )
    if args.untrained:
        model = build_untrained(Path(args.config), tokenizer)
        model.to(args.device)
        step = -1
 
    profiles, cls_masses, dropped_rows = [], [], []
    aa_micro = {s: [] for s in args.min_sep_sweep}
    aa_macro = {s: [] for s in args.min_sep_sweep}
    aa_per_class = {s: [] for s in args.min_sep_sweep}
    ids, lengths = [], []
 
    for protein_id, sequence in sequences.items():
        attention = extract_attention(model, tokenizer, sequence, device=args.device)
 
        profile, cls_mass, dropped = offset_profile(attention, offsets)
        profiles.append(profile)
        cls_masses.append(cls_mass)
        dropped_rows.append(dropped)
 
        for min_sep in args.min_sep_sweep:
            micro, macro, per_class, _ = same_aa_lift(
                attention, sequence, min_sep=min_sep
            )
            aa_micro[min_sep].append(micro)
            aa_macro[min_sep].append(macro)
            aa_per_class[min_sep].append(per_class)
 
        ids.append(protein_id)
        lengths.append(len(sequence))
 
    payload = {
        "offset_profiles": np.stack(profiles),   # [n_prot, n_layers, n_heads, n_off]
        "offsets": np.array(offsets),
        "cls_mass": np.stack(cls_masses),
        "dropped_rows": np.stack(dropped_rows),
        "min_sep_sweep": np.array(args.min_sep_sweep),
        "ids": np.array(ids),
        "lengths": np.array(lengths),
        "step": step,
    }
    for min_sep in args.min_sep_sweep:
        payload[f"aa_micro_ms{min_sep}"] = np.stack(aa_micro[min_sep])
        payload[f"aa_macro_ms{min_sep}"] = np.stack(aa_macro[min_sep])
        payload[f"aa_per_class_ms{min_sep}"] = np.stack(aa_per_class[min_sep])
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"behavioural_{args.tag}.npz"
    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path}")
    print(f"  proteins {len(ids)}, step {step}, "
          f"offset grid {np.stack(profiles).shape}")
 
 
if __name__ == "__main__":
    main()