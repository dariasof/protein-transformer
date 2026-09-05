"""
make_behavioural_figures.py — figures for the offset and same-AA head scores.
 
Consumes the .npz files written by run_behavioural_scoring.py and writes five
figures. Pure presentation: no scoring happens here, so a figure can be
restyled without recomputing anything.
 
Every figure carries the untrained baseline. Both scores are maxima over 64
heads, so the untrained line is not decoration -- it is the zero point, and a
trained number is only meaningful as a distance from it.
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
 
from plm.analysis.aa_score import STANDARD_AA
 
CHANCE_STYLE = dict(color="grey", linestyle=":", linewidth=0.9)
 
 
def best_offset_lift(data):
    """Per-protein best lift over non-zero offsets: [n_prot, n_layers, n_heads].
 
    Offset 0 is excluded because it is self-attention -- a different phenomenon
    from directional local attention, and including it would label identity
    heads as local.
    """
    keep = data["offsets"] != 0
    return data["offset_profiles"][..., keep].max(axis=-1)
 
 
def annotate_chance(ax, label="uniform-attention chance"):
    ax.axhline(1.0, label=label, **CHANCE_STYLE)
 
 
def figure_offset_by_depth(trained, untrained, out_path):
    """Where in the network local structure lives."""
    bt, bu = best_offset_lift(trained), best_offset_lift(untrained)
    n_layers = bt.shape[1]
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(n_layers), bt.mean((0, 2)), "o-", label="trained")
    ax.plot(range(n_layers), bu.mean((0, 2)), "s--", label="untrained (random init)")
    annotate_chance(ax)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean best-offset lift")
    ax.set_title("local attention by depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def figure_offset_atlas(trained, untrained, out_path):
    """Which individual heads carry the local structure."""
    bt = best_offset_lift(trained).mean(0)
    baseline = best_offset_lift(untrained).mean()
 
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(bt, cmap="viridis", vmin=1.0)
    fig.colorbar(image, ax=ax, label="best-offset lift")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"local-offset specialization\n(untrained mean {baseline:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def figure_aa_by_residue(trained, untrained, min_sep, out_path):
    """Same-AA lift per residue type, ordered by how common the residue is.
 
    Rarity is computed from the eval set itself rather than a reference table,
    so the ordering matches the data the lifts were measured on.
    """
    per_class = np.nanmean(trained[f"aa_per_class_ms{min_sep}"], axis=0)
    per_class_u = np.nanmean(untrained[f"aa_per_class_ms{min_sep}"], axis=0)
 
    best = np.array([np.nanmax(per_class[..., a]) for a in range(len(STANDARD_AA))])
    best_u = np.array([np.nanmax(per_class_u[..., a]) for a in range(len(STANDARD_AA))])
 
    # Rarity ordering makes the trend legible: the ceiling on lift is
    # 1 / (chance rate), so rare residues can reach far higher values.
    frequency = residue_frequency(trained["ids"], trained)
    order = np.argsort(-frequency)
 
    x = np.arange(len(STANDARD_AA))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, best[order], 0.4, label="trained")
    ax.bar(x + 0.2, best_u[order], 0.4, label="untrained")
    annotate_chance(ax, label="chance")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([STANDARD_AA[i] for i in order])
    ax.set_xlabel("residue (most to least common in the eval set)")
    ax.set_ylabel("best-head same-AA lift")
    ax.set_title(f"same-amino-acid specialization (min_sep={min_sep})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def residue_frequency(ids, data):
    """Residue frequencies implied by the stored per-class coverage.
 
    Falls back to counting NaNs in the per-class array: a residue absent from a
    protein yields NaN there, so the count of non-NaN entries across proteins is
    proportional to how many proteins contain it. This avoids re-reading the
    FASTA just to order an axis.
    """
    key = [k for k in data.files if k.startswith("aa_per_class_ms")][0]
    present = np.isfinite(data[key]).sum(axis=(0, 1, 2)).astype(float)
    return present / present.sum()
 
 
def figure_aa_micro_macro(trained, min_sep, out_path):
    """Micro and macro side by side; their disagreement is the point.
 
    A head specialized on rare residues is diluted into invisibility by
    micro-averaging over query positions, and surfaces under macro-averaging
    across residue types.
    """
    micro = np.nanmean(trained[f"aa_micro_ms{min_sep}"], axis=0)
    macro = np.nanmean(trained[f"aa_macro_ms{min_sep}"], axis=0)
    layer, head = np.unravel_index(np.nanargmax(macro), macro.shape)
 
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, name in ((axes[0], micro, "micro"), (axes[1], macro, "macro")):
        image = ax.imshow(values, cmap="viridis")
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax.scatter([head], [layer], s=160, facecolors="none", edgecolors="w", linewidths=2)
        ax.set_title(f"{name}-averaged same-AA lift\n"
                     f"layer {layer} head {head}: {values[layer, head]:.2f}")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def figure_min_sep_sweep(trained, untrained, out_path):
    """The control that separates residue identity from local attention.
 
    Protein sequences are locally autocorrelated, so a pure local-offset head
    scores above chance on same-AA with no residue preference at all. If the
    same-AA lift is flat in min_sep, the signal is not locality in disguise.
    """
    sweep = trained["min_sep_sweep"]
    best_t, best_u, errors = [], [], []
    for min_sep in sweep:
        per_protein = trained[f"aa_macro_ms{min_sep}"]
        mean_map = np.nanmean(per_protein, axis=0)
        layer, head = np.unravel_index(np.nanargmax(mean_map), mean_map.shape)
        values = per_protein[:, layer, head]
        values = values[np.isfinite(values)]
        best_t.append(mean_map[layer, head])
        errors.append(values.std(ddof=1) / np.sqrt(len(values)))
        best_u.append(np.nanmax(np.nanmean(untrained[f"aa_macro_ms{min_sep}"], axis=0)))
 
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(sweep, best_t, yerr=errors, fmt="o-", capsize=3, label="trained")
    ax.plot(sweep, best_u, "s--", label="untrained")
    annotate_chance(ax)
    ax.set_xlabel("min_sep (keys closer than this are excluded)")
    ax.set_ylabel("best-head same-AA lift (macro)")
    ax.set_title("same-AA signal is not local autocorrelation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def print_summary(trained, untrained, min_sep):
    """Numbers to quote in captions, so they are never retyped from a plot."""
    bt, bu = best_offset_lift(trained), best_offset_lift(untrained)
    mean_map = bt.mean(0)
    layer, head = np.unravel_index(mean_map.argmax(), mean_map.shape)
    _, p_value = wilcoxon(bt[:, layer, head] - 1.0)
    print(f"offset: best layer {layer} head {head}, lift {mean_map[layer, head]:.2f}, "
          f"p={p_value:.2e}; untrained mean {bu.mean():.3f}")
 
    macro = np.nanmean(trained[f"aa_macro_ms{min_sep}"], axis=0)
    layer, head = np.unravel_index(np.nanargmax(macro), macro.shape)
    values = trained[f"aa_macro_ms{min_sep}"][:, layer, head]
    values = values[np.isfinite(values)]
    _, p_value = wilcoxon(values - 1.0)
    print(f"same-AA (min_sep={min_sep}): best layer {layer} head {head}, "
          f"macro {macro[layer, head]:.2f}, p={p_value:.2e}; "
          f"untrained max {np.nanmax(np.nanmean(untrained[f'aa_macro_ms{min_sep}'], 0)):.3f}")
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", required=True)
    parser.add_argument("--untrained", required=True)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--min-sep", type=int, default=6,
                        help="min_sep used for the per-residue and atlas figures")
    parser.add_argument("--prefix", default="20M")
    args = parser.parse_args()
 
    trained = np.load(args.trained, allow_pickle=True)
    untrained = np.load(args.untrained, allow_pickle=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = args.prefix
 
    figure_offset_by_depth(trained, untrained, out_dir / f"offset_by_depth_{p}.png")
    figure_offset_atlas(trained, untrained, out_dir / f"offset_atlas_{p}.png")
    figure_aa_by_residue(trained, untrained, args.min_sep,
                         out_dir / f"aa_by_residue_{p}.png")
    figure_aa_micro_macro(trained, args.min_sep, out_dir / f"aa_micro_macro_{p}.png")
    figure_min_sep_sweep(trained, untrained, out_dir / f"aa_min_sep_sweep_{p}.png")
 
    print(f"wrote 5 figures to {out_dir}")
    print_summary(trained, untrained, args.min_sep)
 
 
if __name__ == "__main__":
    main()
 