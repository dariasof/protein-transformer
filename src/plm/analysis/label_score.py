
"""
label_score.py — shared machinery for "does this head attend to residues that
share label X with the query?"
 
 
1. Self-attention is always excluded; position i trivially shares every label
   with itself.
2. Nearby positions can be excluded via min_sep. Both amino acid identity and
   secondary structure are locally autocorrelated -- SS especially, since an
   element is a contiguous run -- so without this a pure local-offset head
   scores above chance with no label preference at all.
3. Chance is the fraction of *eligible* keys sharing the query's label. It
   depends on the protein's composition and on the label alphabet, so it is
   computed per query rather than assumed to be 1/L or 1/n_classes.
 
Because the score is a ratio taken inside the eligible set, the [CLS] sink
cancels: it is absent from numerator and denominator alike.
"""
 
from __future__ import annotations
 
import numpy as np
 
 
def categorical_lift(attn, labels, min_sep=1, min_eligible_mass=1e-3, n_classes=None):
    """Same-label attention lift per head, for one protein.
 
    Args:
        attn: [n_layers, n_heads, L, L], stripped of special tokens.
        labels: integer array of length L. Negative entries are unlabelled and
            never match anything, including each other -- two unknown residues
            are not evidence of same-label attention. They are excluded as
            queries and as keys.
        min_sep: minimum |i - j| for a key to be eligible. 1 excludes only the
            query itself.
        min_eligible_mass: queries retaining less than this share of their row
            inside the eligible set are dropped; their ratio would be a
            quotient of two near-zero numbers.
        n_classes: size of the label alphabet, for the per-class breakdown.
            Defaults to max(labels) + 1.
 
    Returns:
        micro: [n_layers, n_heads], lift averaged over query positions, so
            frequent labels dominate in proportion to their frequency.
        macro: [n_layers, n_heads], lift averaged within each class and then
            across classes, so rare classes count equally.
        per_class: [n_layers, n_heads, n_classes], lift per query label.
        dropped: [n_layers, n_heads], fraction of query rows excluded.
    """
    n_layers, n_heads, L, L_check = attn.shape
    if L != L_check:
        raise ValueError(f"attention must be square, got {L}x{L_check}")
    if len(labels) != L:
        raise ValueError(f"labels length {len(labels)} does not match attention size {L}")
 
    attn = np.asarray(attn, dtype=np.float64)
    labels = np.asarray(labels)
    if n_classes is None:
        n_classes = int(labels.max()) + 1 if (labels >= 0).any() else 0
 
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    known = labels >= 0
    eligible = (sep >= max(min_sep, 1)) & known[None, :]
    same = (labels[:, None] == labels[None, :]) & known[:, None]
    same_eligible = same & eligible
 
    n_eligible = eligible.sum(axis=1).astype(np.float64)
    n_same = same_eligible.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.where(n_eligible > 0, n_same / n_eligible, np.nan)
 
    eligible_mass = (attn * eligible).sum(axis=-1)
    same_mass = (attn * same_eligible).sum(axis=-1)
 
    # expected == 0 means no same-label partner is eligible: the ratio is
    # undefined, not zero. Dropping the query is the only correct move.
    usable = (
        (eligible_mass >= min_eligible_mass)
        & known[None, None, :]
        & np.isfinite(expected)[None, None, :]
        & (expected > 0)[None, None, :]
    )
    dropped = 1.0 - usable.mean(axis=-1)
 
    with np.errstate(invalid="ignore", divide="ignore"):
        lift = (same_mass / eligible_mass) / expected[None, None, :]
    lift = np.where(usable, lift, np.nan)
 
    with np.errstate(invalid="ignore"):
        micro = np.nanmean(lift, axis=-1)
 
    per_class = np.full((n_layers, n_heads, n_classes), np.nan)
    for c in range(n_classes):
        cols = labels == c
        if not cols.any():
            continue
        sub = lift[:, :, cols]
        if np.all(np.isnan(sub)):
            continue
        with np.errstate(invalid="ignore"):
            per_class[:, :, c] = np.nanmean(sub, axis=-1)
    with np.errstate(invalid="ignore"):
        macro = np.nanmean(per_class, axis=-1)
 
    return micro, macro, per_class, dropped