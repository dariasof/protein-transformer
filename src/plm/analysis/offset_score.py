"""Per-head local-offset scoring.

For each (layer, head), measures how much attention mass sits on each
sequence-offset diagonal of the attention matrix, expressed as lift over
the uniform-attention expectation of 1/L.

Offset convention: d selects A[i, i+d], so positive d is attention
directed *forward* along the sequence. A previous-token head peaks at
d = -1.
"""

import numpy as np


def offset_profile(attn, offsets, renormalize=True, min_row_mass=0.01):
    """Score one protein's attention matrices against a set of offsets.

    Args:
        attn: [n_layers, n_heads, L, L], already stripped of special
            tokens. Rows sum to <= 1 (the shortfall is mass that went to
            [CLS] before stripping).
        offsets: iterable of ints, e.g. range(-8, 9).
        renormalize: if True, rescale rows to sum to 1 before scoring, so
            the profile describes the *shape* of non-sink attention rather
            than absolute mass. Sink magnitude is returned separately.

    Returns:
        profile: [n_layers, n_heads, len(offsets)] in lift units.
        cls_mass: [n_layers, n_heads], mean per-row mass lost to [CLS].
        dropped: [n_layers, n_heads]
    """
    n_layers, n_heads, L, L_check = attn.shape
    if L != L_check:
        raise ValueError(f"attention matrices must be square, got {L}x{L_check}")

    offsets = list(offsets)
    if any(abs(d) >= L for d in offsets):
        raise ValueError(f"offset magnitude must be < L={L}")

    # Rows summed to 1 before [CLS] was stripped, so whatever is missing
    # now is exactly what the head sent to [CLS]. Free sink diagnostic,
    # and it means extract.py's contract stays untouched.
    row_sums = attn.sum(axis=-1)
    cls_mass = 1.0 - row_sums.mean(axis=-1)

        # A handful of individual query rows send essentially their whole budget to
    # [CLS]. Excluding those rows is not the same as excluding the head: the
    # rest of the head's rows still have shape worth measuring.
    sink_rows = row_sums < min_row_mass
    dropped = sink_rows.mean(axis=-1)

    if renormalize:
        safe = np.where(sink_rows[..., None], 1.0, row_sums[..., None])
        attn = attn / safe

    # NaN, not zero: a dropped row must be absent from the average, not counted
    # as a row that happened to have no mass at this offset.
    attn = np.where(sink_rows[..., None], np.nan, attn)

    profile = np.empty((n_layers, n_heads, len(offsets)), dtype=np.float64)
    for k, d in enumerate(offsets):
        # np.diagonal returns only the L-|d| valid cells and appends them
        # as the last axis, so .mean(-1) divides by the correct count.
        # Dividing by L instead would make a uniform head appear to decay
        # away from the diagonal.
        diag = np.diagonal(attn, offset=d, axis1=-2, axis2=-1)
        profile[:, :, k] = np.nanmean(diag, axis=-1) * L  # * L == / (1/L)

    return profile, cls_mass, dropped


def collapse_profile(profile, offsets):
    """Reduce a profile to (best lift, best offset), excluding d = 0.

    Offset 0 is self-attention, a different phenomenon from directional
    local attention; folding it in would label identity heads as local.
    It stays in the profile array for plotting, just not in the argmax.
    """
    offsets = np.asarray(list(offsets))
    keep = offsets != 0
    sub = profile[:, :, keep]
    best_idx = sub.argmax(axis=-1)
    return np.take_along_axis(sub, best_idx[..., None], axis=-1).squeeze(-1), \
           offsets[keep][best_idx]