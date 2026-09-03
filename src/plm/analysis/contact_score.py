"""
contact_score.py — score attention heads against ground-truth contact maps.

Each (layer, head) produces an [L, L] attention matrix per protein. This module
turns one such matrix into a single number: the fraction of its top-scoring
long-range pairs that are genuine contacts. Averaged over an evaluation set,
that number is the per-head contact recovery score behind the head atlas.

The transforms are exposed separately rather than folded into one pipeline
function so that a single cached extraction can be scored under several
configurations. Symmetrization follows from what a contact is -- a symmetric
physical relation -- and is not optional. APC is a statistical correction whose
benefit is an empirical question on any given model, so it is applied by the
caller rather than assumed here.
"""

from __future__ import annotations

import numpy as np


def symmetrize(attention: np.ndarray) -> np.ndarray:
    """Fold a directional attention matrix into a symmetric one.

    Residue i attending to j and j attending to i are two observations of the
    same physical proposition, so they are summed. Without this the prediction
    would be asymmetric while the ground truth is not.

    Args:
        attention: [L, L] array. Entry (i, j) is attention from i to j.

    Returns:
        [L, L] symmetric array.
    """
    return attention + attention.T


def apply_apc(matrix: np.ndarray) -> np.ndarray:
    """Subtract the average product correction.

    APC estimates the part of each entry explained by its row and column
    marginals alone -- "position i is generally active" and "position j is
    generally attended-to" -- and removes it, leaving the pairwise excess:

        S_ij = a_ij - (a_i. * a_.j) / a_..

    This is the same construction as the expected cell count in a contingency
    table under independence. It exactly annihilates any matrix that
    factorizes as f_i * g_j; on matrices that only approximately factorize the
    cancellation is partial. A symmetrized attention sink is one such case, so
    APC reduces but does not eliminate sink bias.



    Args:
        matrix: [L, L] array, typically the output of symmetrize().

    Returns:
        [L, L] array of corrected scores. Values may be negative; only the
        ranking matters downstream.
    """
    row_means = matrix.mean(axis=1, keepdims=True)   # [L, 1]
    col_means = matrix.mean(axis=0, keepdims=True)   # [1, L]
    grand_mean = matrix.mean()

    correction = (row_means * col_means) / grand_mean  # broadcasts to [L, L]
    return matrix - correction


def precision_at_k(
    scores: np.ndarray,
    contact_map: np.ndarray,
    eligible: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Fraction of the k highest-scoring eligible pairs that are true contacts.

    Both precision and the matching random baseline are returned, because the
    raw precision is not interpretable alone: it depends on how contact-dense
    the protein happens to be. The baseline is the fraction of eligible pairs
    that are contacts, which is exactly the expected precision of drawing k
    pairs at random from the same pool.

    Args:
        scores: [L, L] per-pair scores, higher meaning more contact-like.
        contact_map: [L, L] boolean ground truth from build_contact_map.
        eligible: [L, L] boolean mask from build_eligibility_mask. Upper
            triangular, so each pair is represented once.
        k: Number of pairs to select. Conventionally L // 5.

    Returns:
        (precision, baseline), both in [0, 1].
    """
    n_eligible = int(eligible.sum())
    if k > n_eligible:
        raise ValueError(
            f"cannot select k={k} pairs from {n_eligible} eligible ones -- "
            f"this protein should have been removed by the n_contacts >= k filter"
        )

    # Boolean-mask indexing traverses in row-major order, so masking both
    # arrays with the SAME mask yields two 1-D arrays in matching order:
    # element n of one describes the same pair as element n of the other.
    # Using different masks here would silently misalign scores and labels.
    scores_flat = scores[eligible]
    labels_flat = contact_map[eligible]

    # argpartition is O(n) where a full sort is O(n log n). It leaves the k
    # largest in the tail without ordering them, which is all that is needed:
    # precision counts hits among the selection, not their rank within it.
    top_k = np.argpartition(scores_flat, -k)[-k:]

    precision = float(labels_flat[top_k].mean())
    baseline = float(labels_flat.mean())
    return precision, baseline


def score_head(
    attention: np.ndarray,
    contact_map: np.ndarray,
    eligible: np.ndarray,
    k: int,
    use_apc: bool = True,
) -> tuple[float, float]:
    """Score one attention head on one protein.

    Convenience wrapper over the three steps above. `use_apc` is exposed so the
    same cached attention can be scored with and without the correction, which
    is how its benefit gets measured rather than assumed.
    """
    scores = symmetrize(attention)
    if use_apc:
        scores = apply_apc(scores)
    return precision_at_k(scores, contact_map, eligible, k)