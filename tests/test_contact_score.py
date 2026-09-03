"""
test_contact_score.py — unit tests for per-head contact scoring.

Every expected value here is constructed by hand: the scores matrix is built so
that the identity of the top-k pairs is known in advance, and the contact map is
built so that the number of hits among them is known too. A failure therefore
points at the code rather than at data.

The scoring functions are the most dangerous part of the pipeline to leave
untested, because a sign error, a misaligned mask, or an off-by-one in the
selection all produce plausible-looking precision values rather than crashes.
"""

import numpy as np
import pytest

from plm.analysis.contact_score import (
    apply_apc,
    precision_at_k,
    score_head,
    symmetrize,
)


# ---------------------------------------------------------------------------
# symmetrize
# ---------------------------------------------------------------------------

def test_symmetrize_produces_a_symmetric_matrix():
    rng = np.random.default_rng(0)
    a = rng.random((12, 12))
    s = symmetrize(a)

    assert np.allclose(s, s.T)
    assert np.isclose(s[3, 7], a[3, 7] + a[7, 3])


def test_symmetrize_pools_both_directions():
    """Directional evidence for the same pair must combine, not be discarded.

    Here i->j is strong and j->i is zero. A method that took only the upper
    triangle of the raw matrix, or only the lower, would rank this pair very
    differently depending on which it picked.
    """
    a = np.zeros((4, 4))
    a[0, 3] = 0.8
    s = symmetrize(a)

    assert s[0, 3] == s[3, 0] == 0.8


# ---------------------------------------------------------------------------
# apply_apc
# ---------------------------------------------------------------------------

def test_apc_annihilates_a_separable_matrix():
    """A matrix of the form f_i * g_j is entirely explained by its marginals.

    Substituting a_i. = f_i * mean(g), a_.j = mean(f) * g_j and
    a_.. = mean(f) * mean(g) into the correction gives exactly f_i * g_j, so
    the residual is zero. This is the one case where APC is exact, and it is
    the property the whole method rests on.
    """
    f = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    g = np.array([0.1, 0.4, 0.2, 0.05, 0.3])
    separable = np.outer(f, g)

    assert np.linalg.matrix_rank(separable) == 1
    assert np.allclose(apply_apc(separable), 0.0, atol=1e-12)


def test_apc_only_partially_removes_a_symmetrized_sink():
    """Documents a limitation, so it cannot be quietly forgotten.

    A single sink column is separable and would be removed exactly. Adding its
    transpose -- which is what symmetrize() does -- produces a cross, which is
    rank 2 and survives the correction. This is why use_apc is a measured
    choice rather than an assumed improvement.
    """
    g = np.array([0.05, 0.05, 0.05, 0.90, 0.05, 0.05])
    sink = np.outer(np.ones(6), g)
    crossed = symmetrize(sink)

    assert np.linalg.matrix_rank(sink) == 1
    assert np.allclose(apply_apc(sink), 0.0, atol=1e-12)

    assert np.linalg.matrix_rank(crossed) == 2
    assert np.abs(apply_apc(crossed)).max() > 0.1


def test_apc_preserves_shape_and_symmetry():
    """Ranking happens after APC, so the matrix must stay comparable to itself.

    Negative values are expected and fine -- only the ordering is used.
    """
    rng = np.random.default_rng(1)
    s = symmetrize(rng.random((10, 10)))
    corrected = apply_apc(s)

    assert corrected.shape == s.shape
    assert np.allclose(corrected, corrected.T)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def _upper_triangle_mask(size: int, offset: int = 1) -> np.ndarray:
    """Eligibility mask standing in for build_eligibility_mask.

    Constructed locally rather than imported so that a change in the geometry
    module cannot silently alter what these tests assert.
    """
    idx = np.arange(size)
    return (idx[None, :] - idx[:, None]) >= offset


def test_precision_counts_hits_among_the_selection():
    """Three of the four highest-scoring pairs are contacts, so precision = 0.75.

    Scores are assigned so the ranking is unambiguous: the four target pairs
    get 0.9, 0.8, 0.7, 0.6 and everything else 0.1. The contact map marks
    three of those four, plus one low-scoring pair that must NOT be selected.
    """
    size = 10
    eligible = _upper_triangle_mask(size)

    scores = np.full((size, size), 0.1)
    for (i, j), value in [((0, 5), 0.9), ((1, 6), 0.8), ((2, 7), 0.7), ((3, 8), 0.6)]:
        scores[i, j] = scores[j, i] = value

    contacts = np.zeros((size, size), dtype=bool)
    for i, j in [(0, 5), (1, 6), (2, 7), (4, 9)]:   # (3, 8) is a miss
        contacts[i, j] = contacts[j, i] = True

    precision, _ = precision_at_k(scores, contacts, eligible, k=4)
    assert precision == pytest.approx(3 / 4)


def test_perfect_and_zero_precision():
    """The two extremes, to pin the ends of the range."""
    size = 8
    eligible = _upper_triangle_mask(size)

    scores = np.full((size, size), 0.1)
    scores[0, 4] = scores[4, 0] = 0.9
    scores[1, 5] = scores[5, 1] = 0.8

    hits = np.zeros((size, size), dtype=bool)
    hits[0, 4] = hits[4, 0] = True
    hits[1, 5] = hits[5, 1] = True
    assert precision_at_k(scores, hits, eligible, k=2)[0] == pytest.approx(1.0)

    misses = np.zeros((size, size), dtype=bool)
    misses[2, 6] = misses[6, 2] = True
    assert precision_at_k(scores, misses, eligible, k=2)[0] == pytest.approx(0.0)


def test_baseline_is_the_contact_density_of_the_eligible_pool():
    """The baseline must be computed over the same mask as the selection.

    With offset 1 on a 10x10 matrix there are 45 eligible pairs. Marking 9 of
    them as contacts gives a density of 0.2 -- the expected precision of
    drawing k pairs at random from that pool, for any k.
    """
    size = 10
    eligible = _upper_triangle_mask(size)
    assert eligible.sum() == 45

    contacts = np.zeros((size, size), dtype=bool)
    for n, (i, j) in enumerate(zip(*np.triu_indices(size, k=1))):
        if n < 9:
            contacts[i, j] = contacts[j, i] = True

    rng = np.random.default_rng(2)
    scores = symmetrize(rng.random((size, size)))

    _, baseline = precision_at_k(scores, contacts, eligible, k=5)
    assert baseline == pytest.approx(9 / 45)


def test_baseline_does_not_depend_on_k():
    """k cancels: expected hits are k * p, so expected precision is p.

    This is why the baseline is comparable across proteins of different length,
    and therefore why averaging baselines across an eval set is meaningful.
    """
    size = 12
    eligible = _upper_triangle_mask(size)
    rng = np.random.default_rng(3)
    scores = symmetrize(rng.random((size, size)))
    contacts = rng.random((size, size)) < 0.3
    contacts = contacts | contacts.T

    baselines = {precision_at_k(scores, contacts, eligible, k)[1] for k in (2, 5, 10)}
    assert len(baselines) == 1


def test_ineligible_pairs_are_never_selected():
    """The mask must gate selection, not merely reweight it.

    Every short-range pair is given a score far above every eligible pair. If
    the mask were ignored, the selection would consist entirely of short-range
    pairs and precision would be 1.0 instead of 0.0.
    """
    size = 12
    eligible = _upper_triangle_mask(size, offset=6)

    scores = np.full((size, size), 0.1)
    contacts = np.zeros((size, size), dtype=bool)
    idx = np.arange(size)
    short_range = np.abs(idx[None, :] - idx[:, None]) < 6
    scores[short_range] = 99.0
    contacts[short_range] = True

    precision, baseline = precision_at_k(scores, contacts, eligible, k=3)
    assert precision == 0.0
    assert baseline == 0.0


def test_raises_when_k_exceeds_the_eligible_pool():
    """Clamping would let a protein the filters should have removed report a
    distorted number instead of failing."""
    size = 6
    eligible = _upper_triangle_mask(size, offset=4)   # 3 eligible pairs
    assert eligible.sum() == 3

    scores = np.zeros((size, size))
    contacts = np.zeros((size, size), dtype=bool)

    with pytest.raises(ValueError, match="eligible"):
        precision_at_k(scores, contacts, eligible, k=4)


# ---------------------------------------------------------------------------
# score_head
# ---------------------------------------------------------------------------

def test_score_head_matches_its_parts():
    """The wrapper must be a composition, not a reimplementation."""
    size = 14
    eligible = _upper_triangle_mask(size, offset=4)
    rng = np.random.default_rng(4)
    attention = rng.random((size, size))
    contacts = rng.random((size, size)) < 0.25
    contacts = contacts | contacts.T

    expected = precision_at_k(apply_apc(symmetrize(attention)), contacts, eligible, k=5)
    assert score_head(attention, contacts, eligible, k=5, use_apc=True) == expected

    expected_raw = precision_at_k(symmetrize(attention), contacts, eligible, k=5)
    assert score_head(attention, contacts, eligible, k=5, use_apc=False) == expected_raw


def test_apc_can_rescue_a_signal_buried_under_a_sink():
    """The case that motivates APC, in the setting where it works.

    A directional sink at column 4 dominates the raw scores. One genuine pair
    carries a smaller value. Because the sink is separable before
    symmetrization, applying APC first removes it and the true pair surfaces.
    Note the argument order -- this is APC-then-symmetrize, which is not what
    score_head does; see the rank-2 limitation test above.
    """
    size = 10
    eligible = _upper_triangle_mask(size, offset=4)

    attention = np.outer(np.ones(size), np.full(size, 0.05))
    attention[:, 4] = 0.60                      # sink column
    attention[0, 9] = attention[0, 9] + 0.25    # the genuine pair

    contacts = np.zeros((size, size), dtype=bool)
    contacts[0, 9] = contacts[9, 0] = True

    raw, _ = precision_at_k(symmetrize(attention), contacts, eligible, k=1)
    corrected, _ = precision_at_k(symmetrize(apply_apc(attention)), contacts, eligible, k=1)

    assert raw == 0.0
    assert corrected == 1.0