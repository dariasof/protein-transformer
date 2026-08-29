"""
test_contacts.py — unit tests for contact map and eligibility mask construction.

These tests use hand-constructed coordinates and masks, never real data. Every
expected value is derived by hand from the input, so a failure points at the
code rather than at a dataset. They run in milliseconds and do not require the
TAPE download.
"""

import numpy as np
import pytest

from plm.analysis.contacts import build_contact_map, build_eligibility_mask


# ---------------------------------------------------------------------------
# build_eligibility_mask
# ---------------------------------------------------------------------------

def test_eligible_pair_count_matches_formula():
    """With all residues valid, the eligible count is a closed-form expression.

    Pairs with j - i >= s exist for each separation s, s+1, ..., L-1. There are
    (L - s) pairs at separation s, so the total is the sum from 1 to (L - s),
    i.e. (L - s)(L - s + 1) / 2.

    For L=30, s=24: separations 24..29 contribute 6+5+4+3+2+1 = 21.
    """
    L, s = 30, 24
    mask = build_eligibility_mask(np.ones(L, dtype=bool), min_separation=s)

    expected = (L - s) * (L - s + 1) // 2
    assert expected == 21          
    assert mask.sum() == expected


def test_mask_is_upper_triangular():
    """Each pair must appear exactly once.

    Contacts are symmetric, but if the mask were symmetric too, top-k selection
    over mask would return every pair twice and precision@L/5 would silently
    become precision@L/10.
    """
    mask = build_eligibility_mask(np.ones(30, dtype=bool), min_separation=24)

    assert not (mask & mask.T).any()               # no pair present in both triangles
    assert np.triu(mask, k=1).sum() == mask.sum()  # everything lives above the diagonal


def test_diagonal_is_excluded():
    """A residue is never evaluated against itself."""
    mask = build_eligibility_mask(np.ones(30, dtype=bool), min_separation=24)
    assert not np.diag(mask).any()


def test_invalid_residue_removes_its_row_and_column():
    """An unresolved residue must be excluded as both i and j.

    L=10, s=3 is chosen so that position 5 has eligible pairs in both directions:
    as i it pairs with j in {8, 9} (2 pairs), as j it pairs with i in {0, 1, 2}
    (3 pairs). A mask that only filtered columns would still leave the 2 pairs
    where 5 appears as i.
    """
    L, s, bad = 10, 3, 5
    valid = np.ones(L, dtype=bool)

    n_all_valid = build_eligibility_mask(valid, min_separation=s).sum()
    assert n_all_valid == (L - s) * (L - s + 1) // 2 == 28

    valid[bad] = False
    mask = build_eligibility_mask(valid, min_separation=s)

    assert mask[bad, :].sum() == 0
    assert mask[:, bad].sum() == 0
    assert mask.sum() == 28 - 2 - 3


def test_accepts_non_boolean_mask():
    """A mask arriving as ints must not be treated as a bitwise operand."""
    as_int = np.array([1, 0, 1, 1, 1, 1], dtype=int)
    as_bool = as_int.astype(bool)

    assert np.array_equal(
        build_eligibility_mask(as_int, min_separation=2),
        build_eligibility_mask(as_bool, min_separation=2),
    )


# ---------------------------------------------------------------------------
# build_contact_map
# ---------------------------------------------------------------------------

def test_known_geometry():
    """Four points placed so the contacts are obvious by inspection.

    p0-p1 and p2-p3 are each 5 A apart (3-4-5 triangles). The two pairs are
    separated by 20 A along x, so no cross-pair is within 8 A.
    """
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],    # 5 A from p0
        [20.0, 0.0, 0.0],
        [23.0, 4.0, 0.0],   # 5 A from p2
    ])
    cmap = build_contact_map(coords, np.ones(4, dtype=bool))

    assert cmap[0, 1] and cmap[2, 3]
    assert not cmap[0, 2] and not cmap[0, 3] and not cmap[1, 2] and not cmap[1, 3]


def test_contact_map_is_symmetric_with_true_diagonal():
    """Distance is symmetric and zero from a point to itself.

    The diagonal being True is harmless: short-range pairs are removed by the
    separation filter downstream, not here.
    """
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [20.0, 0.0, 0.0]])
    cmap = build_contact_map(coords, np.ones(3, dtype=bool))

    assert np.array_equal(cmap, cmap.T)
    assert np.diag(cmap).all()


def test_unresolved_residue_does_not_contact_via_the_origin():
    """The regression test for the [0, 0, 0] fill.

    TAPE stores unresolved residues as the origin, which is a real point in
    space. p0 sits 1 A from it, so without masking both endpoints the pair
    (0, 2) would register as a contact. Every other test here passes even if
    this masking is missing.
    """
    coords = np.array([
        [1.0, 0.0, 0.0],     # valid, 1 A from the origin
        [50.0, 0.0, 0.0],    # valid, far away
        [0.0, 0.0, 0.0],     # unresolved -> TAPE's zero fill
    ])
    valid = np.array([True, True, False])
    cmap = build_contact_map(coords, valid)

    assert not cmap[0, 2]
    assert not cmap[2, 0]
    assert not cmap[2, :].any()
    assert cmap[0, 1] == False   # 49 A apart, unrelated to the masking


@pytest.mark.parametrize("distance, expected", [(7.99, True), (8.01, False)])
def test_threshold_boundary(distance, expected):
    """Pins the cutoff convention: strictly less than the threshold."""
    coords = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    cmap = build_contact_map(coords, np.ones(2, dtype=bool))
    assert bool(cmap[0, 1]) is expected