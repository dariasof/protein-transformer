import numpy as np

def build_contact_map(coords: np.ndarray, valid_mask: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Build a binary residue-residue contact map from backbone coordinates.

    A pair (i, j) is a contact if the two residues lie strictly within `threshold`
    Angstrom of each other and both have resolved coordinates.

    Convention note: TAPE/ProteinNet stores only Calpha coordinates, so this is a
    Calpha-Calpha map.

    Unresolved residues are the reason for the `valid_mask` argument. TAPE fills
    their coordinate rows with [0, 0, 0], and the origin is a real point in
    space -- a residue that happens to sit within `threshold` of it would
    otherwise register as being in contact with every unresolved residue. Such
    pairs are set False here.

    Args:
        coords: Float array of shape [L, 3], Calpha coordinates in Angstrom,
            with one row per sequence position.
        valid_mask: Boolean array of shape [L], True where the residue has
            resolved coordinates.
        threshold: Contact distance cutoff in Angstrom. Defaults to 8.0, the
            standard convention.

    Returns:
        Boolean array of shape [L, L], symmetric, True where (i, j) is a
        contact. The diagonal is True, since a residue is at zero distance from
        itself; short-range pairs are removed downstream by the sequence
        separation filter rather than here.
    """
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    contact_map = (distances < threshold) & valid_mask[:, None] & valid_mask[None, :]
    return contact_map

def build_eligibility_mask(valid_mask: np.ndarray, min_separation: int = 24) -> np.ndarray:
    """Build a boolean mask of residue pairs eligible for contact evaluation.

    A pair (i, j) is eligible if both residues have resolved coordinates and
    are separated by at least min_separation positions with j > i.

    Args:
        valid_mask: Boolean array of shape [L], True where the residue has
            resolved coordinates.
        min_separation: Minimum sequence separation for an eligible pair. Defaults
            to 24, the standard convention.

    Returns:
        Boolean array of shape [L, L], upper-triangular — entry (i, j) is True only for j > i.
        Contacts are symmetric, but each pair is represented once so that top-k selection over
        mask counts distinct pairs rather than returning each twice.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    L = len(valid_mask)
    idx = np.arange(L)

    valid_pairs = valid_mask[:, None] & valid_mask[None, :]
    separation = idx[None, :] - idx[:, None]      # entry (i, j) = j - i
    return valid_pairs & (separation >= min_separation)
   