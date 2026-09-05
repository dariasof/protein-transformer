
"""
aa_score.py — per-head same-amino-acid-type scoring.
 
Asks, for each (layer, head): does it attend preferentially to residues sharing
the query's amino acid type? 
 
 
1. Self-attention. Position i trivially matches itself, so j == i must go or
   every head scores high.
2. Local autocorrelation. 
   A pure local-offset head therefore scores above chance on same-AA with no
   amino-acid preference at all. Excluding |i - j| < min_sep separates them.
   
3. Composition. Chance is not 1/L. It is the fraction of *eligible* keys that
   share the query's type, which depends on the protein and varies hugely by
   residue (Leu ~10%, Trp ~1%). So the baseline is per query, not per protein.
 
Because the score is a ratio taken inside the eligible set, the [CLS] sink
cancels: it is absent from numerator and denominator alike.
"""
 
from __future__ import annotations
 
import numpy as np
 
from plm.analysis.label_score import categorical_lift
 
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
 
 
def encode_sequence(sequence: str) -> np.ndarray:
    """Map residues to codes 0-19; anything else becomes -1.
 
    A -1 never matches.
    """
    lookup = {aa: i for i, aa in enumerate(STANDARD_AA)}
    return np.array([lookup.get(c, -1) for c in sequence], dtype=np.int64)
 
 
def same_aa_lift(attn, sequence, min_sep=1, min_eligible_mass=1e-3):
    """Same-AA attention lift per head, for one protein.
 
    Thin wrapper over label_score.categorical_lift; see that module for the
    handling of self-attention, locality and composition.
 
    Returns:
        micro, macro, per_aa [.., 20], dropped.
    """
    if len(sequence) != attn.shape[-1]:
        raise ValueError(
            f"sequence length {len(sequence)} does not match attention size "
            f"{attn.shape[-1]}"
        )
    return categorical_lift(
        attn, encode_sequence(sequence), min_sep=min_sep,
        min_eligible_mass=min_eligible_mass, n_classes=len(STANDARD_AA),
    )