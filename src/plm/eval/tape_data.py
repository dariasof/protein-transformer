"""
tape_data.py — loaders for TAPE benchmark datasets stored in LMDB format.

TAPE (Tasks Assessing Protein Embeddings) provides several supervised benchmark
datasets for evaluating protein embeddings. This module provides lightweight
readers that return raw sequences and labels, with no model-specific logic.

Supported label fields per entry:
    fold_label          — SCOP fold class (int), used for remote homology eval
    superfamily_label   — SCOP superfamily class (int)
    family_label        — SCOP family class (int)
    secondary_structure — per-residue SS labels (list), used in Week 7 probes
    solvent_accessibility — per-residue accessibility (list), used in Week 7 probes

Note: entries are returned in LMDB iteration order. Callers must preserve this
order when aligning sequences to embeddings — do not shuffle between loading
and embedding.
"""

import lmdb, pickle

def load_tape_lmdb(path, label_field="fold_label"):
    """Load sequences and labels from a TAPE LMDB file.

    Args:
        path:        path to the .lmdb directory (e.g. 'data/remote_homology_train.lmdb')
        label_field: which label to extract per entry. Must be one of the fields
                     listed in the module docstring. Defaults to 'fold_label'.

    Returns:
        List of (sequence, label) tuples in iteration order.
        sequence is a raw amino acid string (e.g. 'MKTAYIAKQRQISFVK').
        label is an integer.

    Note:
        LMDB files from TAPE contain a metadata entry at key b'0' whose value is
        an integer count, not a protein record. This entry is silently skipped.
    """
    env = lmdb.open(str(path), readonly=True)
    records = []  # list of (sequence, label) — preserves ordering
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        while True:
            item = pickle.loads(cursor.value())
            if isinstance(item, dict):
                records.append((item["primary"], item[label_field]))
            if not cursor.next():
                break
    env.close()
    return records