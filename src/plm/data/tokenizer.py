"""
Protein tokenizer.

Maps protein sequences (strings of amino acid characters) to integer token IDs
and back. Character-level tokenization following ESM convention.

Vocabulary (24 tokens):
    0  [PAD]   padding
    1  [UNK]   unknown / non-standard residue
    2  [CLS]   prepended to every sequence; used for sequence-level embeddings
    3  [MASK]  replaces residues during MLM training
    4-23       the 20 standard amino acids, alphabetical
"""
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[MASK]']
class ProteinTokenizer: 
    """Character-level tokenizer for protein sequences.""" 
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}

        # Build token-to-ID mapping
        for t in SPECIAL_TOKENS:
            self.token_to_id[t] = len(self.token_to_id)
        for aa in AMINO_ACIDS:
            self.token_to_id[aa] = len(self.token_to_id)
        # Inverse mapping
        self.id_to_token :dict[int, str] = {id: token for token, id in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["[UNK]"]

    @property
    def cls_id(self) -> int:
        return self.token_to_id["[CLS]"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["[MASK]"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)   

    def encode(self, sequence: str, add_cls: bool = True) -> list[int]:
        """
        Convert a protein sequence string to a list of token IDs.

        Non-standard amino acids (B, J, O, U, X, Z, etc.) are mapped to [UNK].
        """
        sequence = sequence.upper()
        ids: list[int] = []
        if add_cls:
            ids.append(self.cls_id)
        for char in sequence:
            ids.append(self.token_to_id.get(char, self.unk_id))
        return ids    

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        """
        Convert a list of token IDs back to a protein sequence string.

        If skip_special=True, special tokens ([PAD], [CLS], etc.) are dropped
        from the output. If False, they are kept.
        """
        tokens: list[str] = []
        for i in ids:
            tok = self.id_to_token[i]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)    