# Protein Transformer

A small protein language model trained from scratch, with attention pattern
analysis to study how structural information emerges across model scale and
training dynamics.

**Status:** Data pipeline done. Model and training loop
coming.

---

## What this project is

This project trains an encoder-only transformer on protein sequences using
masked language modeling (MLM), then analyzes what the model learned by
examining its attention patterns. The central research question: *at what
scale and at what point in training does structural information emerge in
attention heads?*

---

## Project structure

```
protein-transformer/
├── src/plm/               # the plm package
│   ├── data/
│   │   ├── tokenizer.py   # character-level tokenizer, 24-token vocab
│   │   ├── dataset.py     # SwissProt dataset loader
│   │   └── collator.py    # MLM collator, 80/10/10 masking
│   ├── model/             # transformer architecture
│   └── training/          # training loop + checkpointing 
├── scripts/
│   ├── build_dataset.py   # one-time data prep (download → tokenize → .pt)
│   └── inspect_data.py    
├── tests/                 
├── configs/               # YAML hyperparameter configs 
└── data/                  # gitignored — raw FASTA, processed .pt, checkpoints
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/protein-transformer.git
cd protein-transformer
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate     # macOS / Linux
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 2. Download and prepare the data

Downloads SwissProt (~90 MB), filters to 10K proteins of length ≤ 512,
tokenizes, and saves to `data/processed/swissprot_10k.pt`:

```bash
python scripts/build_dataset.py
```


## Design decisions

**Tokenization.** Character-level, one token per amino acid. Vocabulary of 24
tokens: 20 standard amino acids + `[PAD]`, `[UNK]`, `[CLS]`, `[MASK]`.
`[PAD]` is id 0 so PyTorch's default padding behavior works without
configuration.

**MLM objective.** 15% of amino acid positions selected per sequence.
Of those, 80% replaced with `[MASK]`, 10% replaced with a random amino acid,
10% left unchanged. The 80/10/10 split prevents the model from only building
good representations at `[MASK]` positions — it must represent all tokens
well, which is what makes the embeddings useful for downstream tasks.

**Data filtering.** Sequences between 30 and 512 residues, standard amino
acids only (no B/J/O/U/X/Z). Keeps sequences clean for the 24-token vocab
and avoids polluting training with ambiguous residues.


---

## Roadmap

|  | Focus | Status |
|------|-------|--------|
| 1 | Data pipeline: tokenizer, dataset, MLM collator | ✅ Done |
| 2 | Transformer architecture + training loop | ⏳ Next |
| 3 | Config system, 100K proteins, homology-aware splits | — |
| 4 | Train 5M model | — |
| 5 | Train 20M model, apply for cluster access | — |
| 6–8 | Attention analysis pipeline, head atlas | — |
| 9–11 | Scaling study, training dynamics, ESM-2 comparison | — |
| 12–14 | Polish, writeup, HuggingFace model cards | — |

---

## License

MIT — see [LICENSE](LICENSE).