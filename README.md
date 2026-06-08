# Protein Transformer

A small protein language model trained from scratch, with attention pattern
analysis to study how structural information emerges across model scale and
training dynamics.

**Status:** Model and training loop done. First 1M model trained on 10K
SwissProt proteins — loss drops from 3.2 → 2.6, perplexity 24 → 13.
Scaling and homology-aware splits next.

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
│   ├── model/
│   │   ├── embeddings.py  # token + positional embeddings
│   │   ├── attention.py   # multi-head self-attention
│   │   ├── transformer.py # encoder block (attention + FFN + residuals)
│   │   └── mlm.py         # full MLM model + head
│   └── training/
│       ├── trainer.py     # training loop, W&B logging, checkpointing
│       └── checkpoint.py  # save/resume logic
├── scripts/
│   ├── build_dataset.py   # one-time data prep (download → tokenize → .pt)
│   ├── inspect_data.py    # sanity check: print masked batches
│   └── train.py           # training entry point
├── tests/
├── configs/               # YAML hyperparameter configs (Week 3)
└── data/                  # gitignored — raw FASTA, processed .pt, checkpoints
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/dariasof/protein-transformer.git
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

**Data filtering.** Sequences between 30 and 511 residues, standard amino
acids only (no B/J/O/U/X/Z). Keeps sequences clean for the 24-token vocab
and avoids polluting training with ambiguous residues.

**Architecture.** Encoder-only transformer, pre-norm convention (LayerNorm
before each sub-layer, not after). 4 layers, 4 heads, d_model=128 for the
1M parameter baseline. Bidirectional attention — no causal mask — because
MLM prediction benefits from full sequence context in both directions.

**Checkpointing.** Two checkpoint types: a rolling `resume.pt` saved every
500 steps (overwrites each time), and permanent named checkpoints every 1000
steps (`ckpt_step_XXXXXX.pt`). The named checkpoints are the raw material
for the training-dynamics emergence study in Week 10 — they cannot be
retrofitted later.


---

## Roadmap

|  | Focus | Status |
|------|-------|--------|
| 1 | Data pipeline: tokenizer, dataset, MLM collator | ✅ Done |
| 2 | Transformer architecture + training loop | ✅ Done |
| 3 | Config system, 100K proteins, homology-aware splits | ⏳ Next |
| 4 | Train 5M model | — |
| 5 | Train 20M model, apply for cluster access | — |
| 6–8 | Attention analysis pipeline, head atlas | — |
| 9–11 | Scaling study, training dynamics, ESM-2 comparison | — |
| 12–14 | Polish, writeup, HuggingFace model cards | — |

---

## License

MIT — see [LICENSE](LICENSE).