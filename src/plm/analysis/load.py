"""
load.py — reconstruct a trained model from a HuggingFace Hub checkpoint.

Checkpoints are stored as raw state dicts rather than in the `transformers`
repo format, so they cannot be loaded with `AutoModel.from_pretrained`. The
architecture has to be rebuilt from the YAML config first, then the weights
loaded into it. Keeping that in one place means the analysis pipeline and the
training scripts cannot drift apart on architecture arguments.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from plm.config import load_config
from plm.data.tokenizer import ProteinTokenizer
from plm.model.mlm import ProteinMLM
from plm.training.checkpoint import load_weights


def load_from_hub(
    config_path: Path,
    repo_id: str,
    filename: str,
    device: torch.device | str = "cpu",
) -> tuple[ProteinMLM, ProteinTokenizer, int]:
    """Download a checkpoint and return the model it contains.

    Downloads are cached by huggingface_hub, so repeated calls with the same
    arguments hit disk rather than the network. This matters for the emergence
    study, which loads the full retained checkpoint sequence.

    Dropout is set to 0.0 rather than taken from the config. Analysis always
    runs under eval(), where dropout is inert anyway, but hardcoding it here
    means a forgotten eval() cannot silently corrupt attention weights.

    Args:
        config_path: YAML config for the model size being loaded. Must match
            the checkpoint's architecture, or load_state_dict raises.
        repo_id: HuggingFace repo, e.g. 'dariasof/protein-transformer-20M'.
        filename: Checkpoint filename within that repo, including any
            subdirectory. Use HfApi().list_repo_files(repo_id) to see them.
        device: Where to place the model. Weights are always read to CPU
            first, then moved, so this works regardless of where the
            checkpoint was saved.

    Returns:
        (model, tokenizer, step) where step is the global training step at
        which the checkpoint was written. The tokenizer is returned alongside
        the model because attention extraction needs both, and they must be
        the same pair the model was trained with.
    """
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    config = load_config(config_path)
    tokenizer = ProteinTokenizer()

    model = ProteinMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        max_len=config.data.max_len,
        pad_id=tokenizer.pad_id,
        d_ff=config.model.d_ff,
        dropout=0.0,
    )

    step = load_weights(Path(checkpoint_path), model, map_location="cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, step