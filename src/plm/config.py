"""Configuration schema and loader for the protein transformer.

The YAML files in ``configs/`` hold the *values*; this module holds the
*schema* (typed dataclasses) and the loader that reads a YAML file into a
validated ``Config``. Derived values (max_length, keep_token_prob) are
computed here, never stored in the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class MaskingConfig:
    mask_prob: float          # of all tokens, fraction selected for masking
    mask_token_prob: float    # of selected, fraction replaced with [MASK]
    random_token_prob: float  # of selected, fraction replaced with a random token

    def __post_init__(self) -> None:
        # The two replacement fractions must leave a non-negative remainder
        if self.mask_token_prob + self.random_token_prob > 1.0 + 1e-9:
            raise ValueError(
                "mask_token_prob + random_token_prob must be <= 1 "
                f"(got {self.mask_token_prob} + {self.random_token_prob})"
            )

    @property
    def keep_token_prob(self) -> float:
        """Fraction of selected tokens left unchanged — the derived remainder."""
        return 1.0 - self.mask_token_prob - self.random_token_prob


@dataclass
class ModelConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int

    def __post_init__(self) -> None:
        # Attention splits d_model across n_heads; head_dim must be an integer
        # or the per-head reshape crashes at the first forward pass.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    n_epochs: int
    warmup_ratio: float
    precision: str
    checkpoint_every: int
    retain_every: int
    wandb_project: str

    def __post_init__(self) -> None:
        allowed = {"fp16", "bf16", "fp32"}
        if self.precision not in allowed:
            raise ValueError(
                f"precision must be one of {sorted(allowed)} "
                f"(got {self.precision!r})"
            )


@dataclass
class DataConfig:
    n_proteins: int
    min_length: int
    max_len: int
    masking: MaskingConfig

    @property
    def max_length(self) -> int:
        """Per-sequence token budget after reserving one slot for [CLS].

        ``max_len`` is the total budget; the dataloader truncates raw sequences
        to this. Single source of truth: ``max_len``.
        """
        return self.max_len - 1


@dataclass
class Config:
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    """Read a YAML config file into a validated ``Config``.
    

    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Inner-first: the masking dict becomes a MaskingConfig before DataConfig
    # receives it.
    masking = MaskingConfig(**raw["data"]["masking"])
    data = DataConfig(
        n_proteins=raw["data"]["n_proteins"],
        min_length=raw["data"]["min_length"],
        max_len=raw["data"]["max_len"],
        masking=masking,
    )

    return Config(
        seed=raw["seed"],
        model=ModelConfig(**raw["model"]),
        data=data,
        training=TrainingConfig(**raw["training"]),
    )