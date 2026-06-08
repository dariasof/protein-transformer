# src/plm/training/checkpoint.py

"""
Checkpoint save and resume for the protein language model.

A complete checkpoint contains:
    - model state dict      (weights)
    - optimizer state dict  (momentum terms, adaptive learning rates)
    - scheduler state dict  (current step in the LR schedule)
    - global step number    (so resume knows where to continue)

The `save_checkpoint` function saves all of this to a file, and the `load_checkpoint` function restores it
"""

from __future__ import annotations
from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
) -> None:
    """
    Save a complete training checkpoint.

    Args:
        path:       File path to save to (e.g. checkpoints/resume.pt).
        model:      The model being trained.
        optimizer:  The optimizer.
        scheduler:  The LR scheduler.
        step:       Current global step number.
    """
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step":      step,
            "rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> int:
    """
    Load a checkpoint and restore all training state.

    Args:
        path:       Path to the checkpoint file.
        model:      Model instance to restore weights into.
        optimizer:  Optimizer instance to restore state into.
        scheduler:  Scheduler instance to restore state into.
    Returns:
        step: The global step number at the time of saving.
    """
    checkpoint = torch.load(path, weights_only=True)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["rng_state"])  
    return checkpoint["step"]