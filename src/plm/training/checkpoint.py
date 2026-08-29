
"""
Checkpoint save and resume for the protein language model.

A complete checkpoint contains:
    - model state dict      (weights)
    - optimizer state dict  (momentum terms, adaptive learning rates)
    - scheduler state dict  (current position in the LR schedule)
    - global step number    (so resume knows where to continue)
    - schedule horizon      (total_steps / warmup_steps the run was built for)
    - RNG state             (so data order and masking continue deterministically)

Two load paths are provided deliberately:

    load_weights      — evaluation and analysis. Model only. Constructing a
                        dummy optimizer just to satisfy a signature wastes
                        memory proportional to model size, which matters when
                        sweeping many checkpoints for the emergence study.

    load_for_resume   — continuing training. Restores everything, and refuses
                        to proceed if the schedule horizon has changed since
                        the checkpoint was written.
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
    total_steps: int,
    warmup_steps: int,
) -> None:
    """
    Save a complete training checkpoint.

    Args:
        path:         File path to save to (e.g. checkpoints/resume.pt).
        model:        The model being trained.
        optimizer:    The optimizer.
        scheduler:    The LR scheduler.
        step:         Current global step number.
        total_steps:  Schedule horizon this run was built for. Stored so that
                      resume can detect a changed horizon rather than silently
                      evaluating the restored scheduler against a new one.
        warmup_steps: Warmup length this run was built for.
    """
    torch.save(
        {
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "step":         step,
            "total_steps":  total_steps,
            "warmup_steps": warmup_steps,
            "rng_state":    torch.get_rng_state(),
        },
        path,
    )


def load_weights(path: Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> int:
    """
    Load model weights only — for evaluation and analysis.

    Args:
        path:  Path to the checkpoint file.
        model: Model instance to restore weights into.
        map_location: Device to load the model onto. Defaults to CPU.
    Returns:
        step: The global step number at the time of saving. Useful for
              labelling checkpoints in the emergence study.
    """
    checkpoint = torch.load(path, weights_only=True, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    return checkpoint["step"]


def load_for_resume(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    total_steps: int,
) -> int:
    """
    Load a checkpoint and restore all training state.

    Raises if the schedule horizon differs from the one the checkpoint was
    written under. The restored scheduler carries an internal step counter that
    is only meaningful relative to the total_steps it was constructed with;
    resuming against a different horizon silently distorts the LR curve for the
    remainder of training.

    Args:
        path:        Path to the checkpoint file.
        model:       Model instance to restore weights into.
        optimizer:   Optimizer instance to restore state into.
        scheduler:   Scheduler instance to restore state into.
        total_steps: Schedule horizon computed for the current run.
    Returns:
        step: The global step number at the time of saving.
    Raises:
        ValueError: If total_steps does not match the stored value.
    """
    checkpoint = torch.load(path, weights_only=True)

    saved_total_steps = checkpoint["total_steps"]
    if saved_total_steps != total_steps:
        raise ValueError(
            f"Schedule horizon mismatch on resume.\n"
            f"  checkpoint was written with total_steps={saved_total_steps}\n"
            f"  current run computed        total_steps={total_steps}\n"
            f"total_steps = n_epochs * len(train_loader), so the usual cause is "
            f"a changed batch size or a changed dataset/split between sessions. "
            f"Restore the original batch size and n_epochs, or start a fresh run."
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["rng_state"])
    return checkpoint["step"]
def save_weights(path: Path, model: torch.nn.Module, step: int) -> None:
    """
    Save model weights only for retained analysis checkpoints.

    Retained checkpoints are read exclusively by load_weights (emergence study,
    contact analysis, kNN probes), which never touches optimizer or scheduler
    state. Omitting that state cuts the file to roughly a third of its size:
    AdamW carries two moment tensors per parameter, so a full checkpoint is
    ~3x the parameter count in bytes against ~1x here.

    Note this file cannot be resumed from. Resume runs from resume.pt only.
    """
    torch.save({"model": model.state_dict(), "step": step}, path)