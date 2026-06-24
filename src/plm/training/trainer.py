
"""
Training loop.

Handles:
    - forward pass + loss computation
    - gradient clipping
    - optimizer and scheduler steps
    - W&B logging (loss, perplexity, learning rate, gradient norm)
    - periodic checkpointing

Designed for Kaggle's 12-hour session limit — checkpoints every N steps
so a killed session can resume without losing progress.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb

from plm.model.mlm import ProteinMLM
from plm.training.checkpoint import save_checkpoint, load_checkpoint


def get_lr_schedule(
    optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay.

    For the first warmup_steps, lr increases linearly from 0 to target.
    After that, lr follows a cosine curve down to 0.

    Args:
        optimizer:     The optimizer to schedule.
        warmup_steps:  Number of warmup steps.
        total_steps:   Total training steps.
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        # warmup phase — linear increase from 0 to 1
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # cosine decay phase — from 1 to 0
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train(
    model: ProteinMLM,
    train_loader: DataLoader,
    *,
    precision: str = "fp32",
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    warmup_ratio: float = 0.05,
    max_grad_norm: float = 1.0,
    checkpoint_dir: Path,
    checkpoint_every: int = 500,
    retain_every: int = 1000,
    device: str = "cuda",
    wandb_project: str = "protein-mlm",
    resume_from: Path | None = None,
) -> None:
    """
    Full training loop with checkpointing and W&B logging.

    Args:
        model:            ProteinMLM instance (uninitialised weights).
        train_loader:     DataLoader yielding collated batches.
        n_epochs:         Number of passes over the training data.
        learning_rate:    Peak learning rate after warmup.
        warmup_ratio:     Fraction of total steps used for warmup.
        max_grad_norm:    Gradient clipping threshold.
        checkpoint_dir:   Directory to save checkpoints.
        checkpoint_every: Save resume checkpoint every N steps.
        retain_every:     Save a permanent named checkpoint every N steps.
                          These are the checkpoints used in the emergence
                          study, never overwritten.
        device:           'cuda' or 'cpu'.
        wandb_project:    W&B project name.
        resume_from:      Path to checkpoint to resume from, or None.
    """
    model = model.to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer 
    # AdamW: Adam with decoupled weight decay.
    # Weight decay acts as L2 regularization on the weights
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    # Total steps and schedule 
    total_steps   = n_epochs * len(train_loader)  # n_epochs*number of batches per epoch
    warmup_steps  = int(warmup_ratio * total_steps)
    scheduler     = get_lr_schedule(optimizer, warmup_steps, total_steps)

    #  Resume from checkpoint if provided 
    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"Resumed from step {start_step}")

    #  W&B 
    wandb.init(
        project=wandb_project,
        config={
            "n_params":       model.count_parameters(),
            "d_model":        model.embeddings.token_emb.embedding_dim,
            "n_layers":       len(model.blocks),
            "learning_rate":  learning_rate,
            "warmup_steps":   warmup_steps,
            "total_steps":    total_steps,
            "n_epochs":       n_epochs,
        },
        resume="allow",
    )

    #  Training loop 
    model.train()
    global_step = start_step

    for epoch in range(n_epochs):
        for batch in train_loader:

            # skip steps already completed before resume
            if global_step < start_step:
                global_step += 1
                continue

            # move batch to device
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            # forward pass
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(precision == "fp16")):
                output = model(input_ids, labels=labels)
                loss   = output["loss"]
                
            #  backward pass 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            #  gradient clipping 
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            #  optimizer + scheduler step  
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

          
            
           

            #  logging 
            if global_step % 10 == 0:
                perplexity = math.exp(loss.item())
                current_lr = scheduler.get_last_lr()[0]

                wandb.log({
                    "train/loss":       loss.item(),
                    "train/perplexity": perplexity,
                    "train/lr":         current_lr,
                    "train/grad_norm":  grad_norm.item(),
                    "step":             global_step,
                })

                print(
                    f"step {global_step:6d} | "
                    f"loss {loss.item():.4f} | "
                    f"ppl {perplexity:.2f} | "
                    f"lr {current_lr:.2e} | "
                    f"grad_norm {grad_norm.item():.3f}"
                )

            #  checkpointing 
            if global_step % checkpoint_every == 0 and global_step > 0:
                save_checkpoint(
                    path=checkpoint_dir / "resume.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                )

            # retain periodic checkpoints 
            # These are never overwritten and used in the emergence study
            if global_step % retain_every == 0 and global_step > 0:
                save_checkpoint(
                    path=checkpoint_dir / f"ckpt_step_{global_step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                )

            global_step += 1

    #  Final checkpoint 
    save_checkpoint(
        path=checkpoint_dir / f"ckpt_step_{global_step:06d}.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
    )

    wandb.finish()
    print(f"Training complete. Final step: {global_step}")