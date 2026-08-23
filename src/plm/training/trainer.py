
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
    total_steps: int,                   
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
    model = model.to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps)

    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"Resumed from step {start_step}")

    wandb.init(
        project=wandb_project,
        config={
            "n_params":      model.count_parameters(),
            "d_model":       model.embeddings.token_emb.embedding_dim,
            "n_layers":      len(model.blocks),
            "learning_rate": learning_rate,
            "warmup_steps":  warmup_steps,
            "total_steps":   total_steps,
        },
        resume="allow",
    )
    # Tell W&B to use our global_step as the x-axis for every train/* metric,
    
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")

    model.train()
    global_step = start_step

    # Step-driven loop. The DataLoader is treated as a refillable source of
    # batches rather than as an epoch boundary: when it is exhausted we build a
    # fresh iterator, which reshuffles. Resume needs no batch/epoch bookkeeping
    # because position is tracked solely by global_step.
    data_iter = iter(train_loader)

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(precision == "fp16"),
        ):
            output = model(input_ids, labels=labels)
            loss   = output["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1

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

        if global_step % checkpoint_every == 0:
            save_checkpoint(
                path=checkpoint_dir / "resume.pt",
                model=model, optimizer=optimizer,
                scheduler=scheduler, step=global_step,
            )

        
        if global_step % retain_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"ckpt_step_{global_step:06d}.pt",
                model=model, optimizer=optimizer,
                scheduler=scheduler, step=global_step,
            )

    save_checkpoint(
        path=checkpoint_dir / f"ckpt_step_{global_step:06d}.pt",
        model=model, optimizer=optimizer,
        scheduler=scheduler, step=global_step,
    )
    wandb.finish()
    print(f"Training complete. Final step: {global_step}")