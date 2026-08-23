
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
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb

from plm.model.mlm import ProteinMLM
from plm.training.checkpoint import save_checkpoint, load_for_resume, save_weights

# bf16 has the same exponent range as fp32, so it cannot overflow the way fp16
# can and therefore needs no loss scaling. fp16 has a much narrower range and
# does need a GradScaler to keep small gradients from flushing to zero.
_AUTOCAST_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def _upload_checkpoint(path: Path, repo_id: str | None) -> None:
    """
    Push a retained checkpoint to the HuggingFace Hub.

    Synchronous by design. A background thread would be killed mid-flight when
    Kaggle terminates the session — which is precisely the scenario this exists
    to protect against — and would need a CPU copy of the weights to avoid
    reading state the training loop is still mutating.

    Failures are logged, not raised: the upload is a backup, and a transient
    network error should not end a multi-hour run.
    """
    if repo_id is None:
        return
    try:
        HfApi().upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
        )
    except Exception as e:
        print(f"WARNING: checkpoint upload failed for {path.name}: {e}")

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
    run_id: str | None = None,
    resume_from: Path | None = None,
    hf_repo_id: str | None = None,
) -> None:
    """
    Full training loop with checkpointing and W&B logging.

    The loop is step-driven, not epoch-driven: it runs until global_step
    reaches total_steps, pulling batches from a DataLoader iterator that is
    rebuilt (and so reshuffled) whenever it is exhausted. This makes resume
    trivial — position is tracked by global_step alone — and makes it
    structurally impossible for global_step to exceed the horizon the LR
    schedule was constructed for.

    Args:
        model:            ProteinMLM instance.
        train_loader:     DataLoader yielding collated batches.
        precision:        'fp32', 'fp16' or 'bf16'. bf16 is preferred where
                          supported; it needs no loss scaling.
        total_steps:      Schedule horizon and stopping condition. Computed by
                          the caller as n_epochs * len(train_loader); epochs
                          live in the config because matched epochs over the
                          same split give matched token budgets across model
                          sizes regardless of batch size.
        learning_rate:    Peak learning rate after warmup.
        warmup_ratio:     Fraction of total_steps used for warmup.
        max_grad_norm:    Gradient clipping threshold.
        checkpoint_dir:   Directory to save checkpoints.
        checkpoint_every: Overwrite the resume checkpoint every N steps.
        retain_every:     Save a permanent named checkpoint every N steps.
                          Never overwritten — these are the raw material for
                          the emergence study and cannot be reconstructed
                          after the fact.
        device:           'cuda' or 'cpu'.
        wandb_project:    W&B project name.
        run_id:           Stable W&B run id. Pass the same value across
                          sessions so a resumed run continues one curve
                          instead of starting a new one.
        resume_from:      Path to checkpoint to resume from, or None.
    """
    model = model.to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"Unknown precision {precision!r}; expected fp32, fp16 or bf16.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("precision='bf16' requested but this GPU does not support it.")

    autocast_dtype = _AUTOCAST_DTYPES.get(precision)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps)

    start_step = 0
    if resume_from is not None:
        start_step = load_for_resume(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
        )
        print(f"Resumed from step {start_step}")

   
    # resume="allow" only reattaches to an existing run when an explicit id is
    # given; without one W&B mints a fresh run each session and the training
    # curve arrives in disconnected segments.
    wandb.init(
        project=wandb_project,
        id=run_id,
        name=run_id,
        config={ "n_params":      model.count_parameters(),
            "d_model":       model.embeddings.token_emb.embedding_dim,
            "n_layers":      len(model.blocks),
            "learning_rate": learning_rate,
            "warmup_steps":  warmup_steps,
            "total_steps":   total_steps,},
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
            dtype=autocast_dtype or torch.float32,
            enabled=(autocast_dtype is not None),
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
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

        
        if global_step % retain_every == 0:
            ckpt_path = checkpoint_dir / f"ckpt_step_{global_step:06d}.pt"
            save_weights(path=ckpt_path, model=model, step=global_step)
            _upload_checkpoint(ckpt_path, hf_repo_id)

    save_checkpoint(
        path=checkpoint_dir / f"ckpt_step_{global_step:06d}.pt",
        model=model, optimizer=optimizer,
        scheduler=scheduler, step=global_step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    wandb.finish()
    print(f"Training complete. Final step: {global_step}")