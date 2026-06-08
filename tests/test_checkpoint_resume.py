
"""
Tests that save + load round-trips correctly and that resume
produces a continuous loss curve.
"""

import torch
import pytest
from pathlib import Path
from torch.optim import AdamW


from plm.model.mlm import ProteinMLM
from plm.training.trainer import get_lr_schedule
from plm.training.checkpoint import save_checkpoint, load_checkpoint


VOCAB    = 24
D_MODEL  = 64    # small for test speed
N_HEADS  = 4
N_LAYERS = 2
B, L     = 2, 16


@pytest.fixture
def model():
    return ProteinMLM(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )


def test_checkpoint_round_trip(model, tmp_path):
    """Save and load — model weights must be identical."""
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scheduler = get_lr_schedule(optimizer, warmup_steps=10, total_steps=100)

    # take a step so optimizer has non-trivial state
    ids    = torch.randint(1, VOCAB, (B, L))
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[:, 5] = ids[:, 5]
    out = model(ids, labels=labels)
    out["loss"].backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # save
    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, step=1)

    # create fresh model and load
    model_2   = ProteinMLM(
        vocab_size=VOCAB, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS
    )
    optimizer_2 = AdamW(model_2.parameters(), lr=3e-4)
    scheduler_2 = get_lr_schedule(optimizer_2, warmup_steps=10, total_steps=100)

    step = load_checkpoint(ckpt_path, model_2, optimizer_2, scheduler_2)

    assert step == 1

    # weights must match exactly
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model_2.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"


def test_resume_continuous_loss(model, tmp_path):
    """
    Loss at step N+1 after resume must equal loss at step N+1
    without resume — same weights, same batch, same result.
    """
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scheduler = get_lr_schedule(optimizer, warmup_steps=10, total_steps=100)

    # fix a batch
    torch.manual_seed(42)
    ids    = torch.randint(1, VOCAB, (B, L))
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[:, 3] = ids[:, 3]
    labels[:, 7] = ids[:, 7]

    # step 1
    out_1 = model(ids, labels=labels)
    out_1["loss"].backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # save after step 1
    ckpt_path = tmp_path / "resume.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, step=1)

    # step 2 without resume — record loss
    out_2 = model(ids, labels=labels)
    loss_without_resume = out_2["loss"].item()

    # now reload from checkpoint and take step 2 again
    model_r     = ProteinMLM(
        vocab_size=VOCAB, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS
    )
    optimizer_r = AdamW(model_r.parameters(), lr=3e-4)
    scheduler_r = get_lr_schedule(optimizer_r, warmup_steps=10, total_steps=100)
    load_checkpoint(ckpt_path, model_r, optimizer_r, scheduler_r)

    out_r = model_r(ids, labels=labels)
    loss_with_resume = out_r["loss"].item()

    assert abs(loss_without_resume - loss_with_resume) < 1e-5, (
        f"Loss discontinuity on resume: "
        f"{loss_without_resume:.6f} vs {loss_with_resume:.6f}"
    )