"""Small optimizer helpers for transparent local training runs."""

from __future__ import annotations

import math

import torch


LR_DECAYS = ("none", "linear", "cosine")


def validate_optim_controls(
    *,
    max_steps: int,
    lr_warmup_steps: int,
    lr_decay: str,
    min_lr_ratio: float,
    grad_clip: float,
) -> None:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    if lr_decay not in LR_DECAYS:
        raise ValueError(f"lr_decay must be one of: {', '.join(LR_DECAYS)}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")


def learning_rate_for_step(
    *,
    base_learning_rate: float,
    step: int,
    max_steps: int,
    warmup_steps: int = 0,
    decay: str = "none",
    min_lr_ratio: float = 1.0,
) -> float:
    """Return the learning rate for a 1-indexed training step."""
    if step < 1:
        raise ValueError("step must be 1-indexed")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    if decay == "none":
        return base_learning_rate

    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    if decay == "linear":
        multiplier = 1.0 - progress
    elif decay == "cosine":
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported lr decay: {decay}")
    multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * multiplier
    return base_learning_rate * multiplier


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def maybe_clip_grad_norm(model: torch.nn.Module, grad_clip: float) -> float | None:
    if grad_clip <= 0:
        return None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    return float(norm.item())
