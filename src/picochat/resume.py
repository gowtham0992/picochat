"""Resumable training state helpers."""

from __future__ import annotations

from typing import Any

import torch


def make_training_state(
    *,
    step: int,
    losses: list[dict],
    best_metric: float,
    best_checkpoint: dict | None,
    evals_without_improvement: int,
    stop_reason: str,
    elapsed_sec: float,
    optimizer,
    scaler,
    ema,
    batcher,
    device: torch.device,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture mutable training state needed to continue a run."""
    state = {
        "step": step,
        "losses": losses,
        "best_metric": best_metric,
        "best_checkpoint": best_checkpoint,
        "evals_without_improvement": evals_without_improvement,
        "stop_reason": stop_reason,
        "elapsed_sec": elapsed_sec,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
        "ema": ema.state_dict() if ema is not None else None,
        "batcher": batcher.state_dict(),
        "rng": capture_rng_state(device),
    }
    if extra_state:
        state.update(extra_state)
    return state


def restore_training_state(
    state: dict[str, Any],
    *,
    optimizer,
    scaler,
    ema,
    batcher,
) -> None:
    """Restore mutable optimizer, scaler, EMA, RNG, and batcher state."""
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scaler") and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(state["scaler"])
    if state.get("ema") is not None:
        if ema is None:
            raise ValueError("checkpoint contains EMA state but this run has EMA disabled")
        ema.load_state_dict(state["ema"])
    elif ema is not None:
        raise ValueError("checkpoint has no EMA state but this run has EMA enabled")
    batcher.load_state_dict(state["batcher"])
    restore_rng_state(state.get("rng", {}))


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
