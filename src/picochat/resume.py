"""Resumable training state helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    training_fingerprint: dict[str, Any] | None = None,
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
    if training_fingerprint is not None:
        state["training_fingerprint"] = training_fingerprint
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


def make_training_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a stable digest for the data/tokenizer/model identity of a run."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "version": 1,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def validate_training_fingerprint(
    state: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Reject resume attempts against a different dataset/tokenizer/model setup."""
    observed = state.get("training_fingerprint")
    if observed is None:
        return
    if observed.get("sha256") != expected.get("sha256"):
        raise ValueError("resume checkpoint fingerprint does not match this run")


def file_sha256(path: str | Path | None) -> str | None:
    """Hash a file without loading it all into memory."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
