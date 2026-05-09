"""Checkpoint save/load helpers for Picochat."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from picochat.model import GPTConfig, TinyGPT


def save_checkpoint(
    path: str | Path,
    model: TinyGPT,
    step: int,
    train_loss: float,
) -> None:
    """Save model weights and lightweight metadata."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    metadata = {
        "step": step,
        "train_loss": train_loss,
        "model_config": model.config.to_dict(),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[TinyGPT, dict]:
    """Load a TinyGPT model and metadata from a checkpoint directory."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    config = GPTConfig(**metadata["model_config"])
    model = TinyGPT(config)
    state = torch.load(path / "model.pt", map_location=map_location)
    model.load_state_dict(state)
    return model, metadata

