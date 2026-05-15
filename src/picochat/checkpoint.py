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
    extra_metadata: dict | None = None,
    training_state: dict | None = None,
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
    if training_state is not None:
        torch.save(training_state, path / "training_state.pt")
        metadata["has_training_state"] = True
    if extra_metadata:
        metadata.update(extra_metadata)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[TinyGPT, dict]:
    """Load a TinyGPT model and metadata from a checkpoint directory."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    config = GPTConfig(**metadata["model_config"])
    model = TinyGPT(config)
    state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    return model, metadata


def load_training_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load optimizer/EMA/RNG/dataloader state from a resumable checkpoint."""
    state_path = Path(path) / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"checkpoint has no training_state.pt: {Path(path)}")
    return torch.load(state_path, map_location=map_location, weights_only=False)
