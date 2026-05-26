"""Checkpoint save/load helpers for Picochat."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import uuid

import torch

from picochat.model import GPTConfig, TinyGPT


def save_checkpoint(
    path: str | Path,
    model: TinyGPT,
    step: int,
    train_loss: float,
    extra_metadata: dict | None = None,
    training_state: dict | None = None,
    model_state_dict: dict | None = None,
    model_config: GPTConfig | None = None,
) -> None:
    """Save model weights and lightweight metadata with crash-safe directory replacement."""
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    previous_path = _previous_checkpoint_path(path)
    tmp_path.mkdir(parents=True)
    metadata = {
        "step": step,
        "train_loss": train_loss,
        "model_config": (model_config or model.config).to_dict(),
    }
    try:
        state_dict = model_state_dict if model_state_dict is not None else model.state_dict()
        torch.save(state_dict, tmp_path / "model.pt")
        if training_state is not None:
            torch.save(training_state, tmp_path / "training_state.pt")
            metadata["has_training_state"] = True
        if extra_metadata:
            metadata.update(extra_metadata)
        (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if previous_path.exists():
            shutil.rmtree(previous_path)
        if path.exists():
            os.replace(path, previous_path)
        os.replace(tmp_path, path)
    except Exception:
        if not path.exists() and previous_path.exists():
            os.replace(previous_path, path)
        raise
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        if previous_path.exists():
            shutil.rmtree(previous_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[TinyGPT, dict]:
    """Load a TinyGPT model and metadata from a checkpoint directory."""
    path = _resolve_checkpoint_path(Path(path))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    config = GPTConfig(**metadata["model_config"])
    model = TinyGPT(config)
    state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    return model, metadata


def load_training_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load optimizer/EMA/RNG/dataloader state from a resumable checkpoint."""
    path = _resolve_checkpoint_path(Path(path))
    state_path = path / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"checkpoint has no training_state.pt: {Path(path)}")
    return torch.load(state_path, map_location=map_location, weights_only=True)


def _previous_checkpoint_path(path: Path) -> Path:
    return path.parent / f".{path.name}.previous"


def _resolve_checkpoint_path(path: Path) -> Path:
    if (path / "metadata.json").exists():
        return path
    previous_path = _previous_checkpoint_path(path)
    if (previous_path / "metadata.json").exists():
        return previous_path
    return path
