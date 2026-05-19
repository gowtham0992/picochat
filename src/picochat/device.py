"""Device selection helpers for local Picochat runs."""

from __future__ import annotations

import torch


DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(requested: str = "cpu") -> torch.device:
    """Resolve a user-facing device name into an available torch device."""
    name = (requested or "cpu").strip().lower()
    if name not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of: {', '.join(DEVICE_CHOICES)}")

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if name == "mps" and not _mps_available():
        raise ValueError("MPS was requested but torch.backends.mps.is_available() is false")
    return torch.device(name)


def resolved_device_name(requested: str = "cpu") -> str:
    """Return the concrete torch device name for reports and UI labels."""
    return resolve_device(requested).type


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())
