"""Optional training telemetry integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorBoardLogger:
    """Small wrapper around torch.utils.tensorboard with an explicit opt-in."""

    def __init__(self, log_dir: str | Path | None) -> None:
        self.writer = None
        if not log_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "TensorBoard logging requires tensorboard. Install with: "
                "python -m pip install -e '.[monitor]'"
            ) from exc
        self.writer = SummaryWriter(str(log_dir))

    def scalar(self, tag: str, value: Any, step: int) -> None:
        if self.writer is None or value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        self.writer.add_scalar(tag, numeric, step)

    def scalars(self, metrics: dict[str, Any], step: int) -> None:
        for tag, value in metrics.items():
            self.scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
