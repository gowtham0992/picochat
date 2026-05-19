"""Small human-readable progress reports for resumable checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_checkpoint_progress(
    checkpoint_dir: str | Path,
    *,
    stage: str,
    step: int,
    max_steps: int,
    train_loss: float,
    losses: list[dict[str, Any]],
    best_checkpoint: dict[str, Any] | None,
    stop_reason: str,
    resume_from: str | None,
) -> tuple[str, str]:
    """Write JSON/Markdown progress next to a resumable checkpoint."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    latest = dict(losses[-1]) if losses else {}
    payload = {
        "status": "in_progress" if step < max_steps and stop_reason == "max_steps" else stop_reason,
        "stage": stage,
        "step": step,
        "max_steps": max_steps,
        "progress": step / max(1, max_steps),
        "train_loss": train_loss,
        "latest_eval": latest,
        "best_checkpoint": best_checkpoint,
        "resume_checkpoint": str(path),
        "resume_from": resume_from,
        "note": (
            "This file is written beside resume_checkpoint so interrupted runs can be "
            "inspected, moved, and resumed before final reports exist."
        ),
    }
    json_path = path / "progress.json"
    markdown_path = path / "progress.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_progress_markdown(payload), encoding="utf-8")
    return str(json_path), str(markdown_path)


def _progress_markdown(payload: dict[str, Any]) -> str:
    latest = payload.get("latest_eval") or {}
    best = payload.get("best_checkpoint") or {}
    lines = [
        f"# Picochat {payload.get('stage', 'training')} Resume Progress",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Step: {payload.get('step')} / {payload.get('max_steps')}",
        f"- Progress: {float(payload.get('progress') or 0.0) * 100:.2f}%",
        f"- Train loss: {_format_optional(payload.get('train_loss'))}",
        f"- Latest val loss: {_format_optional(latest.get('val_loss'))}",
        f"- Latest val BPB: {_format_optional(latest.get('val_bpb'))}",
        f"- Best checkpoint: `{best.get('path') or '--'}`",
        f"- Best step: {_format_optional(best.get('step'))}",
        f"- Best val loss: {_format_optional(best.get('val_loss'))}",
        f"- Best val BPB: {_format_optional(best.get('val_bpb'))}",
        f"- Resume checkpoint: `{payload.get('resume_checkpoint')}`",
    ]
    if payload.get("resume_from"):
        lines.append(f"- Resumed from: `{payload.get('resume_from')}`")
    lines.extend([
        "",
        str(payload.get("note") or ""),
        "",
    ])
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
