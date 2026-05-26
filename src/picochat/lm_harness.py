"""EleutherAI lm-eval-harness command bridge."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class LMEvalHarnessConfig:
    model_path: str
    tasks: tuple[str, ...]
    out_dir: str
    device: str = "cpu"
    batch_size: str = "auto"
    limit: str | None = None
    num_fewshot: int | None = None
    model: str = "hf"
    trust_remote_code: bool = True
    extra_model_args: tuple[str, ...] = ()
    dry_run: bool = False


def build_lm_eval_command(config: LMEvalHarnessConfig) -> list[str]:
    """Build a reproducible lm-eval-harness command."""
    if not config.tasks:
        raise ValueError("at least one lm-eval task is required")
    model_args = [f"pretrained={config.model_path}"]
    if config.trust_remote_code:
        model_args.append("trust_remote_code=True")
    model_args.extend(config.extra_model_args)
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        config.model,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(config.tasks),
        "--device",
        config.device,
        "--batch_size",
        config.batch_size,
        "--output_path",
        str(Path(config.out_dir)),
    ]
    if config.limit:
        command.extend(["--limit", str(config.limit)])
    if config.num_fewshot is not None:
        command.extend(["--num_fewshot", str(config.num_fewshot)])
    return command


def run_lm_eval_harness(config: LMEvalHarnessConfig) -> dict[str, Any]:
    """Run or print an lm-eval-harness command."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_lm_eval_command(config)
    report = {
        "command": command,
        "command_text": shell_join(command),
        "dry_run": config.dry_run,
        "out_dir": str(out_dir),
        "tasks": list(config.tasks),
        "model_path": config.model_path,
    }
    (out_dir / "lm_eval_command.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if config.dry_run:
        return report
    if importlib.util.find_spec("lm_eval") is None:
        raise RuntimeError(
            "lm-eval-harness is not installed. Install with `pip install lm-eval` "
            "or rerun with --dry-run to write the command only."
        )
    completed = subprocess.run(command, check=False)
    report["returncode"] = completed.returncode
    if completed.returncode != 0:
        raise RuntimeError(f"lm-eval-harness failed with exit code {completed.returncode}")
    return report


def parse_lm_eval_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tasks:
        raise ValueError("tasks cannot be empty")
    return tasks


def shell_join(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
