"""Model registry and release-card reports for Picochat runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegistryEntry:
    run: str
    path: str
    status: str
    gate_profile: str
    parameters: int | None
    planned_tokens: int | None
    tokens_per_parameter: float | None
    eval_pass_rate: float | None
    sft_fit_rate: float | None
    heldout_sft_fit_rate: float | None
    external_benchmark_count: int
    honesty_status: str
    preflight_status: str
    best_checkpoint: str | None
    resume_checkpoint: str | None
    tokenizer: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_run_dirs(runs_dir: str | Path) -> list[Path]:
    """Return immediate child run folders that contain summary.json."""
    root = Path(runs_dir)
    if not root.exists():
        raise FileNotFoundError(f"runs directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if (path / "summary.json").exists())


def build_model_registry(run_dirs: list[str | Path]) -> dict[str, Any]:
    """Build a registry report from run summary.json files."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    entries = [_entry_for_run(Path(path)) for path in run_dirs]
    status_counts: dict[str, int] = {}
    for entry in entries:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
    best = _best_entry(entries)
    return {
        "entries": [entry.to_dict() for entry in sorted(entries, key=_sort_key)],
        "status_counts": status_counts,
        "best_run": best.run if best else None,
        "best_eval_pass_rate": best.eval_pass_rate if best else None,
    }


def registry_table(registry: dict[str, Any]) -> str:
    headers = ["Run", "Status", "Gate", "Params", "Tok/Param", "Eval", "SFT", "Honesty"]
    rows = []
    for row in registry["entries"]:
        rows.append([
            row["run"],
            row["status"],
            row["gate_profile"],
            _fmt_int(row["parameters"]),
            _fmt_float(row["tokens_per_parameter"]),
            _fmt_percent(row["eval_pass_rate"]),
            _fmt_percent(row["sft_fit_rate"]),
            row["honesty_status"],
        ])
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def registry_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# Picochat Model Registry",
        "",
        f"Best run: `{registry.get('best_run') or '--'}`",
        "",
        "| Run | Status | Gate | Params | Planned tokens | Tok/param | Eval | SFT fit | Held-out SFT | External | Honesty | Checkpoint |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in registry["entries"]:
        checkpoint = row.get("best_checkpoint") or row.get("resume_checkpoint") or "--"
        lines.append(
            "| "
            f"`{row['run']}` | `{row['status']}` | `{row['gate_profile']}` | "
            f"{_fmt_int(row.get('parameters'))} | {_fmt_int(row.get('planned_tokens'))} | "
            f"{_fmt_float(row.get('tokens_per_parameter'))} | {_fmt_percent(row.get('eval_pass_rate'))} | "
            f"{_fmt_percent(row.get('sft_fit_rate'))} | {_fmt_percent(row.get('heldout_sft_fit_rate'))} | "
            f"{int(row.get('external_benchmark_count') or 0)} | `{row['honesty_status']}` | `{checkpoint}` |"
        )
    lines.extend([
        "",
        "A registry row is not a release claim. Treat `approved` as usable evidence only when the linked run also includes preflight, honesty, eval, external benchmark, and model-card artifacts.",
    ])
    return "\n".join(lines)


def release_card_markdown(run_dir: str | Path) -> str:
    """Render a single-run release card from summary.json."""
    entry = _entry_for_run(Path(run_dir))
    summary = _load_summary(Path(run_dir))
    gate = summary.get("long_run_gate") or {}
    issues = gate.get("issues") or []
    external = summary.get("external_evals") or []
    artifacts = summary.get("artifacts") or {}
    lines = [
        f"# Picochat Release Card: {entry.run}",
        "",
        f"- Status: `{entry.status}`",
        f"- Gate profile: `{entry.gate_profile}`",
        f"- Parameters: {_fmt_int(entry.parameters)}",
        f"- Planned tokens: {_fmt_int(entry.planned_tokens)}",
        f"- Tokens per parameter: {_fmt_float(entry.tokens_per_parameter)}",
        f"- Eval pass rate: {_fmt_percent(entry.eval_pass_rate)}",
        f"- SFT fit rate: {_fmt_percent(entry.sft_fit_rate)}",
        f"- Held-out SFT fit rate: {_fmt_percent(entry.heldout_sft_fit_rate)}",
        f"- Honesty status: `{entry.honesty_status}`",
        f"- Preflight status: `{entry.preflight_status}`",
        f"- Best checkpoint: `{entry.best_checkpoint or '--'}`",
        f"- Resume checkpoint: `{entry.resume_checkpoint or '--'}`",
        f"- Tokenizer: `{entry.tokenizer or '--'}`",
        "",
        "## Required Evidence",
        "",
        f"- Summary: `{entry.summary}`",
        f"- Preflight: `{artifacts.get('preflight_report') or '--'}`",
        f"- Honesty: `{artifacts.get('honesty_report') or '--'}`",
        f"- Eval: `{artifacts.get('eval_report') or '--'}`",
    ]
    if external:
        lines.append("- External benchmarks:")
        for item in external:
            eval_summary = item.get("summary") or {}
            lines.append(
                f"  - `{item.get('name', 'external')}`: "
                f"{_fmt_percent(eval_summary.get('choice_accuracy') or eval_summary.get('pass_rate'))}"
            )
    else:
        lines.append("- External benchmarks: `--`")
    if issues:
        lines.extend(["", "## Gate Issues", ""])
        for issue in issues:
            lines.append(
                f"- `{issue.get('severity', 'warn')}` `{issue.get('name', 'issue')}`: "
                f"{issue.get('message', '')}"
            )
    lines.extend([
        "",
        "## Release Rule",
        "",
        "Do not present this model as production-ready unless the gate status is `approved` and the evidence above is present.",
    ])
    return "\n".join(lines)


def write_registry_report(registry: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(registry_markdown(registry), encoding="utf-8")
    return path


def write_registry_json(registry: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


def write_release_card(run_dir: str | Path, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(release_card_markdown(run_dir), encoding="utf-8")
    return path


def _entry_for_run(run_dir: Path) -> RegistryEntry:
    summary = _load_summary(run_dir)
    preflight = summary.get("preflight") or {}
    budget = preflight.get("budget") or {}
    gate = summary.get("long_run_gate") or {}
    eval_summary = summary.get("eval") or {}
    status = str(gate.get("status") or "ungated")
    parameters = _optional_int(budget.get("estimated_parameters"))
    planned_tokens = _optional_int(budget.get("base_planned_tokens"))
    tokens_per_parameter = (
        planned_tokens / parameters
        if planned_tokens is not None and parameters not in {None, 0}
        else _optional_float(budget.get("target_param_data_ratio"))
    )
    return RegistryEntry(
        run=run_dir.name,
        path=str(run_dir),
        status=status,
        gate_profile=str(gate.get("profile") or "none"),
        parameters=parameters,
        planned_tokens=planned_tokens,
        tokens_per_parameter=tokens_per_parameter,
        eval_pass_rate=_optional_float(eval_summary.get("pass_rate")),
        sft_fit_rate=_optional_float(gate.get("sft_fit_rate") or (summary.get("sft_fit") or {}).get("pass_rate")),
        heldout_sft_fit_rate=_optional_float(gate.get("sft_heldout_fit_rate")),
        external_benchmark_count=len(summary.get("external_evals") or []),
        honesty_status=str((summary.get("honesty") or {}).get("status") or "missing"),
        preflight_status=str(preflight.get("status") or "missing"),
        best_checkpoint=_find_checkpoint(run_dir, "best_checkpoint"),
        resume_checkpoint=_find_checkpoint(run_dir, "resume_checkpoint"),
        tokenizer=str(run_dir / "tokenizer.json") if (run_dir / "tokenizer.json").exists() else None,
        summary=str(run_dir / "summary.json"),
    )


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing summary.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_checkpoint(run_dir: Path, name: str) -> str | None:
    for stage in ("sft", "base", "dpo"):
        path = run_dir / stage / name
        if (path / "model.pt").exists():
            return str(path)
    return None


def _best_entry(entries: list[RegistryEntry]) -> RegistryEntry | None:
    if not entries:
        return None
    return max(entries, key=lambda row: (_status_rank(row.status), row.eval_pass_rate or 0.0, row.sft_fit_rate or 0.0))


def _sort_key(row: RegistryEntry) -> tuple[int, str]:
    return (-_status_rank(row.status), row.run)


def _status_rank(status: str) -> int:
    return {
        "approved": 4,
        "warn": 3,
        "blocked": 2,
        "ungated": 1,
    }.get(status, 0)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_int(value: Any) -> str:
    parsed = _optional_int(value)
    return "--" if parsed is None else f"{parsed:,}"


def _fmt_float(value: Any) -> str:
    parsed = _optional_float(value)
    return "--" if parsed is None else f"{parsed:.2f}"


def _fmt_percent(value: Any) -> str:
    parsed = _optional_float(value)
    return "--" if parsed is None else f"{parsed * 100:.2f}%"
