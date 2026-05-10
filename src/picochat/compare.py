"""Compare completed Picochat experiment runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CompareRow:
    run: str
    path: str
    tokenizer_type: str
    eval_score: str
    pass_rate: float
    support_match_rate: float | None
    prompt_echo_rate: float | None
    base_val_loss: float
    sft_val_loss: float
    base_val_bpb: float | None
    sft_val_bpb: float | None
    base_best_step: int | None
    sft_best_step: int | None
    base_stop_reason: str
    sft_stop_reason: str
    base_loss_status: str
    sft_loss_status: str
    memorization_status: str
    num_parameters: int
    context_size: int
    truncated_examples: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_compare_row(run_dir: str | Path) -> CompareRow:
    """Load the comparison fields from a run summary."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing run summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_summary = summary["eval"]
    config = summary.get("config", {})
    tokenizer = summary.get("tokenizer", {})
    base = summary.get("base", {})
    sft = summary.get("sft", {})
    return CompareRow(
        run=run_dir.name,
        path=str(run_dir),
        tokenizer_type=str(
            tokenizer.get("tokenizer_type")
            or config.get("tokenizer_type")
            or "unknown"
        ),
        eval_score=f"{eval_summary['num_passed']}/{eval_summary['num_examples']}",
        pass_rate=float(eval_summary["pass_rate"]),
        support_match_rate=_optional_float(eval_summary.get("support_match_rate")),
        prompt_echo_rate=_optional_float(eval_summary.get("prompt_echo_rate")),
        base_val_loss=float(base["final_val_loss"]),
        sft_val_loss=float(sft["final_val_loss"]),
        base_val_bpb=_optional_float(base.get("final_val_bpb")),
        sft_val_bpb=_optional_float(sft.get("final_val_bpb")),
        base_best_step=_best_step(base),
        sft_best_step=_best_step(sft),
        base_stop_reason=str(base.get("stop_reason") or "unknown"),
        sft_stop_reason=str(sft.get("stop_reason") or "unknown"),
        base_loss_status=str(base.get("loss_diagnostics", {}).get("status") or "unknown"),
        sft_loss_status=str(sft.get("loss_diagnostics", {}).get("status") or "unknown"),
        memorization_status=str(base.get("memorization", {}).get("status") or "unknown"),
        num_parameters=int(base["num_parameters"]),
        context_size=int(config["context_size"]),
        truncated_examples=int(sft["truncated_examples"]),
    )


def compare_runs(run_dirs: list[str | Path]) -> dict:
    """Return comparison data for one or more run directories."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    rows = [load_compare_row(path) for path in run_dirs]
    best = max(rows, key=lambda row: (row.pass_rate, -row.sft_val_loss))
    best_base_bpb = _best_optional_bpb(rows, "base_val_bpb")
    best_sft_bpb = _best_optional_bpb(rows, "sft_val_bpb")
    return {
        "rows": [row.to_dict() for row in rows],
        "best_run": best.run,
        "best_eval_run": best.run,
        "best_base_bpb_run": best_base_bpb.run if best_base_bpb else None,
        "best_sft_bpb_run": best_sft_bpb.run if best_sft_bpb else None,
    }


def comparison_table(comparison: dict) -> str:
    """Render a compact fixed-width table for terminal output."""
    rows = comparison["rows"]
    table_rows = [
        [
            row["run"],
            row["tokenizer_type"],
            row["eval_score"],
            f"{row['pass_rate'] * 100:.2f}%",
            _format_optional_percent(row["support_match_rate"]),
            _format_optional_percent(row["prompt_echo_rate"]),
            _format_optional_float(row["base_val_bpb"]),
            _format_optional_float(row["sft_val_bpb"]),
            f"{row['base_val_loss']:.4f}",
            f"{row['sft_val_loss']:.4f}",
            _format_best_steps(row),
            _format_stop_reasons(row),
            row["memorization_status"],
            _short_int(row["num_parameters"]),
            str(row["context_size"]),
            str(row["truncated_examples"]),
        ]
        for row in rows
    ]
    headers = [
        "Run",
        "Tok",
        "Eval",
        "Pass",
        "Support",
        "Echo",
        "Base BPB",
        "SFT BPB",
        "Base Val",
        "SFT Val",
        "Best",
        "Stop",
        "Mem",
        "Params",
        "Ctx",
        "Trunc",
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in table_rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    lines.append(f"\nBest eval run: {comparison['best_eval_run']}")
    if comparison.get("best_base_bpb_run"):
        lines.append(f"Best base BPB run: {comparison['best_base_bpb_run']}")
    if comparison.get("best_sft_bpb_run"):
        lines.append(f"Best SFT BPB run: {comparison['best_sft_bpb_run']}")
    return "\n".join(lines)


def comparison_markdown(comparison: dict) -> str:
    """Render a Markdown report for run comparisons."""
    lines = [
        "# Picochat Run Comparison",
        "",
        f"Best eval run: `{comparison['best_eval_run']}`",
        "",
        f"Best base BPB run: `{comparison.get('best_base_bpb_run') or 'n/a'}`",
        "",
        f"Best SFT BPB run: `{comparison.get('best_sft_bpb_run') or 'n/a'}`",
        "",
        "| Run | Tokenizer | Eval | Pass Rate | Support Match | Prompt Echo | Base Val BPB | SFT Val BPB | Base Val Loss | SFT Val Loss | Best Steps | Stop Reasons | Base Loss Status | SFT Loss Status | Memorization | Params | Context | Truncated Examples |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| `{row['run']}` | `{row['tokenizer_type']}` | {row['eval_score']} | "
            f"{row['pass_rate'] * 100:.2f}% | {_format_optional_percent(row['support_match_rate'])} | "
            f"{_format_optional_percent(row['prompt_echo_rate'])} | "
            f"{_format_optional_float(row['base_val_bpb'])} | "
            f"{_format_optional_float(row['sft_val_bpb'])} | {row['base_val_loss']:.4f} | "
            f"{row['sft_val_loss']:.4f} | {_format_best_steps(row)} | "
            f"{_format_stop_reasons(row)} | `{row['base_loss_status']}` | "
            f"`{row['sft_loss_status']}` | `{row['memorization_status']}` | "
            f"{row['num_parameters']:,} | {row['context_size']} | {row['truncated_examples']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "Higher eval pass rate is better. Raw validation loss is useful within the "
        "same tokenizer, but it is not tokenizer-fair. Compare BPB when judging "
        "char, byte, and BPE runs against each other."
    )
    lines.append("")
    lines.append(
        "Best steps show the validation-best checkpoint step used by the pipeline "
        "when available. Stop reasons and memorization status help distinguish "
        "healthy learning from wasted steps or exact copying."
    )
    lines.append("")
    return "\n".join(lines)


def write_comparison_report(comparison: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(comparison_markdown(comparison), encoding="utf-8")


def _short_int(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_step(stage: dict) -> int | None:
    checkpoint = stage.get("best_checkpoint") or {}
    step = checkpoint.get("step")
    if step is None:
        step = stage.get("loss_diagnostics", {}).get("best_val_step")
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def _best_optional_bpb(rows: list[CompareRow], field_name: str) -> CompareRow | None:
    available = [row for row in rows if getattr(row, field_name) is not None]
    if len(available) < 2:
        return None
    return min(available, key=lambda row: getattr(row, field_name))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}%"


def _format_best_steps(row: dict) -> str:
    return f"{_format_optional_step(row['base_best_step'])}/{_format_optional_step(row['sft_best_step'])}"


def _format_optional_step(value: int | None) -> str:
    return "--" if value is None else str(value)


def _format_stop_reasons(row: dict) -> str:
    return f"{_short_reason(row['base_stop_reason'])}/{_short_reason(row['sft_stop_reason'])}"


def _short_reason(reason: str) -> str:
    if reason == "max_steps":
        return "max"
    if reason == "max_minutes":
        return "time"
    if reason == "early_stop":
        return "early"
    if reason == "unknown":
        return "--"
    return reason
