"""Compare completed Picochat experiment runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CompareRow:
    run: str
    path: str
    eval_score: str
    pass_rate: float
    base_val_loss: float
    sft_val_loss: float
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
    return CompareRow(
        run=run_dir.name,
        path=str(run_dir),
        eval_score=f"{eval_summary['num_passed']}/{eval_summary['num_examples']}",
        pass_rate=float(eval_summary["pass_rate"]),
        base_val_loss=float(summary["base"]["final_val_loss"]),
        sft_val_loss=float(summary["sft"]["final_val_loss"]),
        num_parameters=int(summary["base"]["num_parameters"]),
        context_size=int(summary["config"]["context_size"]),
        truncated_examples=int(summary["sft"]["truncated_examples"]),
    )


def compare_runs(run_dirs: list[str | Path]) -> dict:
    """Return comparison data for one or more run directories."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    rows = [load_compare_row(path) for path in run_dirs]
    best = max(rows, key=lambda row: (row.pass_rate, -row.sft_val_loss))
    return {
        "rows": [row.to_dict() for row in rows],
        "best_run": best.run,
    }


def comparison_table(comparison: dict) -> str:
    """Render a compact fixed-width table for terminal output."""
    rows = comparison["rows"]
    table_rows = [
        [
            row["run"],
            row["eval_score"],
            f"{row['pass_rate'] * 100:.2f}%",
            f"{row['base_val_loss']:.4f}",
            f"{row['sft_val_loss']:.4f}",
            _short_int(row["num_parameters"]),
            str(row["context_size"]),
            str(row["truncated_examples"]),
        ]
        for row in rows
    ]
    headers = ["Run", "Eval", "Pass", "Base Val", "SFT Val", "Params", "Ctx", "Trunc"]
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
    lines.append(f"\nBest eval run: {comparison['best_run']}")
    return "\n".join(lines)


def comparison_markdown(comparison: dict) -> str:
    """Render a Markdown report for run comparisons."""
    lines = [
        "# Picochat Run Comparison",
        "",
        f"Best eval run: `{comparison['best_run']}`",
        "",
        "| Run | Eval | Pass Rate | Base Val Loss | SFT Val Loss | Params | Context | Truncated Examples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| `{row['run']}` | {row['eval_score']} | {row['pass_rate'] * 100:.2f}% | "
            f"{row['base_val_loss']:.4f} | {row['sft_val_loss']:.4f} | "
            f"{row['num_parameters']:,} | {row['context_size']} | {row['truncated_examples']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "Higher eval pass rate is better. Losses are diagnostic: low SFT train loss "
        "with high SFT validation loss usually means the tiny run memorized its chat examples."
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
