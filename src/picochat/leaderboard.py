"""Formal benchmark leaderboard reports for Picochat runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


BENCHMARK_CATEGORY_PREFIXES = (
    "arc",
    "mmlu",
    "gsm8k",
    "bench_",
    "identity",
    "refusal",
    "honesty",
)


@dataclass(frozen=True)
class LeaderboardRow:
    run: str
    path: str
    suite: str
    score: str
    pass_rate: float
    num_passed: int
    num_examples: int
    support_match_rate: float | None
    prompt_echo_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_benchmark_leaderboard(run_dirs: list[str | Path]) -> dict[str, Any]:
    """Build an overall and per-suite leaderboard from run eval reports."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    rows: list[LeaderboardRow] = []
    for run_dir in run_dirs:
        rows.extend(_rows_for_run(Path(run_dir)))
    if not rows:
        raise ValueError("no benchmark eval reports found")
    overall = [row for row in rows if row.suite == "overall"]
    best = max(overall or rows, key=lambda row: (row.pass_rate, row.num_passed, row.num_examples))
    return {
        "rows": [row.to_dict() for row in sorted(rows, key=_sort_key)],
        "best_run": best.run,
        "best_suite": best.suite,
        "best_pass_rate": best.pass_rate,
    }


def leaderboard_table(leaderboard: dict[str, Any]) -> str:
    """Render a compact text table for terminal use."""
    headers = ["Run", "Suite", "Score", "Pass", "Support", "Echo"]
    body = []
    for row in leaderboard["rows"]:
        body.append([
            row["run"],
            row["suite"],
            row["score"],
            f"{row['pass_rate'] * 100:.2f}%",
            _fmt_optional(row["support_match_rate"]),
            _fmt_optional(row["prompt_echo_rate"]),
        ])
    widths = [len(header) for header in headers]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in body:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def write_leaderboard_report(leaderboard: dict[str, Any], out_path: str | Path) -> Path:
    """Write a Markdown leaderboard report."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(leaderboard_markdown(leaderboard), encoding="utf-8")
    return path


def leaderboard_markdown(leaderboard: dict[str, Any]) -> str:
    lines = [
        "# Picochat Benchmark Leaderboard",
        "",
        f"Best run: `{leaderboard['best_run']}`",
        "",
        "| Run | Suite | Score | Pass | Support | Echo |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in leaderboard["rows"]:
        lines.append(
            "| "
            f"{row['run']} | {row['suite']} | {row['score']} | "
            f"{row['pass_rate'] * 100:.2f}% | {_fmt_optional(row['support_match_rate'])} | "
            f"{_fmt_optional(row['prompt_echo_rate'])} |"
        )
    lines.append("")
    lines.append("Scores come from held-out eval reports. Higher is better, but trust requires checking category failures and leakage reports.")
    return "\n".join(lines)


def _rows_for_run(run_dir: Path) -> list[LeaderboardRow]:
    report_path = _find_eval_report(run_dir)
    if not report_path.exists():
        raise FileNotFoundError(f"missing eval report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary") or {}
    rows = [_summary_row(run_dir, "overall", summary)]
    for name, category in sorted((summary.get("category_breakdown") or {}).items()):
        if _is_benchmark_suite(name):
            rows.append(_summary_row(run_dir, name, category))
    for name, level in sorted((summary.get("level_breakdown") or {}).items()):
        if name in {"choice", "math", "spelling", "refusal", "identity"}:
            rows.append(_summary_row(run_dir, f"level:{name}", level))
    return rows


def _find_eval_report(run_dir: Path) -> Path:
    default = run_dir / "eval" / "eval_report.json"
    if default.exists():
        return default
    candidates = sorted(
        run_dir.glob("eval*/eval_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else default


def _summary_row(run_dir: Path, suite: str, summary: dict[str, Any]) -> LeaderboardRow:
    passed = int(summary.get("num_passed") or 0)
    total = int(summary.get("num_examples") or 0)
    pass_rate = float(summary.get("pass_rate") or 0.0)
    return LeaderboardRow(
        run=run_dir.name,
        path=str(run_dir),
        suite=suite,
        score=f"{passed}/{total}",
        pass_rate=pass_rate,
        num_passed=passed,
        num_examples=total,
        support_match_rate=_optional_float(summary.get("support_match_rate")),
        prompt_echo_rate=_optional_float(summary.get("prompt_echo_rate")),
    )


def _is_benchmark_suite(name: str) -> bool:
    return name.startswith(BENCHMARK_CATEGORY_PREFIXES)


def _sort_key(row: LeaderboardRow) -> tuple[int, str, float, int]:
    return (0 if row.suite == "overall" else 1, row.suite, -row.pass_rate, -row.num_passed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}%"
