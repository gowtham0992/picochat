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
    non_choice_pass_rate: float | None
    domain_pass_rate: float | None
    refusal_pass_rate: float | None
    support_match_rate: float | None
    prompt_echo_rate: float | None
    sft_fit_rate: float | None
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
    device: str
    base_effective_batch_size: int | None
    sft_effective_batch_size: int | None
    truncated_examples: int
    skipped_long_examples: int

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
    sft_fit_summary = summary.get("sft_fit") or {}
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
        non_choice_pass_rate=_optional_float(eval_summary.get("non_choice_pass_rate")),
        domain_pass_rate=_optional_float(eval_summary.get("domain_pass_rate")),
        refusal_pass_rate=_optional_float(eval_summary.get("refusal_pass_rate")),
        support_match_rate=_optional_float(eval_summary.get("support_match_rate")),
        prompt_echo_rate=_optional_float(eval_summary.get("prompt_echo_rate")),
        sft_fit_rate=_optional_float(sft_fit_summary.get("pass_rate")),
        base_val_loss=float(_stage_metric(base, "best_val_loss", "val_loss", "final_val_loss")),
        sft_val_loss=float(_stage_metric(sft, "best_val_loss", "val_loss", "final_val_loss")),
        base_val_bpb=_optional_float(_stage_metric(base, "best_val_bpb", "val_bpb", "final_val_bpb")),
        sft_val_bpb=_optional_float(_stage_metric(sft, "best_val_bpb", "val_bpb", "final_val_bpb")),
        base_best_step=_best_step(base),
        sft_best_step=_best_step(sft),
        base_stop_reason=str(base.get("stop_reason") or "unknown"),
        sft_stop_reason=str(sft.get("stop_reason") or "unknown"),
        base_loss_status=str(base.get("loss_diagnostics", {}).get("status") or "unknown"),
        sft_loss_status=str(sft.get("loss_diagnostics", {}).get("status") or "unknown"),
        memorization_status=str(base.get("memorization", {}).get("status") or "unknown"),
        num_parameters=int(base["num_parameters"]),
        context_size=int(config["context_size"]),
        device=str(config.get("device") or "unknown"),
        base_effective_batch_size=_optional_int(base.get("effective_batch_size")),
        sft_effective_batch_size=_optional_int(sft.get("effective_batch_size")),
        truncated_examples=int(sft["truncated_examples"]),
        skipped_long_examples=int(sft.get("skipped_long_examples", 0)),
    )


def compare_runs(run_dirs: list[str | Path]) -> dict:
    """Return comparison data for one or more run directories."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    rows = [load_compare_row(path) for path in run_dirs]
    best = max(rows, key=lambda row: (row.pass_rate, -row.sft_val_loss))
    best_base_bpb = _best_optional_bpb(rows, "base_val_bpb")
    best_sft_bpb = _best_optional_bpb(rows, "sft_val_bpb")
    closed_book = best_base_bpb or min(rows, key=lambda row: row.base_val_loss)
    decision = _comparison_decision(rows, best, best_base_bpb, best_sft_bpb)
    return {
        "rows": [row.to_dict() for row in rows],
        "best_run": best.run,
        "best_eval_run": best.run,
        "best_closed_book_run": closed_book.run,
        "best_base_bpb_run": best_base_bpb.run if best_base_bpb else None,
        "best_sft_bpb_run": best_sft_bpb.run if best_sft_bpb else None,
        "decision": decision,
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
            _format_optional_percent(row["non_choice_pass_rate"]),
            _format_optional_percent(row["domain_pass_rate"]),
            _format_optional_percent(row["refusal_pass_rate"]),
            _format_optional_percent(row["support_match_rate"]),
            _format_optional_percent(row["prompt_echo_rate"]),
            _format_optional_percent(row["sft_fit_rate"]),
            _format_optional_float(row["base_val_bpb"]),
            _format_optional_float(row["sft_val_bpb"]),
            f"{row['base_val_loss']:.4f}",
            f"{row['sft_val_loss']:.4f}",
            _format_best_steps(row),
            _format_stop_reasons(row),
            row["memorization_status"],
            _short_int(row["num_parameters"]),
            str(row["context_size"]),
            row["device"],
            _format_optional_int(row["base_effective_batch_size"]),
            _format_optional_int(row["sft_effective_batch_size"]),
            str(row["truncated_examples"]),
            str(row["skipped_long_examples"]),
        ]
        for row in rows
    ]
    headers = [
        "Run",
        "Tok",
        "Eval",
        "Pass",
        "NonChoice",
        "Domain",
        "Refusal",
        "Support",
        "Echo",
        "SFT Fit",
        "Base BPB",
        "SFT BPB",
        "Base Val",
        "SFT Val",
        "Best",
        "Stop",
        "Mem",
        "Params",
        "Ctx",
        "Device",
        "Base Eff B",
        "SFT Eff B",
        "Trunc",
        "Skip",
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
    lines.append(f"Best closed-book run: {comparison['best_closed_book_run']}")
    if comparison.get("best_sft_bpb_run"):
        lines.append(f"Best SFT BPB run: {comparison['best_sft_bpb_run']}")
    decision = comparison.get("decision") or {}
    if decision:
        lines.append("")
        lines.append(f"Champion gate: {decision.get('champion_title', 'n/a')}")
        lines.append(f"Regression watch: {decision.get('regression_title', 'n/a')}")
        lines.append(f"Next experiment: {decision.get('next_title', 'n/a')}")
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
        f"Best closed-book run: `{comparison.get('best_closed_book_run') or 'n/a'}`",
        "",
    ]
    decision = comparison.get("decision") or {}
    if decision:
        lines.extend([
            "## Decision Gate",
            "",
            "| Gate | Status | Message |",
            "| --- | --- | --- |",
            (
                f"| Champion | `{decision.get('champion_status', 'unknown')}` | "
                f"{_markdown_text(decision.get('champion_title'))}: {_markdown_text(decision.get('champion_message'))} |"
            ),
            (
                f"| Regression | `{decision.get('regression_status', 'unknown')}` | "
                f"{_markdown_text(decision.get('regression_title'))}: {_markdown_text(decision.get('regression_message'))} |"
            ),
            (
                f"| Next | `{decision.get('next_status', 'unknown')}` | "
                f"{_markdown_text(decision.get('next_title'))}: {_markdown_text(decision.get('next_message'))} |"
            ),
            "",
        ])
        issues = decision.get("issues") or []
        if issues:
            lines.append("Watch items:")
            lines.append("")
            for issue in issues:
                lines.append(f"- `{issue.get('severity', 'warn')}` {_markdown_text(issue.get('message'))}")
            lines.append("")
    lines.extend([
        "| Run | Tokenizer | Eval | Pass Rate | Non-Choice Pass | Domain Pass | Refusal Pass | Support Match | Prompt Echo | SFT Fit | Base Val BPB | SFT Val BPB | Base Val Loss | SFT Val Loss | Best Steps | Stop Reasons | Base Loss Status | SFT Loss Status | Memorization | Params | Context | Device | Base Eff B | SFT Eff B | Truncated Examples | Skipped Too-Long |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in comparison["rows"]:
        lines.append(
            f"| `{row['run']}` | `{row['tokenizer_type']}` | {row['eval_score']} | "
            f"{row['pass_rate'] * 100:.2f}% | {_format_optional_percent(row.get('non_choice_pass_rate'))} | "
            f"{_format_optional_percent(row['domain_pass_rate'])} | "
            f"{_format_optional_percent(row['refusal_pass_rate'])} | "
            f"{_format_optional_percent(row['support_match_rate'])} | "
            f"{_format_optional_percent(row['prompt_echo_rate'])} | "
            f"{_format_optional_percent(row.get('sft_fit_rate'))} | "
            f"{_format_optional_float(row['base_val_bpb'])} | "
            f"{_format_optional_float(row['sft_val_bpb'])} | {row['base_val_loss']:.4f} | "
            f"{row['sft_val_loss']:.4f} | {_format_best_steps(row)} | "
            f"{_format_stop_reasons(row)} | `{row['base_loss_status']}` | "
            f"`{row['sft_loss_status']}` | `{row['memorization_status']}` | "
            f"{row['num_parameters']:,} | {row['context_size']} | `{row['device']}` | "
            f"{_format_optional_int(row['base_effective_batch_size'])} | "
            f"{_format_optional_int(row['sft_effective_batch_size'])} | {row['truncated_examples']} | "
            f"{row.get('skipped_long_examples', 0)} |"
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


def _comparison_decision(
    rows: list[CompareRow],
    best: CompareRow,
    best_base_bpb: CompareRow | None,
    best_sft_bpb: CompareRow | None,
) -> dict:
    ranked = sorted(rows, key=lambda row: (row.pass_rate, -row.sft_val_loss), reverse=True)
    baseline = next((row for row in ranked if row.run != best.run), None)
    issues = _regression_issues(best, baseline)
    pass_delta = best.pass_rate - baseline.pass_rate if baseline else None
    champion_status = (
        "warn" if baseline is None
        else "fail" if any(issue["severity"] == "fail" for issue in issues)
        else "warn" if issues
        else "pass"
    )
    champion_title = (
        "Need a baseline" if baseline is None
        else "Promote as reference" if champion_status == "pass"
        else "Promising, inspect regressions" if champion_status == "warn"
        else "Do not promote yet"
    )
    champion_message = (
        "Compare at least two runs so the winner has something to beat."
        if baseline is None
        else f"{best.run} is {_signed_percent(pass_delta or 0.0)} eval pass versus {baseline.run}."
    )
    regression_status = (
        "warn" if baseline is None
        else "fail" if any(issue["severity"] == "fail" for issue in issues)
        else "warn" if issues
        else "pass"
    )
    regression_title = (
        "No regression check" if baseline is None
        else f"{len(issues)} watch item{'s' if len(issues) != 1 else ''}" if issues
        else "No obvious regression"
    )
    regression_message = (
        "Regression checks need both a candidate and a baseline."
        if baseline is None
        else " ".join(issue["message"] for issue in issues) if issues
        else "Eval pass, support, echo, SFT BPB, truncation, and memorization look acceptable."
    )
    next_step = _next_experiment(best, baseline, issues, best_base_bpb, best_sft_bpb)
    return {
        "baseline_run": baseline.run if baseline else None,
        "champion_status": champion_status,
        "champion_title": champion_title,
        "champion_message": champion_message,
        "regression_status": regression_status,
        "regression_title": regression_title,
        "regression_message": regression_message,
        "next_status": next_step["status"],
        "next_title": next_step["title"],
        "next_message": next_step["message"],
        "issues": issues,
    }


def _regression_issues(best: CompareRow, baseline: CompareRow | None) -> list[dict]:
    if baseline is None:
        return []
    issues: list[dict] = []
    pass_delta = best.pass_rate - baseline.pass_rate
    domain_delta = _optional_delta(best.domain_pass_rate, baseline.domain_pass_rate)
    refusal_delta = _optional_delta(best.refusal_pass_rate, baseline.refusal_pass_rate)
    support_delta = _optional_delta(best.support_match_rate, baseline.support_match_rate)
    echo_delta = _optional_delta(best.prompt_echo_rate, baseline.prompt_echo_rate)
    sft_fit_delta = _optional_delta(best.sft_fit_rate, baseline.sft_fit_rate)
    sft_bpb_delta = _optional_delta(best.sft_val_bpb, baseline.sft_val_bpb)
    if pass_delta < 0.02:
        issues.append({"severity": "warn", "message": "Eval gain is under +2 points."})
    if domain_delta is not None and domain_delta < 0.02:
        issues.append({"severity": "warn", "message": "Domain-answer gain is under +2 points."})
    if refusal_delta is not None and refusal_delta < -0.05:
        issues.append({"severity": "fail", "message": f"Refusal/boundary pass dropped {_signed_percent(refusal_delta)}."})
    if support_delta is not None and support_delta < -0.05:
        issues.append({"severity": "fail", "message": f"Support match dropped {_signed_percent(support_delta)}."})
    if echo_delta is not None and echo_delta > 0.02:
        issues.append({"severity": "fail", "message": f"Prompt echo worsened {_signed_percent(echo_delta)}."})
    if sft_fit_delta is not None and sft_fit_delta < -0.05:
        issues.append({"severity": "warn", "message": f"SFT exact-fit dropped {_signed_percent(sft_fit_delta)}."})
    if sft_bpb_delta is not None and sft_bpb_delta > 0.10:
        issues.append({"severity": "warn", "message": f"SFT BPB rose {_signed_float(sft_bpb_delta)}."})
    if best.truncated_examples > baseline.truncated_examples:
        issues.append({"severity": "warn", "message": "More SFT rows were truncated."})
    if best.skipped_long_examples > baseline.skipped_long_examples:
        issues.append({"severity": "warn", "message": "More SFT rows were skipped for exceeding context."})
    if best.memorization_status.lower() != "low":
        issues.append({"severity": "fail", "message": f"Memorization status is {best.memorization_status}."})
    return issues


def _next_experiment(
    best: CompareRow,
    baseline: CompareRow | None,
    issues: list[dict],
    best_base_bpb: CompareRow | None,
    best_sft_bpb: CompareRow | None,
) -> dict:
    if baseline is None:
        return {
            "status": "warn",
            "title": "Add a comparison run",
            "message": "Compare against a previous run before changing model size or training time.",
        }
    if any(issue["severity"] == "fail" for issue in issues):
        return {
            "status": "fail",
            "title": "Repair before scaling",
            "message": "Fix trust regressions before longer runs or larger models.",
        }
    if best.sft_fit_rate is not None and best.sft_fit_rate < 0.70:
        return {
            "status": "warn",
            "title": "Increase SFT fit first",
            "message": (
                "The eval checkpoint cannot reliably reproduce its own SFT rows yet; "
                "raise SFT fit before treating held-out eval as the main blocker."
            ),
        }
    if best_base_bpb and best.run != best_base_bpb.run:
        return {
            "status": "warn",
            "title": "Separate compression from behavior",
            "message": (
                f"{best_base_bpb.run} has better base BPB; compare tokenizer/model "
                f"settings before scaling {best.run}."
            ),
        }
    if best_sft_bpb and best.run != best_sft_bpb.run:
        return {
            "status": "warn",
            "title": "SFT quality mismatch",
            "message": f"{best_sft_bpb.run} has better SFT BPB; inspect SFT curriculum before choosing a champion.",
        }
    if best.pass_rate >= 0.70 and (best.domain_pass_rate is None or best.domain_pass_rate >= 0.50):
        return {
            "status": "pass",
            "title": "Attack harder eval",
            "message": "Keep this as reference, add harder eval rows, then run a stronger preset.",
        }
    if best.domain_pass_rate is not None and best.domain_pass_rate < 0.25:
        return {
            "status": "warn",
            "title": "Improve answer data",
            "message": "The model is not passing enough domain-answer rows yet; improve SFT/eval data before scaling.",
        }
    return {
        "status": "warn",
        "title": "Improve data next",
        "message": "Use failed eval categories to add targeted SFT rows before changing architecture.",
    }


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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stage_metric(stage: dict, summary_key: str, checkpoint_key: str, final_key: str):
    if stage.get(summary_key) is not None:
        return stage.get(summary_key)
    checkpoint = stage.get("best_checkpoint") or {}
    if checkpoint.get(checkpoint_key) is not None:
        return checkpoint.get(checkpoint_key)
    return stage.get(final_key)


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


def _format_optional_int(value: int | None) -> str:
    return "--" if value is None else str(value)


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}%"


def _optional_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def _signed_percent(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _signed_float(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.4f}"


def _markdown_text(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


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
