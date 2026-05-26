"""Starter preference-pair generation for DPO experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from picochat.sft import ChatExample, load_chat_examples


@dataclass(frozen=True)
class PreferenceStarterConfig:
    input_path: str
    output_path: str
    max_rows: int = 0
    force: bool = False


def generate_preference_starter(config: PreferenceStarterConfig) -> dict[str, Any]:
    """Generate conservative chosen/rejected rows from a chat SFT JSONL file."""
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    if output_path.exists() and not config.force:
        raise FileExistsError(f"refusing to overwrite existing preference file: {output_path}")
    examples = load_chat_examples(input_path)
    if config.max_rows > 0:
        examples = examples[:config.max_rows]
    rows = [_preference_row(example, index) for index, example in enumerate(examples)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    category_counts = Counter(row["category"] for row in rows)
    reject_counts = Counter(row["rejected_type"] for row in rows)
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_rows": len(examples),
        "num_examples": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "rejected_type_counts": dict(sorted(reject_counts.items())),
        "warning": (
            "Starter preference rows are synthetic negatives for DPO plumbing and smoke tests. "
            "Use human or judge-reviewed preference data for release alignment claims."
        ),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.md")
    report_path.write_text(preference_starter_markdown(report), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def preference_starter_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Preference Starter Pack",
        "",
        f"- Input: `{report['input_path']}`",
        f"- Output: `{report['output_path']}`",
        f"- Rows: {int(report['num_examples']):,}",
        f"- Warning: {report['warning']}",
        "",
        "## Rejected Response Types",
        "",
    ]
    for name, count in report["rejected_type_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Categories", ""])
    for name, count in report["category_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "Use this pack to verify DPO mechanics. Do not treat synthetic negatives as a substitute for real preference data.",
    ])
    return "\n".join(lines)


def _preference_row(example: ChatExample, index: int) -> dict[str, str]:
    rejected_type, rejected = _rejected_answer(example, index)
    if rejected.strip() == example.assistant.strip():
        rejected_type = "generic_wrong"
        rejected = "I cannot provide a reliable answer to that request."
    group = example.group or f"preference-{index:06d}"
    return {
        "user": example.user,
        "chosen": example.assistant,
        "rejected": rejected,
        "category": f"{example.category}_preference",
        "group": group,
        "source_category": example.category,
        "rejected_type": rejected_type,
    }


def _rejected_answer(example: ChatExample, index: int) -> tuple[str, str]:
    variants = (
        ("prompt_echo", example.user),
        ("empty_answer", "I do not know."),
        ("generic_refusal", "I cannot help with that."),
        ("off_topic", "This question is about something else, so the answer is unavailable."),
    )
    return variants[index % len(variants)]
