"""Converters for external multiple-choice benchmark files."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any


EXTERNAL_BENCHMARK_FORMATS = ("auto", "arc", "mmlu", "mmlu_csv")


@dataclass(frozen=True)
class ExternalBenchmarkConvertConfig:
    input_path: str
    output_path: str
    source_format: str = "auto"
    benchmark_name: str = "external"
    split: str = "external"
    max_rows: int | None = None
    seed: int = 42
    shuffle: bool = False


def convert_external_benchmark(config: ExternalBenchmarkConvertConfig) -> dict[str, Any]:
    """Convert ARC/MMLU-style rows into Picochat choice-eval JSONL."""
    if config.source_format not in EXTERNAL_BENCHMARK_FORMATS:
        raise ValueError(f"source_format must be one of {EXTERNAL_BENCHMARK_FORMATS}")
    if config.max_rows is not None and config.max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    records = _read_records(input_path, config.source_format)
    if config.shuffle:
        rng = random.Random(config.seed)
        rng.shuffle(records)
    if config.max_rows is not None:
        records = records[: config.max_rows]

    rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        row = _convert_record(
            record,
            source_format=config.source_format,
            benchmark_name=config.benchmark_name,
            split=config.split,
            index=index,
        )
        rows.append(row)
        category_counts[str(row["category"])] += 1
        label_counts[str(row["correct_choice"])] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_format": config.source_format,
        "benchmark_name": config.benchmark_name,
        "split": config.split,
        "num_rows": len(rows),
        "categories": dict(sorted(category_counts.items())),
        "correct_choice_labels": dict(sorted(label_counts.items())),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.md")
    report_path.write_text(external_benchmark_report_markdown(report), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def external_benchmark_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Picochat External Benchmark Convert",
        "",
        f"- Input: `{report['input_path']}`",
        f"- Output: `{report['output_path']}`",
        f"- Source format: `{report['source_format']}`",
        f"- Benchmark: `{report['benchmark_name']}`",
        f"- Rows: {report['num_rows']}",
        "",
        "## Categories",
        "",
    ]
    for category, count in report.get("categories", {}).items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Correct Choice Labels", ""])
    for label, count in report.get("correct_choice_labels", {}).items():
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    return "\n".join(lines)


def _read_records(path: Path, source_format: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("data", data) if isinstance(data, dict) else data
    elif suffix == ".csv":
        rows = _read_csv_records(path, source_format)
    else:
        raise ValueError("external benchmark input must be .jsonl, .json, or .csv")
    if not isinstance(rows, list):
        raise ValueError("external benchmark JSON must contain a list or a data list")
    return [dict(row) for row in rows]


def _read_csv_records(path: Path, source_format: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    sample = text.splitlines()[0] if text.splitlines() else ""
    lower_header = {cell.strip().lower() for cell in sample.split(",")}
    has_header = bool({"question", "answer"} & lower_header)
    if source_format == "mmlu_csv" and not has_header:
        rows: list[dict[str, Any]] = []
        for row in csv.reader(text.splitlines()):
            if len(row) < 6:
                continue
            rows.append({
                "question": row[0],
                "A": row[1],
                "B": row[2],
                "C": row[3],
                "D": row[4],
                "answer": row[5],
            })
        return rows
    return list(csv.DictReader(text.splitlines()))


def _convert_record(
    record: dict[str, Any],
    *,
    source_format: str,
    benchmark_name: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    if source_format == "auto":
        source_format = _detect_format(record)
    if source_format == "arc":
        question, labels, choices, answer = _parse_arc_record(record, index)
    elif source_format in {"mmlu", "mmlu_csv"}:
        question, labels, choices, answer = _parse_mmlu_record(record, index)
    else:
        raise ValueError(f"unsupported external benchmark format: {source_format}")

    prompt = _choice_prompt(question, labels, choices)
    category = str(record.get("category") or record.get("subject") or benchmark_name)
    return {
        "user": prompt,
        "must_include": [answer],
        "choice_labels": labels,
        "correct_choice": answer,
        "answerable": True,
        "category": f"external_{category}",
        "split": split,
        "level": "external_choice",
        "curriculum_stage": benchmark_name,
        "reference_answer": f"{answer}. {choices[labels.index(answer)]}",
        "source": str(record.get("id") or record.get("question_id") or f"{benchmark_name}-{index:06d}"),
    }


def _detect_format(record: dict[str, Any]) -> str:
    if "answerKey" in record or "choices" in record:
        return "arc"
    if {"question", "A", "B", "C", "D"} <= set(record):
        return "mmlu"
    raise ValueError("could not auto-detect external benchmark format")


def _parse_arc_record(record: dict[str, Any], index: int) -> tuple[str, list[str], list[str], str]:
    question = _string_field(record, "question", index)
    labels, choices = _parse_choices(record.get("choices"), index)
    answer = str(record.get("answerKey", record.get("answer", ""))).strip()
    if answer not in labels and answer.isdigit():
        answer = _digit_answer_to_label(answer, labels)
    _validate_answer(answer, labels, index)
    return question, labels, choices, answer


def _parse_mmlu_record(record: dict[str, Any], index: int) -> tuple[str, list[str], list[str], str]:
    question = _string_field(record, "question", index)
    labels = ["A", "B", "C", "D"]
    choices = [_string_field(record, label, index) for label in labels]
    answer = str(record.get("answer", record.get("target", ""))).strip()
    if answer.isdigit():
        answer = _digit_answer_to_label(answer, labels)
    answer = answer.upper()
    _validate_answer(answer, labels, index)
    return question, labels, choices, answer


def _parse_choices(raw_choices: Any, index: int) -> tuple[list[str], list[str]]:
    if isinstance(raw_choices, dict):
        labels = [str(label).strip() for label in raw_choices.get("label", [])]
        choices = [str(text).strip() for text in raw_choices.get("text", [])]
    elif isinstance(raw_choices, list):
        labels = []
        choices = []
        for choice_index, item in enumerate(raw_choices):
            if not isinstance(item, dict):
                raise ValueError(f"row {index + 1} choice items must be objects")
            labels.append(str(item.get("label", chr(ord("A") + choice_index))).strip())
            choices.append(str(item.get("text", item.get("choice", ""))).strip())
    else:
        raise ValueError(f"row {index + 1} choices must be a dict or list")
    if not labels or len(labels) != len(choices):
        raise ValueError(f"row {index + 1} choices must have matching labels and text")
    if any(not label for label in labels) or any(not choice for choice in choices):
        raise ValueError(f"row {index + 1} choices cannot be empty")
    return labels, choices


def _choice_prompt(question: str, labels: list[str], choices: list[str]) -> str:
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    return f"{question.strip()}\n\n{options}\n\nAnswer with the single best option letter."


def _string_field(record: dict[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {index + 1} field {field!r} must be a non-empty string")
    return value.strip()


def _digit_answer_to_label(answer: str, labels: list[str]) -> str:
    numeric = int(answer)
    if numeric in range(1, len(labels) + 1):
        return labels[numeric - 1]
    if numeric in range(0, len(labels)):
        return labels[numeric]
    return answer


def _validate_answer(answer: str, labels: list[str], index: int) -> None:
    if answer not in labels:
        raise ValueError(f"row {index + 1} answer {answer!r} is not present in choice labels {labels}")
