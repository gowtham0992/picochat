"""Preflight checks for chat SFT and transparent eval JSONL files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TuningDataIssue:
    line: int
    message: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ChatSFTDataReport:
    path: str
    status: str
    summary: str
    num_rows: int
    num_examples: int
    empty_rows: int
    invalid_rows: int
    average_user_chars: float
    average_assistant_chars: float
    duplicate_user_rate: float
    categories: dict[str, int]
    issues: tuple[TuningDataIssue, ...]
    preview: tuple[dict[str, str], ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "issues": [issue.to_dict() for issue in self.issues],
            "preview": list(self.preview),
        }


@dataclass(frozen=True)
class ChatEvalDataReport:
    path: str
    status: str
    summary: str
    num_rows: int
    num_items: int
    empty_rows: int
    invalid_rows: int
    answerable_items: int
    unanswerable_items: int
    must_include_rules: int
    must_include_any_groups: int
    must_not_include_rules: int
    categories: dict[str, int]
    issues: tuple[TuningDataIssue, ...]
    preview: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "issues": [issue.to_dict() for issue in self.issues],
            "preview": list(self.preview),
        }


def inspect_chat_sft_data(path: str | Path, preview_items: int = 3) -> ChatSFTDataReport:
    """Validate one-turn chat SFT JSONL before training."""
    path = str(path)
    records, empty_rows, read_issue = _read_jsonl_records(path)
    if read_issue:
        return ChatSFTDataReport(
            path=path,
            status="blocked",
            summary="Chat SFT data could not be read.",
            num_rows=0,
            num_examples=0,
            empty_rows=0,
            invalid_rows=1,
            average_user_chars=0.0,
            average_assistant_chars=0.0,
            duplicate_user_rate=0.0,
            categories={},
            issues=(read_issue,),
            preview=(),
        )

    issues: list[TuningDataIssue] = []
    examples: list[dict[str, str]] = []
    users: list[str] = []
    categories: dict[str, int] = {}
    user_chars = 0
    assistant_chars = 0
    for line_number, record in records:
        if not isinstance(record, dict):
            issues.append(TuningDataIssue(line_number, "row must be a JSON object"))
            continue
        if "__invalid_json__" in record:
            issues.append(TuningDataIssue(line_number, f"invalid JSON: {record['__invalid_json__']}"))
            continue
        user = record.get("user")
        assistant = record.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            issues.append(TuningDataIssue(line_number, "row must contain string user and assistant fields"))
            continue
        if not user.strip() or not assistant.strip():
            issues.append(TuningDataIssue(line_number, "user and assistant fields should not be empty"))
            continue
        category = record.get("category", "chat")
        if not isinstance(category, str) or not category.strip():
            issues.append(TuningDataIssue(line_number, "category field must be a non-empty string when present"))
            continue
        category = category.strip()
        examples.append({"user": user, "assistant": assistant, "category": category})
        users.append(user)
        categories[category] = categories.get(category, 0) + 1
        user_chars += len(user)
        assistant_chars += len(assistant)

    num_rows = len(records)
    invalid_rows = len(issues)
    duplicate_user_rate = _duplicate_rate(users)
    status, summary = _chat_sft_status(len(examples), invalid_rows, duplicate_user_rate)
    return ChatSFTDataReport(
        path=path,
        status=status,
        summary=summary,
        num_rows=num_rows,
        num_examples=len(examples),
        empty_rows=empty_rows,
        invalid_rows=invalid_rows,
        average_user_chars=(user_chars / len(examples)) if examples else 0.0,
        average_assistant_chars=(assistant_chars / len(examples)) if examples else 0.0,
        duplicate_user_rate=duplicate_user_rate,
        categories=dict(sorted(categories.items())),
        issues=tuple(issues[:8]),
        preview=tuple(examples[:max(0, preview_items)]),
    )


def inspect_chat_eval_data(path: str | Path, preview_items: int = 3) -> ChatEvalDataReport:
    """Validate transparent chat eval JSONL before scoring."""
    path = str(path)
    records, empty_rows, read_issue = _read_jsonl_records(path)
    if read_issue:
        return ChatEvalDataReport(
            path=path,
            status="blocked",
            summary="Eval data could not be read.",
            num_rows=0,
            num_items=0,
            empty_rows=0,
            invalid_rows=1,
            answerable_items=0,
            unanswerable_items=0,
            must_include_rules=0,
            must_include_any_groups=0,
            must_not_include_rules=0,
            categories={},
            issues=(read_issue,),
            preview=(),
        )

    issues: list[TuningDataIssue] = []
    items: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    answerable_items = 0
    must_include_rules = 0
    must_include_any_groups = 0
    must_not_include_rules = 0
    for line_number, record in records:
        item, item_issues = _parse_eval_item(line_number, record)
        if item_issues:
            issues.extend(item_issues)
            continue
        items.append(item)
        category = item["category"]
        categories[category] = categories.get(category, 0) + 1
        if item["answerable"]:
            answerable_items += 1
        must_include_rules += len(item["must_include"])
        must_include_any_groups += len(item["must_include_any"])
        must_not_include_rules += len(item["must_not_include"])

    num_rows = len(records)
    invalid_rows = len(issues)
    unanswerable_items = len(items) - answerable_items
    status, summary = _chat_eval_status(
        len(items),
        invalid_rows,
        must_include_rules + must_include_any_groups + must_not_include_rules,
        unanswerable_items,
    )
    return ChatEvalDataReport(
        path=path,
        status=status,
        summary=summary,
        num_rows=num_rows,
        num_items=len(items),
        empty_rows=empty_rows,
        invalid_rows=invalid_rows,
        answerable_items=answerable_items,
        unanswerable_items=unanswerable_items,
        must_include_rules=must_include_rules,
        must_include_any_groups=must_include_any_groups,
        must_not_include_rules=must_not_include_rules,
        categories=dict(sorted(categories.items())),
        issues=tuple(issues[:8]),
        preview=tuple(items[:max(0, preview_items)]),
    )


def _read_jsonl_records(path: str) -> tuple[list[tuple[int, Any]], int, TuningDataIssue | None]:
    source = Path(path)
    if not source.exists():
        return [], 0, TuningDataIssue(0, "file does not exist")
    if not source.is_file():
        return [], 0, TuningDataIssue(0, "path is not a file")

    records: list[tuple[int, Any]] = []
    empty_rows = 0
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [], 0, TuningDataIssue(0, "file is not valid UTF-8 text")

    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            empty_rows += 1
            continue
        try:
            records.append((line_number, json.loads(text)))
        except json.JSONDecodeError as error:
            records.append((line_number, {"__invalid_json__": str(error)}))
    return records, empty_rows, None


def _parse_eval_item(line_number: int, record: Any) -> tuple[dict[str, Any], list[TuningDataIssue]]:
    issues: list[TuningDataIssue] = []
    if not isinstance(record, dict):
        return {}, [TuningDataIssue(line_number, "row must be a JSON object")]
    if "__invalid_json__" in record:
        return {}, [TuningDataIssue(line_number, f"invalid JSON: {record['__invalid_json__']}")]

    user = record.get("user")
    if not isinstance(user, str) or not user.strip():
        issues.append(TuningDataIssue(line_number, "row must contain a non-empty string user field"))

    must_include = record.get("must_include", ())
    expected = record.get("expected")
    if expected is not None:
        if isinstance(expected, str):
            if must_include in (None, ()):
                must_include = [expected]
            elif isinstance(must_include, list):
                must_include = [*must_include, expected]
        else:
            issues.append(TuningDataIssue(line_number, "expected field must be a string"))

    must_include_values = _string_list(must_include, line_number, "must_include", issues)
    must_not_include_values = _string_list(record.get("must_not_include", ()), line_number, "must_not_include", issues)
    must_include_any_values = _phrase_groups(record.get("must_include_any", ()), line_number, issues)

    answerable = record.get("answerable", True)
    if not isinstance(answerable, bool):
        issues.append(TuningDataIssue(line_number, "answerable field must be a boolean"))
        answerable = True

    category = record.get("category", "answerable" if answerable else "unanswerable")
    if not isinstance(category, str) or not category.strip():
        issues.append(TuningDataIssue(line_number, "category field must be a non-empty string"))
        category = "answerable" if answerable else "unanswerable"

    if issues:
        return {}, issues
    return {
        "user": user,
        "answerable": answerable,
        "category": category,
        "must_include": must_include_values,
        "must_include_any": must_include_any_values,
        "must_not_include": must_not_include_values,
    }, []


def _string_list(value: Any, line_number: int, field: str, issues: list[TuningDataIssue]) -> list[str]:
    if value in (None, ()):
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        issues.append(TuningDataIssue(line_number, f"{field} field must be a list of non-empty strings"))
        return []
    return value


def _phrase_groups(value: Any, line_number: int, issues: list[TuningDataIssue]) -> list[list[str]]:
    if value in (None, ()):
        return []
    if not isinstance(value, list):
        issues.append(TuningDataIssue(line_number, "must_include_any field must be a list of lists"))
        return []

    groups: list[list[str]] = []
    for group in value:
        if not isinstance(group, list) or not group or not all(isinstance(item, str) and item for item in group):
            issues.append(TuningDataIssue(line_number, "must_include_any groups must contain non-empty strings"))
            return []
        groups.append(group)
    return groups


def _duplicate_rate(values: list[str]) -> float:
    if not values:
        return 0.0
    return (len(values) - len(set(values))) / len(values)


def _chat_sft_status(num_examples: int, invalid_rows: int, duplicate_user_rate: float) -> tuple[str, str]:
    if num_examples == 0:
        return "blocked", "No usable chat SFT examples were found."
    if invalid_rows:
        return "blocked", "Fix invalid chat SFT rows before training."
    if num_examples < 8:
        return "caution", "Chat SFT file is valid but very small."
    if duplicate_user_rate > 0.25:
        return "caution", "Chat SFT file is valid but repeats many user prompts."
    return "ready", "Chat SFT file looks usable for a tiny run."


def _chat_eval_status(
    num_items: int,
    invalid_rows: int,
    total_rules: int,
    unanswerable_items: int,
) -> tuple[str, str]:
    if num_items == 0:
        return "blocked", "No usable eval items were found."
    if invalid_rows:
        return "blocked", "Fix invalid eval rows before scoring."
    if total_rules == 0:
        return "blocked", "Eval items need visible pass/fail rules."
    if num_items < 4:
        return "caution", "Eval file is valid but too small to trust."
    if unanswerable_items == 0:
        return "caution", "Eval file has no unanswerable/refusal checks yet."
    return "ready", "Eval file looks usable for transparent scoring."
