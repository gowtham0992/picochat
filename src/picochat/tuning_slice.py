"""Create auditable tuning-pack slices for staged SFT experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import fnmatch
import json
import os
from pathlib import Path
from typing import Any

from picochat.dataset_pack import load_dataset_pack
from picochat.tuning_data import inspect_chat_eval_data, inspect_chat_sft_data


@dataclass(frozen=True)
class TuningSliceReport:
    source_dataset_pack: str
    dataset_pack: str
    out_dir: str
    chat_input: str
    eval_input: str
    report_path: str
    json_report_path: str
    include_categories: tuple[str, ...]
    exclude_categories: tuple[str, ...]
    chat_rows_in: int
    chat_rows_out: int
    eval_rows_in: int
    eval_rows_out: int
    chat_categories: dict[str, int]
    eval_categories: dict[str, int]
    sft_status: str
    eval_status: str
    created: tuple[str, ...]
    overwritten: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "include_categories": list(self.include_categories),
            "exclude_categories": list(self.exclude_categories),
            "created": list(self.created),
            "overwritten": list(self.overwritten),
        }


def slice_tuning_pack(
    dataset_pack: str | Path,
    out_dir: str | Path,
    *,
    include_categories: tuple[str, ...] = (),
    exclude_categories: tuple[str, ...] = (),
    name: str | None = None,
    description: str | None = None,
    force: bool = False,
) -> TuningSliceReport:
    """Write a dataset pack whose chat/eval files contain only selected categories."""
    include_categories = _normalize_patterns(include_categories)
    exclude_categories = _normalize_patterns(exclude_categories)
    if not include_categories and not exclude_categories:
        raise ValueError("at least one include or exclude category pattern is required")

    pack = load_dataset_pack(dataset_pack)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chat_path = out_dir / "chat_slice.jsonl"
    eval_path = out_dir / "eval_slice.jsonl"
    pack_path = out_dir / "dataset_pack.json"
    report_path = out_dir / "tuning_slice.md"
    json_report_path = out_dir / "tuning_slice.json"
    targets = (chat_path, eval_path, pack_path, report_path, json_report_path)
    existing = tuple(str(path) for path in targets if path.exists())
    if existing and not force:
        raise FileExistsError(f"Refusing to overwrite existing tuning-slice file(s): {', '.join(existing)}")

    chat_rows = _read_jsonl(Path(pack.chat_input), "chat")
    eval_rows = _read_jsonl(Path(pack.eval_input), "eval")
    chat_selected = _filter_rows(chat_rows, include_categories, exclude_categories)
    eval_selected = _filter_rows(eval_rows, include_categories, exclude_categories)
    if not chat_selected:
        raise ValueError("tuning slice selected zero chat rows")
    if not eval_selected:
        raise ValueError("tuning slice selected zero eval rows")

    _write_jsonl(chat_path, chat_selected)
    _write_jsonl(eval_path, eval_selected)
    pack_payload = {
        "name": name or f"{pack.name} tuning slice",
        "description": description or (
            f"Category-sliced tuning pack derived from {pack.path}. "
            "Corpus is unchanged; chat/eval rows are copied only from their original split."
        ),
        "corpus": _corpus_payload_for_slice(pack, out_dir),
        "chat": "chat_slice.jsonl",
        "eval": "eval_slice.jsonl",
        "source_dataset_pack": _relative_path(pack.path, out_dir),
        "slice": {
            "include_categories": list(include_categories),
            "exclude_categories": list(exclude_categories),
        },
    }
    pack_path.write_text(json.dumps(pack_payload, indent=2) + "\n", encoding="utf-8")

    sft_report = inspect_chat_sft_data(chat_path, preview_items=0)
    eval_report = inspect_chat_eval_data(eval_path, preview_items=0)
    report = TuningSliceReport(
        source_dataset_pack=str(dataset_pack),
        dataset_pack=str(pack_path),
        out_dir=str(out_dir),
        chat_input=str(chat_path),
        eval_input=str(eval_path),
        report_path=str(report_path),
        json_report_path=str(json_report_path),
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        chat_rows_in=len(chat_rows),
        chat_rows_out=len(chat_selected),
        eval_rows_in=len(eval_rows),
        eval_rows_out=len(eval_selected),
        chat_categories=dict(sorted(Counter(_category(row) for row in chat_selected).items())),
        eval_categories=dict(sorted(Counter(_category(row) for row in eval_selected).items())),
        sft_status=sft_report.status,
        eval_status=eval_report.status,
        created=tuple(str(path) for path in targets if str(path) not in existing),
        overwritten=tuple(str(path) for path in targets if str(path) in existing),
    )
    json_report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    report_path.write_text(tuning_slice_markdown(report), encoding="utf-8")
    return report


def tuning_slice_markdown(report: TuningSliceReport) -> str:
    """Render a compact human-readable tuning-slice report."""
    lines = [
        "# Picochat Tuning Slice",
        "",
        "This pack copies selected chat/eval categories into a separate dataset pack. "
        "It does not move rows between train and eval, and it does not alter the corpus.",
        "",
        "## Inputs",
        "",
        f"- Source dataset pack: `{report.source_dataset_pack}`",
        f"- Include categories: `{', '.join(report.include_categories) or 'all'}`",
        f"- Exclude categories: `{', '.join(report.exclude_categories) or 'none'}`",
        "",
        "## Outputs",
        "",
        f"- Dataset pack: `{report.dataset_pack}`",
        f"- Chat SFT: `{report.chat_input}`",
        f"- Eval: `{report.eval_input}`",
        f"- Chat rows: {report.chat_rows_out} / {report.chat_rows_in}",
        f"- Eval rows: {report.eval_rows_out} / {report.eval_rows_in}",
        f"- Chat status: `{report.sft_status}`",
        f"- Eval status: `{report.eval_status}`",
        "",
        "## Chat Categories",
        "",
        *_category_lines(report.chat_categories),
        "",
        "## Eval Categories",
        "",
        *_category_lines(report.eval_categories),
        "",
    ]
    return "\n".join(lines)


def parse_category_patterns(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated category pattern list."""
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _normalize_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pattern.strip() for pattern in patterns if pattern and pattern.strip())


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing {label} JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} must be a JSON object")
        rows.append(row)
    return rows


def _filter_rows(
    rows: list[dict[str, Any]],
    include_categories: tuple[str, ...],
    exclude_categories: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        category = _category(row)
        if include_categories and not _matches_any(category, include_categories):
            continue
        if exclude_categories and _matches_any(category, exclude_categories):
            continue
        selected.append(row)
    return selected


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _category(row: dict[str, Any]) -> str:
    value = row.get("category", "chat")
    return value.strip() if isinstance(value, str) and value.strip() else "chat"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _corpus_payload_for_slice(pack, out_dir: Path) -> dict[str, str]:
    if pack.corpus_recipe:
        return {"recipe": _relative_path(pack.corpus_recipe, out_dir)}
    if pack.corpus_input:
        return {"input": _relative_path(pack.corpus_input, out_dir)}
    raise ValueError("source dataset pack has no corpus path")


def _relative_path(path: str | Path, base_dir: Path) -> str:
    source = Path(path)
    target = source if source.is_absolute() else Path.cwd() / source
    try:
        relative = os.path.relpath(target.resolve(strict=False), start=base_dir.resolve(strict=False))
    except ValueError:
        return str(target.resolve(strict=False))
    return Path(relative).as_posix()


def _category_lines(categories: dict[str, int]) -> list[str]:
    if not categories:
        return ["- none"]
    return [f"- `{name}`: {count}" for name, count in categories.items()]
