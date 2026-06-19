"""Optional Hugging Face dataset import utilities."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable


class HFDatasetsMissingError(RuntimeError):
    """Raised when the optional Hugging Face datasets dependency is missing."""


class HFSplitError(ValueError):
    """Raised when the requested Hugging Face split is not available."""

    def __init__(self, dataset: str, requested_split: str, available_splits: list[str], message: str | None = None):
        self.dataset = dataset
        self.requested_split = requested_split
        self.available_splits = available_splits
        detail = message or (
            f"Bad split: {requested_split}. Available splits: {available_splits}"
        )
        super().__init__(detail)


@dataclass(frozen=True)
class HFImportConfig:
    dataset: str
    out_path: str
    config_name: str | None = None
    token: str | None = None
    split: str = "train"
    text_column: str = "text"
    max_rows: int = 1000
    min_chars: int = 20
    streaming: bool = True
    report_path: str | None = None
    documents_dir: str | None = None
    document_shard_rows: int = 1
    data_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class HFImportRow:
    index: int
    included: bool
    reason: str
    num_characters: int
    preview: str
    document_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HFImportDocumentFile:
    path: str
    num_documents: int
    num_characters: int
    num_lines: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HFImportReport:
    dataset: str
    config_name: str | None
    split: str
    text_column: str
    streaming: bool
    max_rows: int
    min_chars: int
    out_path: str
    report_path: str
    documents_dir: str | None
    document_shard_rows: int
    document_files_written: int
    rows_seen: int
    rows_written: int
    rows_skipped: int
    characters_written: int
    rows: tuple[HFImportRow, ...]
    data_files: tuple[str, ...] = ()
    rows_reported: int = 0
    rows_omitted: int = 0
    document_files: tuple[HFImportDocumentFile, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "rows": [row.to_dict() for row in self.rows],
            "document_files": [document.to_dict() for document in self.document_files],
        }


DatasetLoader = Callable[..., Iterable[Any]]
HF_IMPORT_REPORT_ROW_LIMIT = 10_000


def import_hf_dataset(config: HFImportConfig, loader: DatasetLoader | None = None) -> HFImportReport:
    """Import text rows from a Hugging Face dataset into a local corpus file."""
    _validate_config(config)
    out_path = Path(config.out_path)
    report_path = Path(config.report_path) if config.report_path else out_path.with_name("hf_import_report.json")
    documents_dir = Path(config.documents_dir) if config.documents_dir else out_path.parent / "documents"
    dataset = _load_dataset(config, loader)

    rows: list[HFImportRow] = []
    rows_seen = 0
    rows_written = 0
    rows_skipped = 0
    characters_written = 0
    document_counts: dict[Path, int] = {}
    document_file_stats: dict[Path, dict[str, int]] = {}
    document_shard_rows = max(1, config.document_shard_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_document_files(documents_dir)

    with out_path.open("w", encoding="utf-8") as corpus_handle:
        for index, record in enumerate(dataset):
            if index >= config.max_rows:
                break
            rows_seen += 1
            text, reason = _extract_text(record, config.text_column)
            if text is None:
                rows_skipped += 1
                _append_report_row(rows, HFImportRow(
                    index=index,
                    included=False,
                    reason=reason,
                    num_characters=0,
                    preview="",
                ))
                continue
            text = text.strip()
            if len(text) < config.min_chars:
                rows_skipped += 1
                _append_report_row(rows, HFImportRow(
                    index=index,
                    included=False,
                    reason="below_min_chars",
                    num_characters=len(text),
                    preview=_preview(text),
                ))
                continue

            included_index = rows_written
            if document_shard_rows == 1:
                document_path = documents_dir / f"row-{index:06d}.txt"
            else:
                shard_index = included_index // document_shard_rows
                document_path = documents_dir / f"shard-{shard_index:06d}.txt"

            if rows_written:
                corpus_handle.write("\n\n")
                characters_written += 2
            corpus_handle.write(text)
            characters_written += len(text)
            _append_document_file(document_path, text, document_counts, document_file_stats)
            rows_written += 1
            _append_report_row(rows, HFImportRow(
                index=index,
                included=True,
                reason="included",
                num_characters=len(text),
                preview=_preview(text),
                document_path=str(document_path),
            ))
        if rows_written:
            corpus_handle.write("\n")

    _finish_document_files(document_counts)
    document_files_written = len(document_counts)

    report = HFImportReport(
        dataset=config.dataset,
        config_name=config.config_name,
        split=config.split,
        text_column=config.text_column,
        streaming=config.streaming,
        max_rows=config.max_rows,
        min_chars=config.min_chars,
        out_path=str(out_path),
        report_path=str(report_path),
        documents_dir=str(documents_dir),
        document_shard_rows=document_shard_rows,
        document_files_written=document_files_written,
        data_files=tuple(config.data_files),
        rows_seen=rows_seen,
        rows_written=rows_written,
        rows_skipped=rows_skipped,
        characters_written=characters_written,
        rows=tuple(rows),
        rows_reported=len(rows),
        rows_omitted=max(0, rows_seen - len(rows)),
        document_files=tuple(
            HFImportDocumentFile(
                path=str(path),
                num_documents=stats["num_documents"],
                num_characters=stats["num_characters"],
                num_lines=stats["num_lines"],
            )
            for path, stats in document_file_stats.items()
        ),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    report_path.with_suffix(".md").write_text(hf_import_markdown(report), encoding="utf-8")
    return report


def hf_import_markdown(report: HFImportReport) -> str:
    """Render a small human-readable report for the imported dataset sample."""
    lines = [
        "# Picochat Hugging Face Import Report",
        "",
        f"- Dataset: `{report.dataset}`",
        f"- Config: `{report.config_name}`" if report.config_name else "- Config: none",
        f"- Split: `{report.split}`",
        f"- Text column: `{report.text_column}`",
        f"- Streaming: `{report.streaming}`",
        f"- Data files: `{', '.join(report.data_files)}`" if report.data_files else "- Data files: none",
        f"- Rows inspected: {report.rows_seen}",
        f"- Rows written: {report.rows_written}",
        f"- Rows skipped: {report.rows_skipped}",
        f"- Rows reported: {report.rows_reported}",
        f"- Rows omitted from report: {report.rows_omitted}",
        f"- Characters written: {report.characters_written:,}",
        f"- Output corpus: `{report.out_path}`",
        f"- Output documents: `{report.documents_dir}`" if report.documents_dir else "- Output documents: none",
        f"- Document shard rows: {report.document_shard_rows}",
        f"- Document files written: {report.document_files_written}",
        f"- Document file metadata rows: {len(report.document_files)}",
        "",
        "## Rows",
        "",
        "| Index | Included | Chars | Reason | Document | Preview |",
        "| ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in report.rows[:20]:
        preview = row.preview.replace("|", "\\|")
        document_path = row.document_path or ""
        lines.append(
            f"| {row.index} | {str(row.included).lower()} | {row.num_characters} | "
            f"`{row.reason}` | `{document_path}` | {preview} |"
        )
    if len(report.rows) > 20:
        lines.append(f"| ... | ... | ... | ... | ... | {len(report.rows) - 20} more reported row(s) omitted |")
    if report.rows_omitted:
        lines.append(f"| ... | ... | ... | ... | ... | {report.rows_omitted} additional row(s) omitted from the JSON report |")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("Preview the imported documents before training so Picochat can see row-level document boundaries:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"picochat data preview --input {report.documents_dir or report.out_path}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _load_dataset(config: HFImportConfig, loader: DatasetLoader | None) -> Iterable[Any]:
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise HFDatasetsMissingError(
                "Hugging Face import requires the optional dependency: "
                "pip install -e '.[hf]'"
            ) from error
        loader = load_dataset

    args = [config.dataset]
    if config.config_name:
        args.append(config.config_name)
    kwargs = {"split": config.split, "streaming": config.streaming}
    if config.token:
        kwargs["token"] = config.token
    if config.data_files:
        kwargs["data_files"] = list(config.data_files)
    try:
        return loader(*args, **kwargs)
    except Exception as error:
        split_error = _split_error_from_exception(config, error)
        if split_error:
            raise split_error from error
        raise


def _split_error_from_exception(config: HFImportConfig, error: Exception) -> HFSplitError | None:
    message = str(error)
    if "Available splits:" not in message:
        return None
    match = re.search(r"Available splits:\s*(\[[^\]]*\])", message)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    return HFSplitError(
        dataset=config.dataset,
        requested_split=config.split,
        available_splits=parsed,
        message=message,
    )


def _extract_text(record: Any, text_column: str) -> tuple[str | None, str]:
    if not isinstance(record, dict):
        return None, "row_not_object"
    if text_column not in record:
        return None, "missing_text_column"
    value = record[text_column]
    if not isinstance(value, str):
        return None, "text_column_not_string"
    if not value.strip():
        return None, "empty_text"
    return value, "included"


def _validate_config(config: HFImportConfig) -> None:
    if not config.dataset.strip():
        raise ValueError("dataset is required")
    if not config.out_path.strip():
        raise ValueError("out_path is required")
    if not config.text_column.strip():
        raise ValueError("text_column is required")
    if config.max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if config.min_chars < 1:
        raise ValueError("min_chars must be at least 1")
    if config.document_shard_rows < 1:
        raise ValueError("document_shard_rows must be at least 1")


def _append_report_row(rows: list[HFImportRow], row: HFImportRow) -> None:
    if len(rows) < HF_IMPORT_REPORT_ROW_LIMIT:
        rows.append(row)


def _clear_document_files(documents_dir: Path) -> None:
    documents_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("row-*.txt", "shard-*.txt"):
        for stale_path in documents_dir.glob(pattern):
            if stale_path.is_file():
                stale_path.unlink()


def _append_document_file(
    document_path: Path,
    text: str,
    document_counts: dict[Path, int],
    document_file_stats: dict[Path, dict[str, int]],
) -> None:
    document_path.parent.mkdir(parents=True, exist_ok=True)
    count = document_counts.get(document_path, 0)
    with document_path.open("a" if count else "w", encoding="utf-8") as handle:
        if count:
            handle.write("\n\n")
        handle.write(text)
    document_counts[document_path] = count + 1
    stats = document_file_stats.setdefault(
        document_path,
        {"num_documents": 0, "num_characters": 0, "num_lines": 0},
    )
    if count:
        stats["num_characters"] += 2
        stats["num_lines"] += 1
    stats["num_documents"] += 1
    stats["num_characters"] += len(text)
    stats["num_lines"] += len(text.splitlines())


def _finish_document_files(document_counts: dict[Path, int]) -> None:
    for document_path in document_counts:
        with document_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")


def _preview(text: str, limit: int = 90) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."
