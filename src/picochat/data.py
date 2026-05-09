"""Dataset inspection and simple corpus building utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatch
import json
from pathlib import Path


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".text",
    ".jsonl",
    ".csv",
    ".py",
}

DOCUMENT_EXTENSIONS = {
    ".docx",
    ".pdf",
}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS


class DocumentExtractionError(RuntimeError):
    """Raised when a source document cannot be converted into training text."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CorpusStats:
    num_files: int
    num_documents: int
    num_characters: int
    num_lines: int
    average_document_chars: float
    duplicate_line_rate: float
    non_ascii_rate: float
    empty_line_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusFileRecord:
    path: str
    extension: str
    num_characters: int
    num_lines: int
    included: bool
    reason: str
    label: str | None = None

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusReadinessCheck:
    name: str
    status: str
    metric: str
    threshold: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusReadiness:
    status: str
    summary: str
    checks: tuple[CorpusReadinessCheck, ...]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class CorpusBuildReport:
    input_path: str
    output_path: str
    manifest_path: str
    report_path: str
    stats: CorpusStats
    files: tuple[CorpusFileRecord, ...]
    readiness: CorpusReadiness
    warnings: tuple[str, ...]
    recipe_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "manifest_path": self.manifest_path,
            "report_path": self.report_path,
            "recipe_path": self.recipe_path,
            "stats": self.stats.to_dict(),
            "files": [record.to_dict() for record in self.files],
            "readiness": self.readiness.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CorpusPreviewReport:
    input_path: str
    recipe_path: str | None
    stats: CorpusStats
    files: tuple[CorpusFileRecord, ...]
    readiness: CorpusReadiness
    warnings: tuple[str, ...]
    preview: str

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "recipe_path": self.recipe_path,
            "stats": self.stats.to_dict(),
            "files": [record.to_dict() for record in self.files],
            "readiness": self.readiness.to_dict(),
            "warnings": list(self.warnings),
            "preview": self.preview,
        }


@dataclass(frozen=True)
class _CollectedCorpus:
    input_path: str
    recipe_path: str | None
    documents: tuple[str, ...]
    files: tuple[CorpusFileRecord, ...]
    stats: CorpusStats
    readiness: CorpusReadiness
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _SourceCandidate:
    path: Path
    label: str | None = None
    skipped_reason: str | None = None


def find_text_files(path: str | Path) -> list[Path]:
    """Return readable text-like files under a file or directory path."""
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    files = [
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in TEXT_EXTENSIONS
    ]
    return sorted(files)


def find_corpus_files(path: str | Path) -> list[Path]:
    """Return supported corpus source files under a file or directory path."""
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    files = [
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def read_documents(path: str | Path) -> list[str]:
    """Read one document per supported corpus source file."""
    documents: list[str] = []
    for file_path in find_corpus_files(path):
        try:
            text, _reason = _extract_source_text(file_path)
        except DocumentExtractionError:
            continue
        if text.strip():
            documents.append(text)
    return documents


def inspect_documents(documents: list[str], num_files: int | None = None) -> CorpusStats:
    """Compute simple, explainable corpus quality stats."""
    if not documents:
        return CorpusStats(
            num_files=num_files or 0,
            num_documents=0,
            num_characters=0,
            num_lines=0,
            average_document_chars=0.0,
            duplicate_line_rate=0.0,
            non_ascii_rate=0.0,
            empty_line_rate=0.0,
        )

    all_text = "\n".join(documents)
    lines = all_text.splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    duplicate_lines = len(non_empty_lines) - len(set(non_empty_lines))
    non_ascii_chars = sum(1 for char in all_text if ord(char) > 127)
    empty_lines = sum(1 for line in lines if not line.strip())

    return CorpusStats(
        num_files=num_files if num_files is not None else len(documents),
        num_documents=len(documents),
        num_characters=len(all_text),
        num_lines=len(lines),
        average_document_chars=len(all_text) / len(documents),
        duplicate_line_rate=duplicate_lines / max(1, len(non_empty_lines)),
        non_ascii_rate=non_ascii_chars / max(1, len(all_text)),
        empty_line_rate=empty_lines / max(1, len(lines)),
    )


def inspect_path(path: str | Path) -> CorpusStats:
    files = find_corpus_files(path)
    documents = read_documents(path)
    return inspect_documents(documents, num_files=len(files))


def assess_corpus_readiness(
    stats: CorpusStats,
    records: list[CorpusFileRecord] | tuple[CorpusFileRecord, ...],
) -> CorpusReadiness:
    """Return structured, explainable readiness checks for tiny training."""
    skipped = [record for record in records if not record.included]
    checks = [
        _readiness_check(
            "usable_documents",
            "fail" if stats.num_documents == 0 else "pass",
            str(stats.num_documents),
            ">= 1",
            "At least one supported source needs usable text.",
        ),
        _readiness_check(
            "corpus_size",
            "fail" if stats.num_characters == 0 else "warn" if stats.num_characters < 1000 else "pass",
            f"{stats.num_characters:,} chars",
            ">= 1,000 chars",
            "Very small corpora are useful for smoke tests, but they mostly teach memorization.",
        ),
        _readiness_check(
            "document_mix",
            "warn" if stats.num_documents == 1 else "pass" if stats.num_documents > 1 else "fail",
            str(stats.num_documents),
            ">= 2 for broader training",
            "One document is fine for a focused overfit test; more documents give better variation.",
        ),
        _readiness_check(
            "duplicate_lines",
            "warn" if stats.duplicate_line_rate > 0.15 else "pass",
            f"{stats.duplicate_line_rate * 100:.2f}%",
            "<= 15%",
            "Repeated lines can make a tiny model memorize phrasing instead of learning patterns.",
        ),
        _readiness_check(
            "empty_lines",
            "warn" if stats.empty_line_rate > 0.35 else "pass",
            f"{stats.empty_line_rate * 100:.2f}%",
            "<= 35%",
            "Too many empty lines waste context windows during next-token training.",
        ),
        _readiness_check(
            "non_ascii",
            "warn" if stats.non_ascii_rate > 0.05 else "pass",
            f"{stats.non_ascii_rate * 100:.2f}%",
            "<= 5% unless intentional",
            "High non-ASCII text is fine when expected; otherwise it may indicate extraction noise.",
        ),
        _readiness_check(
            "skipped_sources",
            "warn" if skipped else "pass",
            str(len(skipped)),
            "0 skipped preferred",
            "Skipped files may mean unsupported formats, missing extractors, or recipe exclusions.",
        ),
    ]
    if any(check.status == "fail" for check in checks):
        status = "blocked"
        summary = "Corpus is not trainable yet."
    elif any(check.status == "warn" for check in checks):
        status = "caution"
        summary = "Corpus is trainable, but read the cautions before spending time on training."
    else:
        status = "ready"
        summary = "Corpus looks ready for a tiny training run."
    return CorpusReadiness(status=status, summary=summary, checks=tuple(checks))


def _readiness_check(name: str, status: str, metric: str, threshold: str, message: str) -> CorpusReadinessCheck:
    return CorpusReadinessCheck(
        name=name,
        status=status,
        metric=metric,
        threshold=threshold,
        message=message,
    )


def build_corpus(input_path: str | Path, output_path: str | Path) -> CorpusStats:
    """Combine corpus sources into one normalized text file."""
    return build_corpus_artifacts(input_path, output_path, write_manifest=False).stats


def build_corpus_artifacts(
    input_path: str | Path | None,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
    write_manifest: bool = True,
    recipe_path: str | Path | None = None,
) -> CorpusBuildReport:
    """Combine corpus sources and write provenance artifacts."""
    output_path = Path(output_path)
    manifest_path = Path(manifest_path) if manifest_path else output_path.with_name("corpus_manifest.json")
    report_path = Path(report_path) if report_path else output_path.with_name("corpus_report.md")
    collected = _collect_corpus_sources(input_path, recipe_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_text = "\n\n".join(collected.documents)
    output_path.write_text(corpus_text + ("\n" if corpus_text else ""), encoding="utf-8")

    report = CorpusBuildReport(
        input_path=collected.input_path,
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        report_path=str(report_path),
        stats=collected.stats,
        files=collected.files,
        readiness=collected.readiness,
        warnings=collected.warnings,
        recipe_path=collected.recipe_path,
    )
    if write_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        report_path.write_text(corpus_report_markdown(report), encoding="utf-8")
    return report


def preview_corpus_sources(
    input_path: str | Path | None = None,
    recipe_path: str | Path | None = None,
    preview_chars: int = 1000,
) -> CorpusPreviewReport:
    """Inspect corpus sources without writing artifacts."""
    collected = _collect_corpus_sources(input_path, recipe_path)
    corpus_text = "\n\n".join(collected.documents)
    return CorpusPreviewReport(
        input_path=collected.input_path,
        recipe_path=collected.recipe_path,
        stats=collected.stats,
        files=collected.files,
        readiness=collected.readiness,
        warnings=collected.warnings,
        preview=corpus_text[:max(0, preview_chars)],
    )


def corpus_report_markdown(report: CorpusBuildReport) -> str:
    """Render a human-readable corpus provenance report."""
    included = [record for record in report.files if record.included]
    skipped = [record for record in report.files if not record.included]
    stats = report.stats
    lines = [
        "# Picochat Corpus Report",
        "",
        "This report records which local files were used to build the training corpus.",
        "",
        "## Summary",
        "",
        f"- Input path: `{report.input_path}`",
        f"- Output corpus: `{report.output_path}`",
        f"- Recipe: `{report.recipe_path}`" if report.recipe_path else "- Recipe: none",
        f"- Files scanned: {len(report.files)}",
        f"- Files included: {len(included)}",
        f"- Files skipped: {len(skipped)}",
        f"- Documents: {stats.num_documents}",
        f"- Characters: {stats.num_characters:,}",
        f"- Lines: {stats.num_lines:,}",
        f"- Duplicate line rate: {stats.duplicate_line_rate * 100:.2f}%",
        f"- Empty line rate: {stats.empty_line_rate * 100:.2f}%",
        f"- Non-ASCII rate: {stats.non_ascii_rate * 100:.2f}%",
        "",
        "## Readiness",
        "",
        f"- Status: `{report.readiness.status}`",
        f"- Summary: {report.readiness.summary}",
        "",
        "| Check | Status | Metric | Threshold | Note |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for check in report.readiness.checks:
        lines.append(
            f"| `{check.name}` | `{check.status}` | {check.metric} | `{check.threshold}` | {check.message} |"
        )
    lines.extend([
        "",
        "## Warnings",
        "",
    ])
    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Files",
        "",
        "| File | Label | Ext | Included | Chars | Lines | Reason |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for record in report.files:
        lines.append(
            f"| `{record.path}` | `{record.label or ''}` | `{record.extension}` | {str(record.included).lower()} | "
            f"{record.num_characters} | {record.num_lines} | `{record.reason}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob("*") if item.is_file())


def _collect_corpus_sources(
    input_path: str | Path | None,
    recipe_path: str | Path | None = None,
) -> _CollectedCorpus:
    recipe = Path(recipe_path) if recipe_path else None
    input_root = Path(input_path) if input_path else None
    if recipe is None and input_root is None:
        raise ValueError("Either input_path or recipe_path is required.")

    input_display = str(input_root) if input_root is not None else str(recipe)
    candidates = _recipe_source_candidates(recipe) if recipe else _path_source_candidates(input_root)
    documents: list[str] = []
    records: list[CorpusFileRecord] = []
    for candidate in candidates:
        text, record = _read_source_candidate(candidate)
        records.append(record)
        if text is not None:
            documents.append(text)

    stats = inspect_documents(documents, num_files=len(candidates))
    readiness = assess_corpus_readiness(stats, records)
    warnings = _corpus_warnings(stats, records)
    return _CollectedCorpus(
        input_path=input_display,
        recipe_path=str(recipe) if recipe else None,
        documents=tuple(documents),
        files=tuple(records),
        stats=stats,
        readiness=readiness,
        warnings=tuple(warnings),
    )


def _read_source_candidate(candidate: _SourceCandidate) -> tuple[str | None, CorpusFileRecord]:
    file_path = candidate.path
    extension = file_path.suffix.lower()
    if candidate.skipped_reason:
        return None, CorpusFileRecord(
            path=str(file_path),
            extension=extension or "(none)",
            num_characters=0,
            num_lines=0,
            included=False,
            reason=candidate.skipped_reason,
            label=candidate.label,
        )

    if extension not in SUPPORTED_EXTENSIONS:
        return None, CorpusFileRecord(
            path=str(file_path),
            extension=extension or "(none)",
            num_characters=0,
            num_lines=0,
            included=False,
            reason="unsupported_extension",
            label=candidate.label,
        )

    try:
        text, included_reason = _extract_source_text(file_path)
    except DocumentExtractionError as error:
        return None, CorpusFileRecord(
            path=str(file_path),
            extension=extension,
            num_characters=0,
            num_lines=0,
            included=False,
            reason=error.reason,
            label=candidate.label,
        )

    text = text.strip()
    if not text:
        empty_reason = "empty_text" if extension in TEXT_EXTENSIONS else "empty_extracted_text"
        return None, CorpusFileRecord(
            path=str(file_path),
            extension=extension,
            num_characters=0,
            num_lines=0,
            included=False,
            reason=empty_reason,
            label=candidate.label,
        )

    return text, CorpusFileRecord(
        path=str(file_path),
        extension=extension,
        num_characters=len(text),
        num_lines=len(text.splitlines()),
        included=True,
        reason=included_reason,
        label=candidate.label,
    )


def _path_source_candidates(path: Path | None) -> list[_SourceCandidate]:
    if path is None:
        return []
    return [_SourceCandidate(item) for item in _source_files(path)]


def _recipe_source_candidates(recipe_path: Path) -> list[_SourceCandidate]:
    recipe = _load_recipe(recipe_path)
    recipe_dir = recipe_path.parent
    global_excludes = _string_list(recipe.get("exclude", []), "exclude")
    sources = recipe.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Corpus recipe must define a non-empty 'sources' list.")

    candidates: list[_SourceCandidate] = []
    for index, source in enumerate(sources):
        path_value, label, include, local_excludes, reason = _parse_recipe_source(source, index)
        source_path = Path(path_value)
        if not source_path.is_absolute():
            source_path = recipe_dir / source_path

        if not include:
            candidates.extend(_excluded_candidates(source_path, label, reason))
            continue

        if not source_path.exists():
            candidates.append(_SourceCandidate(source_path, label=label, skipped_reason="missing_source"))
            continue

        excludes = tuple(global_excludes + local_excludes)
        for file_path in _source_files(source_path):
            if _matches_any_pattern(file_path, recipe_dir, excludes):
                candidates.append(_SourceCandidate(file_path, label=label, skipped_reason="recipe_excluded"))
            else:
                candidates.append(_SourceCandidate(file_path, label=label))
    return candidates


def _load_recipe(recipe_path: Path) -> dict:
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid corpus recipe JSON: {error}") from error
    if not isinstance(recipe, dict):
        raise ValueError("Corpus recipe must be a JSON object.")
    return recipe


def _parse_recipe_source(source: object, index: int) -> tuple[str, str | None, bool, list[str], str]:
    if isinstance(source, str):
        return source, None, True, [], "recipe_excluded"
    if not isinstance(source, dict):
        raise ValueError(f"Corpus recipe source {index} must be a string or object.")

    path_value = source.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Corpus recipe source {index} needs a non-empty 'path'.")

    label = source.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"Corpus recipe source {index} label must be a string.")

    include = source.get("include", True)
    if not isinstance(include, bool):
        raise ValueError(f"Corpus recipe source {index} include must be true or false.")

    excludes = _string_list(source.get("exclude", []), f"sources[{index}].exclude")
    reason = source.get("reason", "recipe_excluded")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"Corpus recipe source {index} reason must be a non-empty string.")
    return path_value, label, include, excludes, reason


def _string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Corpus recipe field '{field_name}' must be a list of strings.")
    return value


def _excluded_candidates(path: Path, label: str | None, reason: str) -> list[_SourceCandidate]:
    if path.exists() and path.is_dir():
        return [_SourceCandidate(file_path, label=label, skipped_reason=reason) for file_path in _source_files(path)]
    return [_SourceCandidate(path, label=label, skipped_reason=reason)]


def _matches_any_pattern(file_path: Path, recipe_dir: Path, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    relative_path = _relative_posix(file_path, recipe_dir)
    name = file_path.name
    return any(fnmatch(relative_path, pattern) or fnmatch(name, pattern) for pattern in patterns)


def _relative_posix(file_path: Path, root: Path) -> str:
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def _extract_source_text(file_path: Path) -> tuple[str, str]:
    extension = file_path.suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return file_path.read_text(encoding="utf-8", errors="replace"), "included"
    if extension == ".pdf":
        return _extract_pdf_text(file_path), "included_pdf"
    if extension == ".docx":
        return _extract_docx_text(file_path), "included_docx"
    raise DocumentExtractionError("unsupported_extension")


def _extract_pdf_text(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as import_error:
        raise DocumentExtractionError("missing_pdf_dependency") from import_error

    try:
        reader = PdfReader(str(file_path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as error:
        raise DocumentExtractionError(f"extract_failed:{error.__class__.__name__}") from error
    return "\n\n".join(page for page in pages if page)


def _extract_docx_text(file_path: Path) -> str:
    try:
        from docx import Document
    except ImportError as import_error:
        raise DocumentExtractionError("missing_docx_dependency") from import_error

    try:
        document = Document(str(file_path))
        parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
    except Exception as error:
        raise DocumentExtractionError(f"extract_failed:{error.__class__.__name__}") from error
    return "\n".join(parts)


def _corpus_warnings(stats: CorpusStats, records: list[CorpusFileRecord]) -> list[str]:
    warnings: list[str] = []
    skipped = [record for record in records if not record.included]
    if stats.num_documents == 0:
        warnings.append("No usable text documents were included.")
    if stats.num_characters < 1000:
        warnings.append("Corpus is very small; expect memorization and weak generalization.")
    if stats.num_documents == 1:
        warnings.append("Only one usable document was included; add more sources for broader variation.")
    if stats.duplicate_line_rate > 0.15:
        warnings.append("Duplicate line rate is high; repeated text can encourage memorization.")
    if stats.empty_line_rate > 0.35:
        warnings.append("Empty line rate is high; context windows may be wasted.")
    if stats.non_ascii_rate > 0.05:
        warnings.append("Non-ASCII rate is high; confirm this is expected for the corpus.")
    if skipped:
        warnings.append(f"{len(skipped)} source file(s) were skipped.")
    return warnings
