"""Data honesty checks for Picochat experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any


TEXT_EXTENSIONS = {".txt", ".text", ".md", ".jsonl", ".csv", ".py"}
GENERIC_REFUSAL_SUPPORT_PHRASES = {
    "i do not know",
    "not enough information",
    "cannot answer",
    "provided material",
    "do not know from the provided material",
    "cannot answer from the provided material",
    "not enough information in the provided material",
}


@dataclass(frozen=True)
class HonestyFinding:
    kind: str
    severity: str
    message: str
    eval_line: int | None = None
    eval_category: str | None = None
    matched_line: int | None = None
    matched_source: str | None = None
    similarity: float | None = None
    snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataHonestyReport:
    status: str
    summary: str
    corpus_path: str | None
    chat_input: str
    eval_input: str
    num_sft_examples: int
    num_eval_items: int
    exact_prompt_leaks: int
    near_prompt_leaks: int
    corpus_prompt_hits: int
    sft_support_phrase_hits: int
    corpus_support_phrase_hits: int
    duplicate_eval_prompts: int
    max_sft_prompt_similarity: float
    contamination_matrix: dict[str, Any]
    findings: tuple[HonestyFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class _TextRecord:
    source: str
    kind: str
    line: int | None
    category: str | None
    text: str
    normalized: str
    tokens: tuple[str, ...]
    ngrams: frozenset[tuple[str, ...]]


def inspect_data_honesty(
    chat_input: str | Path,
    eval_input: str | Path,
    corpus_path: str | Path | None = None,
    near_threshold: float = 0.86,
    generated_texts: list[str] | None = None,
    ngram_size: int = 8,
) -> DataHonestyReport:
    """Check whether an eval is visibly contaminated by SFT or corpus text."""
    chat_rows = _read_chat_rows(chat_input)
    eval_rows = _read_eval_rows(eval_input)
    corpus_text = _read_corpus_text(corpus_path) if corpus_path else ""
    normalized_corpus = _normalize(corpus_text)

    findings: list[HonestyFinding] = []
    exact_prompt_leaks = 0
    near_prompt_leaks = 0
    corpus_prompt_hits = 0
    sft_support_phrase_hits = 0
    corpus_support_phrase_hits = 0
    duplicate_eval_prompts = 0
    max_similarity = 0.0
    normalized_assistant_rows = [
        (chat_row, _normalize(chat_row["assistant"]))
        for chat_row in chat_rows
    ]

    seen_eval_prompts: dict[str, int] = {}
    for eval_row in eval_rows:
        normalized_prompt = _normalize(eval_row["user"])
        if normalized_prompt in seen_eval_prompts:
            duplicate_eval_prompts += 1
            findings.append(HonestyFinding(
                kind="duplicate_eval_prompt",
                severity="warn",
                eval_line=eval_row["line"],
                eval_category=eval_row["category"],
                matched_line=seen_eval_prompts[normalized_prompt],
                matched_source=str(eval_input),
                message="Eval prompt is duplicated; repeated prompts can inflate or distort the score.",
                snippet=_preview(eval_row["user"]),
            ))
        else:
            seen_eval_prompts[normalized_prompt] = eval_row["line"]

        best_chat_row = None
        best_similarity = 0.0
        for chat_row in chat_rows:
            similarity = _prompt_similarity(normalized_prompt, _normalize(chat_row["user"]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_chat_row = chat_row

        max_similarity = max(max_similarity, best_similarity)
        if best_chat_row and normalized_prompt == _normalize(best_chat_row["user"]):
            exact_prompt_leaks += 1
            findings.append(HonestyFinding(
                kind="exact_sft_prompt_leak",
                severity="fail",
                eval_line=eval_row["line"],
                eval_category=eval_row["category"],
                matched_line=best_chat_row["line"],
                matched_source=str(chat_input),
                similarity=1.0,
                message="Eval prompt exactly matches a chat SFT prompt; this score should not be trusted.",
                snippet=_preview(eval_row["user"]),
            ))
        elif best_chat_row and best_similarity >= near_threshold:
            near_prompt_leaks += 1
            findings.append(HonestyFinding(
                kind="near_sft_prompt_leak",
                severity="warn",
                eval_line=eval_row["line"],
                eval_category=eval_row["category"],
                matched_line=best_chat_row["line"],
                matched_source=str(chat_input),
                similarity=best_similarity,
                message="Eval prompt is very similar to a chat SFT prompt; inspect before trusting the score.",
                snippet=_preview(eval_row["user"]),
            ))

        if (
            normalized_corpus
            and len(normalized_prompt) >= 24
            and normalized_prompt in normalized_corpus
        ):
            corpus_prompt_hits += 1
            findings.append(HonestyFinding(
                kind="eval_prompt_in_corpus",
                severity="warn",
                eval_line=eval_row["line"],
                eval_category=eval_row["category"],
                matched_source=str(corpus_path),
                message="Eval prompt text appears inside the base corpus; this may measure recall instead of generalization.",
                snippet=_preview(eval_row["user"]),
            ))

        for phrase in eval_row["support_phrases"]:
            normalized_phrase = _normalize(phrase)
            if not _is_specific_support_phrase(normalized_phrase):
                continue
            assistant_hit = next(
                (
                    chat_row
                    for chat_row, normalized_assistant in normalized_assistant_rows
                    if normalized_phrase in normalized_assistant
                ),
                None,
            )
            if assistant_hit is not None:
                sft_support_phrase_hits += 1
                findings.append(HonestyFinding(
                    kind="eval_support_phrase_in_sft",
                    severity="warn",
                    eval_line=eval_row["line"],
                    eval_category=eval_row["category"],
                    matched_line=assistant_hit["line"],
                    matched_source=str(chat_input),
                    message=(
                        "A specific eval support phrase appears in a chat SFT answer; "
                        "this can make phrase-based evals easier to pass by memorization."
                    ),
                    snippet=_preview(phrase),
                ))

            if normalized_corpus and normalized_phrase in normalized_corpus:
                corpus_support_phrase_hits += 1
                findings.append(HonestyFinding(
                    kind="eval_support_phrase_in_corpus",
                    severity="warn",
                    eval_line=eval_row["line"],
                    eval_category=eval_row["category"],
                    matched_source=str(corpus_path),
                    message=(
                        "A specific eval support phrase appears in the base corpus; "
                        "inspect whether the eval is measuring recall instead of behavior."
                    ),
                    snippet=_preview(phrase),
                ))

    contamination_matrix = _build_contamination_matrix(
        chat_rows=chat_rows,
        eval_rows=eval_rows,
        corpus_text=corpus_text,
        generated_texts=generated_texts or [],
        near_threshold=near_threshold,
        ngram_size=ngram_size,
    )
    status = _status(findings)
    return DataHonestyReport(
        status=status,
        summary=_summary(status, findings),
        corpus_path=str(corpus_path) if corpus_path else None,
        chat_input=str(chat_input),
        eval_input=str(eval_input),
        num_sft_examples=len(chat_rows),
        num_eval_items=len(eval_rows),
        exact_prompt_leaks=exact_prompt_leaks,
        near_prompt_leaks=near_prompt_leaks,
        corpus_prompt_hits=corpus_prompt_hits,
        sft_support_phrase_hits=sft_support_phrase_hits,
        corpus_support_phrase_hits=corpus_support_phrase_hits,
        duplicate_eval_prompts=duplicate_eval_prompts,
        max_sft_prompt_similarity=max_similarity,
        contamination_matrix=contamination_matrix,
        findings=tuple(findings),
    )


def write_data_honesty_report(report: DataHonestyReport, out_dir: str | Path) -> tuple[str, str]:
    """Write JSON and Markdown honesty reports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "honesty_report.json"
    markdown_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(data_honesty_markdown(report), encoding="utf-8")
    return str(json_path), str(markdown_path)


def data_honesty_markdown(report: DataHonestyReport) -> str:
    """Render a human-readable anti-cheating report."""
    lines = [
        "# Picochat Data Honesty Report",
        "",
        "This report checks whether the eval can be trusted. It does not prove the model is honest; "
        "it looks for obvious leakage between chat SFT data, eval prompts, and the base corpus.",
        "",
        "## Summary",
        "",
        f"- Status: `{report.status}`",
        f"- Summary: {report.summary}",
        f"- Chat SFT: `{report.chat_input}`",
        f"- Eval: `{report.eval_input}`",
        f"- Corpus: `{report.corpus_path}`" if report.corpus_path else "- Corpus: none",
        f"- SFT examples: {report.num_sft_examples}",
        f"- Eval items: {report.num_eval_items}",
        f"- Exact SFT prompt leaks: {report.exact_prompt_leaks}",
        f"- Near SFT prompt leaks: {report.near_prompt_leaks}",
        f"- Eval prompts found in corpus: {report.corpus_prompt_hits}",
        f"- Specific eval support phrases found in SFT answers: {report.sft_support_phrase_hits}",
        f"- Specific eval support phrases found in corpus: {report.corpus_support_phrase_hits}",
        f"- Duplicate eval prompts: {report.duplicate_eval_prompts}",
        f"- Max SFT/eval prompt similarity: {report.max_sft_prompt_similarity:.4f}",
        "",
        "## Contamination Matrix",
        "",
    ]
    matrix = report.contamination_matrix or {}
    pairs = matrix.get("pairs", [])
    if not pairs:
        lines.append("No contamination matrix was recorded.")
        lines.append("")
    else:
        lines.append(f"- N-gram size: {matrix.get('ngram_size', 8)}")
        lines.append(f"- Near-text threshold: {float(matrix.get('near_threshold', 0.0)):.4f}")
        lines.append("")
        lines.extend([
            "| Pair | Risk | Checked | Exact text hits | Near text hits | "
            "Max n-gram overlap | Longest overlap tokens |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for pair in pairs:
            checked = "yes" if pair.get("checked") else f"no: {pair.get('reason', 'not checked')}"
            lines.append(
                f"| `{_escape_table(str(pair.get('name', 'unknown')))}` | "
                f"`{_escape_table(str(pair.get('risk', 'unknown')))}` | "
                f"{_escape_table(checked)} | "
                f"{int(pair.get('exact_text_hits', 0))} | "
                f"{int(pair.get('near_text_hits', 0))} | "
                f"{float(pair.get('max_ngram_overlap_rate', 0.0)):.4f} | "
                f"{int(pair.get('max_longest_overlap_tokens', 0))} |"
            )
        lines.append("")
        for pair in pairs:
            samples = pair.get("nearest_neighbors", [])
            if not samples:
                continue
            lines.append(f"### `{pair.get('name', 'unknown')}` nearest neighbors")
            lines.append("")
            lines.extend([
                "| Target | Reference | Overlap | Longest | Preview |",
                "| --- | --- | ---: | ---: | --- |",
            ])
            for sample in samples[:3]:
                target = _record_label(
                    sample.get("target_source"),
                    sample.get("target_kind"),
                    sample.get("target_line"),
                )
                reference = _record_label(
                    sample.get("reference_source"),
                    sample.get("reference_kind"),
                    sample.get("reference_line"),
                )
                preview = sample.get("overlap_preview") or sample.get("target_preview") or ""
                lines.append(
                    f"| `{_escape_table(target)}` | `{_escape_table(reference)}` | "
                    f"{float(sample.get('ngram_overlap_rate', 0.0)):.4f} | "
                    f"{int(sample.get('longest_overlap_tokens', 0))} | "
                    f"{_escape_table(str(preview))} |"
                )
            lines.append("")

    lines.extend([
        "## Findings",
        "",
    ])
    if not report.findings:
        lines.append("No obvious leakage findings.")
        lines.append("")
    else:
        lines.extend([
            "| Severity | Kind | Eval line | Matched source | Similarity | Message |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ])
        for finding in report.findings:
            similarity = "" if finding.similarity is None else f"{finding.similarity:.4f}"
            lines.append(
                f"| `{finding.severity}` | `{finding.kind}` | "
                f"{finding.eval_line if finding.eval_line is not None else ''} | "
                f"`{_escape_table(finding.matched_source or '')}` | {similarity} | "
                f"{_escape_table(finding.message)} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- `ready` means no obvious text leakage was detected by these simple checks.",
        "- `caution` means the run can proceed, but the score needs manual inspection.",
        "- `blocked` means at least one eval prompt appears to be directly present in SFT data.",
        "",
    ])
    return "\n".join(lines)


def _build_contamination_matrix(
    chat_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    corpus_text: str,
    generated_texts: list[str],
    near_threshold: float,
    ngram_size: int,
) -> dict[str, Any]:
    ngram_size = max(1, int(ngram_size))
    corpus_records = (
        [_make_record("base_corpus", "corpus", corpus_text, ngram_size)]
        if corpus_text.strip()
        else []
    )
    sft_records = _chat_text_records(chat_rows, ngram_size)
    eval_records = _eval_text_records(eval_rows, ngram_size)
    generated_records = [
        _make_record("generated_answers", "reply", text, ngram_size, line=index)
        for index, text in enumerate(generated_texts, start=1)
        if isinstance(text, str) and text.strip()
    ]
    pairs = [
        _matrix_pair(
            name="base_corpus_vs_sft",
            reference_label="base_corpus",
            target_label="chat_sft",
            reference_records=corpus_records,
            target_records=sft_records,
            near_threshold=near_threshold,
            ngram_size=ngram_size,
        ),
        _matrix_pair(
            name="base_corpus_vs_eval",
            reference_label="base_corpus",
            target_label="eval",
            reference_records=corpus_records,
            target_records=eval_records,
            near_threshold=near_threshold,
            ngram_size=ngram_size,
        ),
        _matrix_pair(
            name="sft_vs_eval",
            reference_label="chat_sft",
            target_label="eval",
            reference_records=sft_records,
            target_records=eval_records,
            near_threshold=near_threshold,
            ngram_size=ngram_size,
        ),
        _matrix_pair(
            name="generated_vs_sft",
            reference_label="chat_sft",
            target_label="generated_answers",
            reference_records=sft_records,
            target_records=generated_records,
            near_threshold=near_threshold,
            ngram_size=ngram_size,
        ),
        _matrix_pair(
            name="generated_vs_base_corpus",
            reference_label="base_corpus",
            target_label="generated_answers",
            reference_records=corpus_records,
            target_records=generated_records,
            near_threshold=near_threshold,
            ngram_size=ngram_size,
        ),
    ]
    return {
        "ngram_size": ngram_size,
        "near_threshold": near_threshold,
        "pairs": pairs,
    }


def _chat_text_records(chat_rows: list[dict[str, Any]], ngram_size: int) -> list[_TextRecord]:
    records: list[_TextRecord] = []
    for row in chat_rows:
        records.append(_make_record(
            "chat_sft",
            "prompt",
            row["user"],
            ngram_size,
            line=row["line"],
            category=row["category"],
        ))
        records.append(_make_record(
            "chat_sft",
            "answer",
            row["assistant"],
            ngram_size,
            line=row["line"],
            category=row["category"],
        ))
    return records


def _eval_text_records(eval_rows: list[dict[str, Any]], ngram_size: int) -> list[_TextRecord]:
    records: list[_TextRecord] = []
    for row in eval_rows:
        records.append(_make_record(
            "eval",
            "prompt",
            row["user"],
            ngram_size,
            line=row["line"],
            category=row["category"],
        ))
        for phrase in row["support_phrases"]:
            normalized_phrase = _normalize(phrase)
            if _is_specific_support_phrase(normalized_phrase):
                records.append(_make_record(
                    "eval",
                    "support_phrase",
                    phrase,
                    ngram_size,
                    line=row["line"],
                    category=row["category"],
                ))
    return records


def _make_record(
    source: str,
    kind: str,
    text: str,
    ngram_size: int,
    line: int | None = None,
    category: str | None = None,
) -> _TextRecord:
    tokens = tuple(_tokens(text))
    return _TextRecord(
        source=source,
        kind=kind,
        line=line,
        category=category,
        text=text,
        normalized=_normalize(text),
        tokens=tokens,
        ngrams=frozenset(_ngrams(tokens, ngram_size)),
    )


def _matrix_pair(
    name: str,
    reference_label: str,
    target_label: str,
    reference_records: list[_TextRecord],
    target_records: list[_TextRecord],
    near_threshold: float,
    ngram_size: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "reference": reference_label,
        "target": target_label,
        "reference_items": len(reference_records),
        "target_items": len(target_records),
        "exact_text_hits": 0,
        "near_text_hits": 0,
        "max_text_similarity": 0.0,
        "max_ngram_overlap_rate": 0.0,
        "max_longest_overlap_tokens": 0,
        "nearest_neighbors": [],
    }
    if not reference_records:
        return {
            **base,
            "checked": False,
            "reason": f"no {reference_label} records",
            "risk": "not_checked",
        }
    if not target_records:
        return {
            **base,
            "checked": False,
            "reason": f"no {target_label} records",
            "risk": "not_checked",
        }

    exact_hits = 0
    near_hits = 0
    max_similarity = 0.0
    max_overlap_rate = 0.0
    max_longest = 0
    samples: list[dict[str, Any]] = []
    for target in target_records:
        best_match: dict[str, Any] | None = None
        target_exact_hit = False
        target_near_hit = False
        for reference in reference_records:
            match = _record_match(reference, target, near_threshold, ngram_size)
            target_exact_hit = target_exact_hit or match["exact_text_hit"]
            target_near_hit = target_near_hit or match["near_text_hit"]
            if best_match is None or _match_key(match) > _match_key(best_match):
                best_match = match
        if best_match is None:
            continue
        if target_exact_hit:
            exact_hits += 1
        if target_near_hit:
            near_hits += 1
        max_similarity = max(max_similarity, best_match["text_similarity"])
        max_overlap_rate = max(max_overlap_rate, best_match["ngram_overlap_rate"])
        max_longest = max(max_longest, best_match["longest_overlap_tokens"])
        if _interesting_match(best_match, ngram_size):
            samples.append(_sample_match(best_match))

    samples.sort(key=lambda item: (
        int(item.get("exact_text_hit", False)),
        float(item.get("ngram_overlap_rate", 0.0)),
        int(item.get("longest_overlap_tokens", 0)),
        float(item.get("text_similarity", 0.0)),
    ), reverse=True)
    return {
        **base,
        "checked": True,
        "risk": _matrix_risk(exact_hits, near_hits, max_overlap_rate, max_longest, ngram_size),
        "exact_text_hits": exact_hits,
        "near_text_hits": near_hits,
        "max_text_similarity": round(max_similarity, 6),
        "max_ngram_overlap_rate": round(max_overlap_rate, 6),
        "max_longest_overlap_tokens": max_longest,
        "nearest_neighbors": samples[:5],
    }


def _record_match(
    reference: _TextRecord,
    target: _TextRecord,
    near_threshold: float,
    ngram_size: int,
) -> dict[str, Any]:
    exact_text_hit = _exact_text_hit(reference.normalized, target.normalized)
    text_similarity = 0.0
    if _can_compare_similarity(reference.normalized, target.normalized):
        text_similarity = _prompt_similarity(target.normalized, reference.normalized)
    elif exact_text_hit:
        text_similarity = 1.0
    overlap_rate = _ngram_overlap_rate(reference, target)
    longest_overlap, overlap_preview = _longest_overlap(target.tokens, reference.normalized)
    near_text_hit = (
        not exact_text_hit
        and len(target.normalized) >= 24
        and text_similarity >= near_threshold
    )
    return {
        "reference": reference,
        "target": target,
        "exact_text_hit": exact_text_hit,
        "near_text_hit": near_text_hit,
        "text_similarity": text_similarity,
        "ngram_overlap_rate": overlap_rate,
        "longest_overlap_tokens": longest_overlap,
        "overlap_preview": overlap_preview,
        "ngram_size": ngram_size,
    }


def _sample_match(match: dict[str, Any]) -> dict[str, Any]:
    reference = match["reference"]
    target = match["target"]
    return {
        "target_source": target.source,
        "target_kind": target.kind,
        "target_line": target.line,
        "target_category": target.category,
        "reference_source": reference.source,
        "reference_kind": reference.kind,
        "reference_line": reference.line,
        "reference_category": reference.category,
        "exact_text_hit": match["exact_text_hit"],
        "near_text_hit": match["near_text_hit"],
        "text_similarity": round(match["text_similarity"], 6),
        "ngram_overlap_rate": round(match["ngram_overlap_rate"], 6),
        "longest_overlap_tokens": match["longest_overlap_tokens"],
        "overlap_preview": _preview(match["overlap_preview"]) if match["overlap_preview"] else None,
        "target_preview": _preview(target.text),
        "reference_preview": _preview(reference.text),
    }


def _match_key(match: dict[str, Any]) -> tuple[int, float, int, float]:
    return (
        int(match["exact_text_hit"]),
        float(match["ngram_overlap_rate"]),
        int(match["longest_overlap_tokens"]),
        float(match["text_similarity"]),
    )


def _interesting_match(match: dict[str, Any], ngram_size: int) -> bool:
    return (
        match["exact_text_hit"]
        or match["near_text_hit"]
        or match["ngram_overlap_rate"] > 0.0
        or match["longest_overlap_tokens"] >= ngram_size
    )


def _matrix_risk(
    exact_hits: int,
    near_hits: int,
    max_overlap_rate: float,
    max_longest: int,
    ngram_size: int,
) -> str:
    if (
        exact_hits
        or max_overlap_rate >= 0.80
        or max_longest >= max(40, ngram_size * 5)
    ):
        return "high"
    if near_hits or max_overlap_rate >= 0.35 or max_longest >= max(16, ngram_size * 2):
        return "medium"
    if max_overlap_rate > 0.0 or max_longest >= ngram_size:
        return "low"
    return "clean"


def _exact_text_hit(
    reference_normalized: str,
    target_normalized: str,
    min_chars: int = 24,
) -> bool:
    if len(target_normalized) < min_chars or not reference_normalized:
        return False
    if target_normalized == reference_normalized:
        return True
    return f" {target_normalized} " in f" {reference_normalized} "


def _can_compare_similarity(reference_normalized: str, target_normalized: str) -> bool:
    if len(target_normalized) < 24 or len(reference_normalized) < 24:
        return False
    return max(len(reference_normalized), len(target_normalized)) <= 2000


def _ngram_overlap_rate(reference: _TextRecord, target: _TextRecord) -> float:
    if not target.ngrams:
        return 0.0
    return len(target.ngrams & reference.ngrams) / len(target.ngrams)


def _longest_overlap(
    target_tokens: tuple[str, ...],
    reference_normalized: str,
    max_target_tokens: int = 120,
) -> tuple[int, str | None]:
    if not target_tokens or not reference_normalized:
        return 0, None
    tokens = target_tokens[:max_target_tokens]
    reference_blob = f" {reference_normalized} "
    best_len = 0
    best_text: str | None = None
    for start in range(len(tokens)):
        for end in range(start + best_len + 1, len(tokens) + 1):
            phrase = " ".join(tokens[start:end])
            if f" {phrase} " not in reference_blob:
                break
            best_len = end - start
            best_text = phrase
    return best_len, best_text


def _record_label(source: Any, kind: Any, line: Any) -> str:
    base = f"{source or 'unknown'}:{kind or 'text'}"
    return f"{base}:{line}" if line is not None else base


def _read_chat_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, record in _read_jsonl(path):
        user = record.get("user")
        assistant = record.get("assistant")
        if isinstance(user, str) and isinstance(assistant, str) and user.strip() and assistant.strip():
            category = record.get("category", "chat")
            rows.append({
                "line": line_number,
                "user": user,
                "assistant": assistant,
                "category": category if isinstance(category, str) else "chat",
            })
    return rows


def _read_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, record in _read_jsonl(path):
        user = record.get("user")
        if isinstance(user, str) and user.strip():
            answerable = record.get("answerable", True)
            category = record.get("category", "answerable" if answerable is not False else "unanswerable")
            rows.append({
                "line": line_number,
                "user": user,
                "category": category if isinstance(category, str) else "eval",
                "support_phrases": _support_phrases(record),
            })
    return rows


def _read_jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    source = Path(path)
    if not source.exists() or not source.is_file():
        return rows
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            rows.append((line_number, record))
    return rows


def _support_phrases(record: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    expected = record.get("expected")
    if isinstance(expected, str):
        phrases.append(expected)
    must_include = record.get("must_include")
    if isinstance(must_include, list):
        phrases.extend(item for item in must_include if isinstance(item, str))
    must_include_any = record.get("must_include_any")
    if isinstance(must_include_any, list):
        for group in must_include_any:
            if isinstance(group, list):
                phrases.extend(item for item in group if isinstance(item, str))
    return phrases


def _read_corpus_text(path: str | Path | None) -> str:
    if path is None:
        return ""
    source = Path(path)
    if not source.exists():
        return ""
    if source.is_file():
        try:
            return source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""
    texts: list[str] = []
    for item in sorted(source.rglob("*")):
        if item.is_file() and item.suffix.lower() in TEXT_EXTENSIONS:
            try:
                texts.append(item.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
    return "\n\n".join(texts)


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: tuple[str, ...], ngram_size: int) -> set[tuple[str, ...]]:
    if ngram_size <= 0 or len(tokens) < ngram_size:
        return set()
    return {
        tuple(tokens[index:index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    }


def _prompt_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    left_words = set(left.split())
    right_words = set(right.split())
    if not left_words or not right_words:
        return sequence
    jaccard = len(left_words & right_words) / len(left_words | right_words)
    return max(sequence, jaccard)


def _is_specific_support_phrase(phrase: str) -> bool:
    if phrase in GENERIC_REFUSAL_SUPPORT_PHRASES:
        return False
    if len(phrase) < 18:
        return False
    return len(phrase.split()) >= 3


def _status(findings: list[HonestyFinding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "blocked"
    if findings:
        return "caution"
    return "ready"


def _summary(status: str, findings: list[HonestyFinding]) -> str:
    if status == "ready":
        return "No obvious eval leakage was detected."
    if status == "blocked":
        failures = sum(1 for finding in findings if finding.severity == "fail")
        return f"{failures} blocking leakage finding(s). Do not treat the eval score as clean."
    return f"{len(findings)} caution finding(s). Inspect the eval before trusting the score."


def _preview(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "..."


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
