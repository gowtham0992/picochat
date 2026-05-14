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
    findings: tuple[HonestyFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def inspect_data_honesty(
    chat_input: str | Path,
    eval_input: str | Path,
    corpus_path: str | Path | None = None,
    near_threshold: float = 0.86,
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
        "## Findings",
        "",
    ]
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
