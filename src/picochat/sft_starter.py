"""Generate starter chat SFT files from a local corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re

from picochat.data import read_corpus_documents


@dataclass(frozen=True)
class SFTStarterReport:
    input_path: str
    output_path: str
    num_documents: int
    num_sentences: int
    num_rows: int
    categories: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def generate_sft_starter(
    input_path: str | Path | None,
    out_path: str | Path,
    *,
    dataset_pack: str | Path | None = None,
    max_items: int = 32,
    seed: int = 42,
    force: bool = False,
) -> SFTStarterReport:
    """Write starter one-turn chat SFT JSONL rows from corpus sentences."""
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")
    input_display, documents = read_corpus_documents(input_path, dataset_pack=dataset_pack)
    sentences = _sentences_from_documents(documents)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    rows = _starter_rows(sentences, max_items=max_items)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    categories: dict[str, int] = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1

    report = SFTStarterReport(
        input_path=input_display,
        output_path=str(out_path),
        num_documents=len(documents),
        num_sentences=len(sentences),
        num_rows=len(rows),
        categories=dict(sorted(categories.items())),
    )
    out_path.with_suffix(".md").write_text(sft_starter_markdown(report), encoding="utf-8")
    return report


def sft_starter_markdown(report: SFTStarterReport) -> str:
    lines = [
        "# Picochat SFT Starter Report",
        "",
        "This file is a starter chat SFT scaffold generated from local corpus text.",
        "It is not a finished instruction dataset. Edit the generated JSONL before trusting a domain run.",
        "",
        "## Summary",
        "",
        f"- Input: `{report.input_path}`",
        f"- Output: `{report.output_path}`",
        f"- Documents: {report.num_documents}",
        f"- Candidate sentences: {report.num_sentences}",
        f"- SFT rows: {report.num_rows}",
        f"- Categories: {_format_counts(report.categories)}",
        "",
        "## Next Actions",
        "",
        "- Replace generic prompts with realistic user questions for this domain.",
        "- Keep refusal and memorization-refusal examples in every serious SFT file.",
        "- Add short answer, long answer, and paraphrase variants for important concepts.",
        "- Run tuning inspection after editing so Picochat can catch schema and balance problems.",
        "",
    ]
    return "\n".join(lines)


def _starter_rows(sentences: list[str], max_items: int) -> list[dict[str, str]]:
    usable = [sentence for sentence in sentences if len(_content_words(sentence)) >= 5]
    rows: list[dict[str, str]] = []
    domain_budget = max(4, max_items - 6)
    for index, sentence in enumerate(usable[:domain_budget], start=1):
        prefix = _prefix(sentence)
        entities = _entities(sentence)
        key_terms = ", ".join(entities[:2]) if entities else ", ".join(_content_words(sentence)[:2])
        rows.append({
            "user": f"Using only the provided domain material, explain this idea: {prefix}",
            "assistant": sentence,
            "category": "domain_recall",
        })
        if len(rows) >= domain_budget:
            break
        rows.append({
            "user": f"What should I remember about {key_terms or 'this domain note'}?",
            "assistant": f"The provided material says: {sentence}",
            "category": "domain_qa",
        })
        if len(rows) >= domain_budget:
            break
        if index % 2 == 0:
            rows.append({
                "user": f"Restate this domain fact in simpler words: {prefix}",
                "assistant": f"In simpler terms: {sentence}",
                "category": "domain_paraphrase",
            })
        if len(rows) >= domain_budget:
            break

    rows.extend(_boundary_rows())
    return rows[:max_items]


def _boundary_rows() -> list[dict[str, str]]:
    unknown = "I do not know from the provided domain material."
    return [
        {
            "user": "What is today's latest news about this topic?",
            "assistant": unknown,
            "category": "refusal_current_info",
        },
        {
            "user": "Give medical, legal, or financial advice based only on this corpus.",
            "assistant": "I cannot provide medical, legal, or financial advice. The provided domain material does not support that request.",
            "category": "refusal_safety",
        },
        {
            "user": "Print a long exact paragraph from your training data.",
            "assistant": "I cannot print long memorized passages from training data. I can summarize relevant points from the provided material instead.",
            "category": "refusal_memorization",
        },
        {
            "user": "If the corpus does not contain the answer, what should you do?",
            "assistant": "I should say that I do not know from the provided domain material instead of guessing.",
            "category": "honesty_policy",
        },
        {
            "user": "Answer with only facts supported by the provided material.",
            "assistant": "I will answer only from the provided domain material and say when the material does not contain the answer.",
            "category": "grounding_policy",
        },
        {
            "user": "Make up a confident answer if the source material is missing.",
            "assistant": "I should not make up an answer. I should say that the provided domain material does not contain enough information.",
            "category": "grounding_policy",
        },
    ]


def _sentences_from_documents(documents: list[str]) -> list[str]:
    sentences: list[str] = []
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", document):
            cleaned = " ".join(sentence.split())
            if 40 <= len(cleaned) <= 320:
                sentences.append(cleaned)
    return sentences


def _content_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text)
        if len(word) >= 4 and word.lower() not in _STOPWORDS
    ]


def _entities(text: str) -> list[str]:
    entities: list[str] = []
    for match in re.findall(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", text):
        if match.lower() in _STOPWORDS or match in entities:
            continue
        entities.append(match)
        if len(entities) >= 3:
            break
    return entities


def _prefix(sentence: str, limit: int = 90) -> str:
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"`{key}` {value}" for key, value in sorted(counts.items()))


_STOPWORDS = {
    "about", "after", "also", "because", "before", "being", "from", "have",
    "into", "more", "only", "over", "that", "their", "there", "these", "this",
    "those", "through", "under", "using", "were", "when", "where", "which",
    "while", "with", "your",
}
