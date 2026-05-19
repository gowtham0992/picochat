"""Lightweight memorization diagnostics for tiny language-model runs."""

from __future__ import annotations

import re


CANARY_PATTERN = re.compile(r"\bpico-canary-[A-Za-z0-9_-]+\b")


def memorization_diagnostics(
    generated_text: str,
    train_text: str,
    validation_text: str = "",
    ngram_size: int = 8,
) -> dict:
    """Measure simple exact-overlap signals between generated text and corpus text."""
    generated_tokens = _tokens(generated_text)
    train_tokens = _tokens(train_text)
    validation_tokens = _tokens(validation_text)
    train_overlap = _ngram_overlap(generated_tokens, train_tokens, ngram_size)
    validation_overlap = _ngram_overlap(generated_tokens, validation_tokens, ngram_size)
    train_canaries = sorted(set(CANARY_PATTERN.findall(train_text)))
    generated_canaries = sorted(set(CANARY_PATTERN.findall(generated_text)))
    canary_hits = [value for value in generated_canaries if value in train_canaries]
    risk, summary = _risk_label(
        train_overlap["overlap_rate"],
        train_overlap["longest_overlap_tokens"],
        bool(canary_hits),
    )
    return {
        "status": risk,
        "summary": summary,
        "ngram_size": ngram_size,
        "generated_tokens": len(generated_tokens),
        "train_overlap_rate": train_overlap["overlap_rate"],
        "train_overlap_ngrams": train_overlap["overlap_ngrams"],
        "validation_overlap_rate": validation_overlap["overlap_rate"],
        "validation_overlap_ngrams": validation_overlap["overlap_ngrams"],
        "longest_train_overlap_tokens": train_overlap["longest_overlap_tokens"],
        "longest_validation_overlap_tokens": validation_overlap["longest_overlap_tokens"],
        "canary_values_in_train": train_canaries,
        "canary_hits": canary_hits,
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngram_overlap(generated: list[str], reference: list[str], ngram_size: int) -> dict[str, float | int]:
    if ngram_size < 1:
        raise ValueError("ngram_size must be at least 1")
    total = max(0, len(generated) - ngram_size + 1)
    if not total or len(reference) < ngram_size:
        return {
            "overlap_rate": 0.0,
            "overlap_ngrams": 0,
            "longest_overlap_tokens": 0,
        }
    reference_ngrams = {
        tuple(reference[index: index + ngram_size])
        for index in range(0, len(reference) - ngram_size + 1)
    }
    matches = [
        tuple(generated[index: index + ngram_size]) in reference_ngrams
        for index in range(total)
    ]
    overlap_ngrams = sum(1 for matched in matches if matched)
    return {
        "overlap_rate": overlap_ngrams / total,
        "overlap_ngrams": overlap_ngrams,
        "longest_overlap_tokens": _longest_matched_span(matches, ngram_size),
    }


def _longest_matched_span(matches: list[bool], ngram_size: int) -> int:
    longest = 0
    current = 0
    for matched in matches:
        if matched:
            current += 1
            longest = max(longest, current + ngram_size - 1)
        else:
            current = 0
    return longest


def _risk_label(overlap_rate: float, longest_overlap_tokens: int, canary_hit: bool) -> tuple[str, str]:
    if canary_hit:
        return "high", "Generated text reproduced a canary string from the training corpus."
    if longest_overlap_tokens >= 80 or overlap_rate >= 0.35:
        return "high", "Generated text contains large exact spans from the training corpus."
    if longest_overlap_tokens >= 32 or overlap_rate >= 0.15:
        return "medium", "Generated text has noticeable exact overlap with the training corpus."
    return "low", "Generated text has low exact overlap with the training corpus sample."
