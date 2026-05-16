"""Small transparent evaluation tools for Picochat."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import time

import torch

from picochat.chat import extract_assistant_reply, render_chat_prompt
from picochat.checkpoint import load_checkpoint
from picochat.device import resolve_device
from picochat.precision import (
    autocast_context,
    configure_float32_matmul_precision,
    resolve_precision,
)
from picochat.report import chat_eval_report_markdown
from picochat.tokenizer import Tokenizer, load_tokenizer


@dataclass(frozen=True)
class ChatEvalItem:
    user: str
    must_include: tuple[str, ...] = ()
    must_include_any: tuple[tuple[str, ...], ...] = ()
    must_not_include: tuple[str, ...] = ()
    answerable: bool = True
    category: str = "answerable"
    split: str = "default"
    level: str = "heldout"
    curriculum_stage: str = ""
    reference_answer: str | None = None
    required_entities: tuple[str, ...] = ()
    min_words: int | None = None
    max_words: int | None = None
    min_chars: int | None = None
    max_chars: int | None = None
    require_corpus_support: bool = False
    choice_labels: tuple[str, ...] = ()
    correct_choice: str | None = None


@dataclass(frozen=True)
class ChatEvalConfig:
    input_path: str
    checkpoint_path: str
    tokenizer_path: str
    out_dir: str
    max_new_tokens: int = 80
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    device: str = "cpu"
    case_sensitive: bool = False
    support_corpus_path: str | None = None
    corpus_support_threshold: float = 0.25
    ci_bootstrap_samples: int = 1000
    ci_confidence: float = 0.95
    log_every: int = 0
    precision: str = "float32"
    matmul_precision: str = "default"


@dataclass(frozen=True)
class _ChoiceCandidate:
    variant: str
    token_ids: tuple[int, ...]


def write_sft_fit_eval(
    input_path: str | Path,
    output_path: str | Path,
    max_rows: int | None = None,
    include_indices: Iterable[int] | None = None,
    split_label: str = "sft_train",
) -> dict:
    """Convert chat SFT JSONL into an exact-fit eval file.

    This is a diagnostic, not a leaderboard. A model should score high here
    before its held-out eval score is interpreted as a generalization signal.
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    selected_indices = set(include_indices) if include_indices is not None else None
    if selected_indices is not None and any(index < 0 for index in selected_indices):
        raise ValueError("include_indices must contain non-negative row indices")
    if not split_label.strip():
        raise ValueError("split_label must be non-empty")

    rows: list[dict] = []
    category_counts: dict[str, int] = {}
    selected_count = 0
    for row_index, line in enumerate(Path(input_path).read_text(encoding="utf-8").splitlines()):
        if selected_indices is not None and row_index not in selected_indices:
            continue
        line = line.strip()
        if not line:
            continue
        line_number = row_index + 1
        record = json.loads(line)
        user = record.get("user")
        assistant = record.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise ValueError(f"line {line_number} must contain string user and assistant fields")
        answer = assistant.strip()
        if not answer:
            continue

        category = record.get("category", "chat")
        if not isinstance(category, str):
            raise ValueError(f"line {line_number} category field must be a string when present")
        curriculum_stage = record.get("curriculum_stage", "")
        if not isinstance(curriculum_stage, str):
            raise ValueError(f"line {line_number} curriculum_stage field must be a string when present")
        answerable = record.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"line {line_number} answerable field must be a boolean when present")

        fit_must_include = record.get("fit_must_include")
        must_include = (
            [answer]
            if fit_must_include is None
            else list(_as_string_tuple(fit_must_include, line_number, "fit_must_include"))
        )
        fit_reference_answer = record.get("fit_reference_answer", answer)
        if not isinstance(fit_reference_answer, str):
            raise ValueError(f"line {line_number} fit_reference_answer field must be a string when present")
        fit_max_words = record.get("fit_max_words")
        max_words = (
            _sft_fit_max_words(answer)
            if fit_max_words is None
            else _optional_int(fit_max_words, line_number, "fit_max_words")
        )

        category_counts[category] = category_counts.get(category, 0) + 1
        rows.append({
            "user": user,
            "answerable": answerable,
            "category": category,
            "curriculum_stage": curriculum_stage.strip(),
            "split": split_label.strip(),
            "level": category,
            "reference_answer": fit_reference_answer,
            "must_include": must_include,
            "max_words": max_words,
        })
        selected_count += 1
        if max_rows is not None and len(rows) >= max_rows:
            break

    if not rows:
        raise ValueError("no usable SFT rows found for fit eval")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "input_path": str(input_path),
        "output_path": str(output),
        "num_rows": len(rows),
        "selected_rows": selected_count,
        "selected_from_indices": selected_indices is not None,
        "split_label": split_label.strip(),
        "category_counts": dict(sorted(category_counts.items())),
    }


def _sft_fit_max_words(answer: str) -> int:
    word_count = len(_word_tokens(answer))
    if word_count <= 1:
        return 8
    return min(80, max(12, word_count + 8))


def load_chat_eval_items(path: str | Path) -> list[ChatEvalItem]:
    """Load transparent chat eval items from JSONL."""
    items: list[ChatEvalItem] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        user = record.get("user")
        if not isinstance(user, str):
            raise ValueError(f"line {line_number} must contain a string user field")

        must_include = record.get("must_include", ())
        expected = record.get("expected")
        reference_answer = record.get("reference_answer", record.get("reference", expected))
        if expected is not None:
            if not isinstance(expected, str):
                raise ValueError(f"line {line_number} expected field must be a string")
            must_include = [*must_include, expected]
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise ValueError(f"line {line_number} reference_answer field must be a string")

        must_not_include = record.get("must_not_include", ())
        must_include_any = record.get("must_include_any", ())
        answerable = record.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"line {line_number} answerable field must be a boolean")
        category = record.get("category", "answerable" if answerable else "unanswerable")
        if not isinstance(category, str):
            raise ValueError(f"line {line_number} category field must be a string")
        split = record.get("split", record.get("eval_split", "default"))
        if not isinstance(split, str):
            raise ValueError(f"line {line_number} split field must be a string")
        level = record.get("level", record.get("eval_level", _infer_eval_level(split, category, answerable)))
        if not isinstance(level, str) or not level.strip():
            raise ValueError(f"line {line_number} level field must be a non-empty string when present")
        curriculum_stage = record.get("curriculum_stage", "")
        if not isinstance(curriculum_stage, str):
            raise ValueError(f"line {line_number} curriculum_stage field must be a string when present")
        required_entities = record.get("required_entities", record.get("entities", ()))
        require_corpus_support = record.get("require_corpus_support", False)
        if not isinstance(require_corpus_support, bool):
            raise ValueError(f"line {line_number} require_corpus_support field must be a boolean")
        choice_labels = _as_string_tuple(
            record.get("choice_labels", ()),
            line_number,
            "choice_labels",
        )
        correct_choice = record.get("correct_choice", record.get("answer_choice"))
        if correct_choice is not None:
            if not isinstance(correct_choice, str) or not correct_choice.strip():
                raise ValueError(f"line {line_number} correct_choice field must be a non-empty string")
            correct_choice = correct_choice.strip()
            if not choice_labels:
                raise ValueError(f"line {line_number} correct_choice requires choice_labels")
            if correct_choice not in choice_labels:
                raise ValueError(f"line {line_number} correct_choice must be present in choice_labels")
        items.append(ChatEvalItem(
            user=user,
            must_include=_as_string_tuple(must_include, line_number, "must_include"),
            must_include_any=_as_phrase_groups(must_include_any, line_number, "must_include_any"),
            must_not_include=_as_string_tuple(must_not_include, line_number, "must_not_include"),
            answerable=answerable,
            category=category,
            split=split or "default",
            level=level.strip(),
            curriculum_stage=curriculum_stage.strip(),
            reference_answer=reference_answer,
            required_entities=_as_string_tuple(required_entities, line_number, "required_entities"),
            min_words=_optional_int(record.get("min_words"), line_number, "min_words"),
            max_words=_optional_int(record.get("max_words"), line_number, "max_words"),
            min_chars=_optional_int(record.get("min_chars"), line_number, "min_chars"),
            max_chars=_optional_int(record.get("max_chars"), line_number, "max_chars"),
            require_corpus_support=require_corpus_support,
            choice_labels=choice_labels,
            correct_choice=correct_choice,
        ))

    if not items:
        raise ValueError("chat eval dataset is empty")
    return items


def _as_string_tuple(value, line_number: int, field: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"line {line_number} {field} field must be a list of strings")
    return tuple(value)


def _as_phrase_groups(value, line_number: int, field: str) -> tuple[tuple[str, ...], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"line {line_number} {field} field must be a list of lists")

    groups: list[tuple[str, ...]] = []
    for group in value:
        if (
            not isinstance(group, list)
            or not group
            or not all(isinstance(item, str) for item in group)
        ):
            raise ValueError(
                f"line {line_number} {field} field must contain non-empty string lists"
            )
        groups.append(tuple(group))
    return tuple(groups)


def _optional_int(value, line_number: int, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"line {line_number} {field} field must be a non-negative integer")
    return value


def _infer_eval_level(split: str, category: str, answerable: bool) -> str:
    text = f"{split} {category}".lower()
    if "smoke" in text:
        return "smoke"
    if "memor" in text or "canary" in text:
        return "memorization"
    if "adversarial" in text or "safety" in text or "refusal" in text or not answerable:
        return "adversarial"
    if "transfer" in text or "paraphrase" in text:
        return "transfer"
    if "domain" in text or "knowledge" in text:
        return "domain"
    return "heldout"


def score_reply(
    reply: str,
    item: ChatEvalItem,
    case_sensitive: bool = False,
    support_corpus_text: str | None = None,
    support_corpus_tokens: frozenset[str] | None = None,
    corpus_support_threshold: float = 0.25,
) -> dict:
    """Score a reply with visible word-aware phrase rules."""
    prompt_echo_reasons = detect_prompt_echo(reply, item.user, case_sensitive=case_sensitive)
    prompt_echo = bool(prompt_echo_reasons)

    missing = [
        phrase
        for phrase in item.must_include
        if not _contains_eval_phrase(reply, phrase, case_sensitive=case_sensitive)
    ]
    missing_any = [
        list(group)
        for group in item.must_include_any
        if not any(_contains_eval_phrase(reply, phrase, case_sensitive=case_sensitive) for phrase in group)
    ]
    found_forbidden = [
        phrase
        for phrase in item.must_not_include
        if _contains_eval_phrase(reply, phrase, case_sensitive=case_sensitive)
    ]
    missing_entities = [
        entity
        for entity in item.required_entities
        if not _contains_eval_phrase(reply, entity, case_sensitive=case_sensitive)
    ]
    length_violations = _length_violations(reply, item)
    reference_metrics = _reference_metrics(reply, item.reference_answer)
    repetition = _repetition_diagnostics(reply)
    refusal_match = _detect_refusal(reply, case_sensitive=case_sensitive)
    corpus_support = _corpus_support_diagnostics(
        reply,
        support_corpus_text=support_corpus_text,
        support_corpus_tokens=support_corpus_tokens,
    )
    corpus_support_failed = (
        item.require_corpus_support
        and corpus_support["rate"] is not None
        and corpus_support["rate"] < corpus_support_threshold
    )
    support_total = len(item.must_include) + len(item.must_include_any)
    support_matched = support_total - len(missing) - len(missing_any)
    return {
        "passed": (
            not missing
            and not missing_any
            and not found_forbidden
            and not prompt_echo
            and not missing_entities
            and not length_violations
            and not corpus_support_failed
        ),
        "missing": missing,
        "missing_any": missing_any,
        "found_forbidden": found_forbidden,
        "missing_entities": missing_entities,
        "length_violations": length_violations,
        "prompt_echo": prompt_echo,
        "prompt_echo_reasons": prompt_echo_reasons,
        "support_total": support_total,
        "support_matched": support_matched,
        "support_match_rate": support_matched / support_total if support_total else 1.0,
        "reference_token_f1": reference_metrics["token_f1"],
        "reference_rouge_l": reference_metrics["rouge_l"],
        "entity_total": len(item.required_entities),
        "entity_matched": len(item.required_entities) - len(missing_entities),
        "entity_match_rate": _safe_rate(len(item.required_entities) - len(missing_entities), len(item.required_entities)),
        "word_count": len(_word_tokens(reply)),
        "char_count": len(reply),
        "refusal_match": refusal_match,
        "repetition_ngram_rate": repetition["ngram_rate"],
        "repetition_unique_token_rate": repetition["unique_token_rate"],
        "corpus_support_rate": corpus_support["rate"],
        "corpus_support_tokens": corpus_support["supported"],
        "corpus_support_total": corpus_support["total"],
        "corpus_support_failed": corpus_support_failed,
    }


def detect_prompt_echo(
    reply: str,
    user_message: str,
    case_sensitive: bool = False,
) -> list[str]:
    """Return visible reasons a reply appears to echo the chat prompt."""
    if not reply.strip():
        return []

    if case_sensitive:
        text = reply
        user_text = user_message
        role_pattern = r"(^|\n)\s*(User|Assistant)\s*:"
    else:
        text = reply.lower()
        user_text = user_message.lower()
        role_pattern = r"(^|\n)\s*(user|assistant)\s*:"

    reasons: list[str] = []
    if re.search(role_pattern, text):
        reasons.append("chat_role_label")

    compact_reply = _normalize_for_echo(text)
    compact_user = _normalize_for_echo(user_text)
    if len(compact_user) >= 12 and (
        compact_reply.startswith(compact_user[:80])
        or compact_reply.startswith(f"user {compact_user[:80]}")
    ):
        reasons.append("starts_with_user_prompt")

    return reasons


def run_chat_eval(config: ChatEvalConfig) -> dict:
    """Run a deterministic chat eval and write JSON/Markdown reports."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.tokenizer_path)
    device = resolve_device(config.device)
    model, metadata = load_checkpoint(config.checkpoint_path, map_location=device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("tokenizer vocabulary size does not match checkpoint")

    model = model.to(device)
    model.eval()
    matmul_precision_runtime = configure_float32_matmul_precision(config.matmul_precision)
    precision_runtime = resolve_precision(config.precision, device)
    support_corpus_text = None
    support_corpus_tokens = _read_optional_support_token_set(config.support_corpus_path)

    items = load_chat_eval_items(config.input_path)
    rows = []
    eval_start = time.perf_counter()
    if config.log_every > 0:
        print(
            f"eval 0000/{len(items):04d} | starting | "
            f"{Path(config.out_dir).name}"
        )
    for index, item in enumerate(items):
        choice_scores = None
        choice_details = None
        generation_max_new_tokens = None
        if item.correct_choice:
            reply, choice_scores, choice_details = _predict_choice_reply(
                model,
                tokenizer,
                config,
                item,
                precision_runtime=precision_runtime,
            )
        else:
            reply, generation_max_new_tokens = _generate_eval_reply(
                model,
                tokenizer,
                config,
                item,
                seed=config.seed + index,
                precision_runtime=precision_runtime,
            )
        score = score_reply(
            reply,
            item,
            case_sensitive=config.case_sensitive,
            support_corpus_text=support_corpus_text,
            support_corpus_tokens=support_corpus_tokens,
            corpus_support_threshold=config.corpus_support_threshold,
        )
        rows.append({
            "index": index + 1,
            "user": item.user,
            "answerable": item.answerable,
            "category": item.category,
            "split": item.split,
            "level": item.level,
            "curriculum_stage": item.curriculum_stage,
            "reply": reply,
            "must_include": list(item.must_include),
            "must_include_any": [list(group) for group in item.must_include_any],
            "must_not_include": list(item.must_not_include),
            "reference_answer": item.reference_answer,
            "required_entities": list(item.required_entities),
            "min_words": item.min_words,
            "max_words": item.max_words,
            "min_chars": item.min_chars,
            "max_chars": item.max_chars,
            "require_corpus_support": item.require_corpus_support,
            "choice_labels": list(item.choice_labels),
            "correct_choice": item.correct_choice,
            "choice_logprobs": choice_scores,
            "choice_score_details": choice_details,
            "choice_eval_method": (
                "normalized_logprob_best_of_whitespace_and_eos_variants"
                if item.correct_choice else None
            ),
            "choice_predicted": reply if item.correct_choice else None,
            "generation_max_new_tokens": generation_max_new_tokens,
            **score,
        })
        if config.log_every > 0 and ((index + 1) % config.log_every == 0 or index + 1 == len(items)):
            passed_so_far = sum(1 for row in rows if row["passed"])
            elapsed = max(1e-9, time.perf_counter() - eval_start)
            rows_per_second = (index + 1) / elapsed
            remaining = len(items) - index - 1
            eta_seconds = remaining / rows_per_second if rows_per_second > 0 else 0.0
            print(
                f"eval {index + 1:04d}/{len(items):04d} | "
                f"passed {passed_so_far}/{len(rows)} | "
                f"{elapsed:.1f}s elapsed | eta {eta_seconds:.1f}s | "
                f"{Path(config.out_dir).name}"
            )

    passed = sum(1 for row in rows if row["passed"])
    unsupported_claims = sum(1 for row in rows if row["found_forbidden"])
    prompt_echoes = sum(1 for row in rows if row.get("prompt_echo"))
    missing_support = sum(1 for row in rows if row["missing"] or row["missing_any"])
    length_violations = sum(1 for row in rows if row.get("length_violations"))
    missing_entities = sum(1 for row in rows if row.get("missing_entities"))
    corpus_support_failures = sum(1 for row in rows if row.get("corpus_support_failed"))
    support_total = sum(int(row["support_total"]) for row in rows)
    support_matched = sum(int(row["support_matched"]) for row in rows)
    answerable_support_total = sum(int(row["support_total"]) for row in rows if row["answerable"])
    answerable_support_matched = sum(int(row["support_matched"]) for row in rows if row["answerable"])
    answerable = sum(1 for row in rows if row["answerable"])
    answerable_pass_rate = _filtered_pass_rate(rows, lambda row: bool(row.get("answerable", True)))
    unanswerable_pass_rate = _filtered_pass_rate(rows, lambda row: not bool(row.get("answerable", True)))
    domain_pass_rate = _filtered_pass_rate(rows, lambda row: str(row.get("category", "")).startswith("domain_"))
    refusal_pass_rate = _filtered_pass_rate(rows, lambda row: "refusal" in str(row.get("category", "")) or not bool(row.get("answerable", True)))
    category_breakdown = _breakdown(
        rows,
        "category",
        "answerable",
        bootstrap_samples=config.ci_bootstrap_samples,
        confidence=config.ci_confidence,
        seed=config.seed + 101,
    )
    split_breakdown = _breakdown(
        rows,
        "split",
        "default",
        bootstrap_samples=config.ci_bootstrap_samples,
        confidence=config.ci_confidence,
        seed=config.seed + 202,
    )
    level_breakdown = _breakdown(
        rows,
        "level",
        "heldout",
        bootstrap_samples=config.ci_bootstrap_samples,
        confidence=config.ci_confidence,
        seed=config.seed + 303,
    )
    stage_breakdown = _breakdown(
        rows,
        "curriculum_stage",
        "",
        bootstrap_samples=config.ci_bootstrap_samples,
        confidence=config.ci_confidence,
        seed=config.seed + 404,
    )
    stage_breakdown.pop("", None)
    choice_rows = [row for row in rows if row.get("correct_choice")]
    non_choice_rows = [row for row in rows if not row.get("correct_choice")]
    choice_correct = sum(
        1 for row in choice_rows
        if row.get("choice_predicted") == row.get("correct_choice")
    )
    choice_passed = sum(1 for row in choice_rows if row.get("passed"))
    non_choice_passed = sum(1 for row in non_choice_rows if row.get("passed"))
    analysis = analyze_eval_failures(rows, category_breakdown, split_breakdown, level_breakdown)
    report = {
        "config": {
            **config.__dict__,
            "requested_device": config.device,
            "device": device.type,
            "precision_runtime": precision_runtime.to_dict(),
            "matmul_precision_runtime": matmul_precision_runtime,
        },
        "checkpoint": {
            "path": config.checkpoint_path,
            "step": metadata.get("step"),
            "train_loss": metadata.get("train_loss"),
        },
        "summary": {
            "num_examples": len(rows),
            "num_passed": passed,
            "num_failed": len(rows) - passed,
            "pass_rate": passed / len(rows),
            "pass_rate_ci": _bootstrap_rate_ci(
                [bool(row.get("passed")) for row in rows],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed,
            ),
            "num_answerable": answerable,
            "num_unanswerable": len(rows) - answerable,
            "non_choice_examples": len(non_choice_rows),
            "non_choice_passed": non_choice_passed,
            "non_choice_pass_rate": (
                _safe_rate(non_choice_passed, len(non_choice_rows))
                if non_choice_rows else None
            ),
            "non_choice_pass_rate_ci": _bootstrap_rate_ci(
                [bool(row.get("passed")) for row in non_choice_rows],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed + 10,
            ),
            "answerable_pass_rate": answerable_pass_rate,
            "answerable_pass_rate_ci": _bootstrap_rate_ci(
                [bool(row.get("passed")) for row in rows if bool(row.get("answerable", True))],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed + 11,
            ),
            "unanswerable_pass_rate": unanswerable_pass_rate,
            "unanswerable_pass_rate_ci": _bootstrap_rate_ci(
                [bool(row.get("passed")) for row in rows if not bool(row.get("answerable", True))],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed + 12,
            ),
            "domain_pass_rate": domain_pass_rate,
            "domain_pass_rate_ci": _bootstrap_rate_ci(
                [
                    bool(row.get("passed"))
                    for row in rows
                    if str(row.get("category", "")).startswith("domain_")
                ],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed + 13,
            ),
            "refusal_pass_rate": refusal_pass_rate,
            "refusal_pass_rate_ci": _bootstrap_rate_ci(
                [
                    bool(row.get("passed"))
                    for row in rows
                    if "refusal" in str(row.get("category", "")) or not bool(row.get("answerable", True))
                ],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed + 14,
            ),
            "unsupported_claims": unsupported_claims,
            "unsupported_claim_rate": unsupported_claims / len(rows),
            "prompt_echoes": prompt_echoes,
            "prompt_echo_rate": prompt_echoes / len(rows),
            "missing_support": missing_support,
            "missing_support_rate": missing_support / len(rows),
            "missing_entities": missing_entities,
            "missing_entity_rate": missing_entities / len(rows),
            "length_violations": length_violations,
            "length_violation_rate": length_violations / len(rows),
            "corpus_support_failures": corpus_support_failures,
            "corpus_support_failure_rate": corpus_support_failures / len(rows),
            "support_requirements": support_total,
            "support_matches": support_matched,
            "support_match_rate": _safe_rate(support_matched, support_total),
            "answerable_support_match_rate": _safe_rate(
                answerable_support_matched,
                answerable_support_total,
            ),
            "average_reference_token_f1": _metric_average(rows, "reference_token_f1"),
            "average_reference_rouge_l": _metric_average(rows, "reference_rouge_l"),
            "average_entity_match_rate": _metric_average(rows, "entity_match_rate"),
            "average_corpus_support_rate": _metric_average(rows, "corpus_support_rate"),
            "average_repetition_ngram_rate": _metric_average(rows, "repetition_ngram_rate"),
            "choice_examples": len(choice_rows),
            "choice_correct": choice_correct,
            "choice_accuracy": _safe_rate(choice_correct, len(choice_rows)) if choice_rows else None,
            "choice_accuracy_ci": _bootstrap_rate_ci(
                [
                    row.get("choice_predicted") == row.get("correct_choice")
                    for row in choice_rows
                ],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed + 15,
            ),
            "choice_passed": choice_passed,
            "choice_pass_rate": _safe_rate(choice_passed, len(choice_rows)) if choice_rows else None,
            "choice_pass_rate_ci": _bootstrap_rate_ci(
                [bool(row.get("passed")) for row in choice_rows],
                samples=config.ci_bootstrap_samples,
                confidence=config.ci_confidence,
                seed=config.seed + 16,
            ),
            "choice_scoring": (
                "normalized_logprob_best_of_whitespace_and_eos_variants"
                if choice_rows else None
            ),
            "category_breakdown": category_breakdown,
            "split_breakdown": split_breakdown,
            "level_breakdown": level_breakdown,
            "stage_breakdown": stage_breakdown,
        },
        "analysis": analysis,
        "examples": rows,
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(chat_eval_report_markdown(report), encoding="utf-8")
    return report


def analyze_eval_failures(
    rows: list[dict],
    category_breakdown: dict[str, dict] | None = None,
    split_breakdown: dict[str, dict] | None = None,
    level_breakdown: dict[str, dict] | None = None,
) -> dict:
    """Return compact failure causes and next actions for an eval report."""
    failure_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    failed_examples: list[dict] = []
    for fallback_index, row in enumerate(rows, start=1):
        reasons = _failure_reasons(row)
        if not row.get("passed", False):
            for reason in reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            clusters = _failure_clusters(row, reasons)
            for cluster in clusters:
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            failed_examples.append({
                "index": row.get("index", fallback_index),
                "category": row.get("category", "answerable"),
                "split": row.get("split", "default"),
                "level": row.get("level", "heldout"),
                "answerable": row.get("answerable", True),
                "reasons": reasons,
                "clusters": clusters,
                "missing": list(row.get("missing", [])),
                "missing_any": list(row.get("missing_any", [])),
                "missing_entities": list(row.get("missing_entities", [])),
                "length_violations": list(row.get("length_violations", [])),
                "found_forbidden": list(row.get("found_forbidden", [])),
                "prompt_echo_reasons": list(row.get("prompt_echo_reasons", [])),
                "reply_preview": _preview_text(str(row.get("reply", ""))),
            })

    category_breakdown = category_breakdown or _breakdown(rows, "category", "answerable")
    split_breakdown = split_breakdown or _breakdown(rows, "split", "default")
    level_breakdown = level_breakdown or _breakdown(rows, "level", "heldout")
    weak_categories = _weak_breakdown(category_breakdown, "category")
    weak_splits = _weak_breakdown(split_breakdown, "split")
    weak_levels = _weak_breakdown(level_breakdown, "level")
    return {
        "failure_counts": dict(sorted(failure_counts.items())),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "failed_examples": failed_examples,
        "weak_categories": weak_categories,
        "weak_splits": weak_splits,
        "weak_levels": weak_levels,
        "recommendations": _eval_recommendations(
            rows,
            failure_counts,
            cluster_counts,
            weak_categories,
            weak_splits,
            weak_levels,
        ),
    }


def _filtered_pass_rate(rows: list[dict], predicate) -> float | None:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        return None
    return sum(1 for row in selected if row.get("passed")) / len(selected)


def _breakdown(
    rows: list[dict],
    field: str,
    default: str,
    *,
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for row in rows:
        value = row.get(field) or default
        bucket = buckets.setdefault(
            str(value),
            {
                "num_examples": 0,
                "num_passed": 0,
                "num_failed": 0,
                "pass_rate": 0.0,
                "num_answerable": 0,
                "num_unanswerable": 0,
                "unsupported_claims": 0,
                "unsupported_claim_rate": 0.0,
                "prompt_echoes": 0,
                "prompt_echo_rate": 0.0,
                "missing_support": 0,
                "missing_support_rate": 0.0,
                "missing_entities": 0,
                "missing_entity_rate": 0.0,
                "length_violations": 0,
                "length_violation_rate": 0.0,
                "corpus_support_failures": 0,
                "corpus_support_failure_rate": 0.0,
                "support_requirements": 0,
                "support_matches": 0,
                "support_match_rate": 1.0,
                "_reference_token_f1_values": [],
                "_reference_rouge_l_values": [],
                "_entity_match_rate_values": [],
                "_corpus_support_rate_values": [],
                "_repetition_ngram_rate_values": [],
                "_pass_values": [],
            },
        )
        bucket["num_examples"] += 1
        bucket["num_passed"] += int(row["passed"])
        bucket["num_failed"] += int(not row["passed"])
        bucket["_pass_values"].append(bool(row["passed"]))
        bucket["num_answerable"] += int(row.get("answerable", True))
        bucket["num_unanswerable"] += int(not row.get("answerable", True))
        bucket["unsupported_claims"] += int(bool(row.get("found_forbidden")))
        bucket["prompt_echoes"] += int(bool(row.get("prompt_echo")))
        bucket["missing_support"] += int(bool(row.get("missing") or row.get("missing_any")))
        bucket["missing_entities"] += int(bool(row.get("missing_entities")))
        bucket["length_violations"] += int(bool(row.get("length_violations")))
        bucket["corpus_support_failures"] += int(bool(row.get("corpus_support_failed")))
        bucket["support_requirements"] += int(row.get("support_total", 0))
        bucket["support_matches"] += int(row.get("support_matched", 0))
        _append_metric(bucket, "_reference_token_f1_values", row.get("reference_token_f1"))
        _append_metric(bucket, "_reference_rouge_l_values", row.get("reference_rouge_l"))
        _append_metric(bucket, "_entity_match_rate_values", row.get("entity_match_rate"))
        _append_metric(bucket, "_corpus_support_rate_values", row.get("corpus_support_rate"))
        _append_metric(bucket, "_repetition_ngram_rate_values", row.get("repetition_ngram_rate"))

    for bucket_index, bucket in enumerate(buckets.values()):
        total = bucket["num_examples"]
        bucket["pass_rate"] = bucket["num_passed"] / total
        bucket["pass_rate_ci"] = _bootstrap_rate_ci(
            bucket.pop("_pass_values"),
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed + bucket_index,
        )
        bucket["unsupported_claim_rate"] = bucket["unsupported_claims"] / total
        bucket["prompt_echo_rate"] = bucket["prompt_echoes"] / total
        bucket["missing_support_rate"] = bucket["missing_support"] / total
        bucket["missing_entity_rate"] = bucket["missing_entities"] / total
        bucket["length_violation_rate"] = bucket["length_violations"] / total
        bucket["corpus_support_failure_rate"] = bucket["corpus_support_failures"] / total
        bucket["support_match_rate"] = _safe_rate(
            bucket["support_matches"],
            bucket["support_requirements"],
        )
        bucket["average_reference_token_f1"] = _average_values(bucket.pop("_reference_token_f1_values"))
        bucket["average_reference_rouge_l"] = _average_values(bucket.pop("_reference_rouge_l_values"))
        bucket["average_entity_match_rate"] = _average_values(bucket.pop("_entity_match_rate_values"))
        bucket["average_corpus_support_rate"] = _average_values(bucket.pop("_corpus_support_rate_values"))
        bucket["average_repetition_ngram_rate"] = _average_values(bucket.pop("_repetition_ngram_rate_values"))
    return dict(sorted(buckets.items()))


def _bootstrap_rate_ci(
    values: list[bool],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict | None:
    """Return a deterministic bootstrap CI for a Bernoulli rate."""
    if not values:
        return None
    if samples < 1:
        return None
    if not 0 < confidence < 1:
        raise ValueError("ci_confidence must be in (0, 1)")
    numeric = [1 if value else 0 for value in values]
    size = len(numeric)
    generator = random.Random(seed)
    rates = []
    for _ in range(samples):
        passed = sum(numeric[generator.randrange(size)] for _ in range(size))
        rates.append(passed / size)
    rates.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, min(samples - 1, int((alpha / 2) * (samples - 1))))
    upper_index = max(0, min(samples - 1, int((1.0 - alpha / 2) * (samples - 1))))
    return {
        "low": rates[lower_index],
        "high": rates[upper_index],
        "confidence": confidence,
        "method": "bootstrap",
        "samples": samples,
        "n": size,
    }


def _failure_reasons(row: dict) -> list[str]:
    reasons: list[str] = []
    if row.get("missing"):
        reasons.append("missing_required")
    if row.get("missing_any"):
        reasons.append("missing_any_group")
    if row.get("missing_entities"):
        reasons.append("missing_entity")
    if row.get("length_violations"):
        reasons.append("length_violation")
    if row.get("found_forbidden"):
        reasons.append("forbidden_phrase")
    if row.get("prompt_echo"):
        reasons.append("prompt_echo")
    if row.get("corpus_support_failed"):
        reasons.append("weak_corpus_support")
    if (
        row.get("correct_choice")
        and row.get("choice_predicted") != row.get("correct_choice")
    ):
        reasons.append("wrong_choice")
    if _number(row.get("reference_token_f1")) is not None and float(row.get("reference_token_f1")) < 0.25:
        reasons.append("low_reference_overlap")
    if float(row.get("repetition_ngram_rate") or 0.0) > 0.25:
        reasons.append("repetitive_reply")
    if not str(row.get("reply", "")).strip():
        reasons.append("empty_reply")
    if not row.get("passed", False) and not reasons:
        reasons.append("unknown")
    return reasons


def _failure_clusters(row: dict, reasons: list[str]) -> list[str]:
    clusters: list[str] = []
    reason_set = set(reasons)
    if reason_set & {"missing_required", "missing_any_group", "missing_entity", "low_reference_overlap"}:
        clusters.append("content_mismatch")
    if "wrong_choice" in reason_set:
        clusters.append("choice_mismatch")
    if reason_set & {"forbidden_phrase", "weak_corpus_support"}:
        clusters.append("unsupported_or_forbidden")
    if "prompt_echo" in reason_set:
        clusters.append("prompt_format_failure")
    if reason_set & {"length_violation", "empty_reply"}:
        clusters.append("length_control")
    if "repetitive_reply" in reason_set:
        clusters.append("degenerate_generation")
    if not row.get("answerable", True) and not row.get("refusal_match", False):
        clusters.append("refusal_failure")
    if not clusters:
        clusters.append("unknown")
    return clusters


def _weak_breakdown(breakdown: dict[str, dict], field: str) -> list[dict]:
    weak: list[dict] = []
    for name, row in breakdown.items():
        num_examples = int(row.get("num_examples", 0))
        if num_examples <= 0:
            continue
        pass_rate = float(row.get("pass_rate", 0.0))
        support_match_rate = float(row.get("support_match_rate", 1.0))
        unsupported_claim_rate = float(row.get("unsupported_claim_rate", 0.0))
        prompt_echo_rate = float(row.get("prompt_echo_rate", 0.0))
        corpus_support_failure_rate = float(row.get("corpus_support_failure_rate", 0.0))
        if (
            row.get("num_failed", 0)
            and (
                pass_rate < 0.80
                or support_match_rate < 0.90
                or unsupported_claim_rate > 0.0
                or prompt_echo_rate > 0.0
                or corpus_support_failure_rate > 0.0
            )
        ):
            weak.append({
                field: name,
                "num_examples": num_examples,
                "num_failed": row.get("num_failed", 0),
                "pass_rate": pass_rate,
                "support_match_rate": support_match_rate,
                "unsupported_claim_rate": unsupported_claim_rate,
                "prompt_echo_rate": prompt_echo_rate,
                "corpus_support_failure_rate": corpus_support_failure_rate,
            })
    return sorted(
        weak,
        key=lambda item: (
            item["pass_rate"],
            item["support_match_rate"],
            -int(item["num_examples"]),
            str(item[field]),
        ),
    )


def _eval_recommendations(
    rows: list[dict],
    failure_counts: dict[str, int],
    cluster_counts: dict[str, int],
    weak_categories: list[dict],
    weak_splits: list[dict],
    weak_levels: list[dict],
) -> list[dict]:
    if not rows:
        return [{
            "priority": "high",
            "area": "eval",
            "message": "No eval rows were scored.",
            "action": "Add held-out eval JSONL rows before trusting this run.",
        }]

    recommendations: list[dict] = []
    if not failure_counts:
        recommendations.append({
            "priority": "medium",
            "area": "eval",
            "message": "All visible eval checks passed.",
            "action": "Add harder held-out prompts before scaling the same recipe.",
        })
        return recommendations

    if failure_counts.get("missing_required") or failure_counts.get("missing_any_group"):
        recommendations.append({
            "priority": "high",
            "area": "sft",
            "message": "The model missed required support phrases.",
            "action": "Add more varied SFT rows for the weakest categories, then rerun SFT before increasing base training.",
        })
    if failure_counts.get("wrong_choice"):
        recommendations.append({
            "priority": "high",
            "area": "choice_eval",
            "message": "Some multiple-choice rows selected the wrong label under logprob scoring.",
            "action": "Add more held-out-style multiple-choice SFT rows, then inspect choice logprob margins before changing decoding.",
        })
    if failure_counts.get("forbidden_phrase"):
        recommendations.append({
            "priority": "high",
            "area": "honesty",
            "message": "Some replies contained forbidden phrases.",
            "action": "Add refusal and boundary examples for those request types, and inspect the failed replies for unsupported claims.",
        })
    if failure_counts.get("prompt_echo"):
        recommendations.append({
            "priority": "high",
            "area": "decoding",
            "message": "Some replies echoed chat prompt structure.",
            "action": "Check stop handling and add SFT examples that answer without regenerating User:/Assistant: role labels.",
        })
    if failure_counts.get("empty_reply"):
        recommendations.append({
            "priority": "medium",
            "area": "decoding",
            "message": "Some replies were empty.",
            "action": "Increase max_new_tokens or inspect whether the checkpoint emits EOS too early.",
        })
    if failure_counts.get("missing_entity"):
        recommendations.append({
            "priority": "high",
            "area": "entities",
            "message": "The model missed required entities.",
            "action": "Add SFT rows that force exact names, product terms, locations, or domain entities to appear in answers.",
        })
    if failure_counts.get("length_violation"):
        recommendations.append({
            "priority": "medium",
            "area": "format",
            "message": "Some replies violated length constraints.",
            "action": "Add short/long answer format examples and keep max_new_tokens aligned with the eval row bounds.",
        })
    if failure_counts.get("weak_corpus_support"):
        recommendations.append({
            "priority": "high",
            "area": "grounding",
            "message": "Some replies had weak overlap with the support corpus.",
            "action": "Inspect those replies for drift; add corpus-supported SFT answers or mark the prompt as unanswerable.",
        })
    if failure_counts.get("low_reference_overlap"):
        recommendations.append({
            "priority": "medium",
            "area": "reference",
            "message": "Some replies had low token overlap with reference answers.",
            "action": "Use this as a soft diagnostic, then inspect whether the reply is genuinely wrong or just phrased differently.",
        })
    if failure_counts.get("repetitive_reply"):
        recommendations.append({
            "priority": "medium",
            "area": "decoding",
            "message": "Some replies repeated themselves.",
            "action": "Try a repetition penalty for manual generation, but fix training data if eval replies are degenerate under greedy decoding.",
        })

    if cluster_counts.get("refusal_failure"):
        recommendations.append({
            "priority": "high",
            "area": "refusal",
            "message": "At least one unanswerable item did not look like a refusal.",
            "action": "Add refusal examples that explicitly say the answer is not available from the provided material.",
        })

    weak_category_names = {str(item["category"]) for item in weak_categories}
    category_actions = {
        "required_words": "Add compositional examples with unseen word pairs and check that held-out eval pairs remain absent from SFT.",
        "story_generation": "Add broader story-generation SFT rows with diverse subjects, lessons, and endings.",
        "story_knowledge": "Add direct domain QA examples and verify the answers are supported by the corpus or SFT curriculum.",
        "outside_story_domain": "Add refusal examples with varied wording for live facts, medical, legal, financial, and private-data requests.",
        "memorization_probe": "Strengthen memorization-boundary refusals and keep canary checks enabled.",
        "continuation": "Add continuation examples that preserve named objects, characters, and the requested next event.",
    }
    for category, action in category_actions.items():
        if category in weak_category_names:
            recommendations.append({
                "priority": "medium",
                "area": category,
                "message": f"`{category}` is a weak eval category.",
                "action": action,
            })

    if weak_splits:
        weakest = weak_splits[0]
        recommendations.append({
            "priority": "medium",
            "area": "split",
            "message": f"Weakest eval split is `{weakest['split']}`.",
            "action": "Treat split-level failures as the next curriculum target instead of only optimizing aggregate pass rate.",
        })
    if weak_levels:
        weakest = weak_levels[0]
        recommendations.append({
            "priority": "medium",
            "area": "eval_ladder",
            "message": f"Weakest eval ladder level is `{weakest['level']}`.",
            "action": "Improve the lowest ladder level first; do not scale training because a harder level is failing.",
        })

    return _dedupe_recommendations(recommendations)


def _dedupe_recommendations(recommendations: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in recommendations:
        key = (str(item.get("area")), str(item.get("action")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _preview_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _contains_eval_phrase(text: str, phrase: str, case_sensitive: bool = False) -> bool:
    phrase = phrase.strip()
    if not phrase:
        return True

    literal_text = text if case_sensitive else text.lower()
    literal_phrase = phrase if case_sensitive else phrase.lower()
    if _requires_literal_phrase_match(phrase):
        return literal_phrase in literal_text

    phrase_tokens = _match_tokens(phrase, case_sensitive=case_sensitive)
    if not phrase_tokens:
        return literal_phrase in literal_text
    text_tokens = _match_tokens(text, case_sensitive=case_sensitive)
    return _contains_token_sequence(text_tokens, phrase_tokens)


def _requires_literal_phrase_match(phrase: str) -> bool:
    """Keep format labels like `Story:` precise while matching normal words by token."""
    return ":" in phrase and len(_match_tokens(phrase, case_sensitive=True)) == 1


def _match_tokens(text: str, case_sensitive: bool = False) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if case_sensitive:
        return tokens
    return [token.lower() for token in tokens]


def _contains_token_sequence(tokens: list[str], phrase_tokens: list[str]) -> bool:
    if not phrase_tokens:
        return True
    if len(phrase_tokens) > len(tokens):
        return False
    limit = len(tokens) - len(phrase_tokens) + 1
    for index in range(limit):
        if tokens[index:index + len(phrase_tokens)] == phrase_tokens:
            return True
    return False


def _length_violations(reply: str, item: ChatEvalItem) -> list[str]:
    words = _word_tokens(reply)
    violations: list[str] = []
    if item.min_words is not None and len(words) < item.min_words:
        violations.append(f"min_words:{item.min_words}")
    if item.max_words is not None and len(words) > item.max_words:
        violations.append(f"max_words:{item.max_words}")
    if item.min_chars is not None and len(reply) < item.min_chars:
        violations.append(f"min_chars:{item.min_chars}")
    if item.max_chars is not None and len(reply) > item.max_chars:
        violations.append(f"max_chars:{item.max_chars}")
    return violations


def _reference_metrics(reply: str, reference: str | None) -> dict[str, float | None]:
    if not reference:
        return {"token_f1": None, "rouge_l": None}
    reply_tokens = _word_tokens(reply)
    reference_tokens = _word_tokens(reference)
    if not reply_tokens or not reference_tokens:
        return {"token_f1": 0.0, "rouge_l": 0.0}
    reply_counts = _token_counts(reply_tokens)
    reference_counts = _token_counts(reference_tokens)
    overlap = sum(min(reply_counts.get(token, 0), reference_counts.get(token, 0)) for token in reference_counts)
    precision = overlap / len(reply_tokens)
    recall = overlap / len(reference_tokens)
    token_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    lcs = _lcs_length(reply_tokens, reference_tokens)
    rouge_l = lcs / len(reference_tokens)
    return {"token_f1": token_f1, "rouge_l": rouge_l}


def _repetition_diagnostics(reply: str, ngram_size: int = 3) -> dict[str, float]:
    tokens = _word_tokens(reply)
    if not tokens:
        return {"ngram_rate": 0.0, "unique_token_rate": 1.0}
    unique_token_rate = len(set(tokens)) / len(tokens)
    if len(tokens) < ngram_size:
        return {"ngram_rate": 0.0, "unique_token_rate": unique_token_rate}
    ngrams = [tuple(tokens[index:index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return {
        "ngram_rate": repeated / len(ngrams),
        "unique_token_rate": unique_token_rate,
    }


def _detect_refusal(reply: str, case_sensitive: bool = False) -> bool:
    text = reply if case_sensitive else reply.lower()
    phrases = (
        "i do not know",
        "i don't know",
        "cannot answer",
        "can't answer",
        "not in the provided",
        "provided material",
        "provided domain material",
        "outside",
        "not available",
    )
    return any((phrase if case_sensitive else phrase.lower()) in text for phrase in phrases)


def _corpus_support_token_set(corpus_text: str | None) -> frozenset[str] | None:
    if not corpus_text:
        return None
    return frozenset(_content_tokens(corpus_text))


def _corpus_support_diagnostics(
    reply: str,
    support_corpus_text: str | None = None,
    support_corpus_tokens: frozenset[str] | None = None,
) -> dict[str, float | int | None]:
    corpus_tokens = (
        support_corpus_tokens
        if support_corpus_tokens is not None
        else _corpus_support_token_set(support_corpus_text)
    )
    if not corpus_tokens:
        return {"rate": None, "supported": 0, "total": 0}
    reply_tokens = _content_tokens(reply)
    if not reply_tokens:
        return {"rate": 1.0, "supported": 0, "total": 0}
    supported = sum(1 for token in reply_tokens if token in corpus_tokens)
    return {
        "rate": supported / len(reply_tokens),
        "supported": supported,
        "total": len(reply_tokens),
    }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in _word_tokens(text)
        if len(token) >= 3 and token not in _STOPWORDS
    ]


def _token_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _normalize_for_match(text: str) -> str:
    return " ".join(_word_tokens(text))


def _metric_average(rows: list[dict], field: str) -> float | None:
    values = [_number(row.get(field)) for row in rows]
    return _average_values([value for value in values if value is not None])


def _append_metric(bucket: dict, field: str, value) -> None:
    number = _number(value)
    if number is not None:
        bucket[field].append(number)


def _average_values(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    return source.read_text(encoding="utf-8")


def _read_optional_support_token_set(path: str | None, read_chars: int = 1_000_000) -> frozenset[str] | None:
    if not path:
        return None
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    tokens: set[str] = set()
    carry = ""
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(read_chars)
            if not chunk:
                break
            text = carry + chunk
            carry = ""
            if text and text[-1].isalnum():
                tail = re.search(r"[A-Za-z0-9]+$", text)
                if tail is not None:
                    carry = tail.group(0)
                    text = text[:tail.start()]
            tokens.update(_content_tokens(text))
    if carry:
        tokens.update(_content_tokens(carry))
    return frozenset(tokens) if tokens else None


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 1.0


def _normalize_for_echo(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "you", "your", "are",
    "was", "were", "has", "have", "had", "not", "but", "can", "will", "would",
    "about", "into", "out", "what", "when", "where", "why", "how", "who",
    "assistant", "user", "story", "answer", "using", "provided", "material",
}


@torch.no_grad()
def _predict_choice_reply(
    model,
    tokenizer: Tokenizer,
    config: ChatEvalConfig,
    item: ChatEvalItem,
    precision_runtime=None,
) -> tuple[str, dict[str, float], dict[str, dict]]:
    """Predict a categorical answer by scoring each choice continuation."""
    prompt_ids = tokenizer.encode(render_chat_prompt([], item.user), add_bos=True)
    prefix_logits = None
    prefix_past_kv = None
    if len(prompt_ids) < model.config.context_size:
        device = next(model.parameters()).device
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with autocast_context(precision_runtime) if precision_runtime else torch.no_grad():
            prefix_logits, _, prefix_past_kv = model(prompt_tensor, use_cache=True)
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}
    for label in item.choice_labels:
        variants = _choice_continuation_candidates(tokenizer, label)
        if not variants:
            scores[label] = float("-inf")
            details[label] = {
                "best_variant": None,
                "best_avg_logprob": float("-inf"),
                "best_raw_logprob": float("-inf"),
                "token_count": 0,
                "variants": [],
            }
            continue
        variant_scores = []
        for candidate in variants:
            raw_logprob = _sequence_logprob(
                model,
                prompt_ids,
                list(candidate.token_ids),
                precision_runtime=precision_runtime,
                prefix_logits=prefix_logits,
                prefix_past_kv=prefix_past_kv,
            )
            avg_logprob = raw_logprob / len(candidate.token_ids)
            variant_scores.append({
                "variant": candidate.variant,
                "raw_logprob": raw_logprob,
                "avg_logprob": avg_logprob,
                "token_count": len(candidate.token_ids),
            })
        best = max(
            variant_scores,
            key=lambda row: (float(row["avg_logprob"]), float(row["raw_logprob"])),
        )
        scores[label] = float(best["avg_logprob"])
        details[label] = {
            "best_variant": best["variant"],
            "best_avg_logprob": best["avg_logprob"],
            "best_raw_logprob": best["raw_logprob"],
            "token_count": best["token_count"],
            "variants": variant_scores,
        }
    best_label = max(scores, key=scores.get)
    return best_label, scores, details


def _choice_continuation_candidates(tokenizer: Tokenizer, label: str) -> list[_ChoiceCandidate]:
    """Return fair answer-label continuations for choice likelihood scoring."""
    label = str(label).strip()
    if not label:
        return []

    raw_candidates = (
        ("space+eos", tokenizer.encode(f" {label}", add_bos=False) + [tokenizer.eos_id]),
        ("space", tokenizer.encode(f" {label}", add_bos=False)),
        ("bare+eos", tokenizer.encode(label, add_bos=False) + [tokenizer.eos_id]),
        ("bare", tokenizer.encode(label, add_bos=False)),
    )
    candidates: list[_ChoiceCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for variant, token_ids in raw_candidates:
        ids = tuple(int(token_id) for token_id in token_ids)
        if not ids or ids in seen:
            continue
        candidates.append(_ChoiceCandidate(variant=variant, token_ids=ids))
        seen.add(ids)
    return candidates


@torch.no_grad()
def _sequence_logprob(
    model,
    prompt_ids: list[int],
    continuation_ids: list[int],
    precision_runtime=None,
    prefix_logits=None,
    prefix_past_kv=None,
) -> float:
    """Score a short continuation under the model."""
    device = next(model.parameters()).device
    if (
        prefix_logits is not None
        and prefix_past_kv is not None
        and len(prompt_ids) + len(continuation_ids) <= model.config.context_size
    ):
        logits = prefix_logits
        past_kv = prefix_past_kv
        total = 0.0
        for index, token_id in enumerate(continuation_ids):
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            total += float(log_probs[token_id].item())
            if index == len(continuation_ids) - 1:
                break
            token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
            with autocast_context(precision_runtime) if precision_runtime else torch.no_grad():
                logits, _, past_kv = model(
                    token_tensor,
                    past_kv=past_kv,
                    use_cache=True,
                )
        return total

    context = list(prompt_ids)
    total = 0.0
    context_size = model.config.context_size
    for token_id in continuation_ids:
        input_ids = torch.tensor(
            [context[-context_size:]],
            dtype=torch.long,
            device=device,
        )
        with autocast_context(precision_runtime) if precision_runtime else torch.no_grad():
            logits, _ = model(input_ids)
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)
        total += float(log_probs[token_id].item())
        context.append(token_id)
    return total


@torch.no_grad()
def _generate_eval_reply(
    model,
    tokenizer: Tokenizer,
    config: ChatEvalConfig,
    item: ChatEvalItem,
    seed: int,
    precision_runtime=None,
) -> tuple[str, int]:
    prompt = render_chat_prompt([], item.user)
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_bos=True)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    max_new_tokens = _generation_max_new_tokens(config, tokenizer, item)
    with autocast_context(precision_runtime) if precision_runtime else torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            seed=seed,
            eos_id=tokenizer.eos_id,
        )
    new_token_ids = generated[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(new_token_ids)
    return extract_assistant_reply("", generated_text), max_new_tokens


def _generation_max_new_tokens(
    config: ChatEvalConfig,
    tokenizer: Tokenizer,
    item: ChatEvalItem,
) -> int:
    """Choose a generation cap from public row length constraints."""
    if config.max_new_tokens <= 0:
        return config.max_new_tokens

    candidates: list[int] = []
    if item.max_chars is not None:
        candidates.append(max(1, item.max_chars + 8))

    tokenizer_type = getattr(tokenizer, "tokenizer_type", "")
    if item.max_words is not None and tokenizer_type not in {"char"}:
        candidates.append(max(1, item.max_words * 6 + 8))

    if not candidates:
        return config.max_new_tokens
    constrained = max(8, min(candidates))
    return min(config.max_new_tokens, constrained)
