"""Long-run preflight checks for Picochat experiments.

This module is intentionally conservative. A smoke run should be easy to start,
but a long run should have enough data, clean held-out eval, and a sane budget
before it spends hours recycling the same tiny corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from picochat.data import CorpusBuildReport, CorpusPreviewReport, CorpusStats


LONG_RUN_BASE_STEPS = 5_000
LONG_RUN_PARAMETER_COUNT = 1_000_000
MIN_LONG_RUN_SFT_ROWS = 300
MIN_LONG_RUN_EVAL_ROWS = 80


@dataclass(frozen=True)
class RunPreflightCheck:
    name: str
    status: str
    metric: str
    threshold: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RunBudgetPlan:
    estimated_parameters: int
    estimated_corpus_tokens: int
    corpus_token_note: str
    base_effective_batch_size: int
    sft_effective_batch_size: int
    base_effective_tokens_per_step: int
    base_planned_tokens: int
    sft_planned_example_updates: int
    estimated_base_epochs: float | None
    estimated_sft_example_epochs: float | None
    long_run: bool
    long_run_reason: str

    def to_dict(self) -> dict[str, int | float | str | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class RunPreflightReport:
    status: str
    summary: str
    budget: RunBudgetPlan
    checks: tuple[RunPreflightCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "budget": self.budget.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "blocking_checks": [check.to_dict() for check in self.checks if check.status == "block"],
            "warning_checks": [check.to_dict() for check in self.checks if check.status == "warn"],
        }


def assess_run_preflight(
    config: Any,
    corpus: CorpusBuildReport | CorpusPreviewReport,
) -> RunPreflightReport:
    """Return the combined data, budget, SFT, and eval gate for one run plan."""
    budget = estimate_run_budget(config, corpus.stats, corpus.chat_data.num_examples)
    checks: list[RunPreflightCheck] = []
    checks.extend(_data_source_checks(corpus, budget.long_run))
    checks.extend(_tokenizer_checks(config, corpus.stats, budget.long_run))
    checks.extend(_base_budget_checks(config, corpus.stats, budget))
    checks.extend(_sft_checks(config, corpus, budget))
    checks.extend(_eval_checks(corpus, budget.long_run))
    checks.extend(_closed_book_checks())

    status = _status_from_checks(checks)
    if status == "blocked":
        prefix = "Long-run" if budget.long_run else "Run"
        summary = f"{prefix} preflight blocked this plan. Fix the blocking checks before spending training time."
    elif status == "warn":
        summary = "Long-run preflight has warnings. A smoke run is fine; inspect warnings before an overnight run."
    else:
        summary = "Long-run preflight passed. This plan is eligible for a controlled long run."
    return RunPreflightReport(
        status=status,
        summary=summary,
        budget=budget,
        checks=tuple(checks),
    )


def preflight_markdown(report: RunPreflightReport) -> str:
    """Render a concise checklist for CLI output and Markdown reports."""
    lines = [
        "# Picochat Run Preflight",
        "",
        f"- Status: `{report.status}`",
        f"- Summary: {report.summary}",
        "",
        "## Budget",
        "",
    ]
    budget = report.budget
    lines.extend([
        f"- Estimated parameters: {budget.estimated_parameters:,}",
        f"- Estimated corpus tokens: {budget.estimated_corpus_tokens:,}",
        f"- Corpus token estimate: {budget.corpus_token_note}",
        f"- Base effective batch: {budget.base_effective_batch_size}",
        f"- Base planned tokens: {budget.base_planned_tokens:,}",
        f"- Base estimated epochs: {_format_optional_float(budget.estimated_base_epochs)}",
        f"- SFT effective batch: {budget.sft_effective_batch_size}",
        f"- SFT planned example updates: {budget.sft_planned_example_updates:,}",
        f"- SFT estimated example epochs: {_format_optional_float(budget.estimated_sft_example_epochs)}",
        f"- Long run: {'yes' if budget.long_run else 'no'} ({budget.long_run_reason})",
        "",
        "## Checklist",
        "",
        "| Status | Check | Metric | Threshold | Message |",
        "| --- | --- | ---: | --- | --- |",
    ])
    for check in report.checks:
        lines.append(
            f"| `{check.status}` | `{check.name}` | {check.metric} | {check.threshold} | {check.message} |"
        )
    lines.append("")
    return "\n".join(lines)


def estimate_run_budget(config: Any, stats: CorpusStats, sft_examples: int) -> RunBudgetPlan:
    """Estimate model size and exposure before tokenizer training starts."""
    tokenizer_type = str(_value(config, "tokenizer_type", "char"))
    vocab_size = _tokenizer_vocab_estimate(config)
    n_embd = int(_value(config, "n_embd", 64) or 64)
    n_layer = int(_value(config, "n_layer", 2) or 2)
    context_size = int(_value(config, "context_size", 128) or 128)
    base_steps = int(_value(config, "base_steps", 0) or 0)
    sft_steps = int(_value(config, "sft_steps", 0) or 0)
    base_effective_batch = int(_value(config, "base_batch_size", 1) or 1) * int(_value(config, "base_grad_accum_steps", 1) or 1)
    sft_effective_batch = int(_value(config, "sft_batch_size", 1) or 1) * int(_value(config, "sft_grad_accum_steps", 1) or 1)
    estimated_parameters = _estimate_parameters(vocab_size, n_embd, n_layer)
    corpus_tokens, token_note = _estimate_corpus_tokens(stats, tokenizer_type, vocab_size)
    base_tokens_per_step = base_effective_batch * context_size
    base_planned_tokens = base_steps * base_tokens_per_step
    base_epochs = _safe_ratio(base_planned_tokens, corpus_tokens)
    sft_example_updates = sft_steps * sft_effective_batch
    sft_epochs = _safe_ratio(sft_example_updates, sft_examples)
    long_run = base_steps >= LONG_RUN_BASE_STEPS or estimated_parameters >= LONG_RUN_PARAMETER_COUNT
    if base_steps >= LONG_RUN_BASE_STEPS and estimated_parameters >= LONG_RUN_PARAMETER_COUNT:
        long_reason = f">={LONG_RUN_BASE_STEPS} base steps and >={LONG_RUN_PARAMETER_COUNT:,} params"
    elif base_steps >= LONG_RUN_BASE_STEPS:
        long_reason = f">={LONG_RUN_BASE_STEPS} base steps"
    elif estimated_parameters >= LONG_RUN_PARAMETER_COUNT:
        long_reason = f">={LONG_RUN_PARAMETER_COUNT:,} params"
    else:
        long_reason = "smoke/local run"
    return RunBudgetPlan(
        estimated_parameters=estimated_parameters,
        estimated_corpus_tokens=corpus_tokens,
        corpus_token_note=token_note,
        base_effective_batch_size=base_effective_batch,
        sft_effective_batch_size=sft_effective_batch,
        base_effective_tokens_per_step=base_tokens_per_step,
        base_planned_tokens=base_planned_tokens,
        sft_planned_example_updates=sft_example_updates,
        estimated_base_epochs=base_epochs,
        estimated_sft_example_epochs=sft_epochs,
        long_run=long_run,
        long_run_reason=long_reason,
    )


def _data_source_checks(
    corpus: CorpusBuildReport | CorpusPreviewReport,
    long_run: bool,
) -> list[RunPreflightCheck]:
    stats = corpus.stats
    checks = [
        _check(
            "corpus_readiness",
            "block" if corpus.readiness.status == "blocked" else "warn" if corpus.readiness.status == "caution" else "pass",
            corpus.readiness.status,
            "ready",
            corpus.readiness.summary,
        ),
        _check(
            "documents",
            "block" if stats.num_documents == 0 else "warn" if long_run and stats.num_documents < 100 else "pass",
            str(stats.num_documents),
            ">= 100 for long run",
            "Long runs should hold out many complete documents, not replay a tiny folder.",
        ),
        _check(
            "duplicate_documents",
            "block" if long_run and stats.duplicate_document_rate > 0.10 else "warn" if stats.duplicate_document_rate > 0.03 else "pass",
            _percent(stats.duplicate_document_rate),
            "<= 3% preferred",
            "Duplicate documents inflate apparent progress and can leak eval-like text.",
        ),
        _check(
            "duplicate_lines",
            "block" if long_run and stats.duplicate_line_rate > 0.30 else "warn" if stats.duplicate_line_rate > 0.15 else "pass",
            _percent(stats.duplicate_line_rate),
            "<= 15% preferred",
            "High line replay makes memorization easier than learning.",
        ),
    ]
    if stats.num_characters < 1_000_000 and long_run:
        checks.append(_check(
            "corpus_size",
            "warn",
            f"{stats.num_characters:,} chars",
            ">= 1,000,000 chars preferred",
            "This can run, but a long experiment on a tiny corpus mostly measures replay.",
        ))
    else:
        checks.append(_check(
            "corpus_size",
            "pass" if stats.num_characters > 0 else "block",
            f"{stats.num_characters:,} chars",
            "> 0",
            "The base model needs real raw text before SFT can help.",
        ))
    return checks


def _tokenizer_checks(config: Any, stats: CorpusStats, long_run: bool) -> list[RunPreflightCheck]:
    tokenizer_type = str(_value(config, "tokenizer_type", "char"))
    vocab_size = _value(config, "tokenizer_vocab_size", None)
    split_mode = str(_value(config, "split_mode", "window"))
    checks = [
        _check(
            "document_split",
            "pass" if split_mode == "document" else "block" if long_run else "warn",
            split_mode,
            "document",
            "Document split keeps validation text held out by source document.",
        ),
    ]
    if tokenizer_type == "bpe":
        target = _tokenizer_vocab_estimate(config)
        checks.append(_check(
            "bpe_vocab_size",
            "block" if long_run and target < 512 else "warn" if target < 1024 else "pass",
            str(target),
            ">= 1024 preferred for long runs",
            "Very small BPE vocabularies compress poorly and make long-context learning harder.",
        ))
        chars_per_token = _safe_ratio(stats.num_characters, _estimate_corpus_tokens(stats, tokenizer_type, target)[0])
        checks.append(_check(
            "tokenizer_corpus_fit",
            "pass" if chars_per_token and chars_per_token >= 2.0 else "warn",
            _format_optional_float(chars_per_token),
            ">= 2 chars/token rough estimate",
            "This is a pre-tokenizer estimate; actual BPB after training is the real signal.",
        ))
        if vocab_size is None and long_run:
            checks.append(_check(
                "explicit_vocab",
                "warn",
                "default",
                "explicit --tokenizer-vocab-size",
                "Long runs should record an intentional tokenizer vocabulary target.",
            ))
    else:
        checks.append(_check(
            "tokenizer_type",
            "warn" if long_run and tokenizer_type != "bpe" else "pass",
            tokenizer_type,
            "bpe preferred for long runs",
            "Char/byte tokenizers are educational, but BPE is the better long-run baseline.",
        ))
    return checks


def _base_budget_checks(config: Any, stats: CorpusStats, budget: RunBudgetPlan) -> list[RunPreflightCheck]:
    epochs = budget.estimated_base_epochs
    status = "pass"
    if epochs is None:
        status = "block"
    elif budget.long_run and epochs > 30:
        status = "block"
    elif epochs > 12:
        status = "warn"
    elif budget.long_run and epochs < 0.25:
        status = "warn"
    checks = [
        _check(
            "base_exposure",
            status,
            _format_optional_float(epochs),
            "0.25-12 preferred; block >30",
            "Too many passes over the same text makes long-run score gains suspect.",
        ),
        _check(
            "effective_batch",
            "pass" if budget.base_effective_batch_size >= 8 else "warn",
            str(budget.base_effective_batch_size),
            ">= 8",
            "Effective batch combines batch size and gradient accumulation.",
        ),
    ]
    base_lr = float(_value(config, "base_learning_rate", 0.0) or 0.0)
    if base_lr <= 0:
        checks.append(_check("base_lr", "block", str(base_lr), "> 0", "Base training needs a positive learning rate."))
    elif base_lr > 0.003:
        checks.append(_check("base_lr", "warn", str(base_lr), "<= 0.003", "High LR can destabilize tiny transformers."))
    else:
        checks.append(_check("base_lr", "pass", str(base_lr), "0 < lr <= 0.003", "Base LR is in a plausible local range."))
    if stats.num_documents > 1:
        checks.append(_check(
            "document_boundaries",
            "pass",
            "bos/eos per document",
            "enabled with document split",
            "Picochat packs document split corpora with BOS/EOS around each document.",
        ))
    return checks


def _sft_checks(config: Any, corpus: CorpusBuildReport | CorpusPreviewReport, budget: RunBudgetPlan) -> list[RunPreflightCheck]:
    chat = corpus.chat_data
    epochs = budget.estimated_sft_example_epochs
    checks = [
        _check(
            "chat_sft_readiness",
            "block" if chat.status == "blocked" else "warn" if chat.status == "caution" else "pass",
            chat.status,
            "ready",
            chat.summary,
        ),
        _check(
            "assistant_only_masking",
            "pass",
            "enabled",
            "assistant tokens only",
            "Picochat masks user/system prompt tokens during SFT, matching the no-cheating chat objective.",
        ),
        _check(
            "sft_rows",
            "block" if budget.long_run and chat.num_examples < MIN_LONG_RUN_SFT_ROWS else "warn" if chat.num_examples < 500 else "pass",
            str(chat.num_examples),
            f">= {MIN_LONG_RUN_SFT_ROWS} for long run; >= 500 preferred",
            "Long runs need enough SFT rows to separate behavior learning from replay.",
        ),
        _check(
            "sft_duplicate_prompts",
            "block" if budget.long_run and chat.duplicate_user_rate > 0.25 else "warn" if chat.duplicate_user_rate > 0.10 else "pass",
            _percent(chat.duplicate_user_rate),
            "<= 10% preferred",
            "Over-replayed prompts inflate SFT fit without improving behavior.",
        ),
        _check(
            "sft_category_balance",
            "warn" if budget.long_run and len(chat.categories) < 4 else "pass",
            str(len(chat.categories)),
            ">= 4 categories preferred",
            "Category coverage helps reveal which behavior the SFT stage actually teaches.",
        ),
    ]
    if epochs is None:
        checks.append(_check("sft_exposure", "block", "--", "<= 25 preferred", "No usable SFT rows means SFT cannot be budgeted."))
    else:
        checks.append(_check(
            "sft_exposure",
            "block" if budget.long_run and epochs > 60 else "warn" if epochs > 25 else "pass",
            _format_optional_float(epochs),
            "<= 25 preferred; block >60",
            "Too many SFT example passes can make the model parrot the tuning set.",
        ))
    base_lr = float(_value(config, "base_learning_rate", 0.0) or 0.0)
    sft_lr = float(_value(config, "sft_learning_rate", 0.0) or 0.0)
    checks.append(_check(
        "sft_lr",
        "warn" if sft_lr > base_lr and budget.long_run else "pass" if sft_lr > 0 else "block",
        str(sft_lr),
        "positive; usually <= base LR for long run",
        "SFT should steer behavior without erasing the base model.",
    ))
    return checks


def _eval_checks(corpus: CorpusBuildReport | CorpusPreviewReport, long_run: bool) -> list[RunPreflightCheck]:
    eval_data = corpus.eval_data
    rules = eval_data.must_include_rules + eval_data.must_include_any_groups + eval_data.must_not_include_rules
    eval_status = "pass"
    if eval_data.status == "blocked":
        eval_status = "block" if long_run or "pass/fail rules" not in eval_data.summary else "warn"
    elif eval_data.status == "caution":
        eval_status = "warn"
    return [
        _check(
            "eval_readiness",
            eval_status,
            eval_data.status,
            "ready",
            eval_data.summary,
        ),
        _check(
            "eval_rows",
            "block" if long_run and eval_data.num_items < MIN_LONG_RUN_EVAL_ROWS else "warn" if eval_data.num_items < 100 else "pass",
            str(eval_data.num_items),
            f">= {MIN_LONG_RUN_EVAL_ROWS} for long run; >= 100 preferred",
            "Small eval files are useful for smoke tests, not for overnight-run claims.",
        ),
        _check(
            "eval_rules",
            "block" if long_run and eval_data.num_items and rules < eval_data.num_items
            else "warn" if eval_data.num_items and rules < eval_data.num_items
            else "pass",
            str(rules),
            ">= eval rows",
            "Every transparent eval item needs visible pass/fail evidence.",
        ),
        _check(
            "unanswerable_eval",
            "warn" if long_run and eval_data.unanswerable_items < 2 else "pass",
            str(eval_data.unanswerable_items),
            ">= 2 preferred",
            "Closed-book models need refusal/boundary checks, not only answerable prompts.",
        ),
        _check(
            "eval_splits",
            "warn" if long_run and len(eval_data.splits) < 2 else "pass",
            str(len(eval_data.splits)),
            ">= 2 split labels preferred",
            "Split labels make smoke, held-out, adversarial, and memorization probes visible.",
        ),
    ]


def _closed_book_checks() -> list[RunPreflightCheck]:
    return [
        _check(
            "closed_book_eval",
            "pass",
            "enabled",
            "no retrieval in generation",
            "The eval generator receives only the prompt and checkpoint; support corpus is diagnostic scoring metadata only.",
        ),
        _check(
            "random_baseline",
            "warn",
            "not recorded pre-run",
            "record after eval",
            "For choice-heavy evals, compare against random choice accuracy before claiming intelligence.",
        ),
    ]


def _status_from_checks(checks: list[RunPreflightCheck]) -> str:
    if any(check.status == "block" for check in checks):
        return "blocked"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "ready"


def _check(name: str, status: str, metric: str, threshold: str, message: str) -> RunPreflightCheck:
    if status not in {"pass", "warn", "block"}:
        raise ValueError(f"invalid preflight status: {status}")
    return RunPreflightCheck(name=name, status=status, metric=metric, threshold=threshold, message=message)


def _estimate_parameters(vocab_size: int, n_embd: int, n_layer: int) -> int:
    # Approximate GPT parameter count before the tokenizer exists. Good enough for budget gating.
    embeddings_and_head = 2 * vocab_size * n_embd
    blocks = n_layer * (12 * n_embd * n_embd + 4 * n_embd)
    final_norm = 2 * n_embd
    return int(embeddings_and_head + blocks + final_norm)


def _estimate_corpus_tokens(stats: CorpusStats, tokenizer_type: str, vocab_size: int) -> tuple[int, str]:
    if stats.num_characters <= 0:
        return 0, "empty corpus"
    if tokenizer_type in {"char", "byte"}:
        return max(1, stats.num_characters + 2 * stats.num_documents), f"{tokenizer_type} tokenizer ~= characters"
    if vocab_size >= 2048:
        ratio = 0.28
    elif vocab_size >= 1024:
        ratio = 0.35
    elif vocab_size >= 512:
        ratio = 0.45
    else:
        ratio = 0.60
    return max(1, int(stats.num_characters * ratio) + 2 * stats.num_documents), (
        f"rough BPE estimate at {ratio:.2f} tokens/char before tokenizer training"
    )


def _tokenizer_vocab_estimate(config: Any) -> int:
    tokenizer_type = str(_value(config, "tokenizer_type", "char"))
    vocab_size = _value(config, "tokenizer_vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    if tokenizer_type == "byte":
        return 260
    if tokenizer_type == "bpe":
        return 512
    return 128


def _value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _safe_ratio(numerator: int | float, denominator: int | float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_optional_float(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"
