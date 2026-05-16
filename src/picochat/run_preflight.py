"""Long-run preflight checks for Picochat experiments.

This module is intentionally conservative. A smoke run should be easy to start,
but a long run should have enough data, clean held-out eval, and a sane budget
before it spends hours recycling the same tiny corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from picochat.data import CorpusBuildReport, CorpusPreviewReport, CorpusStats


LONG_RUN_BASE_STEPS = 5_000
LONG_RUN_PARAMETER_COUNT = 1_000_000
MIN_LONG_RUN_SFT_ROWS = 300
MIN_LONG_RUN_EVAL_ROWS = 80
DEFAULT_TARGET_PARAM_DATA_RATIO = 20.0
BASE_LR_REFERENCE_EFFECTIVE_BATCH = 8


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
    corpus_tokens_per_parameter: float | None
    base_effective_batch_size: int
    sft_effective_batch_size: int
    base_effective_tokens_per_step: int
    base_planned_tokens: int
    target_param_data_ratio: float
    target_training_tokens: int
    target_training_epochs: float | None
    recommended_base_steps: int
    recommended_base_tokens: int
    recommended_base_epochs: float | None
    planned_to_target_ratio: float | None
    sft_planned_example_updates: int
    estimated_base_epochs: float | None
    estimated_sft_example_epochs: float | None
    base_lr_reference_effective_batch: int
    base_lr_sqrt_scale: float
    sft_lr_sqrt_scale: float
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
        f"- Corpus tokens / parameter: {_format_optional_float(budget.corpus_tokens_per_parameter)}",
        f"- Corpus token estimate: {budget.corpus_token_note}",
        f"- Base effective batch: {budget.base_effective_batch_size}",
        f"- Base planned tokens: {budget.base_planned_tokens:,}",
        f"- Target token/parameter ratio: {budget.target_param_data_ratio:.2f}",
        f"- Target training tokens: {budget.target_training_tokens:,}",
        f"- Recommended base steps: {budget.recommended_base_steps:,}",
        f"- Recommended base epochs: {_format_optional_float(budget.recommended_base_epochs)}",
        f"- Planned / target tokens: {_format_optional_float(budget.planned_to_target_ratio)}",
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
    n_head = int(_value(config, "n_head", 4) or 4)
    n_kv_head = int(_value(config, "n_kv_head", n_head) or n_head)
    n_layer = int(_value(config, "n_layer", 2) or 2)
    tie_embeddings = bool(_value(config, "tie_embeddings", False))
    qk_norm = bool(_value(config, "qk_norm", False))
    parallel_residual = bool(_value(config, "parallel_residual", False))
    context_size = int(_value(config, "context_size", 128) or 128)
    base_steps = int(_value(config, "base_steps", 0) or 0)
    sft_steps = int(_value(config, "sft_steps", 0) or 0)
    base_effective_batch = int(_value(config, "base_batch_size", 1) or 1) * int(_value(config, "base_grad_accum_steps", 1) or 1)
    sft_effective_batch = int(_value(config, "sft_batch_size", 1) or 1) * int(_value(config, "sft_grad_accum_steps", 1) or 1)
    estimated_parameters = _estimate_parameters(
        vocab_size,
        n_embd,
        n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        tie_embeddings=tie_embeddings,
        qk_norm=qk_norm,
        parallel_residual=parallel_residual,
    )
    corpus_tokens, token_note = _estimate_corpus_tokens(stats, tokenizer_type, vocab_size)
    base_tokens_per_step = base_effective_batch * context_size
    base_planned_tokens = base_steps * base_tokens_per_step
    base_epochs = _safe_ratio(base_planned_tokens, corpus_tokens)
    target_ratio = max(
        1.0,
        float(_value(config, "target_param_data_ratio", DEFAULT_TARGET_PARAM_DATA_RATIO) or DEFAULT_TARGET_PARAM_DATA_RATIO),
    )
    target_training_tokens = max(1, int(round(estimated_parameters * target_ratio)))
    recommended_steps = max(1, math.ceil(target_training_tokens / max(1, base_tokens_per_step)))
    recommended_tokens = recommended_steps * base_tokens_per_step
    recommended_epochs = _safe_ratio(recommended_tokens, corpus_tokens)
    target_epochs = _safe_ratio(target_training_tokens, corpus_tokens)
    planned_to_target = _safe_ratio(base_planned_tokens, target_training_tokens)
    corpus_tokens_per_parameter = _safe_ratio(corpus_tokens, estimated_parameters)
    sft_example_updates = sft_steps * sft_effective_batch
    sft_epochs = _safe_ratio(sft_example_updates, sft_examples)
    base_lr_scale = math.sqrt(base_effective_batch / BASE_LR_REFERENCE_EFFECTIVE_BATCH)
    sft_lr_scale = math.sqrt(sft_effective_batch / BASE_LR_REFERENCE_EFFECTIVE_BATCH)
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
        corpus_tokens_per_parameter=corpus_tokens_per_parameter,
        base_effective_batch_size=base_effective_batch,
        sft_effective_batch_size=sft_effective_batch,
        base_effective_tokens_per_step=base_tokens_per_step,
        base_planned_tokens=base_planned_tokens,
        target_param_data_ratio=target_ratio,
        target_training_tokens=target_training_tokens,
        target_training_epochs=target_epochs,
        recommended_base_steps=recommended_steps,
        recommended_base_tokens=recommended_tokens,
        recommended_base_epochs=recommended_epochs,
        planned_to_target_ratio=planned_to_target,
        sft_planned_example_updates=sft_example_updates,
        estimated_base_epochs=base_epochs,
        estimated_sft_example_epochs=sft_epochs,
        base_lr_reference_effective_batch=BASE_LR_REFERENCE_EFFECTIVE_BATCH,
        base_lr_sqrt_scale=base_lr_scale,
        sft_lr_sqrt_scale=sft_lr_scale,
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
    base_dataset_mode = str(_value(config, "base_dataset_mode", "memory"))
    if base_dataset_mode == "sharded":
        document_split_check = _check(
            "document_split",
            "warn",
            "sharded",
            "document holdout or explicit sharded tradeoff",
            (
                "Sharded base data validates by token shard, not complete source document. "
                "Use memory mode for strict document holdout."
            ),
        )
    else:
        document_split_check = _check(
            "document_split",
            "pass" if split_mode == "document" else "block" if long_run else "warn",
            split_mode,
            "document",
            "Document split keeps validation text held out by source document.",
        )
    checks = [
        document_split_check,
    ]
    if _is_bpe_tokenizer(tokenizer_type):
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
            "warn" if long_run and not _is_bpe_tokenizer(tokenizer_type) else "pass",
            tokenizer_type,
            "BPE preferred for long runs",
            "Char/byte tokenizers are educational; compiled hf_bpe is the preferred long-run baseline.",
        ))
    return checks


def _base_budget_checks(config: Any, stats: CorpusStats, budget: RunBudgetPlan) -> list[RunPreflightCheck]:
    base_dataset_mode = str(_value(config, "base_dataset_mode", "memory"))
    split_mode = str(_value(config, "split_mode", "window"))
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
            "compute_optimal_horizon",
            _horizon_status(budget),
            _format_optional_float(budget.planned_to_target_ratio),
            "0.50-1.50 planned/target preferred; block >3.00",
            (
                f"Target is {budget.target_training_tokens:,} tokens "
                f"({budget.target_param_data_ratio:.1f} tokens/parameter); "
                f"recommended base steps: {budget.recommended_base_steps:,}."
            ),
        ),
        _check(
            "corpus_model_fit",
            _corpus_model_status(budget),
            _format_optional_float(budget.target_training_epochs),
            "<= 12 target epochs preferred; block >30",
            "If target training needs many corpus passes, add data or shrink the model before a long run.",
        ),
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
    checks.append(_check(
        "lr_batch_scaling",
        "warn" if budget.long_run and budget.base_effective_batch_size != BASE_LR_REFERENCE_EFFECTIVE_BATCH else "pass",
        f"{budget.base_lr_sqrt_scale:.2f}x",
        f"sqrt(effective_batch/{BASE_LR_REFERENCE_EFFECTIVE_BATCH})",
        (
            "If this LR was tuned at effective batch "
            f"{BASE_LR_REFERENCE_EFFECTIVE_BATCH}, sqrt-scaled base LR would be "
            f"{base_lr * budget.base_lr_sqrt_scale:.6g}."
        ),
    ))
    if base_dataset_mode == "sharded":
        checks.append(_check(
            "document_boundaries",
            "warn",
            "disabled in sharded mode",
            "memory document split",
            (
                "Sharded token data does not preserve per-document BOS/EOS validation boundaries; "
                "treat validation BPB as token-shard holdout."
            ),
        ))
    elif stats.num_documents > 1 and split_mode == "document":
        checks.append(_check(
            "document_boundaries",
            "pass",
            "bos/eos per document",
            "enabled with document split",
            "Picochat packs document split corpora with BOS/EOS around each document.",
        ))
    elif stats.num_documents > 1:
        checks.append(_check(
            "document_boundaries",
            "warn",
            "disabled",
            "document split",
            "Window-split base training does not preserve per-document BOS/EOS validation boundaries.",
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
    checks.append(_check(
        "sft_lr_batch_scaling",
        "warn" if budget.long_run and budget.sft_effective_batch_size != BASE_LR_REFERENCE_EFFECTIVE_BATCH else "pass",
        f"{budget.sft_lr_sqrt_scale:.2f}x",
        f"sqrt(effective_batch/{BASE_LR_REFERENCE_EFFECTIVE_BATCH})",
        (
            "If this SFT LR was tuned at effective batch "
            f"{BASE_LR_REFERENCE_EFFECTIVE_BATCH}, sqrt-scaled SFT LR would be "
            f"{sft_lr * budget.sft_lr_sqrt_scale:.6g}."
        ),
    ))
    checks.append(_check(
        "post_run_sft_fit_gate",
        "pass",
        ">=70% required after training",
        "block long-run recipe if failed",
        "Picochat will mark completed long runs unapproved when SFT fit is below 70%.",
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
            "pass",
            "post-run interpretation",
            "compare after eval",
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


def _estimate_parameters(
    vocab_size: int,
    n_embd: int,
    n_layer: int,
    *,
    n_head: int = 4,
    n_kv_head: int | None = None,
    tie_embeddings: bool = False,
    qk_norm: bool = False,
    parallel_residual: bool = False,
) -> int:
    # Approximate GPT parameter count before the tokenizer exists. Good enough for budget gating.
    n_kv_head = n_kv_head or n_head
    head_dim = max(1, n_embd // max(1, n_head))
    kv_dim = n_kv_head * head_dim
    embeddings_and_head = vocab_size * n_embd if tie_embeddings else 2 * vocab_size * n_embd
    attention = n_embd * (n_embd + 2 * kv_dim) + n_embd * n_embd
    mlp = 8 * n_embd * n_embd
    norm_params_per_block = 2 * n_embd if parallel_residual else 4 * n_embd
    blocks = n_layer * (attention + mlp + norm_params_per_block)
    qk_norm_params = n_layer * 2 * head_dim if qk_norm else 0
    final_norm = 2 * n_embd
    return int(embeddings_and_head + blocks + qk_norm_params + final_norm)


def _horizon_status(budget: RunBudgetPlan) -> str:
    ratio = budget.planned_to_target_ratio
    if ratio is None:
        return "block"
    if budget.long_run and ratio > 3.0:
        return "block"
    if ratio < 0.50 or ratio > 1.50:
        return "warn"
    return "pass"


def _corpus_model_status(budget: RunBudgetPlan) -> str:
    target_epochs = budget.target_training_epochs
    if target_epochs is None:
        return "block"
    if budget.long_run and target_epochs > 30:
        return "block"
    if target_epochs > 12:
        return "warn"
    return "pass"


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
    if _is_bpe_tokenizer(tokenizer_type):
        return 512
    return 128


def _is_bpe_tokenizer(tokenizer_type: str) -> bool:
    return tokenizer_type in {"bpe", "hf_bpe"}


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
