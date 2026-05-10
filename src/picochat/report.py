"""Human-readable report generation for Picochat runs."""

from __future__ import annotations


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_optional_float(value: float | None) -> str:
    return "--" if value is None else format_float(value)


def loss_diagnostics(losses: list[dict]) -> dict:
    """Summarize a tiny loss trace with stable, explainable diagnostics."""
    if not losses:
        return {
            "status": "unknown",
            "summary": "No loss points were recorded.",
            "first_step": None,
            "final_step": None,
            "best_val_step": None,
            "best_val_loss": None,
            "final_train_loss": None,
            "final_val_loss": None,
            "final_gap": None,
            "val_regression": None,
            "train_improvement": None,
        }

    first = losses[0]
    final = losses[-1]
    valid_val = [item for item in losses if _number(item.get("val_loss")) is not None]
    best = min(valid_val, key=lambda item: _number(item.get("val_loss"))) if valid_val else final
    first_train = _number(first.get("train_loss"))
    final_train = _number(final.get("train_loss"))
    final_val = _number(final.get("val_loss"))
    best_val = _number(best.get("val_loss"))
    final_gap = _diff(final_val, final_train)
    val_regression = _diff(final_val, best_val)
    train_improvement = _diff(first_train, final_train)
    status, summary = _loss_status(final_gap, val_regression)
    return {
        "status": status,
        "summary": summary,
        "first_step": first.get("step"),
        "final_step": final.get("step"),
        "best_val_step": best.get("step"),
        "best_val_loss": best_val,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "final_gap": final_gap,
        "val_regression": val_regression,
        "train_improvement": train_improvement,
    }


def training_report_markdown(report: dict) -> str:
    """Render a base-training report as Markdown."""
    config = report["config"]
    dataset = report["dataset"]
    model = report["model"]
    losses = report["losses"]
    sample = report["sample"]
    diagnostics = report.get("loss_diagnostics") or loss_diagnostics(losses)

    lines: list[str] = []
    lines.append("# Picochat Base Training Report")
    lines.append("")
    lines.append("This run trained a tiny decoder-only language model with next-token prediction.")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Corpus: `{config['corpus_path']}`")
    lines.append(f"- Tokenizer: `{config['tokenizer_path']}`")
    lines.append(f"- Tokens: {dataset['num_tokens']}")
    lines.append(f"- Context size: {dataset['context_size']}")
    lines.append(f"- Training windows: {dataset['num_sequences']}")
    lines.append(f"- Train windows: {dataset['train_sequences']}")
    lines.append(f"- Validation windows: {dataset['val_sequences']}")
    if dataset.get("train_tokens") is not None:
        lines.append(f"- Train tokens: {dataset.get('train_tokens')}")
        lines.append(f"- Validation tokens: {dataset.get('val_tokens')}")
    lines.append(f"- Split mode: `{dataset.get('split_mode', 'window')}`")
    lines.append(f"- Split reason: {dataset.get('split_reason', 'not recorded')}")
    if dataset.get("val_documents") is not None:
        lines.append(f"- Held-out documents: {dataset.get('val_documents')} / {dataset.get('num_documents')}")
    if dataset.get("canaries_enabled"):
        lines.append(f"- Train-only canaries: {len(dataset.get('canary_values', []))}")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    model_config = model["config"]
    lines.append(f"- Parameters: {model['num_parameters']:,}")
    lines.append(f"- Vocabulary size: {model_config['vocab_size']}")
    lines.append(f"- Layers: {model_config['n_layer']}")
    lines.append(f"- Embedding size: {model_config['n_embd']}")
    lines.append(f"- Attention heads: {model_config['n_head']}")
    lines.append(f"- Dropout: {model_config['dropout']}")
    lines.append("")

    lines.append("## Training")
    lines.append("")
    lines.append(f"- Steps: {config['max_steps']}")
    lines.append(f"- Batch size: {config['batch_size']}")
    lines.append(f"- Learning rate: {config['learning_rate']}")
    lines.append(f"- Validation fraction: {config['val_fraction']}")
    lines.append(f"- Early stop patience: {config.get('early_stop_patience', 0)}")
    lines.append(f"- Max minutes: {config.get('max_minutes') or 'disabled'}")
    lines.append(f"- Device: `{config['device']}`")
    lines.append("")
    lines.append("| Step | Train Loss | Train BPB | Val Loss | Val BPB | Elapsed Seconds |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in losses:
        lines.append(
            f"| {item['step']} | {format_float(item['train_loss'])} | "
            f"{format_optional_float(item.get('train_bpb'))} | "
            f"{format_float(item['val_loss'])} | "
            f"{format_optional_float(item.get('val_bpb'))} | "
            f"{format_float(item['elapsed_sec'])} |"
        )
    lines.append("")

    if report.get("coverage"):
        coverage = report["coverage"]
        lines.append("## Coverage")
        lines.append("")
        lines.append(f"- Stop reason: `{report.get('stop_reason', 'unknown')}`")
        lines.append(f"- Actual steps: {coverage.get('actual_steps')} / {coverage.get('planned_steps')}")
        lines.append(f"- Estimated training tokens: {coverage.get('actual_training_tokens')}")
        lines.append(f"- Estimated train epochs: {format_optional_float(coverage.get('estimated_train_epochs'))}")
        lines.append(f"- Estimated dataset passes: {format_optional_float(coverage.get('estimated_dataset_passes'))}")
        lines.append("")

    lines.append("## Loss Diagnostics")
    lines.append("")
    lines.append(f"- Status: `{diagnostics['status']}`")
    lines.append(f"- Summary: {diagnostics['summary']}")
    lines.append(f"- Best validation step: {diagnostics['best_val_step']}")
    lines.append(f"- Best validation loss: {format_optional_float(diagnostics['best_val_loss'])}")
    lines.append(f"- Final train/val gap: {format_optional_float(diagnostics['final_gap'])}")
    lines.append(f"- Validation regression from best step: {format_optional_float(diagnostics['val_regression'])}")
    lines.append(f"- Train loss improvement: {format_optional_float(diagnostics['train_improvement'])}")
    lines.append("")

    if report.get("memorization"):
        memorization = report["memorization"]
        lines.append("## Memorization Diagnostics")
        lines.append("")
        lines.append(f"- Status: `{memorization['status']}`")
        lines.append(f"- Summary: {memorization['summary']}")
        lines.append(f"- N-gram size: {memorization['ngram_size']}")
        lines.append(f"- Generated tokens checked: {memorization['generated_tokens']}")
        lines.append(f"- Train overlap rate: {format_float(memorization['train_overlap_rate'] * 100)}%")
        lines.append(f"- Held-out overlap rate: {format_float(memorization['validation_overlap_rate'] * 100)}%")
        lines.append(f"- Longest train overlap tokens: {memorization['longest_train_overlap_tokens']}")
        lines.append(f"- Longest held-out overlap tokens: {memorization['longest_validation_overlap_tokens']}")
        if memorization.get("canary_hits"):
            lines.append(f"- Canary hits: {_inline_list(memorization['canary_hits'])}")
        else:
            lines.append("- Canary hits: none")
        lines.append("")

    lines.append("## Sample")
    lines.append("")
    lines.append("The text below is generated by the checkpoint at the end of training.")
    lines.append("")
    lines.append("```text")
    lines.append(sample)
    lines.append("```")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    if report.get("best_checkpoint"):
        best = report["best_checkpoint"]
        lines.append(
            f"- Best validation checkpoint: `{best['path']}` "
            f"(step {best['step']}, val {format_optional_float(best.get('val_loss'))}, "
            f"bpb {format_optional_float(best.get('val_bpb'))})"
        )
    lines.append("- Machine-readable report: `train_report.json`")
    lines.append("- Generated sample: `sample.txt`")
    if report.get("canary_probe"):
        lines.append("- Canary probe: `canary_probe.txt`")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "A high or nonsensical sample is normal for very short runs. "
        "The important thing is that this report records the data, model, "
        "training loss, sample, and checkpoint so the run can be inspected."
    )
    lines.append("")
    return "\n".join(lines)


def sft_report_markdown(report: dict) -> str:
    """Render a supervised fine-tuning report as Markdown."""
    config = report["config"]
    dataset = report["dataset"]
    model = report["model"]
    losses = report["losses"]
    sample = report["sample"]
    base_checkpoint = report["base_checkpoint"]
    diagnostics = report.get("loss_diagnostics") or loss_diagnostics(losses)
    best_checkpoint = report.get("best_checkpoint")

    lines: list[str] = []
    lines.append("# Picochat SFT Report")
    lines.append("")
    lines.append(
        "This run fine-tuned a base checkpoint on chat examples using "
        "supervised next-token prediction over assistant replies."
    )
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Chat data: `{config['input_path']}`")
    lines.append(f"- Tokenizer: `{config['tokenizer_path']}`")
    lines.append(f"- Examples: {dataset['num_examples']}")
    lines.append(f"- Context size: {dataset['context_size']}")
    lines.append(f"- Supervised answer tokens: {dataset['supervised_tokens']}")
    lines.append(f"- Truncated examples: {dataset.get('truncated_examples', 0)}")
    lines.append(f"- Train examples: {dataset['train_examples']}")
    lines.append(f"- Validation examples: {dataset['val_examples']}")
    lines.append("")

    lines.append("## Base Checkpoint")
    lines.append("")
    lines.append(f"- Path: `{base_checkpoint['path']}`")
    lines.append(f"- Step: {base_checkpoint['step']}")
    lines.append(f"- Train loss: {base_checkpoint['train_loss']}")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    model_config = model["config"]
    lines.append(f"- Parameters: {model['num_parameters']:,}")
    lines.append(f"- Vocabulary size: {model_config['vocab_size']}")
    lines.append(f"- Layers: {model_config['n_layer']}")
    lines.append(f"- Embedding size: {model_config['n_embd']}")
    lines.append(f"- Attention heads: {model_config['n_head']}")
    lines.append("")

    lines.append("## Training")
    lines.append("")
    lines.append(f"- Steps: {config['max_steps']}")
    lines.append(f"- Batch size: {config['batch_size']}")
    lines.append(f"- Learning rate: {config['learning_rate']}")
    lines.append(f"- Early stop patience: {config.get('early_stop_patience', 0)}")
    lines.append(f"- Max minutes: {config.get('max_minutes') or 'disabled'}")
    lines.append(f"- Device: `{config['device']}`")
    lines.append("")
    lines.append("| Step | Train Loss | Train BPB | Val Loss | Val BPB | Elapsed Seconds |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in losses:
        lines.append(
            f"| {item['step']} | {format_float(item['train_loss'])} | "
            f"{format_optional_float(item.get('train_bpb'))} | "
            f"{format_float(item['val_loss'])} | "
            f"{format_optional_float(item.get('val_bpb'))} | "
            f"{format_float(item['elapsed_sec'])} |"
        )
    lines.append("")

    if report.get("coverage"):
        coverage = report["coverage"]
        lines.append("## Coverage")
        lines.append("")
        lines.append(f"- Stop reason: `{report.get('stop_reason', 'unknown')}`")
        lines.append(f"- Actual steps: {coverage.get('actual_steps')} / {coverage.get('planned_steps')}")
        lines.append(f"- Estimated chat example updates: {coverage.get('actual_example_updates')}")
        lines.append(f"- Estimated train epochs: {format_optional_float(coverage.get('estimated_train_epochs'))}")
        lines.append(f"- Estimated dataset passes: {format_optional_float(coverage.get('estimated_dataset_passes'))}")
        lines.append("")

    lines.append("## Loss Diagnostics")
    lines.append("")
    lines.append(f"- Status: `{diagnostics['status']}`")
    lines.append(f"- Summary: {diagnostics['summary']}")
    lines.append(f"- Best validation step: {diagnostics['best_val_step']}")
    lines.append(f"- Best validation loss: {format_optional_float(diagnostics['best_val_loss'])}")
    lines.append(f"- Final train/val gap: {format_optional_float(diagnostics['final_gap'])}")
    lines.append(f"- Validation regression from best step: {format_optional_float(diagnostics['val_regression'])}")
    lines.append(f"- Train loss improvement: {format_optional_float(diagnostics['train_improvement'])}")
    lines.append("")

    lines.append("## Sample")
    lines.append("")
    lines.append("The text below starts from the configured chat prompt after SFT.")
    lines.append("")
    lines.append("```text")
    lines.append(sample)
    lines.append("```")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    if best_checkpoint:
        lines.append(
            f"- Best validation checkpoint: `{best_checkpoint['path']}` "
            f"(step {best_checkpoint['step']}, val {format_optional_float(best_checkpoint.get('val_loss'))}, "
            f"bpb {format_optional_float(best_checkpoint.get('val_bpb'))})"
        )
    lines.append("- Machine-readable report: `sft_report.json`")
    lines.append("- Generated sample: `sample.txt`")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "SFT does not create knowledge by itself. It teaches the model the desired "
        "conversation behavior present in the examples."
    )
    lines.append("")
    return "\n".join(lines)


def chat_eval_report_markdown(report: dict) -> str:
    """Render a transparent chat evaluation report as Markdown."""
    config = report["config"]
    checkpoint = report["checkpoint"]
    summary = report["summary"]

    lines: list[str] = []
    lines.append("# Picochat Chat Eval Report")
    lines.append("")
    lines.append(
        "This eval uses visible substring checks. It is intentionally simple: "
        "a reply passes when all required phrases appear, each any-phrase group "
        "has at least one match, and no forbidden phrases appear."
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Eval data: `{config['input_path']}`")
    lines.append(f"- Tokenizer: `{config['tokenizer_path']}`")
    lines.append(f"- Checkpoint: `{checkpoint['path']}`")
    lines.append(f"- Checkpoint step: {checkpoint['step']}")
    lines.append(f"- Temperature: {config['temperature']}")
    lines.append(f"- Max new tokens: {config['max_new_tokens']}")
    lines.append(f"- Case sensitive: {config['case_sensitive']}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Examples: {summary['num_examples']}")
    lines.append(f"- Passed: {summary['num_passed']}")
    lines.append(f"- Failed: {summary['num_failed']}")
    lines.append(f"- Pass rate: {format_float(summary['pass_rate'] * 100)}%")
    if "unsupported_claim_rate" in summary:
        lines.append(f"- Answerable examples: {summary.get('num_answerable', 0)}")
        lines.append(f"- Unanswerable examples: {summary.get('num_unanswerable', 0)}")
        lines.append(f"- Unsupported claim rate: {format_float(summary['unsupported_claim_rate'] * 100)}%")
        lines.append(f"- Missing support rate: {format_float(summary.get('missing_support_rate', 0.0) * 100)}%")
    lines.append("")

    if summary.get("category_breakdown"):
        lines.append("## Category Breakdown")
        lines.append("")
        lines.extend(_category_breakdown_table(summary["category_breakdown"]))
        lines.append("")

    if "unsupported_claim_rate" in summary:
        lines.append("## Honesty Metrics")
        lines.append("")
        lines.append(
            "These metrics are deliberately narrow and inspectable. "
            "Unsupported claim rate counts replies that contain forbidden phrases "
            "from the eval row. Missing support rate counts replies that missed "
            "required phrases or any-phrase groups."
        )
        lines.append("")
        lines.append(
            "This does not prove semantic truth. It tells you whether this tiny "
            "checkpoint followed the explicit support/refusal rules in the eval file."
        )
        lines.append("")

    lines.append("## Results")
    lines.append("")
    for index, item in enumerate(report["examples"], start=1):
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"### {index}. {status}")
        lines.append("")
        lines.append(f"User: {item['user']}")
        lines.append("")
        lines.append(f"Category: `{item.get('category', 'answerable')}`")
        lines.append(f"Answerable: `{item.get('answerable', True)}`")
        lines.append("")
        lines.append("Required phrases:")
        lines.append(_phrase_list(item["must_include"]))
        lines.append("")
        lines.append("Required any-phrase groups:")
        lines.append(_phrase_group_list(item.get("must_include_any", [])))
        lines.append("")
        lines.append("Forbidden phrases:")
        lines.append(_phrase_list(item["must_not_include"]))
        lines.append("")
        if item["missing"]:
            lines.append(f"Missing: {_inline_list(item['missing'])}")
            lines.append("")
        if item.get("missing_any"):
            lines.append(f"Missing any-group: {_phrase_group_inline(item['missing_any'])}")
            lines.append("")
        if item["found_forbidden"]:
            lines.append(f"Found forbidden: {_inline_list(item['found_forbidden'])}")
            lines.append("")
        lines.append("Reply:")
        lines.append("")
        lines.append("```text")
        lines.append(item["reply"])
        lines.append("```")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Machine-readable report: `eval_report.json`")
    lines.append("- Human-readable report: `report.md`")
    lines.append("")
    return "\n".join(lines)


def tiny_run_summary_markdown(summary: dict) -> str:
    """Render a one-page summary for an end-to-end tiny run."""
    config = summary["config"]
    eval_summary = summary["eval"]
    base = summary["base"]
    sft = summary["sft"]
    artifacts = summary["artifacts"]
    tokenizer = summary.get("tokenizer", {})
    base_diagnostics = base.get("loss_diagnostics", {})
    sft_diagnostics = sft.get("loss_diagnostics", {})
    base_memorization = base.get("memorization", {})
    base_coverage = base.get("coverage", {})
    sft_coverage = sft.get("coverage", {})

    lines: list[str] = []
    lines.append("# Picochat Tiny Run Summary")
    lines.append("")
    lines.append(
        "This run executes the full educational pipeline: corpus build, "
        "tokenizer training, base LM training, chat SFT, and transparent eval."
    )
    lines.append("")

    lines.append("## Result")
    lines.append("")
    lines.append(f"- Eval passed: {eval_summary['num_passed']} / {eval_summary['num_examples']}")
    lines.append(f"- Eval pass rate: {format_float(eval_summary['pass_rate'] * 100)}%")
    lines.append(f"- Failed examples: {eval_summary['num_failed']}")
    if "unsupported_claim_rate" in eval_summary:
        lines.append(f"- Unsupported claim rate: {format_float(eval_summary['unsupported_claim_rate'] * 100)}%")
        lines.append(f"- Missing support rate: {format_float(eval_summary.get('missing_support_rate', 0.0) * 100)}%")
    lines.append("")

    if eval_summary.get("category_breakdown"):
        lines.append("## Eval Categories")
        lines.append("")
        lines.extend(_category_breakdown_table(eval_summary["category_breakdown"]))
        lines.append("")

    lines.append("## Settings")
    lines.append("")
    lines.append(f"- Context size: {config['context_size']}")
    lines.append(f"- Embedding size: {config['n_embd']}")
    lines.append(f"- Layers: {config['n_layer']}")
    lines.append(f"- Attention heads: {config['n_head']}")
    lines.append(f"- Base steps: {config['base_steps']}")
    lines.append(f"- SFT steps: {config['sft_steps']}")
    lines.append(f"- Tokenizer type: `{tokenizer.get('tokenizer_type', 'unknown')}`")
    lines.append(f"- Base early stop patience: {config.get('base_early_stop_patience', 0)}")
    lines.append(f"- SFT early stop patience: {config.get('sft_early_stop_patience', 0)}")
    lines.append(f"- Train-only canaries: {config.get('canary_count', 0)}")
    lines.append(f"- Device: `{config['device']}`")
    if config.get("dataset_pack"):
        lines.append(f"- Dataset pack: `{config['dataset_pack']}`")
    lines.append("")

    lines.append("## Losses")
    lines.append("")
    lines.append(f"- Base final train loss: {format_float(base['final_train_loss'])}")
    lines.append(f"- Base final val loss: {format_float(base['final_val_loss'])}")
    if base.get("final_val_bpb") is not None:
        lines.append(f"- Base final val BPB: {format_optional_float(base.get('final_val_bpb'))}")
    lines.append(f"- SFT final train loss: {format_float(sft['final_train_loss'])}")
    lines.append(f"- SFT final val loss: {format_float(sft['final_val_loss'])}")
    if sft.get("final_val_bpb") is not None:
        lines.append(f"- SFT final val BPB: {format_optional_float(sft.get('final_val_bpb'))}")
    lines.append(f"- SFT truncated examples: {sft['truncated_examples']}")
    if base_coverage:
        lines.append(f"- Base stop reason: `{base.get('stop_reason', 'unknown')}`")
        lines.append(f"- Base estimated train epochs: {format_optional_float(base_coverage.get('estimated_train_epochs'))}")
    if sft_coverage:
        lines.append(f"- SFT stop reason: `{sft.get('stop_reason', 'unknown')}`")
        lines.append(f"- SFT estimated train epochs: {format_optional_float(sft_coverage.get('estimated_train_epochs'))}")
    if base_diagnostics:
        lines.append(f"- Base loss status: `{base_diagnostics.get('status', 'unknown')}`")
        lines.append(f"- Base final train/val gap: {format_optional_float(base_diagnostics.get('final_gap'))}")
    if base_memorization:
        lines.append(f"- Base memorization status: `{base_memorization.get('status', 'unknown')}`")
        lines.append(f"- Base train copy rate: {format_float(base_memorization.get('train_overlap_rate', 0.0) * 100)}%")
    if sft_diagnostics:
        lines.append(f"- SFT loss status: `{sft_diagnostics.get('status', 'unknown')}`")
        lines.append(f"- SFT final train/val gap: {format_optional_float(sft_diagnostics.get('final_gap'))}")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    if artifacts.get("dataset_pack"):
        lines.append(f"- Dataset pack: `{artifacts['dataset_pack']}`")
    lines.append(f"- Corpus: `{artifacts['corpus']}`")
    lines.append(f"- Tokenizer: `{artifacts['tokenizer']}`")
    lines.append(f"- Base report: `{artifacts['base_report']}`")
    if artifacts.get("base_eval_checkpoint"):
        lines.append(f"- Base checkpoint used for SFT: `{artifacts['base_eval_checkpoint']}`")
    lines.append(f"- SFT report: `{artifacts['sft_report']}`")
    if artifacts.get("sft_eval_checkpoint"):
        lines.append(f"- SFT checkpoint used for eval: `{artifacts['sft_eval_checkpoint']}`")
    lines.append(f"- Eval report: `{artifacts['eval_report']}`")
    lines.append("- Machine-readable summary: `summary.json`")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Use this summary to compare tiny runs. If eval improves but SFT validation "
        "loss rises, the model is likely memorizing the tiny chat data rather than "
        "generalizing. Unsupported claim and missing support rates are rule-based "
        "signals from the eval file, not proof of semantic truth."
    )
    lines.append("")
    return "\n".join(lines)


def _category_breakdown_table(category_breakdown: dict) -> list[str]:
    lines = [
        "| Category | Passed | Pass Rate | Missing Support | Unsupported |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, row in sorted(category_breakdown.items()):
        passed = f"{row.get('num_passed', 0)} / {row.get('num_examples', 0)}"
        pass_rate = format_float(row.get("pass_rate", 0.0) * 100)
        missing = f"{row.get('missing_support', 0)} / {row.get('num_examples', 0)}"
        unsupported = f"{row.get('unsupported_claims', 0)} / {row.get('num_examples', 0)}"
        lines.append(f"| `{category}` | {passed} | {pass_rate}% | {missing} | {unsupported} |")
    return lines


def _phrase_list(phrases: list[str]) -> str:
    if not phrases:
        return "- none"
    return "\n".join(f"- `{phrase}`" for phrase in phrases)


def _phrase_group_list(groups: list[list[str]]) -> str:
    if not groups:
        return "- none"
    return "\n".join(f"- {_inline_list(group)}" for group in groups)


def _phrase_group_inline(groups: list[list[str]]) -> str:
    return "; ".join(f"[{_inline_list(group)}]" for group in groups)


def _inline_list(items: list[str]) -> str:
    return ", ".join(f"`{item}`" for item in items)


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


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _loss_status(final_gap: float | None, val_regression: float | None) -> tuple[str, str]:
    if final_gap is None or val_regression is None:
        return "unknown", "Loss diagnostics need both train and validation loss."
    if final_gap >= 1.0 and val_regression >= 0.25:
        return "memorization-risk", "Validation loss moved away from the best step while train loss stayed much lower."
    if val_regression >= 0.25:
        return "watch-val-regression", "Final validation loss is noticeably worse than the best validation step."
    if final_gap >= 1.0:
        return "watch-gap", "Final validation loss is much higher than final train loss."
    return "stable", "Train and validation losses are moving together for this tiny run."
