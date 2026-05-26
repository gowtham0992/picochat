"""Human-readable report generation for Picochat runs."""

from __future__ import annotations

import math


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_optional_float(value: float | None) -> str:
    return "--" if value is None else format_float(value)


def format_optional_tflops(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value / 1e12:.2f} TFLOP/s"


def format_optional_percent(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value * 100:.2f}%"


def format_duration(seconds: object) -> str:
    value = _number(seconds)
    if value is None:
        return "--"
    if value < 60:
        return f"{value:.1f}s"
    minutes, remainder = divmod(value, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {remainder:.1f}s"


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
            "best_val_bpb": None,
            "final_val_bpb": None,
            "final_gap": None,
            "val_regression": None,
            "train_improvement": None,
            "evals_after_best": 0,
            "recommended_checkpoint_step": None,
            "recommendations": ["No training curve was recorded; rerun this stage before interpreting results."],
        }

    first = losses[0]
    final = losses[-1]
    valid_val = [item for item in losses if _number(item.get("val_loss")) is not None]
    best = min(valid_val, key=lambda item: _number(item.get("val_loss"))) if valid_val else final
    best_index = losses.index(best)
    first_train = _number(first.get("train_loss"))
    final_train = _number(final.get("train_loss"))
    final_val = _number(final.get("val_loss"))
    best_val = _number(best.get("val_loss"))
    final_gap = _diff(final_val, final_train)
    val_regression = _diff(final_val, best_val)
    train_improvement = _diff(first_train, final_train)
    status, summary = _loss_status(final_gap, val_regression)
    evals_after_best = max(0, len(losses) - 1 - best_index)
    return {
        "status": status,
        "summary": summary,
        "first_step": first.get("step"),
        "final_step": final.get("step"),
        "best_val_step": best.get("step"),
        "best_val_loss": best_val,
        "best_val_bpb": _number(best.get("val_bpb")),
        "final_val_bpb": _number(final.get("val_bpb")),
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "final_gap": final_gap,
        "val_regression": val_regression,
        "train_improvement": train_improvement,
        "evals_after_best": evals_after_best,
        "recommended_checkpoint_step": best.get("step"),
        "recommendations": _loss_recommendations(
            final_gap=final_gap,
            val_regression=val_regression,
            evals_after_best=evals_after_best,
            best_step=best.get("step"),
            final_step=final.get("step"),
        ),
    }


def optimization_stability(losses: list[dict], grad_clip: float | int | None = None) -> dict:
    """Summarize optimizer stability signals from the recorded training trace."""
    grad_norms = [
        float(item["grad_norm"])
        for item in losses
        if item.get("grad_norm") is not None and math.isfinite(float(item["grad_norm"]))
    ]
    train_losses = [
        float(item["train_loss"])
        for item in losses
        if item.get("train_loss") is not None and math.isfinite(float(item["train_loss"]))
    ]
    warnings: list[str] = []
    if not grad_norms:
        warnings.append("No finite gradient norm samples were recorded.")
    else:
        max_grad = max(grad_norms)
        final_grad = grad_norms[-1]
        if max_grad > 100:
            warnings.append("Gradient norm exceeded 100; inspect LR, clipping, and batch size.")
        if grad_clip and grad_clip > 0 and max_grad > float(grad_clip) * 10:
            warnings.append("Pre-clip gradient norm was much larger than the clip threshold.")
        if final_grad > max_grad * 0.90 and len(grad_norms) >= 3:
            warnings.append("Final gradient norm is near the run maximum; watch for late instability.")
    loss_spikes = 0
    for prev, current in zip(train_losses, train_losses[1:]):
        if prev > 0 and current > prev * 1.5:
            loss_spikes += 1
    if loss_spikes:
        warnings.append(f"Train loss spiked {loss_spikes} time(s) between checkpoints.")
    status = "warn" if warnings else "stable"
    return {
        "status": status,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "final_grad_norm": grad_norms[-1] if grad_norms else None,
        "loss_spikes": loss_spikes,
        "warnings": warnings,
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
    if dataset.get("packing"):
        lines.append(f"- Packing: `{dataset.get('packing')}`")
    if dataset.get("val_documents") is not None:
        lines.append(f"- Held-out documents: {dataset.get('val_documents')} / {dataset.get('num_documents')}")
    if dataset.get("canaries_enabled"):
        lines.append(f"- Train-only canaries: {len(dataset.get('canary_values', []))}")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    model_config = model["config"]
    parameter_report = model.get("parameter_report") or {}
    lines.append(f"- Parameters: {model['num_parameters']:,}")
    if parameter_report:
        lines.append(f"- Trainable parameters: {int(parameter_report.get('trainable_parameters', 0)):,}")
        lines.append(
            f"- Trainable fraction: "
            f"{format_float(float(parameter_report.get('trainable_fraction', 0.0)) * 100)}%"
        )
    lines.append(f"- Vocabulary size: {model_config['vocab_size']}")
    lines.append(f"- Layers: {model_config['n_layer']}")
    lines.append(f"- Embedding size: {model_config['n_embd']}")
    lines.append(f"- Attention heads: {model_config['n_head']}")
    lines.append(f"- KV heads: {model_config.get('n_kv_head') or model_config['n_head']}")
    lines.append(f"- Dropout: {model_config['dropout']}")
    lines.append(f"- Norm: `{model_config.get('norm_type', 'layernorm')}`")
    lines.append(f"- Position encoding: `{model_config.get('position_encoding', 'learned')}`")
    lines.append(f"- Activation: `{model_config.get('activation', 'gelu')}`")
    lines.append(f"- Tied embeddings: {'enabled' if model_config.get('tie_embeddings') else 'disabled'}")
    lines.append(f"- QK norm: {'enabled' if model_config.get('qk_norm') else 'disabled'}")
    lines.append(f"- Parallel residual: {'enabled' if model_config.get('parallel_residual') else 'disabled'}")
    lines.append(f"- XSA last layers: {model_config.get('xsa_last_n', 0) or 'disabled'}")
    lines.append(f"- Linear biases: {'enabled' if model_config.get('linear_bias', True) else 'disabled'}")
    lines.append(f"- Scaled residual init: {'enabled' if model_config.get('scaled_residual_init') else 'disabled'}")
    lines.append(f"- Logit softcap: {model_config.get('logit_softcap', 0.0) or 'disabled'}")
    lines.append("")

    lines.append("## Training")
    lines.append("")
    lines.append(f"- Steps: {config['max_steps']}")
    lines.append(f"- Batch size: {config['batch_size']}")
    lines.append(f"- Gradient accumulation steps: {config.get('grad_accum_steps', 1)}")
    lines.append(f"- Effective batch size: {config.get('effective_batch_size', config['batch_size'])}")
    lines.append(f"- Effective tokens/update: {config.get('effective_tokens_per_step', config['batch_size'] * config.get('context_size', 0))}")
    throughput = report.get("throughput", {})
    if throughput:
        lines.append(f"- Average tokens/sec: {format_optional_float(throughput.get('avg_tokens_per_sec'))}")
        lines.append(f"- Final tokens/sec: {format_optional_float(throughput.get('final_tokens_per_sec'))}")
        lines.append(f"- Average model FLOP utilization: {format_optional_percent(throughput.get('avg_mfu'))}")
        lines.append(f"- Final model FLOP utilization: {format_optional_percent(throughput.get('final_mfu'))}")
        lines.append(f"- Average estimated FLOP/s: {format_optional_tflops(throughput.get('avg_flops_per_sec'))}")
    if config.get("peak_flops_source"):
        lines.append(f"- MFU reference: {config.get('peak_flops_source')}")
    lines.append(f"- Optimizer: `{config.get('optimizer', 'adamw')}`")
    if config.get("optimizer") == "muon":
        lines.append(f"- Muon matrix LR: {config.get('muon_learning_rate')}")
        lines.append(f"- Muon momentum schedule: `{config.get('muon_momentum_schedule', 'none')}`")
    lines.append(f"- Weight decay: {config.get('weight_decay', 0.01)}")
    lines.append(f"- Weight decay schedule: `{config.get('weight_decay_decay', 'none')}`")
    lines.append(f"- EMA decay: {config.get('ema_decay', 0.0) or 'disabled'}")
    lines.append(f"- Learning rate: {config['learning_rate']}")
    lines.append(f"- LR decay: `{config.get('lr_decay', 'none')}`")
    lines.append(f"- LR warmup steps: {config.get('lr_warmup_steps', 0)}")
    lines.append(f"- Min LR ratio: {config.get('min_lr_ratio', 1.0)}")
    lines.append(f"- Gradient clip: {config.get('grad_clip', 0.0) or 'disabled'}")
    lines.append(f"- Validation fraction: {config['val_fraction']}")
    lines.append(f"- Early stop patience: {config.get('early_stop_patience', 0)}")
    lines.append(f"- Max minutes: {config.get('max_minutes') or 'disabled'}")
    lines.append(f"- Device: `{config['device']}`")
    if config.get("requested_device") and config.get("requested_device") != config.get("device"):
        lines.append(f"- Requested device: `{config['requested_device']}`")
    precision = config.get("precision_runtime", {})
    if precision:
        lines.append(
            f"- Precision: `{precision.get('requested', 'float32')}` "
            f"-> `{precision.get('dtype_name', 'float32')}`"
        )
    matmul_precision = config.get("matmul_precision_runtime", {})
    if matmul_precision:
        lines.append(
            f"- Float32 matmul precision: `{matmul_precision.get('requested', 'default')}` "
            f"-> `{matmul_precision.get('after') or 'unchanged'}`"
        )
    compile_metadata = config.get("torch_compile_metadata", {})
    lines.append(
        f"- Torch compile: "
        f"{'enabled' if compile_metadata.get('enabled') else 'disabled'}"
    )
    lines.append("")
    has_ema = any("ema_val_loss" in item for item in losses)
    if has_ema:
        lines.append("| Step | Train Loss | Train BPB | Val Loss | Val BPB | EMA Val Loss | EMA Val BPB | LR | Grad Norm | Elapsed Seconds |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    else:
        lines.append("| Step | Train Loss | Train BPB | Val Loss | Val BPB | LR | Grad Norm | Elapsed Seconds |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in losses:
        row = (
            f"| {item['step']} | {format_float(item['train_loss'])} | "
            f"{format_optional_float(item.get('train_bpb'))} | "
            f"{format_float(item['val_loss'])} | "
            f"{format_optional_float(item.get('val_bpb'))} | "
        )
        if has_ema:
            row += (
                f"{format_optional_float(item.get('ema_val_loss'))} | "
                f"{format_optional_float(item.get('ema_val_bpb'))} | "
            )
        row += (
            f"{format_optional_float(item.get('learning_rate'))} | "
            f"{format_optional_float(item.get('grad_norm'))} | "
            f"{format_float(item['elapsed_sec'])} |"
        )
        lines.append(row)
    lines.append("")

    stability = report.get("optimization_stability") or optimization_stability(losses, config.get("grad_clip", 0.0))
    lines.append("## Optimization Stability")
    lines.append("")
    lines.append(f"- Status: `{stability.get('status', 'unknown')}`")
    lines.append(f"- Max grad norm: {format_optional_float(stability.get('max_grad_norm'))}")
    lines.append(f"- Final grad norm: {format_optional_float(stability.get('final_grad_norm'))}")
    lines.append(f"- Loss spikes between checkpoints: {stability.get('loss_spikes', 0)}")
    lines.append(f"- Rollbacks: {len(report.get('rollback_events', []))}")
    if report.get("loss_spike_watch"):
        lines.append(f"- Per-step loss spike watch: `{report['loss_spike_watch']}`")
    for warning in stability.get("warnings", []):
        lines.append(f"- Warning: {warning}")
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
        for warning in coverage.get("warnings", []):
            lines.append(f"- Warning: {warning}")
        lines.append("")

    lines.append("## Loss Diagnostics")
    lines.append("")
    lines.append(f"- Status: `{diagnostics['status']}`")
    lines.append(f"- Summary: {diagnostics['summary']}")
    lines.append(f"- Best validation step: {diagnostics['best_val_step']}")
    lines.append(f"- Best validation loss: {format_optional_float(diagnostics['best_val_loss'])}")
    if diagnostics.get("best_val_bpb") is not None:
        lines.append(f"- Best validation BPB: {format_optional_float(diagnostics.get('best_val_bpb'))}")
    lines.append(f"- Final train/val gap: {format_optional_float(diagnostics['final_gap'])}")
    lines.append(f"- Validation regression from best step: {format_optional_float(diagnostics['val_regression'])}")
    lines.append(f"- Train loss improvement: {format_optional_float(diagnostics['train_improvement'])}")
    lines.append(f"- Recommended checkpoint step: {diagnostics.get('recommended_checkpoint_step')}")
    if diagnostics.get("recommendations"):
        lines.append("- Recommendations:")
        lines.extend(f"  - {item}" for item in diagnostics["recommendations"])
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
    lines.append(f"- Packing: `{dataset.get('packing', config.get('packing', 'separate'))}`")
    lines.append(f"- Source examples: {dataset.get('source_examples', dataset['num_examples'])}")
    lines.append(f"- Packed sequences: {dataset.get('packed_sequences', dataset.get('num_sequences', dataset['num_examples']))}")
    lines.append(f"- Packing efficiency: {format_float(float(dataset.get('packing_efficiency', 0.0)) * 100)}%")
    lines.append(f"- Padded tokens: {dataset.get('padded_tokens', 0)}")
    if dataset.get("average_examples_per_sequence"):
        lines.append(
            f"- Average examples/sequence: "
            f"{format_float(float(dataset.get('average_examples_per_sequence', 0.0)))}"
        )
    if dataset.get("mixed_category_sequences"):
        lines.append(f"- Mixed-category packed sequences: {dataset.get('mixed_category_sequences')}")
    packing_warnings = _sft_packing_warnings(dataset, config)
    for warning in packing_warnings:
        lines.append(f"- Warning: {warning}")
    lines.append(f"- Context size: {dataset['context_size']}")
    lines.append(f"- Supervised answer tokens: {dataset['supervised_tokens']}")
    if dataset.get("masked_prompt_tokens") is not None:
        lines.append(f"- Masked prompt tokens: {dataset['masked_prompt_tokens']}")
    lines.append(f"- Truncated examples: {dataset.get('truncated_examples', 0)}")
    lines.append(f"- Skipped too-long examples: {dataset.get('skipped_long_examples', 0)}")
    lines.append(f"- Train examples: {dataset['train_examples']}")
    lines.append(f"- Validation examples: {dataset['val_examples']}")
    if dataset.get("train_sequences") is not None:
        lines.append(f"- Train sequences: {dataset.get('train_sequences')}")
        lines.append(f"- Validation sequences: {dataset.get('val_sequences')}")
    lines.append(f"- SFT sampling: `{dataset.get('sampling', config.get('sampling', 'uniform'))}`")
    if dataset.get("split_method"):
        lines.append(f"- Validation split: `{dataset['split_method']}`")
    if dataset.get("num_groups"):
        lines.append(
            f"- Groups: {dataset['num_groups']} total "
            f"({dataset.get('train_groups', 0)} train / {dataset.get('val_groups', 0)} validation)"
        )
    if dataset.get("category_counts"):
        lines.append(f"- Categories: {_format_counts(dataset['category_counts'])}")
    if dataset.get("train_category_counts"):
        lines.append(f"- Train categories: {_format_counts(dataset['train_category_counts'])}")
    if dataset.get("val_category_counts"):
        lines.append(f"- Validation categories: {_format_counts(dataset['val_category_counts'])}")
    lines.append("")

    label_audit = report.get("label_audit") or {}
    if label_audit:
        lines.append("## SFT Label Audit")
        lines.append("")
        lines.append(
            "This checks the actual training labels after assistant-only masking. "
            "A healthy SFT run has non-zero supervised labels in every train and validation sequence."
        )
        lines.append("")
        lines.append("| Split | Sequences | Supervised Tokens | Active Labels | Zero-Supervised Sequences | Avg Supervised/Seq |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for split_name in ("full", "train", "validation"):
            audit = label_audit.get(split_name) or {}
            lines.append(
                f"| `{split_name}` | {audit.get('sequences', 0)} | "
                f"{audit.get('supervised_tokens', 0)} | "
                f"{format_float(float(audit.get('active_label_fraction', 0.0)) * 100)}% | "
                f"{audit.get('zero_supervised_sequences', 0)} | "
                f"{format_float(float(audit.get('avg_supervised_tokens_per_sequence', 0.0)))} |"
            )
        if label_audit.get("skipped_long_examples"):
            lines.append("")
            lines.append(f"- Skipped too-long source rows: {label_audit.get('skipped_long_examples')}")
            skipped_counts = label_audit.get("skipped_long_category_counts") or {}
            if skipped_counts:
                lines.append(f"- Skipped too-long categories: {_format_counts(skipped_counts)}")
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
    parameter_report = model.get("parameter_report") or {}
    lines.append(f"- Parameters: {model['num_parameters']:,}")
    if parameter_report:
        lines.append(f"- Trainable parameters: {int(parameter_report.get('trainable_parameters', 0)):,}")
        lines.append(
            f"- Trainable fraction: "
            f"{format_float(float(parameter_report.get('trainable_fraction', 0.0)) * 100)}%"
        )
    lines.append(f"- Vocabulary size: {model_config['vocab_size']}")
    lines.append(f"- Layers: {model_config['n_layer']}")
    lines.append(f"- Embedding size: {model_config['n_embd']}")
    lines.append(f"- Attention heads: {model_config['n_head']}")
    lines.append(f"- KV heads: {model_config.get('n_kv_head') or model_config['n_head']}")
    lines.append(f"- Norm: `{model_config.get('norm_type', 'layernorm')}`")
    lines.append(f"- Position encoding: `{model_config.get('position_encoding', 'learned')}`")
    lines.append(f"- Activation: `{model_config.get('activation', 'gelu')}`")
    lines.append(f"- Tied embeddings: {'enabled' if model_config.get('tie_embeddings') else 'disabled'}")
    lines.append(f"- QK norm: {'enabled' if model_config.get('qk_norm') else 'disabled'}")
    lines.append(f"- Parallel residual: {'enabled' if model_config.get('parallel_residual') else 'disabled'}")
    lines.append(f"- XSA last layers: {model_config.get('xsa_last_n', 0) or 'disabled'}")
    lines.append(f"- Linear biases: {'enabled' if model_config.get('linear_bias', True) else 'disabled'}")
    lines.append(f"- Scaled residual init: {'enabled' if model_config.get('scaled_residual_init') else 'disabled'}")
    lines.append(f"- Logit softcap: {model_config.get('logit_softcap', 0.0) or 'disabled'}")
    lines.append("")

    lines.append("## Training")
    lines.append("")
    lines.append(f"- Steps: {config['max_steps']}")
    lines.append(f"- Batch size: {config['batch_size']}")
    lines.append(f"- Gradient accumulation steps: {config.get('grad_accum_steps', 1)}")
    lines.append(f"- Effective batch size: {config.get('effective_batch_size', config['batch_size'])}")
    lines.append(f"- Effective tokens/update: {config.get('effective_tokens_per_step', config['batch_size'] * report.get('dataset', {}).get('context_size', 0))}")
    throughput = report.get("throughput", {})
    if throughput:
        lines.append(f"- Average tokens/sec: {format_optional_float(throughput.get('avg_tokens_per_sec'))}")
        lines.append(f"- Final tokens/sec: {format_optional_float(throughput.get('final_tokens_per_sec'))}")
        lines.append(f"- Average model FLOP utilization: {format_optional_percent(throughput.get('avg_mfu'))}")
        lines.append(f"- Final model FLOP utilization: {format_optional_percent(throughput.get('final_mfu'))}")
        lines.append(f"- Average estimated FLOP/s: {format_optional_tflops(throughput.get('avg_flops_per_sec'))}")
    if config.get("peak_flops_source"):
        lines.append(f"- MFU reference: {config.get('peak_flops_source')}")
    lines.append(f"- Optimizer: `{config.get('optimizer', 'adamw')}`")
    if config.get("optimizer") == "muon":
        lines.append(f"- Muon matrix LR: {config.get('muon_learning_rate')}")
        lines.append(f"- Muon momentum schedule: `{config.get('muon_momentum_schedule', 'none')}`")
    lines.append(f"- Weight decay: {config.get('weight_decay', 0.01)}")
    lines.append(f"- Weight decay schedule: `{config.get('weight_decay_decay', 'none')}`")
    lines.append(f"- EMA decay: {config.get('ema_decay', 0.0) or 'disabled'}")
    lines.append(f"- Learning rate: {config['learning_rate']}")
    lines.append(f"- LR decay: `{config.get('lr_decay', 'none')}`")
    lines.append(f"- LR warmup steps: {config.get('lr_warmup_steps', 0)}")
    lines.append(f"- Min LR ratio: {config.get('min_lr_ratio', 1.0)}")
    lines.append(f"- Gradient clip: {config.get('grad_clip', 0.0) or 'disabled'}")
    lines.append(f"- Early stop patience: {config.get('early_stop_patience', 0)}")
    lines.append(f"- Max minutes: {config.get('max_minutes') or 'disabled'}")
    lines.append(f"- Device: `{config['device']}`")
    if config.get("requested_device") and config.get("requested_device") != config.get("device"):
        lines.append(f"- Requested device: `{config['requested_device']}`")
    precision = config.get("precision_runtime", {})
    if precision:
        lines.append(
            f"- Precision: `{precision.get('requested', 'float32')}` "
            f"-> `{precision.get('dtype_name', 'float32')}`"
        )
    matmul_precision = config.get("matmul_precision_runtime", {})
    if matmul_precision:
        lines.append(
            f"- Float32 matmul precision: `{matmul_precision.get('requested', 'default')}` "
            f"-> `{matmul_precision.get('after') or 'unchanged'}`"
        )
    compile_metadata = config.get("torch_compile_metadata", {})
    lines.append(
        f"- Torch compile: "
        f"{'enabled' if compile_metadata.get('enabled') else 'disabled'}"
    )
    peft = config.get("peft") or {}
    if isinstance(peft, dict) and peft.get("mode") == "lora":
        lines.append(
            f"- PEFT: `lora` rank {peft.get('rank')} alpha {peft.get('alpha')} "
            f"targets `{', '.join(peft.get('targets', []))}`"
        )
        lines.append(f"- LoRA adapted modules: {peft.get('adapted_module_count')}")
    else:
        lines.append("- PEFT: `none`")
    lines.append("")
    has_ema = any("ema_val_loss" in item for item in losses)
    if has_ema:
        lines.append("| Step | Train Loss | Train BPB | Val Loss | Val BPB | EMA Val Loss | EMA Val BPB | LR | Grad Norm | Elapsed Seconds |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    else:
        lines.append("| Step | Train Loss | Train BPB | Val Loss | Val BPB | LR | Grad Norm | Elapsed Seconds |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in losses:
        row = (
            f"| {item['step']} | {format_float(item['train_loss'])} | "
            f"{format_optional_float(item.get('train_bpb'))} | "
            f"{format_float(item['val_loss'])} | "
            f"{format_optional_float(item.get('val_bpb'))} | "
        )
        if has_ema:
            row += (
                f"{format_optional_float(item.get('ema_val_loss'))} | "
                f"{format_optional_float(item.get('ema_val_bpb'))} | "
            )
        row += (
            f"{format_optional_float(item.get('learning_rate'))} | "
            f"{format_optional_float(item.get('grad_norm'))} | "
            f"{format_float(item['elapsed_sec'])} |"
        )
        lines.append(row)
    lines.append("")

    stability = report.get("optimization_stability") or optimization_stability(losses, config.get("grad_clip", 0.0))
    lines.append("## Optimization Stability")
    lines.append("")
    lines.append(f"- Status: `{stability.get('status', 'unknown')}`")
    lines.append(f"- Max grad norm: {format_optional_float(stability.get('max_grad_norm'))}")
    lines.append(f"- Final grad norm: {format_optional_float(stability.get('final_grad_norm'))}")
    lines.append(f"- Loss spikes between checkpoints: {stability.get('loss_spikes', 0)}")
    lines.append(f"- Rollbacks: {len(report.get('rollback_events', []))}")
    if report.get("loss_spike_watch"):
        lines.append(f"- Per-step loss spike watch: `{report['loss_spike_watch']}`")
    for warning in stability.get("warnings", []):
        lines.append(f"- Warning: {warning}")
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
        for warning in coverage.get("warnings", []):
            lines.append(f"- Warning: {warning}")
        lines.append("")

    lines.append("## Loss Diagnostics")
    lines.append("")
    lines.append(f"- Status: `{diagnostics['status']}`")
    lines.append(f"- Summary: {diagnostics['summary']}")
    lines.append(f"- Best validation step: {diagnostics['best_val_step']}")
    lines.append(f"- Best validation loss: {format_optional_float(diagnostics['best_val_loss'])}")
    if diagnostics.get("best_val_bpb") is not None:
        lines.append(f"- Best validation BPB: {format_optional_float(diagnostics.get('best_val_bpb'))}")
    lines.append(f"- Final train/val gap: {format_optional_float(diagnostics['final_gap'])}")
    lines.append(f"- Validation regression from best step: {format_optional_float(diagnostics['val_regression'])}")
    lines.append(f"- Train loss improvement: {format_optional_float(diagnostics['train_improvement'])}")
    lines.append(f"- Recommended checkpoint step: {diagnostics.get('recommended_checkpoint_step')}")
    if diagnostics.get("recommendations"):
        lines.append("- Recommendations:")
        lines.extend(f"  - {item}" for item in diagnostics["recommendations"])
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
    if report.get("adapter_checkpoint"):
        lines.append(f"- LoRA adapter checkpoint: `{report['adapter_checkpoint']}`")
    if best_checkpoint:
        lines.append(
            f"- Best validation checkpoint: `{best_checkpoint['path']}` "
            f"(step {best_checkpoint['step']}, val {format_optional_float(best_checkpoint.get('val_loss'))}, "
            f"bpb {format_optional_float(best_checkpoint.get('val_bpb'))})"
        )
    lines.append("- Machine-readable report: `sft_report.json`")
    if report.get("label_audit"):
        lines.append("- SFT label audit: `sft_label_audit.json`")
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


def _sft_packing_warnings(dataset: dict, config: dict) -> list[str]:
    warnings: list[str] = []
    packing = dataset.get("packing", config.get("packing", "separate"))
    sampling = dataset.get("sampling", config.get("sampling", "uniform"))
    mixed_sequences = int(dataset.get("mixed_category_sequences") or 0)
    packed_sequences = int(dataset.get("packed_sequences", dataset.get("num_sequences", 0)) or 0)
    average_examples = float(dataset.get("average_examples_per_sequence") or 0.0)
    if packing == "bos_bestfit" and average_examples > 1.1:
        warnings.append(
            "best-fit SFT packing trains with multiple chat examples in one causal context; "
            "compare against `separate` packing if standalone generation underfits."
        )
    if packing == "bos_bestfit" and mixed_sequences and sampling != "uniform":
        ratio = mixed_sequences / packed_sequences if packed_sequences else 0.0
        if ratio >= 0.50:
            warnings.append(
                "most packed SFT sequences mix categories, so category-aware row sampling is weakened."
            )
    return warnings


def chat_eval_report_markdown(report: dict) -> str:
    """Render a transparent chat evaluation report as Markdown."""
    config = report["config"]
    checkpoint = report["checkpoint"]
    summary = report["summary"]

    lines: list[str] = []
    lines.append("# Picochat Chat Eval Report")
    lines.append("")
    lines.append(
        "This eval uses visible word-aware phrase checks. It is intentionally simple: "
        "a reply passes when all required phrases appear, each any-phrase group "
        "has at least one match, and no forbidden phrases appear. Format markers "
        "such as `Story:` stay literal."
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Eval data: `{config['input_path']}`")
    lines.append(f"- Tokenizer: `{config['tokenizer_path']}`")
    lines.append(f"- Checkpoint: `{checkpoint['path']}`")
    lines.append(f"- Checkpoint step: {checkpoint['step']}")
    lines.append(f"- Temperature: {config['temperature']}")
    lines.append(f"- Top-k: {config.get('top_k')}")
    lines.append(f"- Top-p: {config.get('top_p', 1.0)}")
    lines.append(f"- Repetition penalty: {config.get('repetition_penalty', 1.0)}")
    lines.append(f"- Max new tokens: {config['max_new_tokens']}")
    if config.get("support_corpus_path"):
        lines.append(f"- Support corpus: `{config['support_corpus_path']}`")
        lines.append(f"- Corpus support threshold: {config.get('corpus_support_threshold', 0.25)}")
    lines.append(f"- Case sensitive: {config['case_sensitive']}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Examples: {summary['num_examples']}")
    lines.append(f"- Passed: {summary['num_passed']}")
    lines.append(f"- Failed: {summary['num_failed']}")
    lines.append(f"- Pass rate: {format_float(summary['pass_rate'] * 100)}%")
    if summary.get("pass_rate_ci"):
        lines.append(f"- Pass rate 95% CI: {_format_ci(summary.get('pass_rate_ci'))}")
    if summary.get("non_choice_pass_rate") is not None:
        ci = _format_ci(summary.get("non_choice_pass_rate_ci"))
        suffix = f" ({ci})" if ci != "--" else ""
        lines.append(
            f"- Non-choice pass rate: {_format_percent_or_dash(summary.get('non_choice_pass_rate'))}"
            f"{suffix} over {summary.get('non_choice_examples', 0)} example(s)"
        )
    if "unsupported_claim_rate" in summary:
        lines.append(f"- Answerable examples: {summary.get('num_answerable', 0)}")
        lines.append(f"- Unanswerable examples: {summary.get('num_unanswerable', 0)}")
        if summary.get("answerable_pass_rate") is not None:
            ci = _format_ci(summary.get("answerable_pass_rate_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(f"- Answerable pass rate: {_format_percent_or_dash(summary.get('answerable_pass_rate'))}{suffix}")
        if summary.get("domain_pass_rate") is not None:
            ci = _format_ci(summary.get("domain_pass_rate_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(f"- Domain answer pass rate: {_format_percent_or_dash(summary.get('domain_pass_rate'))}{suffix}")
        if summary.get("refusal_pass_rate") is not None:
            ci = _format_ci(summary.get("refusal_pass_rate_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(f"- Refusal/boundary pass rate: {_format_percent_or_dash(summary.get('refusal_pass_rate'))}{suffix}")
        if summary.get("unsafe_refusal_pass_rate") is not None:
            ci = _format_ci(summary.get("unsafe_refusal_pass_rate_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(
                f"- Unsafe/refusal prompt pass rate: "
                f"{_format_percent_or_dash(summary.get('unsafe_refusal_pass_rate'))}"
                f"{suffix} over {summary.get('unsafe_refusal_examples', 0)} example(s)"
            )
        if summary.get("benign_non_refusal_rate") is not None:
            ci = _format_ci(summary.get("benign_non_refusal_rate_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(
                f"- Benign non-refusal rate: "
                f"{_format_percent_or_dash(summary.get('benign_non_refusal_rate'))}"
                f"{suffix} over {summary.get('benign_non_refusal_examples', 0)} example(s)"
            )
            lines.append(f"- Over-refusal rate: {_format_percent_or_dash(summary.get('over_refusal_rate'))}")
        lines.append(f"- Unsupported claim rate: {format_float(summary['unsupported_claim_rate'] * 100)}%")
        lines.append(f"- Prompt echo rate: {format_float(summary.get('prompt_echo_rate', 0.0) * 100)}%")
        lines.append(f"- Missing support rate: {format_float(summary.get('missing_support_rate', 0.0) * 100)}%")
        lines.append(f"- Missing entity rate: {format_float(summary.get('missing_entity_rate', 0.0) * 100)}%")
        lines.append(f"- Length violation rate: {format_float(summary.get('length_violation_rate', 0.0) * 100)}%")
        lines.append(f"- Corpus support failure rate: {format_float(summary.get('corpus_support_failure_rate', 0.0) * 100)}%")
        if summary.get("normalized_answer_accuracy") is not None:
            lines.append(
                f"- Normalized final-answer accuracy: "
                f"{_format_percent_or_dash(summary.get('normalized_answer_accuracy'))} "
                f"over {summary.get('normalized_answer_examples', 0)} example(s)"
            )
        if summary.get("normalized_answer_failures"):
            lines.append(f"- Normalized final-answer failures: {summary.get('normalized_answer_failures', 0)}")
        lines.append(f"- Support match rate: {format_float(summary.get('support_match_rate', 0.0) * 100)}%")
        if summary.get("answerable_support_match_rate") is not None:
            lines.append(
                f"- Answerable support match rate: "
                f"{format_float(summary.get('answerable_support_match_rate', 0.0) * 100)}%"
            )
        if summary.get("average_reference_token_f1") is not None:
            lines.append(f"- Avg reference token F1: {format_float(summary['average_reference_token_f1'] * 100)}%")
        if summary.get("average_reference_rouge_l") is not None:
            lines.append(f"- Avg reference ROUGE-L: {format_float(summary['average_reference_rouge_l'] * 100)}%")
        if summary.get("average_entity_match_rate") is not None:
            lines.append(f"- Avg entity match rate: {format_float(summary['average_entity_match_rate'] * 100)}%")
        if summary.get("average_corpus_support_rate") is not None:
            lines.append(f"- Avg corpus support rate: {format_float(summary['average_corpus_support_rate'] * 100)}%")
        if summary.get("choice_examples"):
            lines.append(f"- Choice-likelihood examples: {summary.get('choice_examples', 0)}")
            ci = _format_ci(summary.get("choice_accuracy_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(f"- Choice accuracy: {_format_percent_or_dash(summary.get('choice_accuracy'))}{suffix}")
            if summary.get("choice_random_baseline") is not None:
                lines.append(
                    f"- Choice random baseline: "
                    f"{_format_percent_or_dash(summary.get('choice_random_baseline'))}"
                )
            if summary.get("choice_accuracy_adjusted") is not None:
                lines.append(
                    f"- Random-baseline-adjusted choice accuracy: "
                    f"{_format_percent_or_dash(summary.get('choice_accuracy_adjusted'))}"
                )
            ci = _format_ci(summary.get("choice_pass_rate_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(f"- Choice pass rate: {_format_percent_or_dash(summary.get('choice_pass_rate'))}{suffix}")
            if summary.get("choice_margin_accuracy") is not None:
                ci = _format_ci(summary.get("choice_margin_accuracy_ci"))
                suffix = f" ({ci})" if ci != "--" else ""
                lines.append(
                    f"- Choice likelihood-margin accuracy: "
                    f"{_format_percent_or_dash(summary.get('choice_margin_accuracy'))}{suffix}"
                )
                lines.append(
                    f"- Avg correct-choice logprob margin: "
                    f"{format_optional_float(summary.get('choice_mean_correct_logprob_margin'))}"
                )
                lines.append(
                    f"- Low-margin correct choice rate: "
                    f"{_format_percent_or_dash(summary.get('choice_low_margin_rate'))}"
                )
            lines.append(f"- Choice scorer: `{summary.get('choice_scoring')}`")
    lines.append("")

    if summary.get("category_breakdown"):
        lines.append("## Category Breakdown")
        lines.append("")
        lines.extend(_category_breakdown_table(summary["category_breakdown"]))
        lines.append("")

    if summary.get("split_breakdown"):
        lines.append("## Split Breakdown")
        lines.append("")
        lines.extend(_split_breakdown_table(summary["split_breakdown"]))
        lines.append("")

    if summary.get("level_breakdown"):
        lines.append("## Eval Ladder")
        lines.append("")
        lines.extend(_level_breakdown_table(summary["level_breakdown"]))
        lines.append("")

    if summary.get("skill_breakdown"):
        lines.append("## Skill Breakdown")
        lines.append("")
        lines.extend(_breakdown_table(summary["skill_breakdown"], "Skill"))
        lines.append("")

    if summary.get("stage_breakdown"):
        lines.append("## Curriculum Stages")
        lines.append("")
        lines.extend(_breakdown_table(summary["stage_breakdown"], "Stage"))
        lines.append("")

    if summary.get("skill_stage_breakdown"):
        lines.append("## Skill Stages")
        lines.append("")
        lines.extend(_breakdown_table(summary["skill_stage_breakdown"], "Skill stage"))
        lines.append("")

    if summary.get("robustness_breakdown"):
        lines.append("## Robustness Variants")
        lines.append("")
        lines.extend(_breakdown_table(summary["robustness_breakdown"], "Variant"))
        lines.append("")

    if report.get("analysis"):
        lines.append("## Failure Analysis")
        lines.append("")
        lines.extend(_eval_analysis_markdown(report["analysis"]))
        lines.append("")

    if "unsupported_claim_rate" in summary:
        lines.append("## Honesty Metrics")
        lines.append("")
        lines.append(
            "These metrics are deliberately narrow and inspectable. "
            "Unsupported claim rate counts replies that contain forbidden phrases "
            "from the eval row using word-aware matching. Prompt echo rate counts "
            "replies that regenerate chat role labels or visibly start with the "
            "user prompt. Missing support rate counts replies that missed required "
            "phrases or any-phrase groups."
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
        lines.append(f"Split: `{item.get('split', 'default')}`")
        lines.append(f"Level: `{item.get('level', 'heldout')}`")
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
        if item.get("missing_entities"):
            lines.append(f"Missing entities: {_inline_list(item['missing_entities'])}")
            lines.append("")
        if item.get("length_violations"):
            lines.append(f"Length violations: {_inline_list(item['length_violations'])}")
            lines.append("")
        if item.get("prompt_echo"):
            lines.append(f"Prompt echo: {_inline_list(item.get('prompt_echo_reasons', []))}")
            lines.append("")
        if item.get("support_total") is not None:
            lines.append(
                f"Support matched: {item.get('support_matched', 0)} / "
                f"{item.get('support_total', 0)}"
            )
            lines.append("")
        metric_lines = _eval_metric_lines(item)
        if metric_lines:
            lines.append("Diagnostics:")
            lines.extend(metric_lines)
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
    sft_fit_summary = summary.get("sft_fit") or {}
    sft_fit_heldout_summary = summary.get("sft_fit_heldout") or {}
    base = summary["base"]
    sft = summary["sft"]
    artifacts = summary["artifacts"]
    tokenizer = summary.get("tokenizer", {})
    honesty = summary.get("honesty", {})
    preflight = summary.get("preflight", {})
    long_run_gate = summary.get("long_run_gate", {})
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
    base_primary_bpb = base.get("best_val_bpb")
    base_primary_label = "best-checkpoint"
    if base_primary_bpb is None:
        base_primary_bpb = base.get("final_val_bpb")
        base_primary_label = "final"
    if base_primary_bpb is not None:
        lines.append(
            f"- Primary base metric: {format_optional_float(base_primary_bpb)} "
            f"validation BPB ({base_primary_label})"
        )
    sft_primary_bpb = sft.get("best_val_bpb")
    sft_primary_label = "best-checkpoint"
    if sft_primary_bpb is None:
        sft_primary_bpb = sft.get("final_val_bpb")
        sft_primary_label = "final"
    if sft_primary_bpb is not None:
        lines.append(
            f"- Primary SFT metric: {format_optional_float(sft_primary_bpb)} "
            f"validation BPB ({sft_primary_label})"
        )
    lines.append(f"- Eval passed: {eval_summary['num_passed']} / {eval_summary['num_examples']}")
    lines.append(f"- Eval pass rate: {format_float(eval_summary['pass_rate'] * 100)}%")
    if eval_summary.get("pass_rate_ci"):
        lines.append(f"- Eval pass rate 95% CI: {_format_ci(eval_summary.get('pass_rate_ci'))}")
    if eval_summary.get("non_choice_pass_rate") is not None:
        ci = _format_ci(eval_summary.get("non_choice_pass_rate_ci"))
        suffix = f" ({ci})" if ci != "--" else ""
        lines.append(
            f"- Eval non-choice pass rate: {_format_percent_or_dash(eval_summary.get('non_choice_pass_rate'))}"
            f"{suffix} over {eval_summary.get('non_choice_examples', 0)} example(s)"
        )
    if sft_fit_summary:
        lines.append(
            f"- SFT fit passed: {sft_fit_summary.get('num_passed', 0)} / "
            f"{sft_fit_summary.get('num_examples', 0)}"
        )
        lines.append(
            f"- SFT fit rate: {format_float(float(sft_fit_summary.get('pass_rate', 0.0)) * 100)}%"
        )
        if sft_fit_summary.get("pass_rate_ci"):
            lines.append(f"- SFT fit rate 95% CI: {_format_ci(sft_fit_summary.get('pass_rate_ci'))}")
        if sft_fit_summary.get("non_choice_pass_rate") is not None:
            ci = _format_ci(sft_fit_summary.get("non_choice_pass_rate_ci"))
            suffix = f" ({ci})" if ci != "--" else ""
            lines.append(
                f"- SFT fit non-choice rate: {_format_percent_or_dash(sft_fit_summary.get('non_choice_pass_rate'))}"
                f"{suffix} over {sft_fit_summary.get('non_choice_examples', 0)} example(s)"
            )
    if sft_fit_heldout_summary:
        lines.append(
            f"- SFT heldout fit passed: {sft_fit_heldout_summary.get('num_passed', 0)} / "
            f"{sft_fit_heldout_summary.get('num_examples', 0)}"
        )
        lines.append(
            f"- SFT heldout fit rate: "
            f"{format_float(float(sft_fit_heldout_summary.get('pass_rate', 0.0)) * 100)}%"
        )
    lines.append(f"- Failed examples: {eval_summary['num_failed']}")
    if "unsupported_claim_rate" in eval_summary:
        lines.append(f"- Unsupported claim rate: {format_float(eval_summary['unsupported_claim_rate'] * 100)}%")
        if eval_summary.get("domain_pass_rate") is not None:
            lines.append(f"- Domain answer pass rate: {_format_percent_or_dash(eval_summary.get('domain_pass_rate'))}")
        if eval_summary.get("refusal_pass_rate") is not None:
            lines.append(f"- Refusal/boundary pass rate: {_format_percent_or_dash(eval_summary.get('refusal_pass_rate'))}")
        if eval_summary.get("unsafe_refusal_pass_rate") is not None:
            lines.append(
                f"- Unsafe/refusal prompt pass rate: "
                f"{_format_percent_or_dash(eval_summary.get('unsafe_refusal_pass_rate'))}"
            )
        if eval_summary.get("benign_non_refusal_rate") is not None:
            lines.append(
                f"- Benign non-refusal rate: "
                f"{_format_percent_or_dash(eval_summary.get('benign_non_refusal_rate'))}"
            )
            lines.append(f"- Over-refusal rate: {_format_percent_or_dash(eval_summary.get('over_refusal_rate'))}")
        if eval_summary.get("choice_accuracy_adjusted") is not None:
            lines.append(
                f"- Random-baseline-adjusted choice accuracy: "
                f"{_format_percent_or_dash(eval_summary.get('choice_accuracy_adjusted'))}"
            )
        lines.append(f"- Prompt echo rate: {format_float(eval_summary.get('prompt_echo_rate', 0.0) * 100)}%")
        lines.append(f"- Missing support rate: {format_float(eval_summary.get('missing_support_rate', 0.0) * 100)}%")
        if eval_summary.get("normalized_answer_accuracy") is not None:
            lines.append(
                f"- Normalized final-answer accuracy: "
                f"{_format_percent_or_dash(eval_summary.get('normalized_answer_accuracy'))}"
            )
        lines.append(f"- Support match rate: {format_float(eval_summary.get('support_match_rate', 0.0) * 100)}%")
    lines.append("")

    timing = summary.get("timing") or {}
    timing_stages = timing.get("stages") or []
    if timing_stages:
        lines.append("## Runtime")
        lines.append("")
        lines.append(f"- Total wall time: {format_duration(timing.get('total_seconds'))}")
        lines.append("")
        lines.append("| Stage | Time |")
        lines.append("| --- | ---: |")
        for item in timing_stages:
            lines.append(
                f"| `{item.get('stage', 'unknown')}` | {format_duration(item.get('seconds'))} |"
            )
        lines.append("")

    if preflight:
        budget = preflight.get("budget", {})
        blocking = preflight.get("blocking_checks", [])
        warnings = preflight.get("warning_checks", [])
        lines.append("## Long-Run Preflight")
        lines.append("")
        lines.append(f"- Status: `{preflight.get('status', 'unknown')}`")
        lines.append(f"- Summary: {preflight.get('summary', 'No preflight summary recorded.')}")
        lines.append(f"- Estimated parameters: {int(budget.get('estimated_parameters', 0)):,}")
        lines.append(f"- Estimated corpus tokens: {int(budget.get('estimated_corpus_tokens', 0)):,}")
        lines.append(f"- Corpus tokens / parameter: {format_optional_float(budget.get('corpus_tokens_per_parameter'))}")
        lines.append(f"- Base planned tokens: {int(budget.get('base_planned_tokens', 0)):,}")
        lines.append(f"- Target token/parameter ratio: {format_optional_float(budget.get('target_param_data_ratio'))}")
        lines.append(f"- Target training tokens: {int(budget.get('target_training_tokens', 0)):,}")
        lines.append(f"- Recommended base steps: {int(budget.get('recommended_base_steps', 0)):,}")
        lines.append(f"- Planned / target tokens: {format_optional_float(budget.get('planned_to_target_ratio'))}")
        lines.append(f"- Recommended base epochs: {format_optional_float(budget.get('recommended_base_epochs'))}")
        lines.append(f"- Base estimated epochs: {format_optional_float(budget.get('estimated_base_epochs'))}")
        lines.append(f"- SFT estimated example epochs: {format_optional_float(budget.get('estimated_sft_example_epochs'))}")
        lines.append(f"- Long run: {'yes' if budget.get('long_run') else 'no'} ({budget.get('long_run_reason', '--')})")
        if blocking:
            lines.append("- Blocking checks:")
            lines.extend(f"  - `{item.get('name')}`: {item.get('message')}" for item in blocking[:8])
        if warnings:
            lines.append("- Warning checks:")
            lines.extend(f"  - `{item.get('name')}`: {item.get('message')}" for item in warnings[:8])
        lines.append("")

    if long_run_gate:
        lines.append("## Approved Long-Run Gate")
        lines.append("")
        lines.append(f"- Status: `{long_run_gate.get('status', 'unknown')}`")
        lines.append(f"- Summary: {long_run_gate.get('summary', 'No long-run gate summary recorded.')}")
        lines.append(f"- Gate profile: `{long_run_gate.get('profile', 'research')}`")
        lines.append(f"- SFT fit rate: {format_float(float(long_run_gate.get('sft_fit_rate', 0.0)) * 100)}%")
        lines.append(f"- Required SFT fit rate: {format_float(float(long_run_gate.get('sft_fit_threshold', 0.70)) * 100)}%")
        if long_run_gate.get("sft_heldout_fit_rate") is not None:
            lines.append(
                f"- Held-out SFT fit rate: "
                f"{format_float(float(long_run_gate.get('sft_heldout_fit_rate', 0.0)) * 100)}%"
            )
        if long_run_gate.get("first_release_eval_rate") is not None:
            lines.append(
                f"- First-release eval rate: "
                f"{format_float(float(long_run_gate.get('first_release_eval_rate', 0.0)) * 100)}%"
            )
            lines.append(
                f"- Required first-release eval rate: "
                f"{format_float(float(long_run_gate.get('first_release_eval_threshold', 0.45)) * 100)}%"
            )
        external_results = long_run_gate.get("external_eval_results") or []
        if external_results:
            lines.append("- External benchmark gate:")
            for item in external_results:
                score_key = item.get("score_key", "score")
                score_label = "choice" if score_key == "choice_accuracy" else "pass"
                threshold = item.get("threshold")
                suffix = (
                    f" (required {format_float(float(threshold) * 100)}%)"
                    if threshold is not None else ""
                )
                lines.append(
                    f"  - `{item.get('name', 'external')}`: "
                    f"{_format_percent_or_dash(item.get('score'))} {score_label}, "
                    f"{int(item.get('num_examples') or 0)} rows{suffix}"
                )
        skill_eval_rates = long_run_gate.get("skill_release_eval_rates") or {}
        skill_eval_thresholds = long_run_gate.get("skill_release_eval_thresholds") or {}
        if skill_eval_rates:
            lines.append("- Skill-release held-out rates:")
            for name, rate in skill_eval_rates.items():
                threshold = skill_eval_thresholds.get(name)
                suffix = (
                    f" (required {format_float(float(threshold) * 100)}%)"
                    if threshold is not None else ""
                )
                lines.append(f"  - `{name}`: {_format_percent_or_dash(rate)}{suffix}")
        skill_stage_rates = long_run_gate.get("skill_release_stage_rates") or {}
        skill_stage_thresholds = long_run_gate.get("skill_release_stage_thresholds") or {}
        if skill_stage_rates:
            lines.append("- Skill-release stage rates:")
            for name, rate in skill_stage_rates.items():
                threshold = skill_stage_thresholds.get(name)
                suffix = (
                    f" (required {format_float(float(threshold) * 100)}%)"
                    if threshold is not None else ""
                )
                lines.append(f"  - `{name}`: {_format_percent_or_dash(rate)}{suffix}")
        if long_run_gate.get("benign_non_refusal_rate") is not None:
            lines.append(
                f"- Benign non-refusal gate: "
                f"{_format_percent_or_dash(long_run_gate.get('benign_non_refusal_rate'))} "
                f"(required {_format_percent_or_dash(long_run_gate.get('benign_non_refusal_threshold'))})"
            )
        for issue in long_run_gate.get("issues", [])[:8]:
            lines.append(f"- {issue.get('severity', 'warn').upper()} `{issue.get('name')}`: {issue.get('message')}")
        lines.append("")

    external_evals = summary.get("external_evals") or []
    if external_evals:
        lines.append("## External Benchmarks")
        lines.append("")
        lines.append("| Benchmark | Passed | Pass Rate | Choice Accuracy | Report |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for item in external_evals:
            eval_summary_item = item.get("summary", {})
            passed = (
                f"{int(eval_summary_item.get('num_passed', 0))} / "
                f"{int(eval_summary_item.get('num_examples', 0))}"
            )
            report_path = item.get("artifacts", {}).get("eval_report", "")
            report_text = f"`{report_path}`" if report_path else "--"
            lines.append(
                f"| `{item.get('name', 'external')}` | {passed} | "
                f"{_format_percent_or_dash(eval_summary_item.get('pass_rate'))} | "
                f"{_format_percent_or_dash(eval_summary_item.get('choice_accuracy'))} | "
                f"{report_text} |"
            )
        lines.append("")

    if honesty:
        lines.append("## Data Honesty")
        lines.append("")
        lines.append(f"- Status: `{honesty.get('status', 'unknown')}`")
        lines.append(f"- Summary: {honesty.get('summary', 'No data honesty report was recorded.')}")
        lines.append(f"- Exact SFT prompt leaks: {honesty.get('exact_prompt_leaks', 0)}")
        lines.append(f"- Near SFT prompt leaks: {honesty.get('near_prompt_leaks', 0)}")
        lines.append(f"- Eval prompts found in corpus: {honesty.get('corpus_prompt_hits', 0)}")
        lines.append(f"- Specific eval support phrases found in SFT answers: {honesty.get('sft_support_phrase_hits', 0)}")
        lines.append(f"- Specific eval support phrases found in corpus: {honesty.get('corpus_support_phrase_hits', 0)}")
        lines.append(f"- Duplicate eval prompts: {honesty.get('duplicate_eval_prompts', 0)}")
        if honesty.get("max_sft_prompt_similarity") is not None:
            lines.append(
                f"- Max SFT/eval prompt similarity: "
                f"{format_float(honesty.get('max_sft_prompt_similarity', 0.0))}"
            )
        matrix_pairs = honesty.get("contamination_matrix", {}).get("pairs", [])
        if matrix_pairs:
            lines.append("- Contamination matrix:")
            for pair in matrix_pairs:
                checked = (
                    "checked"
                    if pair.get("checked")
                    else f"not checked: {pair.get('reason', 'unknown')}"
                )
                lines.append(
                    f"  - `{pair.get('name', 'unknown')}`: risk `{pair.get('risk', 'unknown')}`, "
                    f"{checked}, max n-gram overlap "
                    f"{format_float(float(pair.get('max_ngram_overlap_rate', 0.0)))}, "
                    f"longest overlap {int(pair.get('max_longest_overlap_tokens', 0))} tokens"
                )
        lines.append("")

    if eval_summary.get("category_breakdown"):
        lines.append("## Eval Categories")
        lines.append("")
        lines.extend(_category_breakdown_table(eval_summary["category_breakdown"]))
        lines.append("")

    if sft_fit_summary.get("category_breakdown"):
        lines.append("## SFT Fit Categories")
        lines.append("")
        lines.append(
            "This checks whether the checkpoint used for eval can reproduce its own "
            "SFT rows. Low fit means the chat stage is undertrained or too broad "
            "before held-out eval can be trusted."
        )
        lines.append("")
        lines.extend(_category_breakdown_table(sft_fit_summary["category_breakdown"]))
        lines.append("")

    if eval_summary.get("stage_breakdown"):
        lines.append("## Eval Curriculum Stages")
        lines.append("")
        lines.extend(_breakdown_table(eval_summary["stage_breakdown"], "Stage"))
        lines.append("")

    if eval_summary.get("skill_stage_breakdown"):
        lines.append("## Eval Skill Stages")
        lines.append("")
        lines.extend(_breakdown_table(eval_summary["skill_stage_breakdown"], "Skill stage"))
        lines.append("")

    if eval_summary.get("robustness_breakdown"):
        lines.append("## Eval Robustness Variants")
        lines.append("")
        lines.extend(_breakdown_table(eval_summary["robustness_breakdown"], "Variant"))
        lines.append("")

    if sft_fit_summary.get("stage_breakdown"):
        lines.append("## SFT Fit Curriculum Stages")
        lines.append("")
        lines.extend(_breakdown_table(sft_fit_summary["stage_breakdown"], "Stage"))
        lines.append("")

    if eval_summary.get("split_breakdown"):
        lines.append("## Eval Splits")
        lines.append("")
        lines.extend(_split_breakdown_table(eval_summary["split_breakdown"]))
        lines.append("")

    if eval_summary.get("level_breakdown"):
        lines.append("## Eval Ladder")
        lines.append("")
        lines.extend(_level_breakdown_table(eval_summary["level_breakdown"]))
        lines.append("")

    if summary.get("eval_analysis"):
        lines.append("## Eval Recommendations")
        lines.append("")
        lines.extend(_eval_analysis_markdown(summary["eval_analysis"], compact=True))
        lines.append("")

    lines.append("## Settings")
    lines.append("")
    lines.append(f"- Scale: `{config.get('scale', 'custom')}`")
    lines.append(f"- Context size: {config['context_size']}")
    lines.append(f"- Embedding size: {config['n_embd']}")
    lines.append(f"- Layers: {config['n_layer']}")
    lines.append(f"- Attention heads: {config['n_head']}")
    lines.append(f"- KV heads: {config.get('n_kv_head') or config['n_head']}")
    lines.append(f"- Norm: `{config.get('norm_type', 'layernorm')}`")
    lines.append(f"- Position encoding: `{config.get('position_encoding', 'learned')}`")
    lines.append(f"- Activation: `{config.get('activation', 'gelu')}`")
    lines.append(f"- Tied embeddings: {'enabled' if config.get('tie_embeddings') else 'disabled'}")
    lines.append(f"- QK norm: {'enabled' if config.get('qk_norm') else 'disabled'}")
    lines.append(f"- Parallel residual: {'enabled' if config.get('parallel_residual') else 'disabled'}")
    lines.append(f"- XSA last layers: {config.get('xsa_last_n', 0) or 'disabled'}")
    lines.append(f"- Scaled residual init: {'enabled' if config.get('scaled_residual_init') else 'disabled'}")
    lines.append(f"- Logit softcap: {config.get('logit_softcap', 0.0) or 'disabled'}")
    lines.append(f"- Base steps: {config['base_steps']}")
    lines.append(f"- SFT steps: {config['sft_steps']}")
    lines.append(f"- Tokenizer type: `{tokenizer.get('tokenizer_type', 'unknown')}`")
    if config.get("tokenizer_vocab_size"):
        lines.append(f"- Tokenizer vocab size target: {config.get('tokenizer_vocab_size')}")
    if config.get("tokenizer_type") in {"bpe", "hf_bpe"}:
        lines.append(f"- BPE pretokenizer: `{config.get('bpe_pretokenizer', 'char')}`")
    lines.append(f"- Base LR decay: `{config.get('base_lr_decay', 'none')}`")
    lines.append(f"- SFT LR decay: `{config.get('sft_lr_decay', 'none')}`")
    lines.append(f"- Base optimizer: `{config.get('base_optimizer', 'adamw')}`")
    lines.append(f"- SFT optimizer: `{config.get('sft_optimizer', 'adamw')}`")
    if config.get("base_optimizer") == "muon" or config.get("sft_optimizer") == "muon":
        lines.append(f"- Base Muon LR: {config.get('base_muon_learning_rate')}")
        lines.append(f"- SFT Muon LR: {config.get('sft_muon_learning_rate')}")
    lines.append(f"- Base EMA decay: {config.get('base_ema_decay', 0.0) or 'disabled'}")
    lines.append(f"- SFT EMA decay: {config.get('sft_ema_decay', 0.0) or 'disabled'}")
    lines.append(f"- Base grad clip: {config.get('base_grad_clip', 0.0) or 'disabled'}")
    lines.append(f"- SFT grad clip: {config.get('sft_grad_clip', 0.0) or 'disabled'}")
    lines.append(f"- Base grad accumulation: {config.get('base_grad_accum_steps', 1)}")
    lines.append(f"- SFT grad accumulation: {config.get('sft_grad_accum_steps', 1)}")
    lines.append(f"- Base early stop patience: {config.get('base_early_stop_patience', 0)}")
    lines.append(f"- SFT early stop patience: {config.get('sft_early_stop_patience', 0)}")
    lines.append(f"- SFT fit diagnostic max rows: {config.get('sft_fit_max_rows', 1000)}")
    lines.append(f"- Train-only canaries: {config.get('canary_count', 0)}")
    lines.append(f"- Device: `{config['device']}`")
    if config.get("dataset_pack"):
        lines.append(f"- Dataset pack: `{config['dataset_pack']}`")
    lines.append("")

    lines.append("## Losses")
    lines.append("")
    lines.append(f"- Base final train loss: {format_float(base['final_train_loss'])}")
    lines.append(f"- Base final val loss: {format_float(base['final_val_loss'])}")
    if base.get("best_val_bpb") is not None:
        lines.append(f"- Base best val BPB: {format_optional_float(base.get('best_val_bpb'))}")
    if base.get("final_val_bpb") is not None:
        lines.append(f"- Base final val BPB: {format_optional_float(base.get('final_val_bpb'))}")
    if base.get("final_ema_val_bpb") is not None:
        lines.append(f"- Base final EMA val BPB: {format_optional_float(base.get('final_ema_val_bpb'))}")
    lines.append(f"- SFT final train loss: {format_float(sft['final_train_loss'])}")
    lines.append(f"- SFT final val loss: {format_float(sft['final_val_loss'])}")
    if sft.get("best_val_bpb") is not None:
        lines.append(f"- SFT best val BPB: {format_optional_float(sft.get('best_val_bpb'))}")
    if sft.get("final_val_bpb") is not None:
        lines.append(f"- SFT final val BPB: {format_optional_float(sft.get('final_val_bpb'))}")
    if sft.get("final_ema_val_bpb") is not None:
        lines.append(f"- SFT final EMA val BPB: {format_optional_float(sft.get('final_ema_val_bpb'))}")
    lines.append(f"- SFT truncated examples: {sft['truncated_examples']}")
    if "skipped_long_examples" in sft:
        lines.append(f"- SFT skipped too-long examples: {sft['skipped_long_examples']}")
    if base_coverage:
        lines.append(f"- Base stop reason: `{base.get('stop_reason', 'unknown')}`")
        lines.append(f"- Base estimated train epochs: {format_optional_float(base_coverage.get('estimated_train_epochs'))}")
        for warning in base_coverage.get("warnings", []):
            lines.append(f"- Base coverage warning: {warning}")
    if sft_coverage:
        lines.append(f"- SFT stop reason: `{sft.get('stop_reason', 'unknown')}`")
        lines.append(f"- SFT estimated train epochs: {format_optional_float(sft_coverage.get('estimated_train_epochs'))}")
        for warning in sft_coverage.get("warnings", []):
            lines.append(f"- SFT coverage warning: {warning}")
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
    if artifacts.get("preflight_report"):
        lines.append(f"- Run preflight: `{artifacts['preflight_report']}`")
    if artifacts.get("honesty_report"):
        lines.append(f"- Data honesty report: `{artifacts['honesty_report']}`")
    lines.append(f"- Tokenizer: `{artifacts['tokenizer']}`")
    lines.append(f"- Base report: `{artifacts['base_report']}`")
    if artifacts.get("base_eval_checkpoint"):
        lines.append(f"- Base checkpoint used for SFT: `{artifacts['base_eval_checkpoint']}`")
    lines.append(f"- SFT report: `{artifacts['sft_report']}`")
    if artifacts.get("sft_eval_checkpoint"):
        lines.append(f"- SFT checkpoint used for eval: `{artifacts['sft_eval_checkpoint']}`")
    if artifacts.get("sft_fit_report"):
        lines.append(f"- SFT fit report: `{artifacts['sft_fit_report']}`")
    lines.append(f"- Eval report: `{artifacts['eval_report']}`")
    lines.append("- Machine-readable summary: `summary.json`")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Use this summary to compare tiny runs. If eval improves but SFT validation "
        "loss rises, the model is likely memorizing the tiny chat data rather than "
        "generalizing. Unsupported claim, prompt echo, and missing support rates "
        "are rule-based signals from the eval file, not proof of semantic truth."
    )
    lines.append("")
    return "\n".join(lines)


def _category_breakdown_table(category_breakdown: dict) -> list[str]:
    return _breakdown_table(category_breakdown, "Category")


def _split_breakdown_table(split_breakdown: dict) -> list[str]:
    return _breakdown_table(split_breakdown, "Split")


def _level_breakdown_table(level_breakdown: dict) -> list[str]:
    return _breakdown_table(level_breakdown, "Level")


def _breakdown_table(breakdown: dict, label: str) -> list[str]:
    lines = [
        f"| {label} | Passed | Pass Rate | 95% CI | Support Match | Ref F1 | Corpus Support | Missing Support | Prompt Echo | Unsupported |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(breakdown.items()):
        passed = f"{row.get('num_passed', 0)} / {row.get('num_examples', 0)}"
        pass_rate = format_float(row.get("pass_rate", 0.0) * 100)
        pass_ci = _format_ci(row.get("pass_rate_ci"))
        support = format_float(row.get("support_match_rate", 0.0) * 100)
        ref_f1 = _format_percent_or_dash(row.get("average_reference_token_f1"))
        corpus_support = _format_percent_or_dash(row.get("average_corpus_support_rate"))
        missing = f"{row.get('missing_support', 0)} / {row.get('num_examples', 0)}"
        prompt_echo = f"{row.get('prompt_echoes', 0)} / {row.get('num_examples', 0)}"
        unsupported = f"{row.get('unsupported_claims', 0)} / {row.get('num_examples', 0)}"
        lines.append(
            f"| `{name}` | {passed} | {pass_rate}% | {pass_ci} | {support}% | "
            f"{ref_f1} | {corpus_support} | {missing} | {prompt_echo} | {unsupported} |"
        )
    return lines


def _eval_analysis_markdown(analysis: dict, compact: bool = False) -> list[str]:
    lines: list[str] = []
    recommendations = analysis.get("recommendations") or []
    failure_counts = analysis.get("failure_counts") or {}
    cluster_counts = analysis.get("cluster_counts") or {}
    weak_categories = analysis.get("weak_categories") or []
    weak_splits = analysis.get("weak_splits") or []
    weak_levels = analysis.get("weak_levels") or []
    failed_examples = analysis.get("failed_examples") or []

    if recommendations:
        lines.append("Recommendations:")
        for item in recommendations:
            priority = item.get("priority", "medium")
            area = item.get("area", "eval")
            action = item.get("action", "")
            message = item.get("message", "")
            lines.append(f"- `{priority}` `{area}`: {message} {action}".strip())
        lines.append("")

    if failure_counts:
        lines.append("Failure causes:")
        lines.append("")
        lines.append("| Cause | Count |")
        lines.append("| --- | ---: |")
        for name, count in sorted(failure_counts.items()):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")

    if cluster_counts:
        lines.append("Failure clusters:")
        lines.append("")
        lines.append("| Cluster | Count |")
        lines.append("| --- | ---: |")
        for name, count in sorted(cluster_counts.items()):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")

    if weak_categories:
        lines.append("Weak categories:")
        lines.append("")
        lines.extend(_weak_eval_table(weak_categories, "category"))
        lines.append("")

    if weak_splits:
        lines.append("Weak splits:")
        lines.append("")
        lines.extend(_weak_eval_table(weak_splits, "split"))
        lines.append("")

    if weak_levels:
        lines.append("Weak ladder levels:")
        lines.append("")
        lines.extend(_weak_eval_table(weak_levels, "level"))
        lines.append("")

    if not compact and failed_examples:
        lines.append("Failed examples:")
        lines.append("")
        lines.append("| # | Category | Level | Split | Clusters | Reasons | Missing | Forbidden | Reply Preview |")
        lines.append("| ---: | --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in failed_examples:
            missing_parts = list(item.get("missing", []))
            missing_parts.extend(
                " / ".join(group)
                for group in item.get("missing_any", [])
                if isinstance(group, (list, tuple))
            )
            lines.append(
                f"| {item.get('index')} | `{item.get('category')}` | "
                f"`{item.get('level', 'heldout')}` | `{item.get('split')}` | "
                f"{_inline_list(item.get('clusters', [])) or 'none'} | "
                f"{_inline_list(item.get('reasons', [])) or 'none'} | "
                f"{_inline_list(missing_parts) if missing_parts else 'none'} | "
                f"{_inline_list(item.get('found_forbidden', [])) if item.get('found_forbidden') else 'none'} | "
                f"{_escape_table_text(item.get('reply_preview', ''))} |"
            )
        lines.append("")

    if not lines:
        lines.append("- No failure analysis was recorded.")
    return lines


def _weak_eval_table(items: list[dict], field: str) -> list[str]:
    lines = [
        f"| {field.title()} | Failed | Pass Rate | Support Match | Corpus Fail | Prompt Echo | Unsupported |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in items:
        lines.append(
            f"| `{item.get(field)}` | {item.get('num_failed', 0)} / {item.get('num_examples', 0)} | "
            f"{format_float(float(item.get('pass_rate', 0.0)) * 100)}% | "
            f"{format_float(float(item.get('support_match_rate', 0.0)) * 100)}% | "
            f"{format_float(float(item.get('corpus_support_failure_rate', 0.0)) * 100)}% | "
            f"{format_float(float(item.get('prompt_echo_rate', 0.0)) * 100)}% | "
            f"{format_float(float(item.get('unsupported_claim_rate', 0.0)) * 100)}% |"
        )
    return lines


def _eval_metric_lines(item: dict) -> list[str]:
    lines: list[str] = []
    if item.get("reference_token_f1") is not None:
        lines.append(f"- Reference token F1: {_format_percent_or_dash(item.get('reference_token_f1'))}")
    if item.get("reference_rouge_l") is not None:
        lines.append(f"- Reference ROUGE-L: {_format_percent_or_dash(item.get('reference_rouge_l'))}")
    if item.get("entity_total"):
        lines.append(
            f"- Entity match: {item.get('entity_matched', 0)} / {item.get('entity_total', 0)} "
            f"({_format_percent_or_dash(item.get('entity_match_rate'))})"
        )
    if item.get("corpus_support_rate") is not None:
        lines.append(
            f"- Corpus support: {item.get('corpus_support_tokens', 0)} / "
            f"{item.get('corpus_support_total', 0)} content tokens "
            f"({_format_percent_or_dash(item.get('corpus_support_rate'))})"
        )
    if item.get("normalized_answer_match") is not None:
        lines.append(f"- Normalized final answer match: `{bool(item.get('normalized_answer_match'))}`")
        expected = item.get("normalized_answer_expected") or []
        candidates = item.get("normalized_answer_candidates") or []
        if expected:
            lines.append(f"- Normalized expected: {_inline_list(expected)}")
        if candidates:
            lines.append(f"- Normalized candidates: {_inline_list(candidates[:4])}")
    if item.get("repetition_ngram_rate") is not None:
        lines.append(f"- Repeated trigram rate: {_format_percent_or_dash(item.get('repetition_ngram_rate'))}")
    if item.get("word_count") is not None:
        lines.append(f"- Length: {item.get('word_count')} words / {item.get('char_count')} chars")
    if item.get("answerable") is False:
        lines.append(f"- Refusal phrase detected: `{bool(item.get('refusal_match'))}`")
    if item.get("correct_choice"):
        lines.append(
            f"- Choice predicted: `{item.get('choice_predicted')}` "
            f"(correct `{item.get('correct_choice')}`)"
        )
        scores = item.get("choice_logprobs") or {}
        if isinstance(scores, dict) and scores:
            rendered = ", ".join(
                f"{label}:{format_float(float(score))}"
                for label, score in sorted(scores.items())
            )
            lines.append(f"- Choice scores: {rendered}")
        if item.get("choice_correct_logprob_margin") is not None:
            lines.append(
                f"- Correct-choice margin: "
                f"{format_optional_float(item.get('choice_correct_logprob_margin'))}"
            )
        if item.get("choice_eval_method"):
            lines.append(f"- Choice eval method: `{item.get('choice_eval_method')}`")
    return lines


def _format_percent_or_dash(value: object) -> str:
    number = _number(value)
    if number is None:
        return "--"
    return f"{format_float(number * 100)}%"


def _format_ci(value: object) -> str:
    if not isinstance(value, dict):
        return "--"
    low = _number(value.get("low"))
    high = _number(value.get("high"))
    if low is None or high is None:
        return "--"
    confidence = _number(value.get("confidence")) or 0.95
    return f"{format_float(confidence * 100)}% CI {format_float(low * 100)}%-{format_float(high * 100)}%"


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


def _escape_table_text(value: object) -> str:
    return str(value).replace("|", "\\|")


def _format_counts(counts: dict) -> str:
    return ", ".join(f"`{key}` {value}" for key, value in sorted(counts.items()))


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


def _loss_recommendations(
    final_gap: float | None,
    val_regression: float | None,
    evals_after_best: int,
    best_step: object,
    final_step: object,
) -> list[str]:
    recommendations: list[str] = []
    if best_step is not None and final_step is not None and best_step != final_step:
        recommendations.append(
            f"Use the best-validation checkpoint at step {best_step} for downstream stages, not the final step {final_step}."
        )
    if val_regression is not None and val_regression >= 0.10:
        recommendations.append(
            "Shorten this stage or lower early-stop patience; validation moved noticeably away from the best checkpoint."
        )
    if final_gap is not None and final_gap >= 0.75:
        recommendations.append(
            "Add regularization, more data, or fewer training passes before scaling this recipe longer."
        )
    if evals_after_best >= 2:
        recommendations.append(
            f"Validation failed to beat the best checkpoint for {evals_after_best} later eval point(s)."
        )
    if not recommendations:
        recommendations.append("Training curve looks usable; compare eval behavior before changing scale.")
    return recommendations
