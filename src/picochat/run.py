"""End-to-end experiment runners for Picochat."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from picochat.data import (
    DEFAULT_CHAT_INPUT,
    DEFAULT_CORPUS_INPUT,
    DEFAULT_EVAL_INPUT,
    build_corpus_artifacts,
)
from picochat.eval import ChatEvalConfig, run_chat_eval, write_sft_fit_eval
from picochat.honesty import inspect_data_honesty, write_data_honesty_report
from picochat.report import tiny_run_summary_markdown
from picochat.run_preflight import assess_run_preflight, preflight_markdown
from picochat.sft import SFTConfig, SFT_SAMPLING_MODES, train_sft
from picochat.tokenizer import TOKENIZER_TYPES, train_tokenizer
from picochat.train import TrainConfig, train_base


@dataclass(frozen=True)
class TinyRunConfig:
    out_dir: str
    scale: str = "custom"
    dataset_pack: str | None = None
    corpus_input: str = DEFAULT_CORPUS_INPUT
    corpus_recipe: str | None = None
    chat_input: str = "examples/tiny_chat.jsonl"
    eval_input: str = "examples/tiny_eval.jsonl"
    context_size: int = 128
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    activation: str = "gelu"
    base_steps: int = 300
    sft_steps: int = 600
    base_batch_size: int = 8
    sft_batch_size: int = 7
    base_learning_rate: float = 3e-4
    sft_learning_rate: float = 1e-3
    seed: int = 42
    device: str = "cpu"
    eval_max_new_tokens: int = 120
    min_quality_score: int = 0
    split_mode: str = "document"
    tokenizer_type: str = "char"
    tokenizer_vocab_size: int | None = None
    tokenizer_min_freq: int = 1
    base_early_stop_patience: int = 6
    sft_early_stop_patience: int = 6
    early_stop_min_delta: float = 0.0
    base_max_minutes: float | None = None
    sft_max_minutes: float | None = None
    canary_count: int = 1
    allow_leaky_eval: bool = False
    base_lr_warmup_steps: int = 0
    sft_lr_warmup_steps: int = 0
    base_lr_decay: str = "none"
    sft_lr_decay: str = "none"
    base_min_lr_ratio: float = 1.0
    sft_min_lr_ratio: float = 1.0
    base_grad_clip: float = 0.0
    sft_grad_clip: float = 0.0
    base_grad_accum_steps: int = 1
    sft_grad_accum_steps: int = 1
    base_optimizer: str = "adamw"
    sft_optimizer: str = "adamw"
    base_muon_learning_rate: float = 0.02
    sft_muon_learning_rate: float = 0.02
    base_ema_decay: float = 0.0
    sft_ema_decay: float = 0.0
    sft_sampling: str = "uniform"
    sft_fit_max_rows: int = 1000
    allow_default_tuning_data: bool = False
    logit_softcap: float = 0.0
    allow_unsafe_long_run: bool = False


def run_tiny(config: TinyRunConfig) -> dict:
    """Run the tiny educational pipeline from corpus to eval report."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== pico run tiny ==")
    corpus_path = out_dir / "corpus.txt"
    tokenizer_path = out_dir / "tokenizer.json"

    print(f"[1/7] build corpus -> {corpus_path}")
    corpus_build = build_corpus_artifacts(
        None if config.dataset_pack else config.corpus_input,
        corpus_path,
        recipe_path=None if config.dataset_pack else config.corpus_recipe,
        chat_input=None if config.dataset_pack else config.chat_input,
        eval_input=None if config.dataset_pack else config.eval_input,
        dataset_pack=config.dataset_pack,
        min_quality_score=config.min_quality_score,
    )
    chat_input = corpus_build.training_command.chat_input
    eval_input = corpus_build.training_command.eval_input
    _validate_tuning_data_source(config, chat_input=chat_input, eval_input=eval_input)

    preflight_report = assess_run_preflight(config, corpus_build)
    preflight_json_path = out_dir / "preflight.json"
    preflight_markdown_path = out_dir / "preflight.md"
    preflight_json_path.write_text(json.dumps(preflight_report.to_dict(), indent=2), encoding="utf-8")
    preflight_markdown_path.write_text(preflight_markdown(preflight_report), encoding="utf-8")
    print(f"preflight: {preflight_report.status} | {preflight_report.summary}")
    if preflight_report.status == "blocked" and not config.allow_unsafe_long_run:
        raise ValueError(
            "run preflight blocked this plan; inspect "
            f"{preflight_markdown_path} or rerun with --allow-unsafe-long-run "
            "for a diagnostic-only run"
        )

    if config.tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(f"Unsupported tokenizer type: {config.tokenizer_type}")
    if config.sft_sampling not in SFT_SAMPLING_MODES:
        raise ValueError(f"Unsupported SFT sampling mode: {config.sft_sampling}")

    print("[2/7] check data honesty")
    honesty_report = inspect_data_honesty(
        corpus_path=corpus_path,
        chat_input=chat_input,
        eval_input=eval_input,
    )
    honesty_json_path, honesty_markdown_path = write_data_honesty_report(
        honesty_report,
        out_dir / "honesty",
    )
    if honesty_report.status == "blocked" and not config.allow_leaky_eval:
        raise ValueError(
            "data honesty blocked this run; inspect "
            f"{honesty_markdown_path} or rerun with --allow-leaky-eval for a diagnostic-only run"
        )

    print(f"[3/7] train {config.tokenizer_type} tokenizer -> {tokenizer_path}")
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = train_tokenizer(
        config.tokenizer_type,
        [text],
        vocab_size=config.tokenizer_vocab_size,
        min_freq=config.tokenizer_min_freq,
    )
    tokenizer.save(tokenizer_path)

    print("[4/7] train base model")
    base_report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir / "base"),
        context_size=config.context_size,
        batch_size=config.base_batch_size,
        max_steps=config.base_steps,
        learning_rate=config.base_learning_rate,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        norm_type=config.norm_type,
        position_encoding=config.position_encoding,
        activation=config.activation,
        seed=config.seed,
        device=config.device,
        log_every=_validation_log_every(config.base_steps),
        sample_tokens=160,
        split_mode=config.split_mode,
        corpus_manifest_path=corpus_build.manifest_path,
        early_stop_patience=config.base_early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        max_minutes=config.base_max_minutes,
        canary_count=config.canary_count,
        lr_warmup_steps=config.base_lr_warmup_steps,
        lr_decay=config.base_lr_decay,
        min_lr_ratio=config.base_min_lr_ratio,
        grad_clip=config.base_grad_clip,
        grad_accum_steps=config.base_grad_accum_steps,
        optimizer=config.base_optimizer,
        muon_learning_rate=config.base_muon_learning_rate,
        ema_decay=config.base_ema_decay,
        logit_softcap=config.logit_softcap,
    ))
    base_eval_checkpoint = base_report.get("best_checkpoint", {}).get(
        "path",
        str(out_dir / "base" / "checkpoint"),
    )

    print("[5/7] train chat SFT")
    sft_report = train_sft(SFTConfig(
        input_path=chat_input,
        tokenizer_path=str(tokenizer_path),
        checkpoint_path=base_eval_checkpoint,
        out_dir=str(out_dir / "sft"),
        batch_size=config.sft_batch_size,
        max_steps=config.sft_steps,
        learning_rate=config.sft_learning_rate,
        seed=config.seed,
        device=config.device,
        log_every=_validation_log_every(config.sft_steps),
        sample_tokens=160,
        early_stop_patience=config.sft_early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        max_minutes=config.sft_max_minutes,
        lr_warmup_steps=config.sft_lr_warmup_steps,
        lr_decay=config.sft_lr_decay,
        min_lr_ratio=config.sft_min_lr_ratio,
        grad_clip=config.sft_grad_clip,
        sampling=config.sft_sampling,
        grad_accum_steps=config.sft_grad_accum_steps,
        optimizer=config.sft_optimizer,
        muon_learning_rate=config.sft_muon_learning_rate,
        ema_decay=config.sft_ema_decay,
    ))
    sft_eval_checkpoint = sft_report.get("best_checkpoint", {}).get(
        "path",
        str(out_dir / "sft" / "checkpoint"),
    )

    print("[6/7] run SFT fit diagnostic")
    sft_fit_input = out_dir / "sft_fit" / "sft_fit_eval.jsonl"
    sft_fit_dataset = write_sft_fit_eval(
        chat_input,
        sft_fit_input,
        max_rows=None if config.sft_fit_max_rows <= 0 else config.sft_fit_max_rows,
    )
    sft_fit_report = run_chat_eval(ChatEvalConfig(
        input_path=str(sft_fit_input),
        checkpoint_path=sft_eval_checkpoint,
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir / "sft_fit"),
        max_new_tokens=config.eval_max_new_tokens,
        seed=config.seed,
        device=config.device,
        support_corpus_path=str(corpus_path),
    ))

    print("[7/7] run chat eval")
    eval_report = run_chat_eval(ChatEvalConfig(
        input_path=eval_input,
        checkpoint_path=sft_eval_checkpoint,
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir / "eval"),
        max_new_tokens=config.eval_max_new_tokens,
        seed=config.seed,
        device=config.device,
        support_corpus_path=str(corpus_path),
    ))
    long_run_gate = _long_run_gate(
        preflight_report=preflight_report.to_dict(),
        sft_fit_summary=sft_fit_report["summary"],
        eval_summary=eval_report["summary"],
        honesty=honesty_report.to_dict(),
    )

    effective_config = {
        **config.__dict__,
        "requested_device": config.device,
        "device": base_report.get("config", {}).get("device", config.device),
        "corpus_input": corpus_build.input_path,
        "corpus_recipe": corpus_build.recipe_path,
        "dataset_pack": corpus_build.dataset_pack,
        "chat_input": chat_input,
        "eval_input": eval_input,
    }
    summary = {
        "config": effective_config,
        "artifacts": {
            "dataset_pack": corpus_build.dataset_pack,
            "corpus": str(corpus_path),
            "corpus_manifest": corpus_build.manifest_path,
            "corpus_report": corpus_build.report_path,
            "preflight_json": str(preflight_json_path),
            "preflight_report": str(preflight_markdown_path),
            "honesty_json": honesty_json_path,
            "honesty_report": honesty_markdown_path,
            "tokenizer": str(tokenizer_path),
            "base_report": str(out_dir / "base" / "report.md"),
            "base_best_checkpoint": base_report.get("best_checkpoint", {}).get("path"),
            "base_ema_checkpoint": base_report.get("ema_checkpoint"),
            "base_eval_checkpoint": base_eval_checkpoint,
            "sft_report": str(out_dir / "sft" / "report.md"),
            "sft_best_checkpoint": sft_report.get("best_checkpoint", {}).get("path"),
            "sft_ema_checkpoint": sft_report.get("ema_checkpoint"),
            "sft_eval_checkpoint": sft_eval_checkpoint,
            "sft_fit_report": str(out_dir / "sft_fit" / "report.md"),
            "eval_report": str(out_dir / "eval" / "report.md"),
        },
        "corpus": corpus_build.stats.to_dict(),
        "preflight": preflight_report.to_dict(),
        "honesty": honesty_report.to_dict(),
        "tokenizer": tokenizer.stats().__dict__,
        "base": {
            "checkpoint": base_report["checkpoint"],
            "best_checkpoint": base_report.get("best_checkpoint", {}),
            "eval_checkpoint": base_eval_checkpoint,
            "final_train_loss": base_report["losses"][-1]["train_loss"],
            "final_val_loss": base_report["losses"][-1]["val_loss"],
            "final_val_bpb": base_report["losses"][-1].get("val_bpb"),
            "final_ema_val_loss": base_report["losses"][-1].get("ema_val_loss"),
            "final_ema_val_bpb": base_report["losses"][-1].get("ema_val_bpb"),
            "num_parameters": base_report["model"]["num_parameters"],
            "loss_diagnostics": base_report.get("loss_diagnostics", {}),
            "memorization": base_report.get("memorization", {}),
            "coverage": base_report.get("coverage", {}),
            "optimizer": base_report.get("config", {}).get("optimizer"),
            "optimizer_metadata": base_report.get("config", {}).get("optimizer_metadata", {}),
            "ema_checkpoint": base_report.get("ema_checkpoint"),
            "effective_batch_size": base_report.get("config", {}).get("effective_batch_size"),
            "effective_tokens_per_step": base_report.get("config", {}).get("effective_tokens_per_step"),
            "stop_reason": base_report.get("stop_reason"),
        },
        "sft": {
            "checkpoint": sft_report["checkpoint"],
            "final_train_loss": sft_report["losses"][-1]["train_loss"],
            "final_val_loss": sft_report["losses"][-1]["val_loss"],
            "final_val_bpb": sft_report["losses"][-1].get("val_bpb"),
            "final_ema_val_loss": sft_report["losses"][-1].get("ema_val_loss"),
            "final_ema_val_bpb": sft_report["losses"][-1].get("ema_val_bpb"),
            "truncated_examples": sft_report["dataset"]["truncated_examples"],
            "skipped_long_examples": sft_report["dataset"].get("skipped_long_examples", 0),
            "loss_diagnostics": sft_report.get("loss_diagnostics", {}),
            "best_checkpoint": sft_report.get("best_checkpoint", {}),
            "eval_checkpoint": sft_eval_checkpoint,
            "coverage": sft_report.get("coverage", {}),
            "optimizer": sft_report.get("config", {}).get("optimizer"),
            "optimizer_metadata": sft_report.get("config", {}).get("optimizer_metadata", {}),
            "ema_checkpoint": sft_report.get("ema_checkpoint"),
            "effective_batch_size": sft_report.get("config", {}).get("effective_batch_size"),
            "effective_tokens_per_step": sft_report.get("config", {}).get("effective_tokens_per_step"),
            "stop_reason": sft_report.get("stop_reason"),
        },
        "sft_fit": sft_fit_report["summary"],
        "sft_fit_dataset": sft_fit_dataset,
        "sft_fit_analysis": sft_fit_report.get("analysis", {}),
        "eval": eval_report["summary"],
        "eval_analysis": eval_report.get("analysis", {}),
        "long_run_gate": long_run_gate,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(tiny_run_summary_markdown(summary), encoding="utf-8")
    print(
        f"done: {summary['eval']['num_passed']}/{summary['eval']['num_examples']} "
        f"passed ({summary['eval']['pass_rate'] * 100:.2f}%)"
    )
    print(f"summary: {out_dir / 'summary.md'}")
    return summary


def _validation_log_every(max_steps: int, target_points: int = 24) -> int:
    """Choose enough validation points to catch long-run regressions."""
    if max_steps <= 1:
        return 1
    return max(1, max_steps // target_points)


def _long_run_gate(
    *,
    preflight_report: dict,
    sft_fit_summary: dict,
    eval_summary: dict,
    honesty: dict,
) -> dict:
    """Decide whether this completed run should be used as a long-run recipe."""
    issues: list[dict[str, str]] = []
    budget = preflight_report.get("budget", {})
    long_run = bool(budget.get("long_run"))
    if preflight_report.get("status") == "blocked":
        issues.append({
            "name": "preflight",
            "severity": "block",
            "message": "The run was launched despite blocked preflight checks.",
        })
    if honesty.get("status") == "blocked":
        issues.append({
            "name": "data_honesty",
            "severity": "block",
            "message": "Data honesty found leakage or tuning contamination.",
        })
    sft_fit_rate = float(sft_fit_summary.get("pass_rate") or 0.0)
    if sft_fit_rate < 0.70:
        issues.append({
            "name": "sft_fit",
            "severity": "block" if long_run else "warn",
            "message": "SFT fit is below 70%; fix behavior data before scaling this recipe.",
        })
    if float(eval_summary.get("prompt_echo_rate") or 0.0) > 0.05:
        issues.append({
            "name": "prompt_echo",
            "severity": "block",
            "message": "Eval found prompt echoing; this is not a trustworthy long-run recipe.",
        })
    if float(eval_summary.get("unsupported_claim_rate") or 0.0) > 0.05:
        issues.append({
            "name": "unsupported_claims",
            "severity": "warn",
            "message": "Unsupported claim rate is above 5%; inspect failed replies before scaling.",
        })

    status = "blocked" if any(item["severity"] == "block" for item in issues) else "warn" if issues else "approved"
    summary = (
        "Approved long-run recipe."
        if status == "approved"
        else "Do not use this as the approved long-run recipe yet."
        if status == "blocked"
        else "Promising, but inspect warnings before using this recipe."
    )
    return {
        "status": status,
        "summary": summary,
        "long_run": long_run,
        "sft_fit_threshold": 0.70,
        "sft_fit_rate": sft_fit_rate,
        "issues": issues,
    }


def _validate_tuning_data_source(
    config: TinyRunConfig,
    *,
    chat_input: str,
    eval_input: str,
) -> None:
    if config.dataset_pack or config.allow_default_tuning_data:
        return

    using_default_corpus = (
        config.corpus_recipe is None
        and _same_path_text(config.corpus_input, DEFAULT_CORPUS_INPUT)
    )
    using_default_chat = _same_path_text(chat_input, DEFAULT_CHAT_INPUT)
    using_default_eval = _same_path_text(eval_input, DEFAULT_EVAL_INPUT)
    if using_default_corpus and using_default_chat and using_default_eval:
        return

    if using_default_chat or using_default_eval:
        defaults = []
        if using_default_chat:
            defaults.append(DEFAULT_CHAT_INPUT)
        if using_default_eval:
            defaults.append(DEFAULT_EVAL_INPUT)
        raise ValueError(
            "custom corpus runs must not silently use Picochat demo tuning data; "
            f"provide domain --chat-input and --eval-input or use --dataset-pack. "
            f"Default file(s) still selected: {', '.join(defaults)}. "
            "Use --allow-default-tuning-data only for a diagnostic wiring check."
        )


def _same_path_text(left: str | None, right: str) -> bool:
    return str(left or "").strip() == right
