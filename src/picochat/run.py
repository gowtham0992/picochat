"""End-to-end experiment runners for Picochat."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from picochat.data import build_corpus_artifacts
from picochat.eval import ChatEvalConfig, run_chat_eval
from picochat.honesty import inspect_data_honesty, write_data_honesty_report
from picochat.report import tiny_run_summary_markdown
from picochat.sft import SFTConfig, train_sft
from picochat.tokenizer import TOKENIZER_TYPES, train_tokenizer
from picochat.train import TrainConfig, train_base


@dataclass(frozen=True)
class TinyRunConfig:
    out_dir: str
    scale: str = "custom"
    dataset_pack: str | None = None
    corpus_input: str = "examples/tiny_corpus.txt"
    corpus_recipe: str | None = None
    chat_input: str = "examples/tiny_chat.jsonl"
    eval_input: str = "examples/tiny_eval.jsonl"
    context_size: int = 128
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
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


def run_tiny(config: TinyRunConfig) -> dict:
    """Run the tiny educational pipeline from corpus to eval report."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== pico run tiny ==")
    corpus_path = out_dir / "corpus.txt"
    tokenizer_path = out_dir / "tokenizer.json"

    print(f"[1/5] build corpus -> {corpus_path}")
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

    if config.tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(f"Unsupported tokenizer type: {config.tokenizer_type}")

    print("[2/6] check data honesty")
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

    print(f"[3/6] train {config.tokenizer_type} tokenizer -> {tokenizer_path}")
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = train_tokenizer(
        config.tokenizer_type,
        [text],
        vocab_size=config.tokenizer_vocab_size,
        min_freq=config.tokenizer_min_freq,
    )
    tokenizer.save(tokenizer_path)

    print("[4/6] train base model")
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
        seed=config.seed,
        device=config.device,
        log_every=max(1, config.base_steps // 6),
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
    ))
    base_eval_checkpoint = base_report.get("best_checkpoint", {}).get(
        "path",
        str(out_dir / "base" / "checkpoint"),
    )

    print("[5/6] train chat SFT")
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
        log_every=max(1, config.sft_steps // 6),
        sample_tokens=160,
        early_stop_patience=config.sft_early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        max_minutes=config.sft_max_minutes,
        lr_warmup_steps=config.sft_lr_warmup_steps,
        lr_decay=config.sft_lr_decay,
        min_lr_ratio=config.sft_min_lr_ratio,
        grad_clip=config.sft_grad_clip,
    ))
    sft_eval_checkpoint = sft_report.get("best_checkpoint", {}).get(
        "path",
        str(out_dir / "sft" / "checkpoint"),
    )

    print("[6/6] run chat eval")
    eval_report = run_chat_eval(ChatEvalConfig(
        input_path=eval_input,
        checkpoint_path=sft_eval_checkpoint,
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir / "eval"),
        max_new_tokens=config.eval_max_new_tokens,
        seed=config.seed,
        device=config.device,
    ))

    effective_config = {
        **config.__dict__,
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
            "honesty_json": honesty_json_path,
            "honesty_report": honesty_markdown_path,
            "tokenizer": str(tokenizer_path),
            "base_report": str(out_dir / "base" / "report.md"),
            "base_best_checkpoint": base_report.get("best_checkpoint", {}).get("path"),
            "base_eval_checkpoint": base_eval_checkpoint,
            "sft_report": str(out_dir / "sft" / "report.md"),
            "sft_best_checkpoint": sft_report.get("best_checkpoint", {}).get("path"),
            "sft_eval_checkpoint": sft_eval_checkpoint,
            "eval_report": str(out_dir / "eval" / "report.md"),
        },
        "corpus": corpus_build.stats.to_dict(),
        "honesty": honesty_report.to_dict(),
        "tokenizer": tokenizer.stats().__dict__,
        "base": {
            "checkpoint": base_report["checkpoint"],
            "best_checkpoint": base_report.get("best_checkpoint", {}),
            "eval_checkpoint": base_eval_checkpoint,
            "final_train_loss": base_report["losses"][-1]["train_loss"],
            "final_val_loss": base_report["losses"][-1]["val_loss"],
            "final_val_bpb": base_report["losses"][-1].get("val_bpb"),
            "num_parameters": base_report["model"]["num_parameters"],
            "loss_diagnostics": base_report.get("loss_diagnostics", {}),
            "memorization": base_report.get("memorization", {}),
            "coverage": base_report.get("coverage", {}),
            "stop_reason": base_report.get("stop_reason"),
        },
        "sft": {
            "checkpoint": sft_report["checkpoint"],
            "final_train_loss": sft_report["losses"][-1]["train_loss"],
            "final_val_loss": sft_report["losses"][-1]["val_loss"],
            "final_val_bpb": sft_report["losses"][-1].get("val_bpb"),
            "truncated_examples": sft_report["dataset"]["truncated_examples"],
            "loss_diagnostics": sft_report.get("loss_diagnostics", {}),
            "best_checkpoint": sft_report.get("best_checkpoint", {}),
            "eval_checkpoint": sft_eval_checkpoint,
            "coverage": sft_report.get("coverage", {}),
            "stop_reason": sft_report.get("stop_reason"),
        },
        "eval": eval_report["summary"],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(tiny_run_summary_markdown(summary), encoding="utf-8")
    print(
        f"done: {summary['eval']['num_passed']}/{summary['eval']['num_examples']} "
        f"passed ({summary['eval']['pass_rate'] * 100:.2f}%)"
    )
    print(f"summary: {out_dir / 'summary.md'}")
    return summary
