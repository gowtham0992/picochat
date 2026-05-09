"""End-to-end experiment runners for Picochat."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from picochat.data import build_corpus_artifacts
from picochat.eval import ChatEvalConfig, run_chat_eval
from picochat.report import tiny_run_summary_markdown
from picochat.sft import SFTConfig, train_sft
from picochat.tokenizer import CharTokenizer
from picochat.train import TrainConfig, train_base


@dataclass(frozen=True)
class TinyRunConfig:
    out_dir: str
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


def run_tiny(config: TinyRunConfig) -> dict:
    """Run the tiny educational pipeline from corpus to eval report."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== pico run tiny ==")
    corpus_path = out_dir / "corpus.txt"
    tokenizer_path = out_dir / "tokenizer.json"

    print(f"[1/5] build corpus -> {corpus_path}")
    corpus_build = build_corpus_artifacts(
        config.corpus_input,
        corpus_path,
        recipe_path=config.corpus_recipe,
    )

    print(f"[2/5] train tokenizer -> {tokenizer_path}")
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train([text])
    tokenizer.save(tokenizer_path)

    print("[3/5] train base model")
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
    ))

    print("[4/5] train chat SFT")
    sft_report = train_sft(SFTConfig(
        input_path=config.chat_input,
        tokenizer_path=str(tokenizer_path),
        checkpoint_path=str(out_dir / "base" / "checkpoint"),
        out_dir=str(out_dir / "sft"),
        batch_size=config.sft_batch_size,
        max_steps=config.sft_steps,
        learning_rate=config.sft_learning_rate,
        seed=config.seed,
        device=config.device,
        log_every=max(1, config.sft_steps // 6),
        sample_tokens=160,
    ))

    print("[5/5] run chat eval")
    eval_report = run_chat_eval(ChatEvalConfig(
        input_path=config.eval_input,
        checkpoint_path=str(out_dir / "sft" / "checkpoint"),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir / "eval"),
        max_new_tokens=config.eval_max_new_tokens,
        seed=config.seed,
        device=config.device,
    ))

    summary = {
        "config": config.__dict__,
        "artifacts": {
            "corpus": str(corpus_path),
            "corpus_manifest": corpus_build.manifest_path,
            "corpus_report": corpus_build.report_path,
            "tokenizer": str(tokenizer_path),
            "base_report": str(out_dir / "base" / "report.md"),
            "sft_report": str(out_dir / "sft" / "report.md"),
            "eval_report": str(out_dir / "eval" / "report.md"),
        },
        "corpus": corpus_build.stats.to_dict(),
        "tokenizer": tokenizer.stats().__dict__,
        "base": {
            "checkpoint": base_report["checkpoint"],
            "final_train_loss": base_report["losses"][-1]["train_loss"],
            "final_val_loss": base_report["losses"][-1]["val_loss"],
            "num_parameters": base_report["model"]["num_parameters"],
        },
        "sft": {
            "checkpoint": sft_report["checkpoint"],
            "final_train_loss": sft_report["losses"][-1]["train_loss"],
            "final_val_loss": sft_report["losses"][-1]["val_loss"],
            "truncated_examples": sft_report["dataset"]["truncated_examples"],
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
