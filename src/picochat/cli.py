"""Command-line entrypoint for picochat."""

from __future__ import annotations

import argparse

from picochat.data import (
    DEFAULT_CHAT_INPUT,
    DEFAULT_EVAL_INPUT,
    build_corpus_artifacts,
    inspect_path,
    preview_corpus_sources,
)
from picochat.batching import load_token_dataset
from picochat.tokenizer import TOKENIZER_TYPES, load_tokenizer, train_tokenizer as build_tokenizer
from picochat.train import TrainConfig, train_base
from picochat.sft import SFTConfig, train_sft
from picochat.generate import GenerateConfig, generate_text
from picochat.chat import ChatConfig, chat_loop
from picochat.eval import ChatEvalConfig, run_chat_eval
from picochat.run import TinyRunConfig, run_tiny
from picochat.compare import compare_runs, comparison_table, write_comparison_report
from picochat.dataset_pack import init_dataset_pack
from picochat.hf_import import HFImportConfig, import_hf_dataset
from picochat.web import WebConfig, serve_web


SOURCE_PLAN_PREVIEW_LIMIT = 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pico",
        description="Run picochat training, evaluation, and chat commands.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print picochat version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run the default local Picochat demo.")
    demo_parser.add_argument(
        "--out-dir",
        default="runs/pico-demo",
        help="Output run directory for the demo artifacts.",
    )
    demo_parser.add_argument("--device", default="cpu")

    data_parser = subparsers.add_parser("data", help="Dataset commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    data_inspect = data_subparsers.add_parser("inspect", help="Inspect a corpus source.")
    data_inspect.add_argument(
        "--input",
        required=True,
        help="Path to a supported corpus file or folder.",
    )

    data_preview = data_subparsers.add_parser("preview", help="Preview corpus sources before building.")
    data_preview.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        default=None,
        help="Path to a dataset pack JSON with corpus, chat, and eval inputs.",
    )
    data_preview.add_argument(
        "--input",
        default=None,
        help="Path to a supported corpus file or folder.",
    )
    data_preview.add_argument(
        "--recipe",
        default=None,
        help="Path to a JSON corpus recipe with explicit sources and exclude rules.",
    )
    data_preview.add_argument(
        "--preview-chars",
        type=int,
        default=800,
        help="Maximum number of combined corpus characters to print.",
    )
    data_preview.add_argument(
        "--chat-input",
        default=DEFAULT_CHAT_INPUT,
        help="Chat SFT JSONL path to place in the suggested run command.",
    )
    data_preview.add_argument(
        "--eval-input",
        default=DEFAULT_EVAL_INPUT,
        help="Eval JSONL path to place in the suggested run command.",
    )
    data_preview.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum 0-100 source quality score required for a file to enter the preview corpus.",
    )

    data_build = data_subparsers.add_parser("build", help="Build a normalized text corpus.")
    data_build.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        default=None,
        help="Path to a dataset pack JSON with corpus, chat, and eval inputs.",
    )
    data_build.add_argument(
        "--input",
        default=None,
        help="Path to a supported corpus file or folder.",
    )
    data_build.add_argument(
        "--recipe",
        default=None,
        help="Path to a JSON corpus recipe with explicit sources and exclude rules.",
    )
    data_build.add_argument(
        "--out",
        required=True,
        help="Where to write the combined corpus text file.",
    )
    data_build.add_argument(
        "--manifest",
        default=None,
        help="Where to write corpus_manifest.json. Defaults beside --out.",
    )
    data_build.add_argument(
        "--report",
        default=None,
        help="Where to write corpus_report.md. Defaults beside --out.",
    )
    data_build.add_argument(
        "--chat-input",
        default=DEFAULT_CHAT_INPUT,
        help="Chat SFT JSONL path to place in the suggested run command.",
    )
    data_build.add_argument(
        "--eval-input",
        default=DEFAULT_EVAL_INPUT,
        help="Eval JSONL path to place in the suggested run command.",
    )
    data_build.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum 0-100 source quality score required for a file to enter the built corpus.",
    )

    data_init_pack = data_subparsers.add_parser("init-pack", help="Create a starter dataset pack folder.")
    data_init_pack.add_argument("--name", default="picochat-pack", help="Human-readable dataset pack name.")
    data_init_pack.add_argument("--corpus", required=True, help="Corpus file or folder to reference from the pack.")
    data_init_pack.add_argument("--out", required=True, help="Output folder for dataset_pack.json and starter files.")
    data_init_pack.add_argument(
        "--description",
        default="Starter Picochat dataset pack.",
        help="Short description written into dataset_pack.json.",
    )
    data_init_pack.add_argument("--force", action="store_true", help="Overwrite existing starter files.")

    data_hf_import = data_subparsers.add_parser(
        "hf-import",
        help="Import a Hugging Face dataset split into a local text corpus.",
    )
    data_hf_import.add_argument("--dataset", required=True, help="Hugging Face dataset name, for example wikimedia/wikipedia.")
    data_hf_import.add_argument("--config", dest="config_name", default=None, help="Optional dataset configuration name.")
    data_hf_import.add_argument("--split", default="train", help="Dataset split to read.")
    data_hf_import.add_argument("--text-column", default="text", help="Column containing training text.")
    data_hf_import.add_argument("--out", required=True, help="Output local corpus text path.")
    data_hf_import.add_argument("--report", default=None, help="Optional JSON report path. Markdown is written beside it.")
    data_hf_import.add_argument(
        "--documents-dir",
        default=None,
        help="Optional folder for one text file per accepted row. Defaults to a documents folder beside --out.",
    )
    data_hf_import.add_argument("--max-rows", type=int, default=1000, help="Maximum rows to inspect from the split.")
    data_hf_import.add_argument("--min-chars", type=int, default=20, help="Minimum text length required for a row to be written.")
    data_hf_import.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/load the split normally instead of streaming rows.",
    )

    tok_parser = subparsers.add_parser("tok", help="Tokenizer commands.")
    tok_subparsers = tok_parser.add_subparsers(dest="tok_command")

    tok_train = tok_subparsers.add_parser("train", help="Train a tokenizer from text.")
    tok_train.add_argument(
        "--input",
        required=True,
        help="Path to a UTF-8 text file used to train the tokenizer.",
    )
    tok_train.add_argument(
        "--out",
        required=True,
        help="Where to write tokenizer JSON.",
    )
    tok_train.add_argument(
        "--type",
        choices=TOKENIZER_TYPES,
        default="char",
        help="Tokenizer type to train. Use char for the baseline, byte for UTF-8 bytes, or bpe for learned merges.",
    )
    tok_train.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Optional maximum vocabulary size including special tokens.",
    )
    tok_train.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum character frequency for char, or merge frequency for bpe.",
    )

    batch_parser = subparsers.add_parser("batch", help="Token batching commands.")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command")
    batch_inspect = batch_subparsers.add_parser(
        "inspect",
        help="Inspect next-token training windows.",
    )
    batch_inspect.add_argument("--corpus", required=True, help="Path to corpus text.")
    batch_inspect.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    batch_inspect.add_argument(
        "--context-size",
        type=int,
        default=64,
        help="Number of input tokens per training example.",
    )
    batch_inspect.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Number of example windows to print.",
    )

    train_parser = subparsers.add_parser("train", help="Training commands.")
    train_subparsers = train_parser.add_subparsers(dest="train_command")
    train_base_parser = train_subparsers.add_parser("base", help="Train a base LM.")
    train_base_parser.add_argument("--corpus", required=True, help="Path to corpus text.")
    train_base_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    train_base_parser.add_argument("--out-dir", required=True, help="Output run directory.")
    train_base_parser.add_argument("--context-size", type=int, default=64)
    train_base_parser.add_argument("--batch-size", type=int, default=16)
    train_base_parser.add_argument("--max-steps", type=int, default=200)
    train_base_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_base_parser.add_argument("--n-embd", type=int, default=128)
    train_base_parser.add_argument("--n-head", type=int, default=4)
    train_base_parser.add_argument("--n-layer", type=int, default=2)
    train_base_parser.add_argument("--dropout", type=float, default=0.0)
    train_base_parser.add_argument("--seed", type=int, default=42)
    train_base_parser.add_argument("--device", default="cpu")
    train_base_parser.add_argument("--log-every", type=int, default=20)
    train_base_parser.add_argument("--val-fraction", type=float, default=0.1)
    train_base_parser.add_argument("--eval-batches", type=int, default=10)
    train_base_parser.add_argument("--sample-tokens", type=int, default=120)
    train_base_parser.add_argument("--early-stop-patience", type=int, default=0)
    train_base_parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_base_parser.add_argument("--max-minutes", type=float, default=None)
    train_base_parser.add_argument(
        "--canary-count",
        type=int,
        default=0,
        help="Number of fake train-only canary phrases to inject when document split is available.",
    )
    train_base_parser.add_argument(
        "--split-mode",
        choices=("window", "document"),
        default="window",
        help="Use random token windows or hold out complete corpus documents when a manifest is available.",
    )
    train_base_parser.add_argument(
        "--corpus-manifest",
        default=None,
        help="Path to corpus_manifest.json for document-level holdout.",
    )

    train_sft_parser = train_subparsers.add_parser("sft", help="Fine-tune on chat JSONL.")
    train_sft_parser.add_argument("--input", required=True, help="Path to chat JSONL.")
    train_sft_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    train_sft_parser.add_argument("--checkpoint", required=True, help="Base checkpoint directory.")
    train_sft_parser.add_argument("--out-dir", required=True, help="Output run directory.")
    train_sft_parser.add_argument("--batch-size", type=int, default=8)
    train_sft_parser.add_argument("--max-steps", type=int, default=100)
    train_sft_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_sft_parser.add_argument("--seed", type=int, default=42)
    train_sft_parser.add_argument("--device", default="cpu")
    train_sft_parser.add_argument("--log-every", type=int, default=10)
    train_sft_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_sft_parser.add_argument("--eval-batches", type=int, default=10)
    train_sft_parser.add_argument("--sample-prompt", default="What is Picochat?")
    train_sft_parser.add_argument("--sample-tokens", type=int, default=120)
    train_sft_parser.add_argument("--early-stop-patience", type=int, default=0)
    train_sft_parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_sft_parser.add_argument("--max-minutes", type=float, default=None)

    generate_parser = subparsers.add_parser("generate", help="Generate from a checkpoint.")
    generate_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    generate_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    generate_parser.add_argument("--prompt", default="", help="Text prompt.")
    generate_parser.add_argument("--max-new-tokens", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=20)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--device", default="cpu")

    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat.")
    chat_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    chat_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    chat_parser.add_argument("--max-new-tokens", type=int, default=120)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--top-k", type=int, default=20)
    chat_parser.add_argument("--seed", type=int, default=42)
    chat_parser.add_argument("--device", default="cpu")

    eval_parser = subparsers.add_parser("eval", help="Evaluation commands.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    eval_chat_parser = eval_subparsers.add_parser("chat", help="Run transparent chat eval.")
    eval_chat_parser.add_argument("--input", required=True, help="Path to eval JSONL.")
    eval_chat_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    eval_chat_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    eval_chat_parser.add_argument("--out-dir", required=True, help="Output eval directory.")
    eval_chat_parser.add_argument("--max-new-tokens", type=int, default=80)
    eval_chat_parser.add_argument("--temperature", type=float, default=0.0)
    eval_chat_parser.add_argument("--top-k", type=int, default=0)
    eval_chat_parser.add_argument("--seed", type=int, default=42)
    eval_chat_parser.add_argument("--device", default="cpu")
    eval_chat_parser.add_argument("--case-sensitive", action="store_true")

    run_parser = subparsers.add_parser("run", help="End-to-end experiment runners.")
    run_subparsers = run_parser.add_subparsers(dest="run_command")
    run_tiny_parser = run_subparsers.add_parser("tiny", help="Run the full tiny pipeline.")
    run_tiny_parser.add_argument("--out-dir", required=True, help="Output run directory.")
    run_tiny_parser.add_argument("--dataset-pack", "--pack", dest="dataset_pack", default=None)
    run_tiny_parser.add_argument("--corpus-input", default="examples/tiny_corpus.txt")
    run_tiny_parser.add_argument("--corpus-recipe", default=None)
    run_tiny_parser.add_argument("--chat-input", default="examples/tiny_chat.jsonl")
    run_tiny_parser.add_argument("--eval-input", default="examples/tiny_eval.jsonl")
    run_tiny_parser.add_argument("--context-size", type=int, default=128)
    run_tiny_parser.add_argument("--n-embd", type=int, default=64)
    run_tiny_parser.add_argument("--n-head", type=int, default=4)
    run_tiny_parser.add_argument("--n-layer", type=int, default=2)
    run_tiny_parser.add_argument("--base-steps", type=int, default=300)
    run_tiny_parser.add_argument("--sft-steps", type=int, default=600)
    run_tiny_parser.add_argument("--base-batch-size", type=int, default=8)
    run_tiny_parser.add_argument("--sft-batch-size", type=int, default=7)
    run_tiny_parser.add_argument("--base-learning-rate", type=float, default=3e-4)
    run_tiny_parser.add_argument("--sft-learning-rate", type=float, default=1e-3)
    run_tiny_parser.add_argument("--base-early-stop-patience", type=int, default=6)
    run_tiny_parser.add_argument("--sft-early-stop-patience", type=int, default=6)
    run_tiny_parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    run_tiny_parser.add_argument("--base-max-minutes", type=float, default=None)
    run_tiny_parser.add_argument("--sft-max-minutes", type=float, default=None)
    run_tiny_parser.add_argument("--canary-count", type=int, default=1)
    run_tiny_parser.add_argument("--seed", type=int, default=42)
    run_tiny_parser.add_argument("--device", default="cpu")
    run_tiny_parser.add_argument("--eval-max-new-tokens", type=int, default=120)
    run_tiny_parser.add_argument(
        "--tokenizer-type",
        choices=TOKENIZER_TYPES,
        default="char",
        help="Tokenizer used for this run. Compare char, byte, and bpe on the same dataset pack.",
    )
    run_tiny_parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum 0-100 source quality score required for a file to enter the training corpus.",
    )
    run_tiny_parser.add_argument(
        "--split-mode",
        choices=("window", "document"),
        default="document",
        help="Base training validation split. 'document' holds out complete corpus documents when possible.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare completed run summaries.")
    compare_parser.add_argument("runs", nargs="+", help="Run directories containing summary.json.")
    compare_parser.add_argument("--out", default=None, help="Optional Markdown report output path.")

    web_parser = subparsers.add_parser("web", help="Start the local run dashboard.")
    web_parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8765)
    return parser


def print_stats(stats) -> None:
    for key, value in stats.to_dict().items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def print_readiness(readiness) -> None:
    print(f"readiness: {readiness.status}")
    print(f"readiness_summary: {readiness.summary}")
    print("\nreadiness checks:")
    for check in readiness.checks:
        print(
            f"- {check.status} {check.name}: {check.metric} "
            f"(target {check.threshold}) - {check.message}"
        )


def print_budget(budget) -> None:
    print("\ntraining budget:")
    print(f"- preset: {budget.preset}")
    print(f"- estimated_tokens: {budget.estimated_tokens}")
    print(f"- suggested_context_size: {budget.suggested_context_size}")
    print(f"- estimated_windows: {budget.estimated_windows}")
    print(f"- suggested_batch_size: {budget.suggested_batch_size}")
    print(f"- suggested_base_steps: {budget.suggested_base_steps}")
    print(f"- estimated_tokens_per_step: {budget.estimated_tokens_per_step}")
    print(f"- estimated_passes: {budget.estimated_passes:.2f}")
    print(f"- note: {budget.note}")


def print_training_command(training_command) -> None:
    print("\nsuggested run:")
    print(f"- out_dir: {training_command.out_dir}")
    print(f"- chat_input: {training_command.chat_input}")
    print(f"- eval_input: {training_command.eval_input}")
    print(f"- note: {training_command.note}")
    if training_command.command:
        print(training_command.command)
    else:
        print("(no command until corpus readiness is unblocked)")


def print_tuning_data(chat_data, eval_data) -> None:
    print("\nchat/eval data:")
    print(
        f"- chat_sft: {chat_data.status} | {chat_data.num_examples}/{chat_data.num_rows} usable rows "
        f"| avg_user_chars {chat_data.average_user_chars:.1f} "
        f"| avg_assistant_chars {chat_data.average_assistant_chars:.1f}"
    )
    print(f"  {chat_data.path}: {chat_data.summary}")
    if chat_data.categories:
        print(f"  categories: {_format_counts(chat_data.categories)}")
    for issue in chat_data.issues[:3]:
        print(f"  issue line {issue.line}: {issue.message}")
    print(
        f"- eval: {eval_data.status} | {eval_data.num_items}/{eval_data.num_rows} usable rows "
        f"| answerable {eval_data.answerable_items} "
        f"| unanswerable {eval_data.unanswerable_items}"
    )
    print(
        f"  rules: include {eval_data.must_include_rules}, "
        f"include_any {eval_data.must_include_any_groups}, "
        f"forbidden {eval_data.must_not_include_rules}"
    )
    print(f"  {eval_data.path}: {eval_data.summary}")
    if eval_data.categories:
        print(f"  categories: {_format_counts(eval_data.categories)}")
    for issue in eval_data.issues[:3]:
        print(f"  issue line {issue.line}: {issue.message}")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def inspect_data(args: argparse.Namespace) -> int:
    stats = inspect_path(args.input)
    print_stats(stats)
    return 0


def preview_data(args: argparse.Namespace) -> int:
    if not args.input and not args.recipe and not args.dataset_pack:
        raise SystemExit("data preview requires --input, --recipe, or --dataset-pack")
    report = preview_corpus_sources(
        args.input,
        args.recipe,
        preview_chars=args.preview_chars,
        chat_input=args.chat_input,
        eval_input=args.eval_input,
        dataset_pack=args.dataset_pack,
        min_quality_score=args.min_score,
    )
    included = [record for record in report.files if record.included]
    skipped = [record for record in report.files if not record.included]

    print(f"input: {report.input_path}")
    print(f"recipe: {report.recipe_path or 'none'}")
    print(f"dataset_pack: {report.dataset_pack or 'none'}")
    print(f"min_quality_score: {report.min_quality_score}")
    print(f"files_included: {len(included)}")
    print(f"files_skipped: {len(skipped)}")
    print_stats(report.stats)
    print_readiness(report.readiness)
    print_budget(report.budget)
    print_training_command(report.training_command)
    print_tuning_data(report.chat_data, report.eval_data)

    print("\nsource plan:")
    visible_records = report.files[:SOURCE_PLAN_PREVIEW_LIMIT]
    for record in visible_records:
        status = "include" if record.included else "skip"
        label = f" label={record.label}" if record.label else ""
        print(
            f"- {status} {record.path}{label} ext={record.extension} "
            f"score={record.quality_score} chars={record.num_characters} "
            f"lines={record.num_lines} reason={record.reason}"
            f"{' flags=' + ','.join(record.quality_flags) if record.quality_flags else ''}"
        )
    omitted = len(report.files) - len(visible_records)
    if omitted:
        print(f"- ... {omitted} more source file(s) omitted from CLI preview")

    if report.warnings:
        print("\nwarnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    print("\npreview:")
    print(report.preview or "(empty)")
    return 0


def build_data(args: argparse.Namespace) -> int:
    if not args.input and not args.recipe and not args.dataset_pack:
        raise SystemExit("data build requires --input, --recipe, or --dataset-pack")
    report = build_corpus_artifacts(
        args.input,
        args.out,
        args.manifest,
        args.report,
        recipe_path=args.recipe,
        chat_input=args.chat_input,
        eval_input=args.eval_input,
        dataset_pack=args.dataset_pack,
        min_quality_score=args.min_score,
    )
    print(f"built corpus: {args.out}")
    print(f"manifest: {report.manifest_path}")
    print(f"report: {report.report_path}")
    print(f"min_quality_score: {report.min_quality_score}")
    print_stats(report.stats)
    print_readiness(report.readiness)
    print_budget(report.budget)
    print_training_command(report.training_command)
    print_tuning_data(report.chat_data, report.eval_data)
    return 0


def init_pack_data(args: argparse.Namespace) -> int:
    report = init_dataset_pack(
        out_dir=args.out,
        corpus_path=args.corpus,
        name=args.name,
        description=args.description,
        force=args.force,
    )
    print(f"initialized dataset pack: {report.dataset_pack}")
    print(f"corpus recipe: {report.corpus_recipe}")
    print(f"chat sft jsonl: {report.chat_input}")
    print(f"eval jsonl: {report.eval_input}")
    if report.overwritten:
        print("overwritten:")
        for path in report.overwritten:
            print(f"- {path}")
    print("\nnext:")
    print(f"PYTHONPATH=src python -m picochat.cli data preview --dataset-pack {report.dataset_pack}")
    return 0


def hf_import_data(args: argparse.Namespace) -> int:
    report = import_hf_dataset(HFImportConfig(
        dataset=args.dataset,
        config_name=args.config_name,
        split=args.split,
        text_column=args.text_column,
        out_path=args.out,
        report_path=args.report,
        documents_dir=args.documents_dir,
        max_rows=args.max_rows,
        min_chars=args.min_chars,
        streaming=not args.no_streaming,
    ))
    print(f"imported dataset: {report.dataset}")
    print(f"split: {report.split}")
    print(f"text_column: {report.text_column}")
    print(f"streaming: {report.streaming}")
    print(f"rows_seen: {report.rows_seen}")
    print(f"rows_written: {report.rows_written}")
    print(f"rows_skipped: {report.rows_skipped}")
    print(f"characters_written: {report.characters_written}")
    print(f"corpus: {report.out_path}")
    if report.documents_dir:
        print(f"documents_dir: {report.documents_dir}")
    print(f"report: {report.report_path}")
    print("\nnext:")
    print(f"PYTHONPATH=src python -m picochat.cli data preview --input {report.documents_dir or report.out_path}")
    return 0


def train_tokenizer(args: argparse.Namespace) -> int:
    from pathlib import Path

    input_path = Path(args.input)
    output_path = Path(args.out)
    text = input_path.read_text(encoding="utf-8")

    tokenizer = build_tokenizer(
        args.type,
        [text],
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )
    tokenizer.save(output_path)

    stats = tokenizer.stats()
    print(f"trained tokenizer: {output_path}")
    print(f"type: {stats.tokenizer_type}")
    print(f"vocab_size: {stats.vocab_size}")
    print(f"text_tokens: {stats.num_text_tokens}")
    print(f"special_tokens: {stats.num_special_tokens}")
    return 0


def inspect_batches(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    dataset = load_token_dataset(args.corpus, args.tokenizer, args.context_size)
    stats = dataset.stats()

    print(f"num_tokens: {stats.num_tokens}")
    print(f"context_size: {stats.context_size}")
    print(f"num_sequences: {stats.num_sequences}")

    for index in range(min(args.examples, len(dataset))):
        x, y = dataset[index]
        print(f"\nexample {index}")
        print(f"x ids: {x.tolist()}")
        print(f"y ids: {y.tolist()}")
        print(f"x text: {tokenizer.decode(x.tolist())!r}")
        print(f"y text: {tokenizer.decode(y.tolist())!r}")
    return 0


def run_train_base(args: argparse.Namespace) -> int:
    config = TrainConfig(
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        context_size=args.context_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        val_fraction=args.val_fraction,
        eval_batches=args.eval_batches,
        sample_tokens=args.sample_tokens,
        split_mode=args.split_mode,
        corpus_manifest_path=args.corpus_manifest,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        max_minutes=args.max_minutes,
        canary_count=args.canary_count,
    )
    report = train_base(config)
    print(f"saved checkpoint: {report['checkpoint']}")
    print(f"sample: {report['sample']!r}")
    return 0


def run_train_sft(args: argparse.Namespace) -> int:
    config = SFTConfig(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        val_fraction=args.val_fraction,
        eval_batches=args.eval_batches,
        sample_prompt=args.sample_prompt,
        sample_tokens=args.sample_tokens,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        max_minutes=args.max_minutes,
    )
    report = train_sft(config)
    print(f"saved sft checkpoint: {report['checkpoint']}")
    print(f"sample: {report['sample']!r}")
    return 0


def run_generate(args: argparse.Namespace) -> int:
    top_k = None if args.top_k <= 0 else args.top_k
    text = generate_text(GenerateConfig(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
        device=args.device,
    ))
    print(text)
    return 0


def run_chat(args: argparse.Namespace) -> int:
    top_k = None if args.top_k <= 0 else args.top_k
    return chat_loop(ChatConfig(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
        device=args.device,
    ))


def run_eval_chat(args: argparse.Namespace) -> int:
    top_k = None if args.top_k <= 0 else args.top_k
    report = run_chat_eval(ChatEvalConfig(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
        device=args.device,
        case_sensitive=args.case_sensitive,
    ))
    summary = report["summary"]
    print(
        f"chat eval: {summary['num_passed']}/{summary['num_examples']} passed "
        f"({summary['pass_rate'] * 100:.2f}%)"
    )
    print(f"saved eval report: {args.out_dir}")
    return 0


def run_tiny_command(args: argparse.Namespace) -> int:
    summary = run_tiny(TinyRunConfig(
        out_dir=args.out_dir,
        dataset_pack=args.dataset_pack,
        corpus_input=args.corpus_input,
        corpus_recipe=args.corpus_recipe,
        chat_input=args.chat_input,
        eval_input=args.eval_input,
        context_size=args.context_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        base_steps=args.base_steps,
        sft_steps=args.sft_steps,
        base_batch_size=args.base_batch_size,
        sft_batch_size=args.sft_batch_size,
        base_learning_rate=args.base_learning_rate,
        sft_learning_rate=args.sft_learning_rate,
        seed=args.seed,
        device=args.device,
        eval_max_new_tokens=args.eval_max_new_tokens,
        min_quality_score=args.min_score,
        split_mode=args.split_mode,
        tokenizer_type=args.tokenizer_type,
        base_early_stop_patience=args.base_early_stop_patience,
        sft_early_stop_patience=args.sft_early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        base_max_minutes=args.base_max_minutes,
        sft_max_minutes=args.sft_max_minutes,
        canary_count=args.canary_count,
    ))
    print(
        f"tiny run: {summary['eval']['num_passed']}/{summary['eval']['num_examples']} "
        f"passed ({summary['eval']['pass_rate'] * 100:.2f}%)"
    )
    return 0


def run_demo(args: argparse.Namespace) -> int:
    summary = run_tiny(TinyRunConfig(
        out_dir=args.out_dir,
        device=args.device,
    ))
    print(
        f"demo run: {summary['eval']['num_passed']}/{summary['eval']['num_examples']} "
        f"passed ({summary['eval']['pass_rate'] * 100:.2f}%)"
    )
    print(f"open workbench: PYTHONPATH=src python -m picochat.cli web --runs-dir runs --port 8765")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    comparison = compare_runs(args.runs)
    print(comparison_table(comparison))
    if args.out:
        write_comparison_report(comparison, args.out)
        print(f"saved comparison report: {args.out}")
    return 0


def run_web(args: argparse.Namespace) -> int:
    serve_web(WebConfig(
        runs_dir=args.runs_dir,
        host=args.host,
        port=args.port,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from picochat import __version__

        print(f"picochat {__version__}")
        return 0

    if args.command == "data" and args.data_command == "inspect":
        return inspect_data(args)

    if args.command == "data" and args.data_command == "preview":
        return preview_data(args)

    if args.command == "demo":
        return run_demo(args)

    if args.command == "data" and args.data_command == "build":
        return build_data(args)

    if args.command == "data" and args.data_command == "init-pack":
        return init_pack_data(args)

    if args.command == "data" and args.data_command == "hf-import":
        return hf_import_data(args)

    if args.command == "tok" and args.tok_command == "train":
        return train_tokenizer(args)

    if args.command == "batch" and args.batch_command == "inspect":
        return inspect_batches(args)

    if args.command == "train" and args.train_command == "base":
        return run_train_base(args)

    if args.command == "train" and args.train_command == "sft":
        return run_train_sft(args)

    if args.command == "generate":
        return run_generate(args)

    if args.command == "chat":
        return run_chat(args)

    if args.command == "eval" and args.eval_command == "chat":
        return run_eval_chat(args)

    if args.command == "run" and args.run_command == "tiny":
        return run_tiny_command(args)

    if args.command == "compare":
        return run_compare(args)

    if args.command == "web":
        return run_web(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
