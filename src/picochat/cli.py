"""Command-line entrypoint for picochat."""

from __future__ import annotations

import argparse

from picochat.data import build_corpus_artifacts, inspect_path, preview_corpus_sources
from picochat.batching import load_token_dataset
from picochat.tokenizer import CharTokenizer
from picochat.train import TrainConfig, train_base
from picochat.sft import SFTConfig, train_sft
from picochat.generate import GenerateConfig, generate_text
from picochat.chat import ChatConfig, chat_loop
from picochat.eval import ChatEvalConfig, run_chat_eval
from picochat.run import TinyRunConfig, run_tiny
from picochat.compare import compare_runs, comparison_table, write_comparison_report
from picochat.web import WebConfig, serve_web


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

    data_build = data_subparsers.add_parser("build", help="Build a normalized text corpus.")
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
        "--vocab-size",
        type=int,
        default=None,
        help="Optional maximum vocabulary size including special tokens.",
    )
    tok_train.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum character frequency required to enter the vocabulary.",
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
    run_tiny_parser.add_argument("--seed", type=int, default=42)
    run_tiny_parser.add_argument("--device", default="cpu")
    run_tiny_parser.add_argument("--eval-max-new-tokens", type=int, default=120)

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


def inspect_data(args: argparse.Namespace) -> int:
    stats = inspect_path(args.input)
    print_stats(stats)
    return 0


def preview_data(args: argparse.Namespace) -> int:
    if not args.input and not args.recipe:
        raise SystemExit("data preview requires --input or --recipe")
    report = preview_corpus_sources(args.input, args.recipe, preview_chars=args.preview_chars)
    included = [record for record in report.files if record.included]
    skipped = [record for record in report.files if not record.included]

    print(f"input: {report.input_path}")
    print(f"recipe: {report.recipe_path or 'none'}")
    print(f"files_included: {len(included)}")
    print(f"files_skipped: {len(skipped)}")
    print_stats(report.stats)

    print("\nsource plan:")
    for record in report.files:
        status = "include" if record.included else "skip"
        label = f" label={record.label}" if record.label else ""
        print(
            f"- {status} {record.path}{label} ext={record.extension} "
            f"chars={record.num_characters} lines={record.num_lines} reason={record.reason}"
        )

    if report.warnings:
        print("\nwarnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    print("\npreview:")
    print(report.preview or "(empty)")
    return 0


def build_data(args: argparse.Namespace) -> int:
    if not args.input and not args.recipe:
        raise SystemExit("data build requires --input or --recipe")
    report = build_corpus_artifacts(
        args.input,
        args.out,
        args.manifest,
        args.report,
        recipe_path=args.recipe,
    )
    print(f"built corpus: {args.out}")
    print(f"manifest: {report.manifest_path}")
    print(f"report: {report.report_path}")
    print_stats(report.stats)
    return 0


def train_tokenizer(args: argparse.Namespace) -> int:
    from pathlib import Path

    input_path = Path(args.input)
    output_path = Path(args.out)
    text = input_path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.train(
        [text],
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )
    tokenizer.save(output_path)

    stats = tokenizer.stats()
    print(f"trained tokenizer: {output_path}")
    print(f"vocab_size: {stats.vocab_size}")
    print(f"text_tokens: {stats.num_text_tokens}")
    print(f"special_tokens: {stats.num_special_tokens}")
    return 0


def inspect_batches(args: argparse.Namespace) -> int:
    tokenizer = CharTokenizer.load(args.tokenizer)
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
