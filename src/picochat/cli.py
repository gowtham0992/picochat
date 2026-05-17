"""Command-line entrypoint for picochat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picochat.data import (
    DEFAULT_CHAT_INPUT,
    DEFAULT_EVAL_INPUT,
    build_corpus_artifacts,
    inspect_path,
    preview_corpus_sources,
)
from picochat.batching import load_token_dataset
from picochat.tokenizer import (
    BPE_PRETOKENIZERS,
    DEFAULT_BPE_PRETOKENIZER,
    TOKENIZER_TYPES,
    load_tokenizer,
    train_tokenizer as build_tokenizer,
)
from picochat.train import TrainConfig, train_base
from picochat.sft import SFTConfig, SFT_PACKING_MODES, SFT_SAMPLING_MODES, train_sft
from picochat.generate import GenerateConfig, generate_text
from picochat.chat import ChatConfig, chat_loop
from picochat.eval import ChatEvalConfig, run_chat_eval, write_sft_fit_eval
from picochat.external_benchmark import (
    EXTERNAL_BENCHMARK_FORMATS,
    ExternalBenchmarkConvertConfig,
    convert_external_benchmark,
)
from picochat.lora import DEFAULT_LORA_TARGETS, LORA_TARGETS, PEFT_MODES, parse_lora_targets
from picochat.eval_starter import generate_eval_starter
from picochat.sft_starter import generate_sft_starter
from picochat.benchmark_pack import (
    BENCHMARK_PROFILES,
    BENCHMARK_SKILL_ANSWER_STYLES,
    BENCHMARK_SOURCES,
    DEFAULT_BENCHMARK_EVAL_ROWS,
    DEFAULT_BENCHMARK_SFT_ROWS,
    generate_benchmark_tuning_pack,
)
from picochat.run import (
    LONG_RUN_GATE_PROFILES,
    TinyRunConfig,
    run_tiny,
    run_tiny_multiseed,
)
from picochat.run_preflight import assess_run_preflight, preflight_markdown
from picochat.compare import compare_runs, comparison_table, write_comparison_report
from picochat.dataset_pack import init_dataset_pack, load_dataset_pack
from picochat.device import DEVICE_CHOICES
from picochat.hf_export import HFExportConfig, export_hf_checkpoint
from picochat.hf_import import HFImportConfig, import_hf_dataset
from picochat.honesty import inspect_data_honesty, write_data_honesty_report
from picochat.leaderboard import build_benchmark_leaderboard, leaderboard_table, write_leaderboard_report
from picochat.model import SDPA_BACKENDS
from picochat.artifacts import (
    RunBundleConfig,
    bundle_inspection_markdown,
    create_run_bundle,
    inspect_run_bundle,
)
from picochat.optim import (
    LR_DECAYS,
    MUON_MOMENTUM_SCHEDULES,
    OPTIMIZER_TYPES,
    WEIGHT_DECAY_DECAYS,
)
from picochat.precision import COMPILE_MODES, MATMUL_PRECISION_MODES, PRECISION_MODES
from picochat.sanity import PreH100SanityConfig, run_preh100_sanity
from picochat.scales import RUN_SCALE_NAMES, RUN_SCALES
from picochat.scale_planner import parse_count, plan_scale, render_scale_plan_markdown
from picochat.sft_sweep import SFTSweepConfig, run_sft_sweep
from picochat.skills_corpus import generate_skills_corpus
from picochat.tuning_slice import (
    DEFAULT_TUNING_STAGE_NAMES,
    parse_category_patterns,
    parse_stage_names,
    slice_tuning_pack,
    stage_tuning_pack,
)
from picochat.task_mixture import (
    DEFAULT_TASK_MIXTURE_EVAL_ROWS,
    DEFAULT_TASK_MIXTURE_SFT_ROWS,
    TASK_MIXTURE_PROFILES,
    generate_task_mixture_pack,
)
from picochat.web import WebConfig, serve_web


SOURCE_PLAN_PREVIEW_LIMIT = 25
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_MAX_SHARD = 6542
CLIMBMIX_LARGE_IMPORT_ROWS = 100_000
CLIMBMIX_LARGE_IMPORT_DOCUMENT_SHARD_ROWS = 1000


def _add_eval_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default="float32",
        help="Eval precision. Use bf16 on H100/CUDA to speed SFT-fit and chat eval.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISION_MODES,
        default="default",
        help="torch.set_float32_matmul_precision setting for eval forward passes.",
    )


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
    demo_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")

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

    data_honesty = data_subparsers.add_parser(
        "honesty",
        help="Check chat/eval leakage before trusting a score.",
    )
    data_honesty.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        default=None,
        help="Dataset pack whose corpus, chat, and eval files should be checked.",
    )
    data_honesty.add_argument("--corpus", default=None, help="Optional corpus file or folder.")
    data_honesty.add_argument("--chat-input", default=DEFAULT_CHAT_INPUT, help="Chat SFT JSONL path.")
    data_honesty.add_argument("--eval-input", default=DEFAULT_EVAL_INPUT, help="Eval JSONL path.")
    data_honesty.add_argument("--out-dir", default=None, help="Optional output folder for honesty_report.json and report.md.")
    data_honesty.add_argument(
        "--near-threshold",
        type=float,
        default=0.86,
        help="Similarity threshold for near-duplicate prompt warnings.",
    )
    data_honesty.add_argument(
        "--ngram-size",
        type=int,
        default=8,
        help="Token n-gram size for contamination matrix overlap checks.",
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

    data_eval_starter = data_subparsers.add_parser("eval-starter", help="Generate a starter eval JSONL from a corpus.")
    data_eval_starter.add_argument("--input", default=None, help="Corpus file or folder to sample.")
    data_eval_starter.add_argument("--dataset-pack", "--pack", dest="dataset_pack", default=None, help="Dataset pack whose corpus should be sampled.")
    data_eval_starter.add_argument("--out", required=True, help="Output eval JSONL path.")
    data_eval_starter.add_argument("--max-items", type=int, default=24, help="Maximum eval rows to write.")
    data_eval_starter.add_argument("--seed", type=int, default=42)
    data_eval_starter.add_argument("--force", action="store_true", help="Overwrite an existing output file.")

    data_sft_starter = data_subparsers.add_parser("sft-starter", help="Generate a starter chat SFT JSONL from a corpus.")
    data_sft_starter.add_argument("--input", default=None, help="Corpus file or folder to sample.")
    data_sft_starter.add_argument("--dataset-pack", "--pack", dest="dataset_pack", default=None, help="Dataset pack whose corpus should be sampled.")
    data_sft_starter.add_argument("--out", required=True, help="Output chat SFT JSONL path.")
    data_sft_starter.add_argument("--max-items", type=int, default=32, help="Maximum chat SFT rows to write.")
    data_sft_starter.add_argument("--seed", type=int, default=42)
    data_sft_starter.add_argument("--force", action="store_true", help="Overwrite an existing output file.")

    data_benchmark_pack = data_subparsers.add_parser(
        "benchmark-pack",
        help="Generate a nanochat-style curated SFT/eval curriculum for a dataset pack.",
    )
    data_benchmark_pack.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        required=True,
        help="Dataset pack whose chat/eval paths should be replaced or supplemented.",
    )
    data_benchmark_pack.add_argument("--chat-out", default=None, help="Optional output path for chat SFT JSONL.")
    data_benchmark_pack.add_argument("--eval-out", default=None, help="Optional output path for eval JSONL.")
    data_benchmark_pack.add_argument(
        "--sft-rows",
        type=int,
        default=DEFAULT_BENCHMARK_SFT_ROWS,
        help="Number of curated SFT rows to write.",
    )
    data_benchmark_pack.add_argument(
        "--eval-rows",
        type=int,
        default=DEFAULT_BENCHMARK_EVAL_ROWS,
        help="Number of held-out eval rows to write.",
    )
    data_benchmark_pack.add_argument(
        "--source",
        choices=BENCHMARK_SOURCES,
        default="offline",
        help="Curriculum source: offline deterministic rows, auto HF with offline fallback, or HF-required.",
    )
    data_benchmark_pack.add_argument(
        "--profile",
        choices=BENCHMARK_PROFILES,
        default="full",
        help=(
            "Curriculum profile. release_behavior focuses identity/refusal for first-release gates; "
            "release_skills includes identity/refusal/choice/math/spelling for skill-release claims; "
            "behavior excludes broad long-form chat rows for first-stage SFT fit; weak_skills "
            "over-samples math and spelling after behavior fit is healthy."
        ),
    )
    data_benchmark_pack.add_argument(
        "--skill-answer-style",
        choices=BENCHMARK_SKILL_ANSWER_STYLES,
        default="direct",
        help=(
            "How local math/spelling rows should answer. direct keeps terse answers; "
            "scratchpad teaches a short work trace ending in `Final answer:`."
        ),
    )
    data_benchmark_pack.add_argument("--seed", type=int, default=42)
    data_benchmark_pack.add_argument("--force", action="store_true", help="Overwrite existing benchmark pack files.")
    data_benchmark_pack.add_argument(
        "--no-promote",
        action="store_true",
        help="Write files but do not point dataset_pack.json at them.",
    )

    data_task_pack = data_subparsers.add_parser(
        "task-pack",
        help="Generate a staged task-mixture SFT/eval pack for release or capability tuning.",
    )
    data_task_pack.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        required=True,
        help="Dataset pack whose chat/eval paths should be replaced or supplemented.",
    )
    data_task_pack.add_argument("--out-dir", default=None, help="Optional output directory for task-mixture files.")
    data_task_pack.add_argument("--chat-out", default=None, help="Optional output path for chat SFT JSONL.")
    data_task_pack.add_argument("--eval-out", default=None, help="Optional output path for eval JSONL.")
    data_task_pack.add_argument(
        "--sft-rows",
        type=int,
        default=DEFAULT_TASK_MIXTURE_SFT_ROWS,
        help="Number of staged SFT rows to write.",
    )
    data_task_pack.add_argument(
        "--eval-rows",
        type=int,
        default=DEFAULT_TASK_MIXTURE_EVAL_ROWS,
        help="Number of held-out eval rows to write.",
    )
    data_task_pack.add_argument(
        "--source",
        choices=BENCHMARK_SOURCES,
        default="offline",
        help="Source for benchmark components: offline deterministic rows, auto HF fallback, or HF-required.",
    )
    data_task_pack.add_argument(
        "--profile",
        choices=TASK_MIXTURE_PROFILES,
        default="capability",
        help=(
            "Task-mixture profile. release is identity/refusal only; capability adds scratchpad "
            "math/spelling; balanced mixes release, weak skills, and benchmark rows."
        ),
    )
    data_task_pack.add_argument(
        "--skill-answer-style",
        choices=BENCHMARK_SKILL_ANSWER_STYLES,
        default="scratchpad",
        help="Default answer style for components that do not override it.",
    )
    data_task_pack.add_argument("--seed", type=int, default=42)
    data_task_pack.add_argument("--force", action="store_true", help="Overwrite existing task-mixture files.")
    data_task_pack.add_argument(
        "--no-promote",
        action="store_true",
        help="Write files but do not point dataset_pack.json at them.",
    )

    data_slice_pack = data_subparsers.add_parser(
        "slice-pack",
        help="Create an audited dataset-pack slice for staged SFT by category.",
    )
    data_slice_pack.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        required=True,
        help="Source dataset pack whose chat/eval rows should be sliced.",
    )
    data_slice_pack.add_argument("--out-dir", required=True, help="Output folder for the sliced dataset pack.")
    data_slice_pack.add_argument(
        "--include-categories",
        default="",
        help="Comma-separated category globs to keep, e.g. identity,refusal or bench_choice_*.",
    )
    data_slice_pack.add_argument(
        "--exclude-categories",
        default="",
        help="Comma-separated category globs to remove after include filtering.",
    )
    data_slice_pack.add_argument("--name", default=None, help="Optional name for the sliced dataset_pack.json.")
    data_slice_pack.add_argument("--description", default=None, help="Optional description for the sliced dataset_pack.json.")
    data_slice_pack.add_argument("--force", action="store_true", help="Overwrite existing slice files.")

    data_stage_pack = data_subparsers.add_parser(
        "stage-pack",
        help="Create standard behavior/choice/skill dataset-pack slices for staged SFT.",
    )
    data_stage_pack.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        required=True,
        help="Source dataset pack whose chat/eval rows should be staged.",
    )
    data_stage_pack.add_argument("--out-dir", required=True, help="Output folder for staged dataset packs.")
    data_stage_pack.add_argument(
        "--stages",
        default=",".join(DEFAULT_TUNING_STAGE_NAMES),
        help=(
            "Comma-separated stage names to create. "
            f"Available: {', '.join(DEFAULT_TUNING_STAGE_NAMES)}."
        ),
    )
    data_stage_pack.add_argument("--force", action="store_true", help="Overwrite existing staged pack files.")

    data_skills_corpus = data_subparsers.add_parser(
        "skills-corpus",
        help="Generate arithmetic/spelling/choice drills for base pretraining.",
    )
    data_skills_corpus.add_argument("--out", required=True, help="Output micro-skills corpus text path.")
    data_skills_corpus.add_argument("--math-rows", type=int, default=50_000)
    data_skills_corpus.add_argument("--spelling-rows", type=int, default=50_000)
    data_skills_corpus.add_argument("--choice-rows", type=int, default=10_000)
    data_skills_corpus.add_argument("--seed", type=int, default=42)
    data_skills_corpus.add_argument("--force", action="store_true", help="Overwrite existing skills corpus files.")
    data_skills_corpus.add_argument(
        "--base-corpus",
        default=None,
        help="Optional base corpus or documents folder to include in a generated recipe.",
    )
    data_skills_corpus.add_argument(
        "--recipe-out",
        default=None,
        help="Optional corpus recipe path that mixes --base-corpus with the skills corpus.",
    )
    data_skills_corpus.add_argument(
        "--documents-dir",
        default=None,
        help="Optional folder for sharded micro-skills documents. Recipes use this folder when provided.",
    )
    data_skills_corpus.add_argument(
        "--rows-per-shard",
        type=int,
        default=1000,
        help="Rows per micro-skills shard when --documents-dir is set.",
    )

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
        help="Optional folder for accepted-row document files. Defaults to a documents folder beside --out.",
    )
    data_hf_import.add_argument(
        "--document-shard-rows",
        type=int,
        default=1,
        help=(
            "Accepted rows per document text file. Use 1000+ for large imports to avoid "
            "hundreds of thousands of tiny files; 1 preserves per-row document files."
        ),
    )
    data_hf_import.add_argument("--max-rows", type=int, default=1000, help="Maximum rows to inspect from the split.")
    data_hf_import.add_argument("--min-chars", type=int, default=20, help="Minimum text length required for a row to be written.")
    data_hf_import.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/load the split normally instead of streaming rows.",
    )

    data_climbmix_import = data_subparsers.add_parser(
        "climbmix-import",
        help="Import a bounded nanochat-compatible ClimbMix shard sample into a Picochat dataset pack.",
    )
    data_climbmix_import.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for corpus.txt, row documents, import report, and dataset_pack.json.",
    )
    data_climbmix_import.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Number of shuffled ClimbMix shards to read starting at shard_00000.parquet.",
    )
    data_climbmix_import.add_argument("--max-rows", type=int, default=1000, help="Maximum rows/documents to inspect.")
    data_climbmix_import.add_argument("--min-chars", type=int, default=20, help="Minimum document length.")
    data_climbmix_import.add_argument(
        "--document-shard-rows",
        type=int,
        default=None,
        help=(
            "Accepted rows per local document file. Defaults to 1 for small imports and "
            "1000 for H100-scale imports to avoid hundreds of thousands of tiny files."
        ),
    )
    data_climbmix_import.add_argument("--force", action="store_true", help="Overwrite existing import/pack artifacts.")
    data_climbmix_import.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/load the selected shard files normally instead of streaming rows.",
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
        help=(
            "Tokenizer type to train. Use char for the baseline, byte for UTF-8 bytes, "
            "bpe for inspectable Python BPE, or hf_bpe for compiled long-run BPE."
        ),
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
    tok_train.add_argument(
        "--bpe-pretokenizer",
        choices=BPE_PRETOKENIZERS,
        default=DEFAULT_BPE_PRETOKENIZER,
        help="Pretokenizer used before BPE merges. regex is the stronger long-run default.",
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
    train_base_parser.add_argument(
        "--resume-from",
        default=None,
        help="Resumable checkpoint directory containing training_state.pt.",
    )
    train_base_parser.add_argument("--context-size", type=int, default=64)
    train_base_parser.add_argument("--batch-size", type=int, default=16)
    train_base_parser.add_argument("--max-steps", type=int, default=200)
    train_base_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_base_parser.add_argument("--n-embd", type=int, default=128)
    train_base_parser.add_argument("--n-head", type=int, default=4)
    train_base_parser.add_argument("--n-kv-head", type=int, default=None)
    train_base_parser.add_argument("--n-layer", type=int, default=2)
    train_base_parser.add_argument("--dropout", type=float, default=0.0)
    train_base_parser.add_argument("--norm-type", choices=("layernorm", "rmsnorm"), default="layernorm")
    train_base_parser.add_argument("--position-encoding", choices=("learned", "rope"), default="learned")
    train_base_parser.add_argument("--activation", choices=("gelu", "relu2", "swiglu"), default="gelu")
    train_base_parser.add_argument("--tie-embeddings", action="store_true")
    train_base_parser.add_argument("--qk-norm", action="store_true")
    train_base_parser.add_argument(
        "--no-linear-bias",
        dest="linear_bias",
        action="store_false",
        default=True,
        help="Disable biases in transformer linear layers and the untied LM head.",
    )
    train_base_parser.add_argument(
        "--parallel-residual",
        action="store_true",
        help="Use a parallel residual block: x + attn(norm(x)) + mlp(norm(x)).",
    )
    train_base_parser.add_argument(
        "--attn-backend",
        choices=SDPA_BACKENDS,
        default="auto",
        help="PyTorch SDPA backend. Use flash on H100 after sanity passes; auto is portable.",
    )
    train_base_parser.add_argument("--seed", type=int, default=42)
    train_base_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    train_base_parser.add_argument("--log-every", type=int, default=20)
    train_base_parser.add_argument("--val-fraction", type=float, default=0.1)
    train_base_parser.add_argument("--eval-batches", type=int, default=10)
    train_base_parser.add_argument("--sample-tokens", type=int, default=120)
    train_base_parser.add_argument("--early-stop-patience", type=int, default=0)
    train_base_parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_base_parser.add_argument("--max-minutes", type=float, default=None)
    train_base_parser.add_argument("--lr-warmup-steps", type=int, default=0)
    train_base_parser.add_argument("--lr-decay", choices=LR_DECAYS, default="none")
    train_base_parser.add_argument("--min-lr-ratio", type=float, default=1.0)
    train_base_parser.add_argument("--grad-clip", type=float, default=0.0)
    train_base_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_base_parser.add_argument("--optimizer", choices=OPTIMIZER_TYPES, default="adamw")
    train_base_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_base_parser.add_argument("--weight-decay-decay", choices=WEIGHT_DECAY_DECAYS, default="none")
    train_base_parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    train_base_parser.add_argument("--muon-momentum-schedule", choices=MUON_MOMENTUM_SCHEDULES, default="none")
    train_base_parser.add_argument("--ema-decay", type=float, default=0.0)
    train_base_parser.add_argument(
        "--loss-spike-rollback",
        action="store_true",
        help="Restore the previous weights and reduce LR scale when train loss spikes.",
    )
    train_base_parser.add_argument("--loss-spike-threshold", type=float, default=2.5)
    train_base_parser.add_argument("--loss-spike-lr-decay", type=float, default=0.5)
    train_base_parser.add_argument("--loss-spike-min-lr-scale", type=float, default=0.1)
    train_base_parser.add_argument("--loss-spike-snapshot-every", type=int, default=10)
    train_base_parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default="float32",
        help="Training precision. Use bf16/fp16/auto only on supported accelerators.",
    )
    train_base_parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISION_MODES,
        default="default",
        help="torch.set_float32_matmul_precision setting. Use high on H100/CUDA for TF32 tensor cores.",
    )
    train_base_parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the model forward path with torch.compile.",
    )
    train_base_parser.add_argument(
        "--torch-compile-mode",
        choices=COMPILE_MODES,
        default="default",
        help="torch.compile mode when --torch-compile is enabled.",
    )
    train_base_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Checkpoint transformer blocks during training to reduce activation memory.",
    )
    train_base_parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wrap training in DistributedDataParallel. Launch with torchrun.",
    )
    train_base_parser.add_argument(
        "--logit-softcap",
        type=float,
        default=0.0,
        help="Optional tanh softcap applied to logits during training and generation. 0 disables it.",
    )
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
        "--dataset-mode",
        choices=("memory", "sharded", "packed"),
        default="memory",
        help=(
            "Use in-memory windows, disk-backed token shards, or disk-backed "
            "BOS-bestfit packed rows with complete-document holdout."
        ),
    )
    train_base_parser.add_argument(
        "--shard-token-size",
        type=int,
        default=1_000_000,
        help="Target tokens per disk shard when --dataset-mode sharded is used.",
    )
    train_base_parser.add_argument(
        "--shard-cache-size",
        type=int,
        default=2,
        help="Number of token shards to keep hot in memory when --dataset-mode sharded is used.",
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
    train_sft_parser.add_argument(
        "--resume-from",
        default=None,
        help="Resumable SFT checkpoint directory containing training_state.pt.",
    )
    train_sft_parser.add_argument("--batch-size", type=int, default=8)
    train_sft_parser.add_argument("--max-steps", type=int, default=100)
    train_sft_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_sft_parser.add_argument("--seed", type=int, default=42)
    train_sft_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    train_sft_parser.add_argument("--log-every", type=int, default=10)
    train_sft_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_sft_parser.add_argument("--eval-batches", type=int, default=10)
    train_sft_parser.add_argument("--sample-prompt", default="What is Picochat?")
    train_sft_parser.add_argument("--sample-tokens", type=int, default=120)
    train_sft_parser.add_argument("--early-stop-patience", type=int, default=0)
    train_sft_parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_sft_parser.add_argument("--max-minutes", type=float, default=None)
    train_sft_parser.add_argument("--lr-warmup-steps", type=int, default=0)
    train_sft_parser.add_argument("--lr-decay", choices=LR_DECAYS, default="none")
    train_sft_parser.add_argument("--min-lr-ratio", type=float, default=1.0)
    train_sft_parser.add_argument("--grad-clip", type=float, default=0.0)
    train_sft_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_sft_parser.add_argument("--optimizer", choices=OPTIMIZER_TYPES, default="adamw")
    train_sft_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_sft_parser.add_argument("--weight-decay-decay", choices=WEIGHT_DECAY_DECAYS, default="none")
    train_sft_parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    train_sft_parser.add_argument("--muon-momentum-schedule", choices=MUON_MOMENTUM_SCHEDULES, default="none")
    train_sft_parser.add_argument("--ema-decay", type=float, default=0.0)
    train_sft_parser.add_argument(
        "--loss-spike-rollback",
        action="store_true",
        help="Restore the previous weights and reduce LR scale when train loss spikes.",
    )
    train_sft_parser.add_argument("--loss-spike-threshold", type=float, default=2.5)
    train_sft_parser.add_argument("--loss-spike-lr-decay", type=float, default=0.5)
    train_sft_parser.add_argument("--loss-spike-min-lr-scale", type=float, default=0.1)
    train_sft_parser.add_argument("--loss-spike-snapshot-every", type=int, default=10)
    train_sft_parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default="float32",
        help="Training precision. Use bf16/fp16/auto only on supported accelerators.",
    )
    train_sft_parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISION_MODES,
        default="default",
        help="torch.set_float32_matmul_precision setting. Use high on H100/CUDA for TF32 tensor cores.",
    )
    train_sft_parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the model forward path with torch.compile.",
    )
    train_sft_parser.add_argument(
        "--torch-compile-mode",
        choices=COMPILE_MODES,
        default="default",
        help="torch.compile mode when --torch-compile is enabled.",
    )
    train_sft_parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wrap SFT training in DistributedDataParallel. Launch with torchrun.",
    )
    train_sft_parser.add_argument(
        "--sampling",
        choices=SFT_SAMPLING_MODES,
        default="uniform",
        help=(
            "SFT row sampling strategy. category_sqrt softly boosts rare categories; "
            "category_balanced gives rare categories equal training probability."
        ),
    )
    train_sft_parser.add_argument(
        "--packing",
        "--sft-packing",
        dest="packing",
        choices=SFT_PACKING_MODES,
        default="separate",
        help="SFT sequence packing. Keep separate as the verified default; bos_bestfit packs multiple BOS-delimited examples per context.",
    )
    train_sft_parser.add_argument(
        "--peft",
        choices=PEFT_MODES,
        default="none",
        help="Parameter-efficient SFT mode. Use lora to train adapter weights while saving merged full checkpoints.",
    )
    train_sft_parser.add_argument("--lora-rank", type=int, default=8)
    train_sft_parser.add_argument("--lora-alpha", type=float, default=16.0)
    train_sft_parser.add_argument("--lora-dropout", type=float, default=0.0)
    train_sft_parser.add_argument(
        "--lora-targets",
        default=",".join(DEFAULT_LORA_TARGETS),
        help=f"Comma-separated LoRA targets: {', '.join(LORA_TARGETS)}.",
    )
    train_sft_sweep_parser = train_subparsers.add_parser(
        "sft-sweep",
        help="Run a controlled SFT schedule sweep from one base checkpoint.",
    )
    train_sft_sweep_parser.add_argument(
        "--dataset-pack",
        "--pack",
        dest="dataset_pack",
        default=None,
        help="Optional dataset pack. Uses its chat/eval files unless --input or --eval-input override them.",
    )
    train_sft_sweep_parser.add_argument("--input", default=None, help="Path to chat SFT JSONL.")
    train_sft_sweep_parser.add_argument("--eval-input", default=None, help="Optional held-out eval JSONL.")
    train_sft_sweep_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    train_sft_sweep_parser.add_argument("--checkpoint", required=True, help="Base checkpoint directory.")
    train_sft_sweep_parser.add_argument("--out-dir", required=True, help="Output sweep directory.")
    train_sft_sweep_parser.add_argument("--support-corpus", default=None, help="Optional corpus text file for support diagnostics.")
    train_sft_sweep_parser.add_argument(
        "--learning-rates",
        default="0.00003,0.00005,0.0001",
        help="Comma-separated SFT learning rates.",
    )
    train_sft_sweep_parser.add_argument(
        "--steps",
        default="160,400,800",
        help="Comma-separated SFT step counts.",
    )
    train_sft_sweep_parser.add_argument(
        "--samplings",
        default="category_sqrt",
        help="Comma-separated sampling modes: uniform, category_sqrt, category_balanced.",
    )
    train_sft_sweep_parser.add_argument("--batch-size", type=int, default=8)
    train_sft_sweep_parser.add_argument("--seed", type=int, default=42)
    train_sft_sweep_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    train_sft_sweep_parser.add_argument("--eval-max-new-tokens", type=int, default=120)
    train_sft_sweep_parser.add_argument("--fit-max-rows", type=int, default=500)
    train_sft_sweep_parser.add_argument("--eval-log-every", type=int, default=50, help="Print progress every N eval rows. 0 disables progress logs.")
    train_sft_sweep_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_sft_sweep_parser.add_argument("--eval-batches", type=int, default=10)
    train_sft_sweep_parser.add_argument("--sample-prompt", default="What is Picochat?")
    train_sft_sweep_parser.add_argument("--sample-tokens", type=int, default=120)
    train_sft_sweep_parser.add_argument("--early-stop-patience", type=int, default=0)
    train_sft_sweep_parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_sft_sweep_parser.add_argument("--max-minutes", type=float, default=None)
    train_sft_sweep_parser.add_argument("--lr-warmup-steps", type=int, default=20)
    train_sft_sweep_parser.add_argument("--lr-decay", choices=LR_DECAYS, default="cosine")
    train_sft_sweep_parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    train_sft_sweep_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_sft_sweep_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_sft_sweep_parser.add_argument("--optimizer", choices=OPTIMIZER_TYPES, default="adamw")
    train_sft_sweep_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_sft_sweep_parser.add_argument("--weight-decay-decay", choices=WEIGHT_DECAY_DECAYS, default="none")
    train_sft_sweep_parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    train_sft_sweep_parser.add_argument("--muon-momentum-schedule", choices=MUON_MOMENTUM_SCHEDULES, default="none")
    train_sft_sweep_parser.add_argument("--ema-decay", type=float, default=0.0)
    train_sft_sweep_parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default="float32",
        help="Training precision for each SFT candidate. Use bf16/fp16/auto only on supported accelerators.",
    )
    train_sft_sweep_parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISION_MODES,
        default="default",
        help="torch.set_float32_matmul_precision setting for each candidate.",
    )
    train_sft_sweep_parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile each SFT candidate model forward path with torch.compile.",
    )
    train_sft_sweep_parser.add_argument(
        "--torch-compile-mode",
        choices=COMPILE_MODES,
        default="default",
        help="torch.compile mode when --torch-compile is enabled.",
    )
    train_sft_sweep_parser.add_argument(
        "--packing",
        "--sft-packing",
        dest="packing",
        choices=SFT_PACKING_MODES,
        default="separate",
        help="SFT sequence packing used for every sweep candidate.",
    )
    train_sft_sweep_parser.add_argument(
        "--peft",
        choices=PEFT_MODES,
        default="none",
        help="Parameter-efficient SFT mode for every sweep candidate.",
    )
    train_sft_sweep_parser.add_argument("--lora-rank", type=int, default=8)
    train_sft_sweep_parser.add_argument("--lora-alpha", type=float, default=16.0)
    train_sft_sweep_parser.add_argument("--lora-dropout", type=float, default=0.0)
    train_sft_sweep_parser.add_argument(
        "--lora-targets",
        default=",".join(DEFAULT_LORA_TARGETS),
        help=f"Comma-separated LoRA targets: {', '.join(LORA_TARGETS)}.",
    )

    generate_parser = subparsers.add_parser("generate", help="Generate from a checkpoint.")
    generate_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    generate_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    generate_parser.add_argument("--prompt", default="", help="Text prompt.")
    generate_parser.add_argument("--max-new-tokens", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=20)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    generate_parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable incremental KV-cache decoding during generation.",
    )

    export_parser = subparsers.add_parser("export", help="Export model artifacts.")
    export_subparsers = export_parser.add_subparsers(dest="export_command")
    export_hf_parser = export_subparsers.add_parser(
        "hf",
        help="Export a checkpoint as a HuggingFace-style release folder.",
    )
    export_hf_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    export_hf_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    export_hf_parser.add_argument("--out-dir", required=True, help="Output model folder.")
    export_hf_parser.add_argument("--model-name", default="picochat", help="Model card title.")
    export_hf_parser.add_argument(
        "--fine-tuned",
        action="store_true",
        help="Mark this export as a fine-tuned model instead of a base model.",
    )
    export_hf_parser.add_argument("--license", default="unknown", help="License string for the model card.")
    export_hf_parser.add_argument("--dataset-summary", default="Not provided.", help="Training data summary.")
    export_hf_parser.add_argument("--eval-summary", default="Not provided.", help="Evaluation summary.")
    export_hf_parser.add_argument(
        "--dynamic-int8",
        action="store_true",
        help="Also write a Picochat PyTorch dynamic-int8 serving artifact.",
    )
    export_hf_parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Do not attempt to write model.safetensors.",
    )
    export_hf_parser.add_argument(
        "--no-transformers-adapter",
        action="store_true",
        help="Do not write Transformers trust_remote_code adapter files.",
    )

    sanity_parser = subparsers.add_parser("sanity", help="Run local readiness sanity checks.")
    sanity_subparsers = sanity_parser.add_subparsers(dest="sanity_command")
    sanity_preh100_parser = sanity_subparsers.add_parser(
        "preh100",
        help="Run fast checks before spending H100 time on a long run.",
    )
    sanity_preh100_parser.add_argument(
        "--out-dir",
        default="runs/preh100-sanity",
        help="Output folder for sanity reports and scratch artifacts.",
    )
    sanity_preh100_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    sanity_preh100_parser.add_argument("--precision", choices=PRECISION_MODES, default="auto")
    sanity_preh100_parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISION_MODES,
        default="default",
        help="torch.set_float32_matmul_precision setting to verify before a GPU run.",
    )
    sanity_preh100_parser.add_argument(
        "--attn-backend",
        choices=SDPA_BACKENDS,
        default="auto",
        help="Attention backend to force during sanity checks. Use flash on H100 smoke tests.",
    )
    sanity_preh100_parser.add_argument(
        "--include-compile",
        action="store_true",
        help="Also run a torch.compile smoke test.",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat.")
    chat_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    chat_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    chat_parser.add_argument("--max-new-tokens", type=int, default=120)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--top-k", type=int, default=20)
    chat_parser.add_argument("--top-p", type=float, default=1.0)
    chat_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    chat_parser.add_argument("--seed", type=int, default=42)
    chat_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    chat_parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable incremental KV-cache decoding during chat generation.",
    )

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
    eval_chat_parser.add_argument("--top-p", type=float, default=1.0)
    eval_chat_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    eval_chat_parser.add_argument("--seed", type=int, default=42)
    eval_chat_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    _add_eval_runtime_args(eval_chat_parser)
    eval_chat_parser.add_argument("--log-every", type=int, default=0, help="Print progress every N eval rows. 0 disables progress logs.")
    eval_chat_parser.add_argument("--case-sensitive", action="store_true")
    eval_chat_parser.add_argument("--support-corpus", default=None, help="Optional corpus text file for support-overlap diagnostics.")
    eval_chat_parser.add_argument("--corpus-support-threshold", type=float, default=0.25)
    eval_chat_parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples for eval pass-rate confidence intervals. Use 0 to disable.",
    )
    eval_chat_parser.add_argument("--ci-confidence", type=float, default=0.95)

    eval_external_parser = eval_subparsers.add_parser(
        "external",
        help="Convert and run an external ARC/MMLU-style multiple-choice benchmark.",
    )
    eval_external_parser.add_argument("--input", required=True, help="Path to external benchmark JSONL/JSON/CSV.")
    eval_external_parser.add_argument("--format", choices=EXTERNAL_BENCHMARK_FORMATS, default="auto")
    eval_external_parser.add_argument("--benchmark-name", default="external")
    eval_external_parser.add_argument("--split", default="external")
    eval_external_parser.add_argument("--max-rows", type=int, default=0)
    eval_external_parser.add_argument("--shuffle", action="store_true")
    eval_external_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    eval_external_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    eval_external_parser.add_argument("--out-dir", required=True, help="Output eval directory.")
    eval_external_parser.add_argument("--max-new-tokens", type=int, default=1)
    eval_external_parser.add_argument("--temperature", type=float, default=0.0)
    eval_external_parser.add_argument("--top-k", type=int, default=0)
    eval_external_parser.add_argument("--top-p", type=float, default=1.0)
    eval_external_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    eval_external_parser.add_argument("--seed", type=int, default=42)
    eval_external_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    _add_eval_runtime_args(eval_external_parser)
    eval_external_parser.add_argument("--log-every", type=int, default=0, help="Print progress every N eval rows. 0 disables progress logs.")
    eval_external_parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples for external benchmark confidence intervals. Use 0 to disable.",
    )
    eval_external_parser.add_argument("--ci-confidence", type=float, default=0.95)

    eval_sft_fit_parser = eval_subparsers.add_parser(
        "sft-fit",
        help="Run an exact-fit diagnostic on chat SFT rows.",
    )
    eval_sft_fit_parser.add_argument("--input", required=True, help="Path to chat SFT JSONL.")
    eval_sft_fit_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    eval_sft_fit_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    eval_sft_fit_parser.add_argument("--out-dir", required=True, help="Output eval directory.")
    eval_sft_fit_parser.add_argument("--max-rows", type=int, default=0, help="Optional row limit for quick diagnostics.")
    eval_sft_fit_parser.add_argument("--max-new-tokens", type=int, default=80)
    eval_sft_fit_parser.add_argument("--temperature", type=float, default=0.0)
    eval_sft_fit_parser.add_argument("--top-k", type=int, default=0)
    eval_sft_fit_parser.add_argument("--top-p", type=float, default=1.0)
    eval_sft_fit_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    eval_sft_fit_parser.add_argument("--seed", type=int, default=42)
    eval_sft_fit_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    _add_eval_runtime_args(eval_sft_fit_parser)
    eval_sft_fit_parser.add_argument("--log-every", type=int, default=50, help="Print progress every N eval rows. 0 disables progress logs.")
    eval_sft_fit_parser.add_argument("--case-sensitive", action="store_true")
    eval_sft_fit_parser.add_argument("--support-corpus", default=None, help="Optional corpus text file for support-overlap diagnostics.")
    eval_sft_fit_parser.add_argument("--corpus-support-threshold", type=float, default=0.25)
    eval_sft_fit_parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples for SFT-fit pass-rate confidence intervals. Use 0 to disable.",
    )
    eval_sft_fit_parser.add_argument("--ci-confidence", type=float, default=0.95)

    scale_parser = subparsers.add_parser("scale", help="Plan larger GPU training recipes.")
    scale_subparsers = scale_parser.add_subparsers(dest="scale_command")
    scale_plan_parser = scale_subparsers.add_parser(
        "plan",
        help="Compute a parameter/data/batch scale recipe without training.",
    )
    scale_plan_parser.add_argument(
        "--target-params",
        required=True,
        help="Target model size, e.g. 100m, 300m, or 1b.",
    )
    scale_plan_parser.add_argument(
        "--dataset-tokens",
        default=None,
        help="Optional estimated corpus tokens, e.g. 667m. Used only for epoch reporting.",
    )
    scale_plan_parser.add_argument("--depth", type=int, default=None, help="Optional fixed layer count.")
    scale_plan_parser.add_argument("--aspect-ratio", type=int, default=48, help="n_embd ~= depth * aspect ratio.")
    scale_plan_parser.add_argument("--head-dim", type=int, default=64)
    scale_plan_parser.add_argument("--vocab-size", type=int, default=8192)
    scale_plan_parser.add_argument("--context-size", type=int, default=512)
    scale_plan_parser.add_argument("--world-size", type=int, default=1)
    scale_plan_parser.add_argument("--per-device-batch-size", type=int, default=8)
    scale_plan_parser.add_argument("--grad-accum-steps", type=int, default=16)
    scale_plan_parser.add_argument("--target-param-data-ratio", type=float, default=20.0)
    scale_plan_parser.add_argument("--attn-backend", choices=SDPA_BACKENDS, default="flash")
    scale_plan_parser.add_argument("--long-run-gate-profile", choices=LONG_RUN_GATE_PROFILES, default="skill_release")
    scale_plan_parser.add_argument("--out", default=None, help="Optional Markdown report path.")

    run_parser = subparsers.add_parser("run", help="End-to-end experiment runners.")
    run_subparsers = run_parser.add_subparsers(dest="run_command")
    run_tiny_parser = run_subparsers.add_parser("tiny", help="Run the full tiny pipeline.")
    run_tiny_parser.add_argument("--out-dir", required=True, help="Output run directory.")
    run_tiny_parser.add_argument(
        "--scale",
        choices=("custom", *RUN_SCALE_NAMES),
        default="custom",
        help="Named local run scale. Explicit numeric flags override preset values.",
    )
    run_tiny_parser.add_argument("--dataset-pack", "--pack", dest="dataset_pack", default=None)
    run_tiny_parser.add_argument("--corpus-input", default="examples/tiny_corpus.txt")
    run_tiny_parser.add_argument("--corpus-recipe", default=None)
    run_tiny_parser.add_argument("--chat-input", default="examples/tiny_chat.jsonl")
    run_tiny_parser.add_argument("--eval-input", default="examples/tiny_eval.jsonl")
    run_tiny_parser.add_argument(
        "--external-eval",
        action="append",
        default=None,
        help=(
            "Optional ARC/MMLU-style external benchmark file to run after the main eval. "
            "Repeat for multiple files. Use name=path to control the summary label."
        ),
    )
    run_tiny_parser.add_argument(
        "--external-eval-format",
        choices=EXTERNAL_BENCHMARK_FORMATS,
        default=None,
        help="Format for --external-eval files. auto detects ARC or MMLU records.",
    )
    run_tiny_parser.add_argument(
        "--external-eval-max-rows",
        type=int,
        default=None,
        help="Optional row limit per external benchmark. 0 means all rows.",
    )
    run_tiny_parser.add_argument(
        "--external-eval-shuffle",
        action="store_true",
        help="Shuffle external benchmark rows before applying --external-eval-max-rows.",
    )
    run_tiny_parser.add_argument(
        "--external-eval-max-new-tokens",
        type=int,
        default=None,
        help="Max generated tokens for external choice evals. Choice scoring uses logprobs; 1 is usually enough.",
    )
    run_tiny_parser.add_argument("--context-size", type=int, default=None)
    run_tiny_parser.add_argument("--n-embd", type=int, default=None)
    run_tiny_parser.add_argument("--n-head", type=int, default=None)
    run_tiny_parser.add_argument("--n-kv-head", type=int, default=None)
    run_tiny_parser.add_argument("--n-layer", type=int, default=None)
    run_tiny_parser.add_argument("--dropout", type=float, default=None)
    run_tiny_parser.add_argument("--norm-type", choices=("layernorm", "rmsnorm"), default=None)
    run_tiny_parser.add_argument("--position-encoding", choices=("learned", "rope"), default=None)
    run_tiny_parser.add_argument("--activation", choices=("gelu", "relu2", "swiglu"), default=None)
    run_tiny_parser.add_argument("--tie-embeddings", action="store_true")
    run_tiny_parser.add_argument("--qk-norm", action="store_true")
    run_tiny_bias_group = run_tiny_parser.add_mutually_exclusive_group()
    run_tiny_bias_group.add_argument(
        "--linear-bias",
        dest="linear_bias",
        action="store_true",
        default=None,
        help="Enable biases in transformer linear layers and the untied LM head.",
    )
    run_tiny_bias_group.add_argument(
        "--no-linear-bias",
        dest="linear_bias",
        action="store_false",
        default=None,
        help="Disable biases in transformer linear layers and the untied LM head.",
    )
    run_tiny_parser.add_argument(
        "--parallel-residual",
        action="store_true",
        help="Use a parallel residual block: x + attn(norm(x)) + mlp(norm(x)).",
    )
    run_tiny_parser.add_argument(
        "--attn-backend",
        choices=SDPA_BACKENDS,
        default=None,
        help="PyTorch SDPA backend. Use flash on H100 after sanity passes; auto is portable.",
    )
    run_tiny_parser.add_argument("--base-steps", type=int, default=None)
    run_tiny_parser.add_argument("--sft-steps", type=int, default=None)
    run_tiny_parser.add_argument("--base-batch-size", type=int, default=None)
    run_tiny_parser.add_argument("--sft-batch-size", type=int, default=None)
    run_tiny_parser.add_argument("--base-learning-rate", type=float, default=None)
    run_tiny_parser.add_argument("--sft-learning-rate", type=float, default=None)
    run_tiny_parser.add_argument("--base-early-stop-patience", type=int, default=None)
    run_tiny_parser.add_argument("--sft-early-stop-patience", type=int, default=None)
    run_tiny_parser.add_argument("--early-stop-min-delta", type=float, default=None)
    run_tiny_parser.add_argument("--base-max-minutes", type=float, default=None)
    run_tiny_parser.add_argument("--sft-max-minutes", type=float, default=None)
    run_tiny_parser.add_argument("--canary-count", type=int, default=None)
    run_tiny_parser.add_argument("--seed", type=int, default=42)
    run_tiny_parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Run consecutive seeds and write mean/std aggregate reports.",
    )
    run_tiny_parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    run_tiny_parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    run_tiny_parser.add_argument(
        "--tokenizer-type",
        choices=TOKENIZER_TYPES,
        default=None,
        help=(
            "Tokenizer used for this run. Compare char, byte, inspectable bpe, "
            "and compiled hf_bpe on the same dataset pack."
        ),
    )
    run_tiny_parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=None,
        help="Optional tokenizer vocabulary size including special tokens. Useful for BPE.",
    )
    run_tiny_parser.add_argument(
        "--tokenizer-min-freq",
        type=int,
        default=None,
        help="Minimum BPE merge frequency or character frequency.",
    )
    run_tiny_parser.add_argument(
        "--bpe-pretokenizer",
        choices=BPE_PRETOKENIZERS,
        default=None,
        help="Pretokenizer used before BPE merges. regex is the stronger long-run default.",
    )
    run_tiny_parser.add_argument("--base-lr-warmup-steps", type=int, default=None)
    run_tiny_parser.add_argument("--sft-lr-warmup-steps", type=int, default=None)
    run_tiny_parser.add_argument("--base-lr-decay", choices=LR_DECAYS, default=None)
    run_tiny_parser.add_argument("--sft-lr-decay", choices=LR_DECAYS, default=None)
    run_tiny_parser.add_argument("--base-min-lr-ratio", type=float, default=None)
    run_tiny_parser.add_argument("--sft-min-lr-ratio", type=float, default=None)
    run_tiny_parser.add_argument("--base-grad-clip", type=float, default=None)
    run_tiny_parser.add_argument("--sft-grad-clip", type=float, default=None)
    run_tiny_parser.add_argument("--base-grad-accum-steps", type=int, default=None)
    run_tiny_parser.add_argument("--sft-grad-accum-steps", type=int, default=None)
    run_tiny_parser.add_argument("--base-optimizer", choices=OPTIMIZER_TYPES, default=None)
    run_tiny_parser.add_argument("--sft-optimizer", choices=OPTIMIZER_TYPES, default=None)
    run_tiny_parser.add_argument("--base-weight-decay", type=float, default=None)
    run_tiny_parser.add_argument("--sft-weight-decay", type=float, default=None)
    run_tiny_parser.add_argument("--base-weight-decay-decay", choices=WEIGHT_DECAY_DECAYS, default=None)
    run_tiny_parser.add_argument("--sft-weight-decay-decay", choices=WEIGHT_DECAY_DECAYS, default=None)
    run_tiny_parser.add_argument("--base-muon-learning-rate", type=float, default=None)
    run_tiny_parser.add_argument("--sft-muon-learning-rate", type=float, default=None)
    run_tiny_parser.add_argument("--base-muon-momentum-schedule", choices=MUON_MOMENTUM_SCHEDULES, default=None)
    run_tiny_parser.add_argument("--sft-muon-momentum-schedule", choices=MUON_MOMENTUM_SCHEDULES, default=None)
    run_tiny_parser.add_argument("--base-ema-decay", type=float, default=None)
    run_tiny_parser.add_argument("--sft-ema-decay", type=float, default=None)
    run_tiny_parser.add_argument(
        "--target-param-data-ratio",
        type=float,
        default=None,
        help=(
            "Training-budget target in tokens per model parameter. "
            "Used by long-run preflight to recommend base steps."
        ),
    )
    run_tiny_parser.add_argument(
        "--auto-lr-scaling",
        action="store_true",
        help="Apply preflight sqrt effective-batch LR scaling to base and SFT learning rates.",
    )
    run_tiny_parser.add_argument(
        "--loss-spike-rollback",
        action="store_true",
        help="Restore the previous weights and reduce LR scale when train loss spikes.",
    )
    run_tiny_parser.add_argument("--loss-spike-threshold", type=float, default=None)
    run_tiny_parser.add_argument("--loss-spike-lr-decay", type=float, default=None)
    run_tiny_parser.add_argument("--loss-spike-min-lr-scale", type=float, default=None)
    run_tiny_parser.add_argument("--loss-spike-snapshot-every", type=int, default=None)
    run_tiny_parser.add_argument(
        "--logit-softcap",
        type=float,
        default=None,
        help="Optional tanh softcap applied to model logits. 0 disables it.",
    )
    run_tiny_parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default=None,
        help="Base and SFT training precision. Defaults to the selected scale.",
    )
    run_tiny_parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISION_MODES,
        default=None,
        help="torch.set_float32_matmul_precision setting. Use high on H100/CUDA for TF32 tensor cores.",
    )
    run_tiny_parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile base and SFT model forward paths with torch.compile.",
    )
    run_tiny_parser.add_argument(
        "--torch-compile-mode",
        choices=COMPILE_MODES,
        default=None,
        help="torch.compile mode when --torch-compile is enabled.",
    )
    run_tiny_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Checkpoint base model transformer blocks during training.",
    )
    run_tiny_parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wrap base and SFT training in DistributedDataParallel. Launch with torchrun.",
    )
    run_tiny_parser.add_argument(
        "--ddp-world-size",
        type=int,
        default=None,
        help=(
            "Preflight-only DDP world-size simulation. Use this to estimate an 8-GPU "
            "budget without launching eight preflight workers."
        ),
    )
    run_tiny_parser.add_argument(
        "--sft-sampling",
        choices=SFT_SAMPLING_MODES,
        default=None,
        help=(
            "SFT row sampling strategy. category_sqrt softly boosts rare categories; "
            "category_balanced gives rare categories equal training probability."
        ),
    )
    run_tiny_parser.add_argument(
        "--sft-packing",
        choices=SFT_PACKING_MODES,
        default=None,
        help=(
            "SFT sequence packing. Default separate keeps one chat row per sequence; "
            "bos_bestfit packs multiple BOS-delimited rows per context."
        ),
    )
    run_tiny_parser.add_argument(
        "--sft-fit-max-rows",
        type=int,
        default=None,
        help=(
            "Maximum SFT rows to score in the automatic fit diagnostic. "
            "Use 0 to score every SFT row."
        ),
    )
    run_tiny_parser.add_argument(
        "--sft-peft",
        choices=PEFT_MODES,
        default=None,
        help="Parameter-efficient SFT mode for the chat stage. Use lora for adapter tuning.",
    )
    run_tiny_parser.add_argument("--sft-lora-rank", type=int, default=None)
    run_tiny_parser.add_argument("--sft-lora-alpha", type=float, default=None)
    run_tiny_parser.add_argument("--sft-lora-dropout", type=float, default=None)
    run_tiny_parser.add_argument(
        "--sft-lora-targets",
        default=None,
        help=f"Comma-separated LoRA targets: {', '.join(LORA_TARGETS)}.",
    )
    run_tiny_parser.add_argument(
        "--base-resume-from",
        default=None,
        help=(
            "Resume the base-training phase from a checkpoint directory, usually "
            "<run>/base/resume_checkpoint after an interrupted expensive run."
        ),
    )
    run_tiny_parser.add_argument(
        "--sft-resume-from",
        default=None,
        help=(
            "Resume the chat-SFT phase from a checkpoint directory, usually "
            "<run>/sft/resume_checkpoint after an interrupted run."
        ),
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
    run_tiny_parser.add_argument(
        "--base-dataset-mode",
        choices=("memory", "sharded", "packed"),
        default=None,
        help=(
            "Base training dataset path. memory preserves document-split behavior; "
            "sharded writes token shards to disk for larger H100-scale corpora; "
            "packed holds out complete documents, then writes BOS-bestfit rows."
        ),
    )
    run_tiny_parser.add_argument(
        "--base-shard-token-size",
        type=int,
        default=None,
        help="Target tokens per disk shard when --base-dataset-mode sharded is used.",
    )
    run_tiny_parser.add_argument(
        "--base-shard-cache-size",
        type=int,
        default=None,
        help="Number of token shards to keep hot in memory during sharded base training.",
    )
    run_tiny_parser.add_argument(
        "--allow-leaky-eval",
        action="store_true",
        help="Allow a diagnostic run to continue when data honesty detects blocking eval leakage.",
    )
    run_tiny_parser.add_argument(
        "--allow-default-tuning-data",
        action="store_true",
        help=(
            "Allow a custom corpus run to keep Picochat's demo chat/eval files. "
            "Use only for diagnostic wiring checks."
        ),
    )
    run_tiny_parser.add_argument(
        "--allow-unsafe-long-run",
        action="store_true",
        help=(
            "Bypass long-run preflight blocking checks. Use only for diagnostic runs; "
            "the run summary will still record the failed checklist."
        ),
    )
    run_tiny_parser.add_argument(
        "--long-run-gate-profile",
        choices=LONG_RUN_GATE_PROFILES,
        default=None,
        help=(
            "Completed-run approval profile. research gates all behavior metrics; "
            "first_release gates release behavior while keeping math and spelling diagnostic; "
            "skill_release blocks unless identity/refusal/choice/math/spelling all clear."
        ),
    )
    run_tiny_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Inspect the long-run checklist and exit without training.",
    )
    run_bundle_parser = run_subparsers.add_parser(
        "bundle",
        help="Package completed or interrupted run artifacts for copying off a GPU box.",
    )
    run_bundle_parser.add_argument("--run-dir", required=True, help="Run directory to package.")
    run_bundle_parser.add_argument("--out", default=None, help="Output .tgz path. Defaults to <run>-bundle.tgz.")
    run_bundle_parser.add_argument(
        "--logs-dir",
        default=None,
        help="Optional logs directory to include in the bundle.",
    )
    run_bundle_parser.add_argument(
        "--include-corpus",
        action="store_true",
        help="Also include corpus.txt and corpus_manifest.json. This can make very large bundles.",
    )
    run_bundle_parser.add_argument(
        "--include-token-shards",
        action="store_true",
        help="Also include base token-shard caches. Usually unnecessary when the corpus can be rebuilt.",
    )
    run_bundle_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no checkpoint payload exists in the run directory.",
    )
    inspect_bundle_parser = run_subparsers.add_parser(
        "inspect-bundle",
        help="Inspect a copied .tgz run bundle and show resumable checkpoints.",
    )
    inspect_bundle_parser.add_argument("--bundle", required=True, help="Bundle .tgz path to inspect.")
    inspect_bundle_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of Markdown.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare completed run summaries.")
    compare_parser.add_argument("runs", nargs="+", help="Run directories containing summary.json.")
    compare_parser.add_argument("--out", default=None, help="Optional Markdown report output path.")

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Build a formal benchmark leaderboard from eval reports.")
    leaderboard_parser.add_argument("runs", nargs="+", help="Run directories containing eval/eval_report.json.")
    leaderboard_parser.add_argument("--out", default=None, help="Optional Markdown leaderboard output path.")

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
        print("(no training command generated; see note above)")


def print_tuning_data(chat_data, eval_data) -> None:
    print("\nchat/eval data:")
    print(
        f"- chat_sft: {chat_data.status} | {chat_data.num_examples}/{chat_data.num_rows} usable rows "
        f"| avg_user_chars {chat_data.average_user_chars:.1f} "
        f"| avg_assistant_chars {chat_data.average_assistant_chars:.1f} "
        f"| curriculum {chat_data.curriculum_label}"
    )
    print(f"  {chat_data.path}: {chat_data.summary}")
    if chat_data.categories:
        print(
            f"  categories: {_format_counts(chat_data.categories)} "
            f"(entropy {chat_data.category_entropy:.2f}, normalized {chat_data.category_entropy_normalized:.2f})"
        )
    if chat_data.template_families:
        print(f"  template_families: {_format_counts(chat_data.template_families)}")
    if chat_data.answer_styles:
        print(f"  answer_styles: {_format_counts(chat_data.answer_styles)}")
    print(
        f"  duplicates: exact {chat_data.duplicate_user_prompts} "
        f"({chat_data.duplicate_user_rate * 100:.2f}%), "
        f"near {chat_data.near_duplicate_user_pairs}"
    )
    if chat_data.assistant_length_distribution.get("count"):
        lengths = chat_data.assistant_length_distribution
        print(
            f"  answer_lengths: avg_words {lengths['avg_words']:.1f}, "
            f"min/max_words {lengths['min_words']}/{lengths['max_words']}"
        )
    for warning in chat_data.quality_warnings[:3]:
        print(f"  warning: {warning}")
    for issue in chat_data.issues[:3]:
        print(f"  issue line {issue.line}: {issue.message}")
    print(
        f"- eval: {eval_data.status} | {eval_data.num_items}/{eval_data.num_rows} usable rows "
        f"| answerable {eval_data.answerable_items} "
        f"| unanswerable {eval_data.unanswerable_items} "
        f"| curriculum {eval_data.curriculum_label}"
    )
    print(
        f"  rules: include {eval_data.must_include_rules}, "
        f"include_any {eval_data.must_include_any_groups}, "
        f"forbidden {eval_data.must_not_include_rules}"
    )
    print(f"  {eval_data.path}: {eval_data.summary}")
    if eval_data.categories:
        print(
            f"  categories: {_format_counts(eval_data.categories)} "
            f"(entropy {eval_data.category_entropy:.2f}, normalized {eval_data.category_entropy_normalized:.2f})"
        )
    if eval_data.heldout_categories:
        print(f"  heldout_categories: {_format_counts(eval_data.heldout_categories)}")
    if eval_data.template_families:
        print(f"  template_families: {_format_counts(eval_data.template_families)}")
    print(
        f"  duplicates: exact {eval_data.duplicate_user_prompts} "
        f"({eval_data.duplicate_user_rate * 100:.2f}%), "
        f"near {eval_data.near_duplicate_user_pairs}"
    )
    if eval_data.answer_length_distribution.get("count"):
        lengths = eval_data.answer_length_distribution
        print(
            f"  answer_lengths: avg_words {lengths['avg_words']:.1f}, "
            f"min/max_words {lengths['min_words']}/{lengths['max_words']}"
        )
    if eval_data.splits:
        print(f"  splits: {_format_counts(eval_data.splits)}")
    for warning in eval_data.quality_warnings[:3]:
        print(f"  warning: {warning}")
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


def eval_starter_data(args: argparse.Namespace) -> int:
    if not args.input and not args.dataset_pack:
        raise SystemExit("data eval-starter requires --input or --dataset-pack")
    report = generate_eval_starter(
        args.input,
        args.out,
        dataset_pack=args.dataset_pack,
        max_items=args.max_items,
        seed=args.seed,
        force=args.force,
    )
    print(f"eval starter: {report.output_path}")
    print(f"input: {report.input_path}")
    print(f"documents: {report.num_documents}")
    print(f"candidate_sentences: {report.num_sentences}")
    print(f"rows: {report.num_rows}")
    print("categories:")
    for name, count in report.categories.items():
        print(f"- {name}: {count}")
    print("levels:")
    for name, count in report.levels.items():
        print(f"- {name}: {count}")
    print(f"report: {report.output_path.rsplit('.', 1)[0]}.md")
    return 0


def sft_starter_data(args: argparse.Namespace) -> int:
    if not args.input and not args.dataset_pack:
        raise SystemExit("data sft-starter requires --input or --dataset-pack")
    report = generate_sft_starter(
        args.input,
        args.out,
        dataset_pack=args.dataset_pack,
        max_items=args.max_items,
        seed=args.seed,
        force=args.force,
    )
    print(f"sft starter: {report.output_path}")
    print(f"input: {report.input_path}")
    print(f"documents: {report.num_documents}")
    print(f"candidate_sentences: {report.num_sentences}")
    print(f"rows: {report.num_rows}")
    print("categories:")
    for name, count in report.categories.items():
        print(f"- {name}: {count}")
    print(f"report: {report.output_path.rsplit('.', 1)[0]}.md")
    return 0


def benchmark_pack_data(args: argparse.Namespace) -> int:
    report = generate_benchmark_tuning_pack(
        dataset_pack=args.dataset_pack,
        chat_out=args.chat_out,
        eval_out=args.eval_out,
        sft_rows=args.sft_rows,
        eval_rows=args.eval_rows,
        seed=args.seed,
        source=args.source,
        profile=args.profile,
        skill_answer_style=args.skill_answer_style,
        force=args.force,
        promote_to_pack=not args.no_promote,
    )
    print(f"benchmark chat SFT: {report.chat_output_path}")
    print(f"benchmark eval: {report.eval_output_path}")
    print(f"sft_rows: {report.sft_rows}")
    print(f"eval_rows: {report.eval_rows}")
    print(f"source: {report.source_mode}")
    print(f"source_status: {report.source_status}")
    print(f"profile: {report.profile}")
    print(f"skill_answer_style: {report.skill_answer_style}")
    if report.fallback_reason:
        print(f"fallback_reason: {report.fallback_reason}")
    print(f"contamination: {report.contamination['status']}")
    print(f"promoted_to_pack: {report.promoted_to_pack}")
    print("chat categories:")
    for name, count in report.chat_categories.items():
        print(f"- {name}: {count}")
    print("eval categories:")
    for name, count in report.eval_categories.items():
        print(f"- {name}: {count}")
    print(f"report: {report.report_path}")
    print("\nnext:")
    print(f"PYTHONPATH=src python -m picochat.cli data preview --dataset-pack {report.dataset_pack}")
    return 0


def task_pack_data(args: argparse.Namespace) -> int:
    report = generate_task_mixture_pack(
        dataset_pack=args.dataset_pack,
        out_dir=args.out_dir,
        chat_out=args.chat_out,
        eval_out=args.eval_out,
        sft_rows=args.sft_rows,
        eval_rows=args.eval_rows,
        seed=args.seed,
        source=args.source,
        profile=args.profile,
        skill_answer_style=args.skill_answer_style,
        force=args.force,
        promote_to_pack=not args.no_promote,
    )
    print(f"task-mixture chat SFT: {report.chat_output_path}")
    print(f"task-mixture eval: {report.eval_output_path}")
    print(f"sft_rows: {report.sft_rows}")
    print(f"eval_rows: {report.eval_rows}")
    print(f"profile: {report.profile}")
    print(f"source: {report.source_mode}")
    print(f"source_status: {report.source_status}")
    if report.fallback_reason:
        print(f"fallback_reason: {report.fallback_reason}")
    print(f"contamination: {report.contamination['status']}")
    print(f"promoted_to_pack: {report.promoted_to_pack}")
    print("components:")
    for component in report.components:
        chat_count = report.chat_component_counts.get(component.name, 0)
        eval_count = report.eval_component_counts.get(component.name, 0)
        print(f"- {component.name}: train={chat_count}, eval={eval_count}, profile={component.benchmark_profile}")
    print("chat categories:")
    for name, count in report.chat_categories.items():
        print(f"- {name}: {count}")
    print("eval categories:")
    for name, count in report.eval_categories.items():
        print(f"- {name}: {count}")
    print(f"report: {report.report_path}")
    print("\nnext:")
    print(f"PYTHONPATH=src python -m picochat.cli data preview --dataset-pack {report.dataset_pack}")
    return 0


def slice_pack_data(args: argparse.Namespace) -> int:
    report = slice_tuning_pack(
        args.dataset_pack,
        args.out_dir,
        include_categories=parse_category_patterns(args.include_categories),
        exclude_categories=parse_category_patterns(args.exclude_categories),
        name=args.name,
        description=args.description,
        force=args.force,
    )
    print(f"sliced dataset pack: {report.dataset_pack}")
    print(f"source dataset pack: {report.source_dataset_pack}")
    print(f"chat rows: {report.chat_rows_out}/{report.chat_rows_in}")
    print(f"eval rows: {report.eval_rows_out}/{report.eval_rows_in}")
    print(f"chat status: {report.sft_status}")
    print(f"eval status: {report.eval_status}")
    print("chat categories:")
    for name, count in report.chat_categories.items():
        print(f"- {name}: {count}")
    print("eval categories:")
    for name, count in report.eval_categories.items():
        print(f"- {name}: {count}")
    print(f"report: {report.report_path}")
    print("\nnext:")
    print(f"PYTHONPATH=src python -m picochat.cli data preview --dataset-pack {report.dataset_pack}")
    return 0


def stage_pack_data(args: argparse.Namespace) -> int:
    report = stage_tuning_pack(
        args.dataset_pack,
        args.out_dir,
        stages=parse_stage_names(args.stages),
        force=args.force,
    )
    print(f"staged tuning pack: {report.out_dir}")
    print(f"source dataset pack: {report.source_dataset_pack}")
    print(f"stages: {len(report.stages)}")
    for stage in report.stages:
        print(
            f"- {Path(stage.out_dir).name}: "
            f"chat {stage.chat_rows_out}/{stage.chat_rows_in}, "
            f"eval {stage.eval_rows_out}/{stage.eval_rows_in}, "
            f"sft {stage.sft_status}, eval {stage.eval_status}"
        )
        print(f"  dataset_pack: {stage.dataset_pack}")
    print(f"report: {report.report_path}")
    print("\nnext:")
    for stage in report.stages:
        print(f"PYTHONPATH=src python -m picochat.cli data preview --dataset-pack {stage.dataset_pack}")
    return 0


def skills_corpus_data(args: argparse.Namespace) -> int:
    report = generate_skills_corpus(
        args.out,
        math_rows=args.math_rows,
        spelling_rows=args.spelling_rows,
        choice_rows=args.choice_rows,
        seed=args.seed,
        force=args.force,
        base_corpus=args.base_corpus,
        recipe_out=args.recipe_out,
        documents_dir=args.documents_dir,
        rows_per_shard=args.rows_per_shard,
    )
    print(f"skills corpus: {report.output_path}")
    if report.documents_dir:
        print(f"documents_dir: {report.documents_dir}")
        print(f"shards: {report.num_shards}")
        print(f"rows_per_shard: {report.rows_per_shard}")
    print(f"rows: {report.total_rows}")
    print(f"characters: {report.characters_written}")
    print("categories:")
    for name, count in report.categories.items():
        print(f"- {name}: {count}")
    print(f"report: {report.report_path}")
    if report.recipe_path:
        print(f"recipe: {report.recipe_path}")
        print("\nnext:")
        print(f"PYTHONPATH=src python -m picochat.cli data preview --recipe {report.recipe_path}")
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
        document_shard_rows=args.document_shard_rows,
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
    print(f"document_shard_rows: {report.document_shard_rows}")
    print(f"document_files_written: {report.document_files_written}")
    print(f"report: {report.report_path}")
    print("\nnext:")
    print(f"PYTHONPATH=src python -m picochat.cli data preview --input {report.documents_dir or report.out_path}")
    return 0


def climbmix_import_data(args: argparse.Namespace) -> int:
    if args.shards < 1 or args.shards > CLIMBMIX_MAX_SHARD + 1:
        raise SystemExit(f"--shards must be between 1 and {CLIMBMIX_MAX_SHARD + 1}")
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output folder already exists: {out_dir}; use --force to overwrite import artifacts")

    data_files = tuple(f"shard_{index:05d}.parquet" for index in range(args.shards))
    corpus_path = out_dir / "corpus.txt"
    documents_dir = out_dir / "documents"
    document_shard_rows = (
        args.document_shard_rows
        if args.document_shard_rows is not None
        else _default_climbmix_document_shard_rows(args.max_rows)
    )
    report = import_hf_dataset(HFImportConfig(
        dataset=CLIMBMIX_DATASET,
        split="train",
        text_column="text",
        out_path=str(corpus_path),
        report_path=str(out_dir / "hf_import_report.json"),
        documents_dir=str(documents_dir),
        document_shard_rows=document_shard_rows,
        max_rows=args.max_rows,
        min_chars=args.min_chars,
        streaming=not args.no_streaming,
        data_files=data_files,
    ))
    pack_report = init_dataset_pack(
        out_dir=str(out_dir),
        corpus_path=str(documents_dir),
        name=f"ClimbMix sample ({args.shards} shard{'s' if args.shards != 1 else ''})",
        description=(
            "Nanochat-compatible public ClimbMix data sample from "
            "karpathy/climbmix-400b-shuffle, repackaged from nvidia/Nemotron-ClimbMix."
        ),
        force=args.force,
    )
    print(f"imported dataset: {report.dataset}")
    print(f"source: nvidia/Nemotron-ClimbMix via {CLIMBMIX_DATASET}")
    print(f"shards: {', '.join(data_files)}")
    print(f"rows_seen: {report.rows_seen}")
    print(f"rows_written: {report.rows_written}")
    print(f"characters_written: {report.characters_written}")
    print(f"dataset_pack: {pack_report.dataset_pack}")
    print(f"documents_dir: {report.documents_dir}")
    print(f"document_shard_rows: {report.document_shard_rows}")
    print(f"document_files_written: {report.document_files_written}")
    print(f"report: {report.report_path}")
    print("\nnext:")
    print(f"PYTHONPATH=src python -m picochat.cli data preview --dataset-pack {pack_report.dataset_pack}")
    return 0


def _default_climbmix_document_shard_rows(max_rows: int) -> int:
    if max_rows >= CLIMBMIX_LARGE_IMPORT_ROWS:
        return CLIMBMIX_LARGE_IMPORT_DOCUMENT_SHARD_ROWS
    return 1


def honesty_data(args: argparse.Namespace) -> int:
    corpus = args.corpus
    chat_input = args.chat_input
    eval_input = args.eval_input
    if args.dataset_pack:
        if corpus is not None or args.chat_input != DEFAULT_CHAT_INPUT or args.eval_input != DEFAULT_EVAL_INPUT:
            raise SystemExit("data honesty --dataset-pack cannot be combined with explicit corpus/chat/eval paths")
        pack = load_dataset_pack(args.dataset_pack)
        corpus = pack.corpus_input
        chat_input = pack.chat_input
        eval_input = pack.eval_input

    report = inspect_data_honesty(
        corpus_path=corpus,
        chat_input=chat_input,
        eval_input=eval_input,
        near_threshold=args.near_threshold,
        ngram_size=args.ngram_size,
    )
    print(f"honesty: {report.status}")
    print(f"summary: {report.summary}")
    print(f"sft_examples: {report.num_sft_examples}")
    print(f"eval_items: {report.num_eval_items}")
    print(f"exact_sft_prompt_leaks: {report.exact_prompt_leaks}")
    print(f"near_sft_prompt_leaks: {report.near_prompt_leaks}")
    print(f"eval_prompts_found_in_corpus: {report.corpus_prompt_hits}")
    print(f"sft_support_phrase_hits: {report.sft_support_phrase_hits}")
    print(f"corpus_support_phrase_hits: {report.corpus_support_phrase_hits}")
    print(f"duplicate_eval_prompts: {report.duplicate_eval_prompts}")
    print(f"max_sft_prompt_similarity: {report.max_sft_prompt_similarity:.4f}")
    print("contamination_matrix:")
    for pair in report.contamination_matrix.get("pairs", []):
        checked = (
            "checked"
            if pair.get("checked")
            else f"not_checked({pair.get('reason', 'unknown')})"
        )
        print(
            f"- {pair.get('name')}: risk={pair.get('risk')} {checked} "
            f"exact={pair.get('exact_text_hits', 0)} near={pair.get('near_text_hits', 0)} "
            f"max_ngram_overlap={float(pair.get('max_ngram_overlap_rate', 0.0)):.4f} "
            f"longest={pair.get('max_longest_overlap_tokens', 0)}"
        )
    for finding in report.findings[:8]:
        message = (
            f"- {finding.severity} {finding.kind} eval_line={finding.eval_line} "
            f"matched={finding.matched_source or 'none'}"
        )
        if finding.similarity is not None:
            message += f" similarity={finding.similarity:.4f}"
        print(message)
    if len(report.findings) > 8:
        print(f"- ... {len(report.findings) - 8} more finding(s)")
    if args.out_dir:
        json_path, markdown_path = write_data_honesty_report(report, args.out_dir)
        print(f"json_report: {json_path}")
        print(f"markdown_report: {markdown_path}")
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
        bpe_pretokenizer=args.bpe_pretokenizer,
    )
    tokenizer.save(output_path)

    stats = tokenizer.stats()
    print(f"trained tokenizer: {output_path}")
    print(f"type: {stats.tokenizer_type}")
    print(f"vocab_size: {stats.vocab_size}")
    print(f"text_tokens: {stats.num_text_tokens}")
    print(f"special_tokens: {stats.num_special_tokens}")
    if hasattr(tokenizer, "pretokenizer"):
        print(f"pretokenizer: {tokenizer.pretokenizer}")
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
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        norm_type=args.norm_type,
        position_encoding=args.position_encoding,
        activation=args.activation,
        tie_embeddings=args.tie_embeddings,
        qk_norm=args.qk_norm,
        attn_backend=args.attn_backend,
        parallel_residual=args.parallel_residual,
        linear_bias=args.linear_bias,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        val_fraction=args.val_fraction,
        eval_batches=args.eval_batches,
        sample_tokens=args.sample_tokens,
        split_mode=args.split_mode,
        corpus_manifest_path=args.corpus_manifest,
        dataset_mode=args.dataset_mode,
        shard_token_size=args.shard_token_size,
        shard_cache_size=args.shard_cache_size,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        max_minutes=args.max_minutes,
        canary_count=args.canary_count,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        weight_decay_decay=args.weight_decay_decay,
        muon_learning_rate=args.muon_learning_rate,
        muon_momentum_schedule=args.muon_momentum_schedule,
        ema_decay=args.ema_decay,
        logit_softcap=args.logit_softcap,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        resume_from=args.resume_from,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp=args.ddp,
        loss_spike_rollback=args.loss_spike_rollback,
        loss_spike_threshold=args.loss_spike_threshold,
        loss_spike_lr_decay=args.loss_spike_lr_decay,
        loss_spike_min_lr_scale=args.loss_spike_min_lr_scale,
        loss_spike_snapshot_every=args.loss_spike_snapshot_every,
    )
    report = train_base(config)
    if report.get("config", {}).get("artifacts_written", True):
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
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        sampling=args.sampling,
        grad_accum_steps=args.grad_accum_steps,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        weight_decay_decay=args.weight_decay_decay,
        muon_learning_rate=args.muon_learning_rate,
        muon_momentum_schedule=args.muon_momentum_schedule,
        ema_decay=args.ema_decay,
        packing=args.packing,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        resume_from=args.resume_from,
        ddp=args.ddp,
        loss_spike_rollback=args.loss_spike_rollback,
        loss_spike_threshold=args.loss_spike_threshold,
        loss_spike_lr_decay=args.loss_spike_lr_decay,
        loss_spike_min_lr_scale=args.loss_spike_min_lr_scale,
        loss_spike_snapshot_every=args.loss_spike_snapshot_every,
        peft=args.peft,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=parse_lora_targets(args.lora_targets),
    )
    report = train_sft(config)
    if report.get("config", {}).get("artifacts_written", True):
        print(f"saved sft checkpoint: {report['checkpoint']}")
        print(f"sample: {report['sample']!r}")
    return 0


def run_train_sft_sweep(args: argparse.Namespace) -> int:
    input_path = args.input
    eval_input_path = args.eval_input
    support_corpus_path = args.support_corpus
    if args.dataset_pack:
        pack = load_dataset_pack(args.dataset_pack)
        input_path = input_path or pack.chat_input
        eval_input_path = eval_input_path or pack.eval_input
        support_corpus_path = support_corpus_path or pack.corpus_input
    if not input_path:
        raise SystemExit("train sft-sweep requires --input or --dataset-pack")

    report = run_sft_sweep(SFTSweepConfig(
        input_path=input_path,
        eval_input_path=eval_input_path,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        support_corpus_path=support_corpus_path,
        learning_rates=_parse_float_csv(args.learning_rates, "learning-rates"),
        step_counts=_parse_int_csv(args.steps, "steps"),
        samplings=_parse_sampling_csv(args.samplings),
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        eval_max_new_tokens=args.eval_max_new_tokens,
        fit_max_rows=args.fit_max_rows,
        val_fraction=args.val_fraction,
        eval_batches=args.eval_batches,
        sample_prompt=args.sample_prompt,
        sample_tokens=args.sample_tokens,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        max_minutes=args.max_minutes,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        weight_decay_decay=args.weight_decay_decay,
        muon_learning_rate=args.muon_learning_rate,
        muon_momentum_schedule=args.muon_momentum_schedule,
        ema_decay=args.ema_decay,
        packing=args.packing,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        eval_log_every=args.eval_log_every,
        peft=args.peft,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=parse_lora_targets(args.lora_targets),
    ))
    print(f"sft sweep report: {Path(args.out_dir) / 'sft_sweep.md'}")
    best_fit = report.get("best_sft_fit") or {}
    best_eval = report.get("best_eval") or {}
    if best_fit:
        print(
            f"best sft fit: {best_fit['candidate']} "
            f"({best_fit['sft_fit_pass_rate'] * 100:.2f}%)"
        )
    if best_eval:
        print(
            f"best eval: {best_eval['candidate']} "
            f"({best_eval['eval_pass_rate'] * 100:.2f}%)"
        )
    return 0


def _parse_float_csv(value: str, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit(f"--{label} must be a comma-separated list of numbers") from error
    if not values:
        raise SystemExit(f"--{label} must include at least one number")
    return values


def _parse_int_csv(value: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit(f"--{label} must be a comma-separated list of integers") from error
    if not values:
        raise SystemExit(f"--{label} must include at least one integer")
    return values


def _parse_sampling_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise SystemExit("--samplings must include at least one sampling mode")
    unsupported = [item for item in values if item not in SFT_SAMPLING_MODES]
    if unsupported:
        raise SystemExit(f"unsupported --samplings value(s): {', '.join(unsupported)}")
    return values


def run_generate(args: argparse.Namespace) -> int:
    top_k = None if args.top_k <= 0 else args.top_k
    text = generate_text(GenerateConfig(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=args.device,
        use_kv_cache=not args.no_kv_cache,
    ))
    print(text)
    return 0


def run_export_hf(args: argparse.Namespace) -> int:
    report = export_hf_checkpoint(HFExportConfig(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        model_name=args.model_name,
        base_model=not args.fine_tuned,
        license_name=args.license,
        dataset_summary=args.dataset_summary,
        eval_summary=args.eval_summary,
        dynamic_int8=args.dynamic_int8,
        safetensors=not args.no_safetensors,
        transformers_adapter=not args.no_transformers_adapter,
    ))
    print(f"exported: {report['out_dir']}")
    print(f"manifest: {report['manifest']}")
    print(f"model_card: {report['model_card']}")
    return 0


def run_bundle(args: argparse.Namespace) -> int:
    report = create_run_bundle(RunBundleConfig(
        run_dir=args.run_dir,
        out_path=args.out,
        logs_dir=args.logs_dir,
        include_corpus=args.include_corpus,
        include_token_shards=args.include_token_shards,
        strict=args.strict,
    ))
    print(f"bundle: {report['bundle']}")
    print(f"manifest_json: {report['manifest_json']}")
    print(f"manifest_md: {report['manifest_md']}")
    if report.get("excluded_large"):
        print("excluded_large: " + ", ".join(report["excluded_large"]))
    if report.get("missing_expected"):
        print("missing_optional: " + ", ".join(report["missing_expected"]))
    return 0


def run_inspect_bundle(args: argparse.Namespace) -> int:
    report = inspect_run_bundle(args.bundle)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(bundle_inspection_markdown(report))
    return 0


def run_sanity_preh100(args: argparse.Namespace) -> int:
    report = run_preh100_sanity(PreH100SanityConfig(
        out_dir=args.out_dir,
        device=args.device,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        attn_backend=args.attn_backend,
        include_compile=args.include_compile,
    ))
    print(f"sanity: {report['status']}")
    print(f"json_report: {report['report_path']}")
    print(f"markdown_report: {report['markdown_path']}")
    for check in report["checks"]:
        detail = check.get("detail") or check.get("error") or ""
        print(f"- {check['name']}: {check['status']} {detail}".rstrip())
    return 1 if report["status"] == "failed" else 0


def run_chat(args: argparse.Namespace) -> int:
    top_k = None if args.top_k <= 0 else args.top_k
    return chat_loop(ChatConfig(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=args.device,
        use_kv_cache=not args.no_kv_cache,
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
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        case_sensitive=args.case_sensitive,
        support_corpus_path=args.support_corpus,
        corpus_support_threshold=args.corpus_support_threshold,
        ci_bootstrap_samples=args.ci_bootstrap_samples,
        ci_confidence=args.ci_confidence,
        log_every=args.log_every,
    ))
    summary = report["summary"]
    print(
        f"chat eval: {summary['num_passed']}/{summary['num_examples']} passed "
        f"({summary['pass_rate'] * 100:.2f}%)"
    )
    print(f"saved eval report: {args.out_dir}")
    return 0


def run_eval_external(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    eval_input = out_dir / "external_eval.jsonl"
    convert_report = convert_external_benchmark(ExternalBenchmarkConvertConfig(
        input_path=args.input,
        output_path=str(eval_input),
        source_format=args.format,
        benchmark_name=args.benchmark_name,
        split=args.split,
        max_rows=None if args.max_rows <= 0 else args.max_rows,
        seed=args.seed,
        shuffle=args.shuffle,
    ))
    top_k = None if args.top_k <= 0 else args.top_k
    report = run_chat_eval(ChatEvalConfig(
        input_path=str(eval_input),
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        ci_bootstrap_samples=args.ci_bootstrap_samples,
        ci_confidence=args.ci_confidence,
        log_every=args.log_every,
    ))
    summary = report["summary"]
    choice_accuracy = summary.get("choice_accuracy")
    choice_text = "n/a" if choice_accuracy is None else f"{choice_accuracy * 100:.2f}%"
    print(f"converted external eval: {convert_report['output_path']}")
    print(
        f"external eval: {summary['num_passed']}/{summary['num_examples']} passed "
        f"({summary['pass_rate'] * 100:.2f}%), choice accuracy {choice_text}"
    )
    print(f"saved eval report: {args.out_dir}")
    return 0


def run_eval_sft_fit(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    fit_input = out_dir / "sft_fit_eval.jsonl"
    fit_report = write_sft_fit_eval(
        args.input,
        fit_input,
        max_rows=None if args.max_rows <= 0 else args.max_rows,
    )
    top_k = None if args.top_k <= 0 else args.top_k
    report = run_chat_eval(ChatEvalConfig(
        input_path=str(fit_input),
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        case_sensitive=args.case_sensitive,
        support_corpus_path=args.support_corpus,
        corpus_support_threshold=args.corpus_support_threshold,
        ci_bootstrap_samples=args.ci_bootstrap_samples,
        ci_confidence=args.ci_confidence,
        log_every=args.log_every,
    ))
    summary = report["summary"]
    print(
        f"sft fit eval: {summary['num_passed']}/{summary['num_examples']} passed "
        f"({summary['pass_rate'] * 100:.2f}%)"
    )
    print(f"converted SFT rows: {fit_report['output_path']}")
    print(f"saved fit report: {args.out_dir}")
    return 0


def run_scale_plan(args: argparse.Namespace) -> int:
    dataset_tokens = parse_count(args.dataset_tokens) if args.dataset_tokens else None
    plan = plan_scale(
        target_parameters=parse_count(args.target_params),
        dataset_tokens=dataset_tokens,
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        context_size=args.context_size,
        world_size=args.world_size,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        target_param_data_ratio=args.target_param_data_ratio,
        attn_backend=args.attn_backend,
        long_run_gate_profile=args.long_run_gate_profile,
    )
    markdown = render_scale_plan_markdown(plan)
    print(markdown, end="")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"saved scale plan: {out_path}")
    return 0


def _resolve_tiny_value(args: argparse.Namespace, defaults: TinyRunConfig, field: str):
    value = getattr(args, field)
    if value is not None:
        return value
    if args.scale != "custom":
        preset = RUN_SCALES[args.scale].tiny_run_values()
        if field in preset:
            return preset[field]
    return getattr(defaults, field)


def _resolve_tiny_bool(args: argparse.Namespace, defaults: TinyRunConfig, field: str) -> bool:
    if bool(getattr(args, field)):
        return True
    if args.scale != "custom":
        preset = RUN_SCALES[args.scale].tiny_run_values()
        if field in preset:
            return bool(preset[field])
    return bool(getattr(defaults, field))


def run_tiny_command(args: argparse.Namespace) -> int:
    config = _tiny_config_from_args(args)
    if args.n_seeds < 1:
        raise SystemExit("--n-seeds must be at least 1")
    if args.preflight_only:
        preview = preview_corpus_sources(
            None if config.dataset_pack else config.corpus_input,
            None if config.dataset_pack else config.corpus_recipe,
            chat_input=None if config.dataset_pack else config.chat_input,
            eval_input=None if config.dataset_pack else config.eval_input,
            dataset_pack=config.dataset_pack,
            preview_chars=0,
            min_quality_score=config.min_quality_score,
        )
        report = assess_run_preflight(config, preview)
        print(preflight_markdown(report))
        return 1 if report.status == "blocked" else 0
    if args.n_seeds > 1:
        summary = run_tiny_multiseed(config, args.n_seeds)
        stats = summary["aggregate"]["eval_pass_rate"]
        if stats.get("mean") is None:
            print(f"multi-seed tiny run: n={stats['n']} eval pass unavailable")
        else:
            print(
                f"multi-seed tiny run: n={stats['n']} eval pass "
                f"mean {stats['mean'] * 100:.2f}% std {stats['std'] * 100:.2f}%"
            )
        return 0
    summary = run_tiny(config)
    if summary.get("status") == "ddp_worker_complete":
        return 0
    if "eval" not in summary:
        print("tiny run: completed without eval summary")
        return 0
    print(
        f"tiny run: {summary['eval']['num_passed']}/{summary['eval']['num_examples']} "
        f"passed ({summary['eval']['pass_rate'] * 100:.2f}%)"
    )
    return 0


def _tiny_config_from_args(args: argparse.Namespace) -> TinyRunConfig:
    defaults = TinyRunConfig(out_dir=args.out_dir)
    return TinyRunConfig(
        out_dir=args.out_dir,
        scale=args.scale,
        dataset_pack=args.dataset_pack,
        corpus_input=args.corpus_input,
        corpus_recipe=args.corpus_recipe,
        chat_input=args.chat_input,
        eval_input=args.eval_input,
        external_eval_inputs=tuple(args.external_eval or ()),
        external_eval_format=_resolve_tiny_value(args, defaults, "external_eval_format"),
        external_eval_max_rows=_resolve_tiny_value(args, defaults, "external_eval_max_rows"),
        external_eval_shuffle=bool(args.external_eval_shuffle),
        external_eval_max_new_tokens=_resolve_tiny_value(args, defaults, "external_eval_max_new_tokens"),
        context_size=_resolve_tiny_value(args, defaults, "context_size"),
        n_embd=_resolve_tiny_value(args, defaults, "n_embd"),
        n_head=_resolve_tiny_value(args, defaults, "n_head"),
        n_kv_head=_resolve_tiny_value(args, defaults, "n_kv_head"),
        n_layer=_resolve_tiny_value(args, defaults, "n_layer"),
        dropout=_resolve_tiny_value(args, defaults, "dropout"),
        norm_type=_resolve_tiny_value(args, defaults, "norm_type"),
        position_encoding=_resolve_tiny_value(args, defaults, "position_encoding"),
        activation=_resolve_tiny_value(args, defaults, "activation"),
        tie_embeddings=_resolve_tiny_bool(args, defaults, "tie_embeddings"),
        qk_norm=_resolve_tiny_bool(args, defaults, "qk_norm"),
        attn_backend=_resolve_tiny_value(args, defaults, "attn_backend"),
        parallel_residual=_resolve_tiny_bool(args, defaults, "parallel_residual"),
        linear_bias=_resolve_tiny_value(args, defaults, "linear_bias"),
        base_steps=_resolve_tiny_value(args, defaults, "base_steps"),
        sft_steps=_resolve_tiny_value(args, defaults, "sft_steps"),
        base_batch_size=_resolve_tiny_value(args, defaults, "base_batch_size"),
        sft_batch_size=_resolve_tiny_value(args, defaults, "sft_batch_size"),
        base_learning_rate=_resolve_tiny_value(args, defaults, "base_learning_rate"),
        sft_learning_rate=_resolve_tiny_value(args, defaults, "sft_learning_rate"),
        seed=args.seed,
        device=args.device,
        eval_max_new_tokens=_resolve_tiny_value(args, defaults, "eval_max_new_tokens"),
        min_quality_score=args.min_score,
        split_mode=args.split_mode,
        base_dataset_mode=_resolve_tiny_value(args, defaults, "base_dataset_mode"),
        base_shard_token_size=_resolve_tiny_value(args, defaults, "base_shard_token_size"),
        base_shard_cache_size=_resolve_tiny_value(args, defaults, "base_shard_cache_size"),
        tokenizer_type=_resolve_tiny_value(args, defaults, "tokenizer_type"),
        tokenizer_vocab_size=_resolve_tiny_value(args, defaults, "tokenizer_vocab_size"),
        tokenizer_min_freq=_resolve_tiny_value(args, defaults, "tokenizer_min_freq"),
        bpe_pretokenizer=_resolve_tiny_value(args, defaults, "bpe_pretokenizer"),
        base_early_stop_patience=_resolve_tiny_value(args, defaults, "base_early_stop_patience"),
        sft_early_stop_patience=_resolve_tiny_value(args, defaults, "sft_early_stop_patience"),
        early_stop_min_delta=_resolve_tiny_value(args, defaults, "early_stop_min_delta"),
        base_max_minutes=args.base_max_minutes,
        sft_max_minutes=args.sft_max_minutes,
        canary_count=_resolve_tiny_value(args, defaults, "canary_count"),
        allow_leaky_eval=args.allow_leaky_eval,
        base_lr_warmup_steps=_resolve_tiny_value(args, defaults, "base_lr_warmup_steps"),
        sft_lr_warmup_steps=_resolve_tiny_value(args, defaults, "sft_lr_warmup_steps"),
        base_lr_decay=_resolve_tiny_value(args, defaults, "base_lr_decay"),
        sft_lr_decay=_resolve_tiny_value(args, defaults, "sft_lr_decay"),
        base_min_lr_ratio=_resolve_tiny_value(args, defaults, "base_min_lr_ratio"),
        sft_min_lr_ratio=_resolve_tiny_value(args, defaults, "sft_min_lr_ratio"),
        base_grad_clip=_resolve_tiny_value(args, defaults, "base_grad_clip"),
        sft_grad_clip=_resolve_tiny_value(args, defaults, "sft_grad_clip"),
        base_grad_accum_steps=_resolve_tiny_value(args, defaults, "base_grad_accum_steps"),
        sft_grad_accum_steps=_resolve_tiny_value(args, defaults, "sft_grad_accum_steps"),
        base_optimizer=_resolve_tiny_value(args, defaults, "base_optimizer"),
        sft_optimizer=_resolve_tiny_value(args, defaults, "sft_optimizer"),
        base_weight_decay=_resolve_tiny_value(args, defaults, "base_weight_decay"),
        sft_weight_decay=_resolve_tiny_value(args, defaults, "sft_weight_decay"),
        base_weight_decay_decay=_resolve_tiny_value(args, defaults, "base_weight_decay_decay"),
        sft_weight_decay_decay=_resolve_tiny_value(args, defaults, "sft_weight_decay_decay"),
        base_muon_learning_rate=_resolve_tiny_value(args, defaults, "base_muon_learning_rate"),
        sft_muon_learning_rate=_resolve_tiny_value(args, defaults, "sft_muon_learning_rate"),
        base_muon_momentum_schedule=_resolve_tiny_value(args, defaults, "base_muon_momentum_schedule"),
        sft_muon_momentum_schedule=_resolve_tiny_value(args, defaults, "sft_muon_momentum_schedule"),
        base_ema_decay=_resolve_tiny_value(args, defaults, "base_ema_decay"),
        sft_ema_decay=_resolve_tiny_value(args, defaults, "sft_ema_decay"),
        sft_sampling=_resolve_tiny_value(args, defaults, "sft_sampling"),
        sft_packing=_resolve_tiny_value(args, defaults, "sft_packing"),
        sft_fit_max_rows=_resolve_tiny_value(args, defaults, "sft_fit_max_rows"),
        sft_peft=_resolve_tiny_value(args, defaults, "sft_peft"),
        sft_lora_rank=_resolve_tiny_value(args, defaults, "sft_lora_rank"),
        sft_lora_alpha=_resolve_tiny_value(args, defaults, "sft_lora_alpha"),
        sft_lora_dropout=_resolve_tiny_value(args, defaults, "sft_lora_dropout"),
        sft_lora_targets=parse_lora_targets(_resolve_tiny_value(args, defaults, "sft_lora_targets")),
        allow_default_tuning_data=args.allow_default_tuning_data,
        base_resume_from=args.base_resume_from,
        sft_resume_from=args.sft_resume_from,
        logit_softcap=_resolve_tiny_value(args, defaults, "logit_softcap"),
        precision=_resolve_tiny_value(args, defaults, "precision"),
        matmul_precision=_resolve_tiny_value(args, defaults, "matmul_precision"),
        torch_compile=_resolve_tiny_bool(args, defaults, "torch_compile"),
        torch_compile_mode=_resolve_tiny_value(args, defaults, "torch_compile_mode"),
        gradient_checkpointing=_resolve_tiny_bool(args, defaults, "gradient_checkpointing"),
        ddp=_resolve_tiny_bool(args, defaults, "ddp"),
        ddp_world_size=args.ddp_world_size if args.ddp_world_size is not None else defaults.ddp_world_size,
        allow_unsafe_long_run=args.allow_unsafe_long_run,
        target_param_data_ratio=_resolve_tiny_value(args, defaults, "target_param_data_ratio"),
        auto_lr_scaling=_resolve_tiny_bool(args, defaults, "auto_lr_scaling"),
        loss_spike_rollback=_resolve_tiny_bool(args, defaults, "loss_spike_rollback"),
        loss_spike_threshold=_resolve_tiny_value(args, defaults, "loss_spike_threshold"),
        loss_spike_lr_decay=_resolve_tiny_value(args, defaults, "loss_spike_lr_decay"),
        loss_spike_min_lr_scale=_resolve_tiny_value(args, defaults, "loss_spike_min_lr_scale"),
        loss_spike_snapshot_every=_resolve_tiny_value(args, defaults, "loss_spike_snapshot_every"),
        long_run_gate_profile=_resolve_tiny_value(args, defaults, "long_run_gate_profile"),
    )


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


def run_leaderboard(args: argparse.Namespace) -> int:
    leaderboard = build_benchmark_leaderboard(args.runs)
    print(leaderboard_table(leaderboard))
    print(f"\nBest benchmark run: {leaderboard['best_run']}")
    if args.out:
        write_leaderboard_report(leaderboard, args.out)
        print(f"saved leaderboard report: {args.out}")
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

    if args.command == "data" and args.data_command == "eval-starter":
        return eval_starter_data(args)

    if args.command == "data" and args.data_command == "sft-starter":
        return sft_starter_data(args)

    if args.command == "data" and args.data_command == "benchmark-pack":
        return benchmark_pack_data(args)

    if args.command == "data" and args.data_command == "task-pack":
        return task_pack_data(args)

    if args.command == "data" and args.data_command == "slice-pack":
        return slice_pack_data(args)

    if args.command == "data" and args.data_command == "stage-pack":
        return stage_pack_data(args)

    if args.command == "data" and args.data_command == "skills-corpus":
        return skills_corpus_data(args)

    if args.command == "data" and args.data_command == "hf-import":
        return hf_import_data(args)

    if args.command == "data" and args.data_command == "climbmix-import":
        return climbmix_import_data(args)

    if args.command == "data" and args.data_command == "honesty":
        return honesty_data(args)

    if args.command == "tok" and args.tok_command == "train":
        return train_tokenizer(args)

    if args.command == "batch" and args.batch_command == "inspect":
        return inspect_batches(args)

    if args.command == "train" and args.train_command == "base":
        return run_train_base(args)

    if args.command == "train" and args.train_command == "sft":
        return run_train_sft(args)

    if args.command == "train" and args.train_command == "sft-sweep":
        return run_train_sft_sweep(args)

    if args.command == "generate":
        return run_generate(args)

    if args.command == "export" and args.export_command == "hf":
        return run_export_hf(args)

    if args.command == "sanity" and args.sanity_command == "preh100":
        return run_sanity_preh100(args)

    if args.command == "chat":
        return run_chat(args)

    if args.command == "eval" and args.eval_command == "chat":
        return run_eval_chat(args)

    if args.command == "eval" and args.eval_command == "external":
        return run_eval_external(args)

    if args.command == "eval" and args.eval_command == "sft-fit":
        return run_eval_sft_fit(args)

    if args.command == "scale" and args.scale_command == "plan":
        return run_scale_plan(args)

    if args.command == "run" and args.run_command == "tiny":
        return run_tiny_command(args)

    if args.command == "run" and args.run_command == "bundle":
        return run_bundle(args)

    if args.command == "run" and args.run_command == "inspect-bundle":
        return run_inspect_bundle(args)

    if args.command == "compare":
        return run_compare(args)

    if args.command == "leaderboard":
        return run_leaderboard(args)

    if args.command == "web":
        return run_web(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
