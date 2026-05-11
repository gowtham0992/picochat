"""Local web dashboard for inspecting Picochat runs."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.resources
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

from picochat.compare import compare_runs
from picochat.data import DEFAULT_CHAT_INPUT, DEFAULT_EVAL_INPUT, preview_corpus_sources
from picochat.dataset_pack import init_dataset_pack, load_dataset_pack
from picochat.eval_starter import generate_eval_starter
from picochat.generate import GenerateConfig, generate_text_with_trace
from picochat.optim import LR_DECAYS
from picochat.scales import RUN_SCALES
from picochat.sft import SFT_SAMPLING_MODES
from picochat.tokenizer import TOKENIZER_TYPES
from picochat.tuning_data import inspect_chat_eval_data, inspect_chat_sft_data

_RUN_JOBS: dict[str, dict] = {}
_RUN_JOBS_LOCK = threading.Lock()
RUN_PRESETS = {
    "smoke": {
        "label": "Smoke",
        "description": "Fast CPU sanity check for UI and data wiring.",
        "context_size": 64,
        "base_steps": 40,
        "sft_steps": 60,
        "base_batch_size": 4,
        "sft_batch_size": 4,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 1,
        "tokenizer_type": "char",
        "tokenizer_vocab_size": None,
        "tokenizer_min_freq": 1,
        "base_lr_warmup_steps": 0,
        "sft_lr_warmup_steps": 0,
        "base_lr_decay": "none",
        "sft_lr_decay": "none",
        "base_min_lr_ratio": 1.0,
        "sft_min_lr_ratio": 1.0,
        "base_grad_clip": 0.0,
        "sft_grad_clip": 0.0,
        "sft_sampling": "uniform",
        "base_early_stop_patience": 4,
        "sft_early_stop_patience": 4,
        "eval_max_new_tokens": 80,
    },
    "tiny": {
        "label": "Tiny",
        "description": "Default educational tiny run.",
        "context_size": 128,
        "base_steps": 300,
        "sft_steps": 600,
        "base_batch_size": 8,
        "sft_batch_size": 7,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "tokenizer_type": "char",
        "tokenizer_vocab_size": None,
        "tokenizer_min_freq": 1,
        "base_lr_warmup_steps": 0,
        "sft_lr_warmup_steps": 0,
        "base_lr_decay": "none",
        "sft_lr_decay": "none",
        "base_min_lr_ratio": 1.0,
        "sft_min_lr_ratio": 1.0,
        "base_grad_clip": 0.0,
        "sft_grad_clip": 0.0,
        "sft_sampling": "uniform",
        "base_early_stop_patience": 3,
        "sft_early_stop_patience": 4,
        "eval_max_new_tokens": 120,
    },
    "small-local": {
        "label": "Small Local",
        "description": "Still local, but slower; use after the workflow is proven.",
        "context_size": 128,
        "base_steps": 800,
        "sft_steps": 1200,
        "base_batch_size": 8,
        "sft_batch_size": 8,
        "n_embd": 96,
        "n_head": 4,
        "n_layer": 3,
        "tokenizer_type": "bpe",
        "tokenizer_vocab_size": 512,
        "tokenizer_min_freq": 2,
        "base_lr_warmup_steps": 200,
        "sft_lr_warmup_steps": 50,
        "base_lr_decay": "cosine",
        "sft_lr_decay": "cosine",
        "base_min_lr_ratio": 0.1,
        "sft_min_lr_ratio": 0.1,
        "base_grad_clip": 1.0,
        "sft_grad_clip": 1.0,
        "sft_sampling": "category_balanced",
        "base_early_stop_patience": 3,
        "sft_early_stop_patience": 4,
        "eval_max_new_tokens": 120,
    },
    "small": RUN_SCALES["small"].to_dict(),
    "medium": RUN_SCALES["medium"].to_dict(),
}


@dataclass(frozen=True)
class WebConfig:
    runs_dir: str = "runs"
    host: str = "127.0.0.1"
    port: int = 8765


def discover_runs(runs_dir: str | Path) -> list[dict]:
    """Return summary rows for run folders under a runs directory."""
    root = Path(runs_dir)
    if not root.exists():
        return []

    run_dirs = sorted(path for path in root.iterdir() if (path / "summary.json").exists())
    rows = []
    for run_dir in run_dirs:
        summary = _read_json(run_dir / "summary.json")
        eval_summary = summary["eval"]
        rows.append({
            "name": run_dir.name,
            "path": str(run_dir),
            "eval_score": f"{eval_summary['num_passed']}/{eval_summary['num_examples']}",
            "pass_rate": eval_summary["pass_rate"],
            "base_val_loss": summary["base"]["final_val_loss"],
            "sft_val_loss": summary["sft"]["final_val_loss"],
            "num_parameters": summary["base"]["num_parameters"],
            "context_size": summary["config"]["context_size"],
            "truncated_examples": summary["sft"]["truncated_examples"],
        })
    return rows


def load_run_detail(runs_dir: str | Path, run_name: str) -> dict:
    """Load summary, eval details, and samples for one run."""
    run_dir = _safe_child(Path(runs_dir), run_name)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing run summary: {summary_path}")

    summary = _read_json(summary_path)
    artifacts = summary.get("artifacts", {})
    corpus_path = Path(artifacts.get("corpus", run_dir / "corpus.txt"))
    corpus_manifest_path = Path(artifacts.get("corpus_manifest", run_dir / "corpus_manifest.json"))
    corpus_report_path = Path(artifacts.get("corpus_report", run_dir / "corpus_report.md"))
    tokenizer_path = Path(artifacts.get("tokenizer", run_dir / "tokenizer.json"))
    detail = {
        "summary": summary,
        "corpus_preview": _read_text_preview(corpus_path),
        "corpus_manifest": _read_json_if_exists(corpus_manifest_path),
        "corpus_report": _read_text_preview(corpus_report_path),
        "tokenizer_detail": _load_tokenizer_detail(tokenizer_path),
        "base_report": _read_json_if_exists(run_dir / "base" / "train_report.json"),
        "sft_report": _read_json_if_exists(run_dir / "sft" / "sft_report.json"),
        "eval": _read_json_if_exists(run_dir / "eval" / "eval_report.json"),
        "eval_reports": _load_eval_reports(run_dir),
        "base_sample": _read_text_if_exists(run_dir / "base" / "sample.txt"),
        "sft_sample": _read_text_if_exists(run_dir / "sft" / "sample.txt"),
        "reports": _load_report_status(run_dir, summary),
        "artifact_inventory": _load_artifact_inventory(run_dir, summary),
    }
    return detail


def load_run_report(runs_dir: str | Path, run_name: str, report_name: str) -> dict:
    """Load one known Markdown report for display in the web UI."""
    run_dir = _safe_child(Path(runs_dir), run_name)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing run summary: {summary_path}")

    summary = _read_json(summary_path)
    reports = _load_report_status(run_dir, summary)
    if report_name not in reports:
        raise ValueError("report must be one of: summary, honesty, base, sft, eval")

    report = reports[report_name]
    path = Path(report["path"])
    if not report["exists"]:
        raise FileNotFoundError(f"missing report: {path}")

    return {
        "run": run_name,
        "report": report_name,
        "path": str(path),
        "markdown": _read_text_preview(path, limit=20000) or "",
    }


def generate_run_text(runs_dir: str | Path, payload: dict) -> dict:
    """Generate text from a run checkpoint for the web UI."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    run_name = str(payload.get("run", ""))
    run_dir = _safe_child(Path(runs_dir), run_name)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing run summary: {summary_path}")

    checkpoint = str(payload.get("checkpoint", "sft")).lower()
    if checkpoint not in {"base", "sft"}:
        raise ValueError("checkpoint must be 'base' or 'sft'")

    checkpoint_path = run_dir / checkpoint / "checkpoint"
    tokenizer_path = run_dir / "tokenizer.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_path}")

    max_new_tokens = _bounded_int(payload.get("max_new_tokens", 80), 1, 240)
    temperature = _bounded_float(payload.get("temperature", 0.8), 0.0, 2.0)
    top_k = _bounded_int(payload.get("top_k", 20), 0, 500)
    top_p = _bounded_float(payload.get("top_p", 1.0), 0.01, 1.0)
    repetition_penalty = _bounded_float(payload.get("repetition_penalty", 1.0), 0.1, 3.0)
    seed = _bounded_int(payload.get("seed", 42), 0, 9999)
    prompt = str(payload.get("prompt", ""))[:4000]

    result = generate_text_with_trace(GenerateConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=None if top_k <= 0 else top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        device="cpu",
    ))
    return {
        "run": run_name,
        "checkpoint": checkpoint,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
        **result,
    }


def preview_corpus_plan(payload: dict) -> dict:
    """Preview corpus source decisions for the web UI without writing artifacts."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    input_path = _optional_string(payload.get("input_path"))
    recipe_path = _optional_string(payload.get("recipe_path"))
    dataset_pack = _optional_string(payload.get("dataset_pack"))
    chat_input = _optional_string(payload.get("chat_input"))
    eval_input = _optional_string(payload.get("eval_input"))
    if not input_path and not recipe_path and not dataset_pack:
        raise ValueError("input_path, recipe_path, or dataset_pack is required")

    preview_chars = _bounded_int(payload.get("preview_chars", 1200), 0, 10000)
    min_quality_score = _bounded_int(payload.get("min_quality_score", 0), 0, 100)
    return preview_corpus_sources(
        input_path=input_path,
        recipe_path=recipe_path,
        preview_chars=preview_chars,
        chat_input=chat_input,
        eval_input=eval_input,
        dataset_pack=dataset_pack,
        min_quality_score=min_quality_score,
    ).to_dict()


def init_dataset_pack_plan(payload: dict) -> dict:
    """Create starter dataset pack files from a web request."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    name = _optional_string(payload.get("name")) or "picochat-pack"
    description = _optional_string(payload.get("description")) or "Starter Picochat dataset pack."
    corpus_path = _optional_string(payload.get("corpus_path"))
    out_dir = _optional_string(payload.get("out_dir"))
    if not corpus_path:
        raise ValueError("corpus_path is required")
    if not out_dir:
        raise ValueError("out_dir is required")

    force = payload.get("force", False)
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")

    report = init_dataset_pack(
        out_dir=out_dir,
        corpus_path=corpus_path,
        name=name,
        description=description,
        force=force,
    )
    return {
        **report.to_dict(),
        "name": name,
        "description": description,
        "preview_command": _shell_command(
            "PYTHONPATH=src",
            "python",
            "-m",
            "picochat.cli",
            "data",
            "preview",
            "--dataset-pack",
            report.dataset_pack,
        ),
    }


def inspect_tuning_plan(payload: dict) -> dict:
    """Inspect chat SFT and eval JSONL inputs for the web UI."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_pack = _optional_string(payload.get("dataset_pack"))
    chat_input = _optional_string(payload.get("chat_input"))
    eval_input = _optional_string(payload.get("eval_input"))
    if dataset_pack:
        if chat_input or eval_input:
            raise ValueError("dataset_pack cannot be combined with chat_input or eval_input")
        pack = load_dataset_pack(dataset_pack)
        chat_input = pack.chat_input
        eval_input = pack.eval_input
    else:
        chat_input = chat_input or DEFAULT_CHAT_INPUT
        eval_input = eval_input or DEFAULT_EVAL_INPUT

    chat_data = inspect_chat_sft_data(chat_input)
    eval_data = inspect_chat_eval_data(eval_input)
    status = _combined_tuning_status(chat_data.status, eval_data.status)
    return {
        "status": status,
        "summary": _tuning_summary(status),
        "training_ready": status == "ready",
        "can_train": status != "blocked",
        "dataset_pack": dataset_pack,
        "chat_input": chat_input,
        "eval_input": eval_input,
        "chat_data": chat_data.to_dict(),
        "eval_data": eval_data.to_dict(),
        "next_actions": _tuning_next_actions(chat_data, eval_data, bool(dataset_pack)),
        "preview_command": (
            _shell_command(
                "PYTHONPATH=src",
                "python",
                "-m",
                "picochat.cli",
                "data",
                "preview",
                "--dataset-pack",
                dataset_pack,
            )
            if dataset_pack else None
        ),
    }


def eval_starter_plan(payload: dict) -> dict:
    """Generate a starter eval JSONL file from a web request."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_pack = _optional_string(payload.get("dataset_pack"))
    input_path = _optional_string(payload.get("input_path"))
    out_path = _optional_string(payload.get("out_path"))
    if not out_path:
        raise ValueError("out_path is required")
    if not dataset_pack and not input_path:
        raise ValueError("dataset_pack or input_path is required")

    max_items = _bounded_int(payload.get("max_items", 24), 4, 200)
    seed = _bounded_int(payload.get("seed", 42), 0, 9999)
    force = payload.get("force", False)
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")

    report = generate_eval_starter(
        input_path=input_path,
        dataset_pack=dataset_pack,
        out_path=out_path,
        max_items=max_items,
        seed=seed,
        force=force,
    )
    command_parts = [
        "PYTHONPATH=src",
        "python",
        "-m",
        "picochat.cli",
        "data",
        "eval-starter",
    ]
    if dataset_pack:
        command_parts.extend(["--dataset-pack", dataset_pack])
    else:
        command_parts.extend(["--input", input_path or ""])
    command_parts.extend(["--out", out_path, "--max-items", str(max_items), "--seed", str(seed)])
    if force:
        command_parts.append("--force")
    return {
        **report.to_dict(),
        "dataset_pack": dataset_pack,
        "force": force,
        "command": _shell_command(*command_parts),
        "report_path": str(Path(out_path).with_suffix(".md")),
        "next_actions": [
            "Open the generated eval JSONL and replace generic wording with domain-specific prompts.",
            "Keep refusal and memorization-probe rows before trusting a custom SLM score.",
            "Run tuning inspection after editing so Picochat can grade the eval file before training.",
        ],
    }


def load_pack_editor_plan(payload: dict) -> dict:
    """Load editable chat/eval JSONL text for a dataset pack or explicit paths."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_pack, chat_input, eval_input = _resolve_tuning_paths(payload)
    return _pack_editor_payload(dataset_pack, chat_input, eval_input, saved=False)


def save_pack_editor_plan(payload: dict) -> dict:
    """Save edited chat/eval JSONL text and return fresh validation reports."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_pack, chat_input, eval_input = _resolve_tuning_paths(payload)
    chat_text = _bounded_text(payload.get("chat_text", ""), "chat_text")
    eval_text = _bounded_text(payload.get("eval_text", ""), "eval_text")
    _validate_jsonl_text(chat_text, "chat_text")
    _validate_jsonl_text(eval_text, "eval_text")

    _write_text_file(chat_input, _normalized_jsonl_text(chat_text))
    _write_text_file(eval_input, _normalized_jsonl_text(eval_text))
    return _pack_editor_payload(dataset_pack, chat_input, eval_input, saved=True)


def start_run_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Start a local tiny run as a background process."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_pack = _optional_string(payload.get("dataset_pack"))
    if not dataset_pack:
        raise ValueError("dataset_pack is required")
    if not Path(dataset_pack).exists():
        raise FileNotFoundError(f"missing dataset pack: {dataset_pack}")
    load_dataset_pack(dataset_pack)
    min_quality_score = _bounded_int(payload.get("min_quality_score", 0), 0, 100)
    try:
        launch_preview = preview_corpus_sources(
            dataset_pack=dataset_pack,
            preview_chars=0,
            min_quality_score=min_quality_score,
        )
    except FileNotFoundError as error:
        raise ValueError(f"corpus readiness blocked: missing source {error}") from error
    _validate_launch_readiness(launch_preview)

    run_name = _slug(_optional_string(payload.get("run_name")) or _default_run_name(dataset_pack))
    out_dir = _safe_child(Path(runs_dir), run_name)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"run output already exists: {out_dir}")

    preset_name = _optional_string(payload.get("preset")) or "smoke"
    preset = _run_preset(preset_name)
    context_size = _bounded_int(payload.get("context_size", preset["context_size"]), 8, 1024)
    base_steps = _bounded_int(payload.get("base_steps", preset["base_steps"]), 1, 100000)
    sft_steps = _bounded_int(payload.get("sft_steps", preset["sft_steps"]), 1, 10000)
    base_batch_size = _bounded_int(payload.get("base_batch_size", preset["base_batch_size"]), 1, 64)
    sft_batch_size = _bounded_int(payload.get("sft_batch_size", preset["sft_batch_size"]), 1, 64)
    seed = _bounded_int(payload.get("seed", 42), 0, 9999)
    eval_max_new_tokens = _bounded_int(payload.get("eval_max_new_tokens", preset["eval_max_new_tokens"]), 1, 320)
    n_embd = _bounded_int(payload.get("n_embd", preset["n_embd"]), 16, 512)
    n_head = _bounded_int(payload.get("n_head", preset["n_head"]), 1, 16)
    n_layer = _bounded_int(payload.get("n_layer", preset["n_layer"]), 1, 12)
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    tokenizer_type = str(payload.get("tokenizer_type", preset.get("tokenizer_type", "char")))
    if tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(f"tokenizer_type must be one of {', '.join(TOKENIZER_TYPES)}")
    tokenizer_vocab_size = _optional_int(
        payload.get("tokenizer_vocab_size", preset.get("tokenizer_vocab_size")),
        minimum=4,
        maximum=8192,
    )
    tokenizer_min_freq = _bounded_int(payload.get("tokenizer_min_freq", preset.get("tokenizer_min_freq", 1)), 1, 1000)
    base_lr_warmup_steps = _bounded_int(payload.get("base_lr_warmup_steps", preset.get("base_lr_warmup_steps", 0)), 0, base_steps)
    sft_lr_warmup_steps = _bounded_int(payload.get("sft_lr_warmup_steps", preset.get("sft_lr_warmup_steps", 0)), 0, sft_steps)
    base_lr_decay = str(payload.get("base_lr_decay", preset.get("base_lr_decay", "none")))
    sft_lr_decay = str(payload.get("sft_lr_decay", preset.get("sft_lr_decay", "none")))
    if base_lr_decay not in LR_DECAYS or sft_lr_decay not in LR_DECAYS:
        raise ValueError(f"lr decay must be one of {', '.join(LR_DECAYS)}")
    base_min_lr_ratio = _bounded_float(payload.get("base_min_lr_ratio", preset.get("base_min_lr_ratio", 1.0)), 0.0, 1.0)
    sft_min_lr_ratio = _bounded_float(payload.get("sft_min_lr_ratio", preset.get("sft_min_lr_ratio", 1.0)), 0.0, 1.0)
    base_grad_clip = _bounded_float(payload.get("base_grad_clip", preset.get("base_grad_clip", 0.0)), 0.0, 100.0)
    sft_grad_clip = _bounded_float(payload.get("sft_grad_clip", preset.get("sft_grad_clip", 0.0)), 0.0, 100.0)
    base_early_stop_patience = _bounded_int(
        payload.get("base_early_stop_patience", preset.get("base_early_stop_patience", 3)),
        0,
        100,
    )
    sft_early_stop_patience = _bounded_int(
        payload.get("sft_early_stop_patience", preset.get("sft_early_stop_patience", 4)),
        0,
        100,
    )
    sft_sampling = str(payload.get("sft_sampling", preset.get("sft_sampling", "uniform")))
    if sft_sampling not in SFT_SAMPLING_MODES:
        raise ValueError(f"sft_sampling must be one of {', '.join(SFT_SAMPLING_MODES)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "web_run.log"
    command = [
        sys.executable,
        "-m",
        "picochat.cli",
        "run",
        "tiny",
        "--out-dir",
        str(out_dir),
        "--dataset-pack",
        dataset_pack,
        "--context-size",
        str(context_size),
        "--n-embd",
        str(n_embd),
        "--n-head",
        str(n_head),
        "--n-layer",
        str(n_layer),
        "--base-steps",
        str(base_steps),
        "--sft-steps",
        str(sft_steps),
        "--base-batch-size",
        str(base_batch_size),
        "--sft-batch-size",
        str(sft_batch_size),
        "--seed",
        str(seed),
        "--eval-max-new-tokens",
        str(eval_max_new_tokens),
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-min-freq",
        str(tokenizer_min_freq),
        "--base-lr-warmup-steps",
        str(base_lr_warmup_steps),
        "--sft-lr-warmup-steps",
        str(sft_lr_warmup_steps),
        "--base-lr-decay",
        base_lr_decay,
        "--sft-lr-decay",
        sft_lr_decay,
        "--base-min-lr-ratio",
        str(base_min_lr_ratio),
        "--sft-min-lr-ratio",
        str(sft_min_lr_ratio),
        "--base-grad-clip",
        str(base_grad_clip),
        "--sft-grad-clip",
        str(sft_grad_clip),
        "--base-early-stop-patience",
        str(base_early_stop_patience),
        "--sft-early-stop-patience",
        str(sft_early_stop_patience),
        "--sft-sampling",
        sft_sampling,
        "--split-mode",
        "document",
        "--min-score",
        str(min_quality_score),
        "--device",
        "cpu",
    ]
    if preset_name in RUN_SCALES:
        command.extend(["--scale", preset_name])
    if tokenizer_vocab_size is not None:
        command.extend(["--tokenizer-vocab-size", str(tokenizer_vocab_size)])
    log_path.write_text(f"$ {_shell_command(*command)}\n\n", encoding="utf-8")
    log_file = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=_child_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_file.close()

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "run_name": run_name,
        "out_dir": str(out_dir),
        "dataset_pack": dataset_pack,
        "log_path": str(log_path),
        "command": _shell_command(*command),
        "started_at": time.time(),
        "process": process,
        "preset": preset_name,
        "min_quality_score": min_quality_score,
        "launch_config": {
            "context_size": context_size,
            "base_steps": base_steps,
            "sft_steps": sft_steps,
            "tokenizer_type": tokenizer_type,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "base_lr_decay": base_lr_decay,
            "sft_lr_decay": sft_lr_decay,
            "base_grad_clip": base_grad_clip,
            "sft_grad_clip": sft_grad_clip,
            "base_early_stop_patience": base_early_stop_patience,
            "sft_early_stop_patience": sft_early_stop_patience,
            "sft_sampling": sft_sampling,
        },
        "launch_readiness": launch_preview.readiness.to_dict(),
        "launch_tuning": {
            "chat": launch_preview.chat_data.to_dict(),
            "eval": launch_preview.eval_data.to_dict(),
        },
    }
    with _RUN_JOBS_LOCK:
        _RUN_JOBS[job_id] = job
    return run_status_plan(job_id, runs_dir)


def run_status_plan(job_id: str | None = None, runs_dir: str | Path = "runs") -> dict:
    """Return status and log tail for a background run."""
    runs_root = Path(runs_dir).resolve()
    with _RUN_JOBS_LOCK:
        if job_id:
            job = _RUN_JOBS.get(job_id)
            jobs = [job] if job is not None else []
        else:
            jobs = [job for job in _RUN_JOBS.values() if _job_in_runs_dir(job, runs_root)]
    active_statuses = [_run_job_status(job) for job in jobs]
    if job_id and not active_statuses:
        persisted = [job for job in _discover_run_jobs(runs_dir) if job["id"] == job_id or job["run_name"] == job_id]
        if not persisted:
            raise ValueError(f"unknown run job: {job_id}")
        active_statuses = persisted[:1]

    known_out_dirs = {str(Path(status["out_dir"]).resolve()) for status in active_statuses}
    discovered = [
        job for job in _discover_run_jobs(runs_dir)
        if not job_id and str(Path(job["out_dir"]).resolve()) not in known_out_dirs
    ]
    statuses = sorted(
        [*active_statuses, *discovered],
        key=lambda item: item.get("updated_at", 0),
    )
    return {
        "jobs": statuses,
        "job": statuses[-1] if statuses else None,
    }


def cancel_run_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Terminate a running web-launched job."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    job_id = _optional_string(payload.get("job_id"))
    if not job_id:
        raise ValueError("job_id is required")
    with _RUN_JOBS_LOCK:
        job = _RUN_JOBS.get(job_id)
    if job is None:
        raise ValueError(f"unknown active run job: {job_id}")
    process = job["process"]
    if process.poll() is None:
        process.terminate()
        time.sleep(0.05)
    return run_status_plan(job_id, runs_dir)


def run_presets_plan() -> dict:
    return {"presets": RUN_PRESETS}


def serve_web(config: WebConfig) -> None:
    """Start the blocking local web server."""
    handler = _make_handler(config)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    url = f"http://{config.host}:{config.port}"
    print(f"Picochat web UI: {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def _make_handler(config: WebConfig):
    class PicoWebHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
            else:
                self.send_error(404, "Not found")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_asset("index.html", "text/html; charset=utf-8")
                elif parsed.path == "/assets/style.css":
                    self._send_asset("style.css", "text/css; charset=utf-8")
                elif parsed.path == "/assets/app.js":
                    self._send_asset("app.js", "application/javascript; charset=utf-8")
                elif parsed.path == "/api/runs":
                    self._send_json({"runs": discover_runs(config.runs_dir)})
                elif parsed.path == "/api/run":
                    query = parse_qs(parsed.query)
                    run_name = query.get("name", [""])[0]
                    self._send_json(load_run_detail(config.runs_dir, run_name))
                elif parsed.path == "/api/report":
                    query = parse_qs(parsed.query)
                    run_name = query.get("name", [""])[0]
                    report_name = query.get("report", ["summary"])[0]
                    self._send_json(load_run_report(config.runs_dir, run_name, report_name))
                elif parsed.path == "/api/compare":
                    query = parse_qs(parsed.query)
                    names = query.get("run", [])
                    run_paths = [str(_safe_child(Path(config.runs_dir), name)) for name in names]
                    self._send_json(compare_runs(run_paths))
                elif parsed.path == "/api/run/status":
                    query = parse_qs(parsed.query)
                    job_id = query.get("job", [None])[0]
                    self._send_json(run_status_plan(job_id, config.runs_dir))
                elif parsed.path == "/api/run/presets":
                    self._send_json(run_presets_plan())
                else:
                    self.send_error(404, "Not found")
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/generate":
                    self._send_json(generate_run_text(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/corpus/preview":
                    self._send_json(preview_corpus_plan(self._read_json_body()))
                elif parsed.path == "/api/dataset-pack/init":
                    self._send_json(init_dataset_pack_plan(self._read_json_body()))
                elif parsed.path == "/api/tuning/inspect":
                    self._send_json(inspect_tuning_plan(self._read_json_body()))
                elif parsed.path == "/api/eval/starter":
                    self._send_json(eval_starter_plan(self._read_json_body()))
                elif parsed.path == "/api/pack/editor/load":
                    self._send_json(load_pack_editor_plan(self._read_json_body()))
                elif parsed.path == "/api/pack/editor/save":
                    self._send_json(save_pack_editor_plan(self._read_json_body()))
                elif parsed.path == "/api/run/start":
                    self._send_json(start_run_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/run/cancel":
                    self._send_json(cancel_run_plan(config.runs_dir, self._read_json_body()))
                else:
                    self.send_error(404, "Not found")
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def log_message(self, format: str, *args) -> None:
            return

        def _send_asset(self, name: str, content_type: str) -> None:
            data = (
                importlib.resources.files("picochat.web_assets")
                .joinpath(name)
                .read_bytes()
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

    return PicoWebHandler


def _safe_child(root: Path, name: str) -> Path:
    if not name:
        raise ValueError("run name is required")
    root = root.resolve()
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError("run path escapes runs directory")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounded_int(value: object, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _optional_int(value: object, minimum: int, maximum: int) -> int | None:
    if value is None or value == "":
        return None
    return _bounded_int(value, minimum, maximum)


def _bounded_float(value: object, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _shell_command(*parts: str) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _combined_tuning_status(chat_status: str, eval_status: str) -> str:
    statuses = {chat_status, eval_status}
    if "blocked" in statuses:
        return "blocked"
    if "caution" in statuses:
        return "caution"
    return "ready"


def _run_preset(name: str) -> dict:
    if name not in RUN_PRESETS:
        known = ", ".join(sorted(RUN_PRESETS))
        raise ValueError(f"preset must be one of: {known}")
    return RUN_PRESETS[name]


def _tuning_summary(status: str) -> str:
    if status == "ready":
        return "Chat SFT and eval files look ready for a tiny run."
    if status == "caution":
        return "Files are readable, but improve them before trusting a run."
    return "Fix blocked chat/eval data before training."


def _tuning_next_actions(chat_data, eval_data, from_pack: bool) -> list[str]:
    actions = []
    for label, report in (("Chat SFT", chat_data), ("Eval", eval_data)):
        if report.status == "blocked":
            actions.append(f"Fix {label}: {report.summary}")
        elif report.status == "caution":
            actions.append(f"Improve {label}: {report.summary}")
    if not actions:
        actions.append("Tuning data is ready for the next tiny run.")
    if from_pack:
        actions.append("Run Source Preview next to inspect corpus readiness and get the training command.")
    return actions


def _resolve_tuning_paths(payload: dict) -> tuple[str | None, str, str]:
    dataset_pack = _optional_string(payload.get("dataset_pack"))
    chat_input = _optional_string(payload.get("chat_input"))
    eval_input = _optional_string(payload.get("eval_input"))
    if dataset_pack:
        if chat_input or eval_input:
            raise ValueError("dataset_pack cannot be combined with chat_input or eval_input")
        pack = load_dataset_pack(dataset_pack)
        return dataset_pack, pack.chat_input, pack.eval_input
    return None, chat_input or DEFAULT_CHAT_INPUT, eval_input or DEFAULT_EVAL_INPUT


def _pack_editor_payload(dataset_pack: str | None, chat_input: str, eval_input: str, saved: bool) -> dict:
    chat_text, chat_truncated = _read_editable_text(chat_input)
    eval_text, eval_truncated = _read_editable_text(eval_input)
    chat_data = inspect_chat_sft_data(chat_input)
    eval_data = inspect_chat_eval_data(eval_input)
    status = _combined_tuning_status(chat_data.status, eval_data.status)
    return {
        "saved": saved,
        "status": status,
        "summary": _tuning_summary(status),
        "dataset_pack": dataset_pack,
        "chat_input": chat_input,
        "eval_input": eval_input,
        "chat_text": chat_text,
        "eval_text": eval_text,
        "chat_truncated": chat_truncated,
        "eval_truncated": eval_truncated,
        "chat_lines": _line_count(chat_text),
        "eval_lines": _line_count(eval_text),
        "chat_data": chat_data.to_dict(),
        "eval_data": eval_data.to_dict(),
        "next_actions": _tuning_next_actions(chat_data, eval_data, bool(dataset_pack)),
    }


def _read_editable_text(path: str, limit: int = 200_000) -> tuple[str, bool]:
    source = Path(path)
    if not source.exists():
        return "", False
    if not source.is_file():
        raise ValueError(f"path is not a file: {path}")
    text = source.read_text(encoding="utf-8")
    if len(text) > limit:
        return text[:limit], True
    return text, False


def _bounded_text(value: object, field: str, limit: int = 300_000) -> str:
    text = str(value or "")
    if len(text) > limit:
        raise ValueError(f"{field} is too large for the web editor")
    return text


def _validate_jsonl_text(text: str, field: str) -> None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} line {line_number} is invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{field} line {line_number} must be a JSON object")


def _normalized_jsonl_text(text: str) -> str:
    stripped = text.strip()
    return f"{stripped}\n" if stripped else ""


def _write_text_file(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "pico-run"


def _default_run_name(dataset_pack: str) -> str:
    pack_path = Path(dataset_pack)
    base = pack_path.parent.name if pack_path.parent.name else pack_path.stem
    return f"{base}-run"


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{current}" if current else src_path
    return env


def _run_job_status(job: dict) -> dict:
    process = job["process"]
    returncode = process.poll()
    state = "running" if returncode is None else "succeeded" if returncode == 0 else "failed"
    summary_path = Path(job["out_dir"]) / "summary.json"
    if returncode is not None:
        _write_returncode(Path(job["out_dir"]) / "web_returncode.txt", returncode)
    return {
        "id": job["id"],
        "run_name": job["run_name"],
        "out_dir": job["out_dir"],
        "dataset_pack": job["dataset_pack"],
        "log_path": job["log_path"],
        "command": job["command"],
        "pid": process.pid,
        "state": state,
        "returncode": returncode,
        "elapsed_seconds": max(0, round(time.time() - job["started_at"], 1)),
        "summary_exists": summary_path.exists(),
        "log_tail": _read_log_tail(Path(job["log_path"])),
        "can_cancel": state == "running",
        "source": "active",
        "updated_at": time.time(),
        "preset": job.get("preset"),
        "min_quality_score": job.get("min_quality_score", 0),
        "launch_config": job.get("launch_config"),
        "launch_readiness": job.get("launch_readiness"),
        "launch_tuning": job.get("launch_tuning"),
    }


def _job_in_runs_dir(job: dict, runs_root: Path) -> bool:
    try:
        out_dir = Path(job["out_dir"]).resolve()
    except (KeyError, OSError):
        return False
    return out_dir == runs_root or runs_root in out_dir.parents


def _discover_run_jobs(runs_dir: str | Path, limit: int = 20) -> list[dict]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    jobs = []
    for run_dir in sorted(root.iterdir()):
        log_path = run_dir / "web_run.log"
        if not run_dir.is_dir() or not log_path.exists():
            continue
        summary_path = run_dir / "summary.json"
        returncode_path = run_dir / "web_returncode.txt"
        returncode = _read_returncode(returncode_path)
        state = (
            "succeeded" if summary_path.exists()
            else "failed" if returncode not in (None, 0)
            else "stopped"
        )
        updated_at = max(log_path.stat().st_mtime, summary_path.stat().st_mtime if summary_path.exists() else 0)
        jobs.append({
            "id": f"run-{_slug(run_dir.name)}",
            "run_name": run_dir.name,
            "out_dir": str(run_dir),
            "dataset_pack": _dataset_pack_from_summary(summary_path),
            "log_path": str(log_path),
            "command": _command_from_log(log_path),
            "pid": None,
            "state": state,
            "returncode": returncode,
            "elapsed_seconds": None,
            "summary_exists": summary_path.exists(),
            "log_tail": _read_log_tail(log_path),
            "can_cancel": False,
            "source": "disk",
            "updated_at": updated_at,
            "preset": None,
            "min_quality_score": None,
            "launch_readiness": None,
            "launch_tuning": None,
        })
    return sorted(jobs, key=lambda item: item["updated_at"])[-limit:]


def _validate_launch_readiness(report) -> None:
    problems = []
    if report.readiness.status == "blocked":
        problems.append(f"corpus readiness blocked: {report.readiness.summary}")
    if report.chat_data.status == "blocked":
        problems.append(f"chat SFT blocked: {report.chat_data.summary}")
    if report.eval_data.status == "blocked":
        problems.append(f"eval blocked: {report.eval_data.summary}")
    if problems:
        raise ValueError("; ".join(problems))


def _read_returncode(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _write_returncode(path: Path, returncode: int) -> None:
    if path.exists():
        return
    try:
        path.write_text(f"{returncode}\n", encoding="utf-8")
    except OSError:
        pass


def _dataset_pack_from_summary(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    try:
        return _read_json(summary_path).get("config", {}).get("dataset_pack")
    except (OSError, json.JSONDecodeError):
        return None


def _command_from_log(log_path: Path) -> str:
    try:
        first_line = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return ""
    return first_line[2:] if first_line.startswith("$ ") else first_line


def _read_log_tail(path: Path, limit: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _read_text_preview(path: Path, limit: int = 1600) -> str | None:
    text = _read_text_if_exists(path)
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def _load_tokenizer_detail(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = _read_json(path)
    token_to_id = data.get("token_to_id", {})
    text_tokens = [
        token
        for token, _ in sorted(token_to_id.items(), key=lambda item: int(item[1]))
        if token not in set(data.get("special_tokens", []))
    ]
    return {
        "type": data.get("type"),
        "special_tokens": data.get("special_tokens", []),
        "vocab_size": len(token_to_id),
        "token_to_id": {token: int(idx) for token, idx in token_to_id.items()},
        "merges": data.get("merges", []),
        "sample_tokens": text_tokens[:32],
    }


def _load_eval_reports(run_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(run_dir.iterdir()):
        report_path = path / "eval_report.json"
        if path.is_dir() and path.name.startswith("eval") and report_path.exists():
            reports.append({
                "name": path.name,
                "report": _read_json(report_path),
            })
    return reports


def _load_report_status(run_dir: Path, summary: dict) -> dict:
    artifacts = summary.get("artifacts", {})
    report_paths = {
        "summary": run_dir / "summary.md",
        "honesty": Path(artifacts.get("honesty_report", run_dir / "honesty" / "report.md")),
        "base": Path(artifacts.get("base_report", run_dir / "base" / "report.md")),
        "sft": Path(artifacts.get("sft_report", run_dir / "sft" / "report.md")),
        "eval": Path(artifacts.get("eval_report", run_dir / "eval" / "report.md")),
    }
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
        }
        for name, path in report_paths.items()
    }


def _load_artifact_inventory(run_dir: Path, summary: dict) -> dict:
    artifacts = summary.get("artifacts", {})
    config = summary.get("config", {})
    out_dir = Path(config.get("out_dir") or run_dir)
    base_checkpoint = Path(summary.get("base", {}).get("checkpoint", out_dir / "base" / "checkpoint"))
    base_best_checkpoint = Path(summary.get("base", {}).get("best_checkpoint", {}).get("path", out_dir / "base" / "best_checkpoint"))
    sft_checkpoint = Path(summary.get("sft", {}).get("checkpoint", out_dir / "sft" / "checkpoint"))

    known_paths = {
        "dataset_pack": config.get("dataset_pack"),
        "corpus_source": config.get("corpus_recipe") or config.get("corpus_input"),
        "chat_input": config.get("chat_input"),
        "eval_input": config.get("eval_input"),
        "corpus": artifacts.get("corpus", run_dir / "corpus.txt"),
        "corpus_manifest": artifacts.get("corpus_manifest", run_dir / "corpus_manifest.json"),
        "corpus_report": artifacts.get("corpus_report", run_dir / "corpus_report.md"),
        "honesty_json": artifacts.get("honesty_json", run_dir / "honesty" / "honesty_report.json"),
        "honesty_report": artifacts.get("honesty_report", run_dir / "honesty" / "report.md"),
        "tokenizer": artifacts.get("tokenizer", run_dir / "tokenizer.json"),
        "summary_json": run_dir / "summary.json",
        "summary_report": run_dir / "summary.md",
        "base_checkpoint": base_checkpoint,
        "base_best_checkpoint": base_best_checkpoint,
        "base_trace": out_dir / "base" / "train_report.json",
        "base_report": artifacts.get("base_report", out_dir / "base" / "report.md"),
        "base_sample": out_dir / "base" / "sample.txt",
        "sft_checkpoint": sft_checkpoint,
        "sft_trace": out_dir / "sft" / "sft_report.json",
        "sft_report": artifacts.get("sft_report", out_dir / "sft" / "report.md"),
        "sft_sample": out_dir / "sft" / "sample.txt",
        "eval_json": out_dir / "eval" / "eval_report.json",
        "eval_report": artifacts.get("eval_report", out_dir / "eval" / "report.md"),
    }
    items = [
        _artifact_record(key, Path(path))
        for key, path in known_paths.items()
        if path
    ]
    return {
        "items": items,
        "by_path": _artifact_records_by_path(items),
    }


def _artifact_record(key: str, path: Path) -> dict:
    exists = path.exists()
    kind = "missing"
    size_bytes = 0
    if exists and path.is_dir():
        kind = "directory"
        size_bytes = _path_size(path)
    elif exists:
        kind = "file"
        size_bytes = path.stat().st_size
    return {
        "key": key,
        "path": str(path),
        "exists": exists,
        "kind": kind,
        "size_bytes": size_bytes,
    }


def _artifact_records_by_path(items: list[dict]) -> dict:
    by_path = {}
    for item in items:
        for path in _path_aliases(Path(item["path"])):
            by_path[path] = item
    return by_path


def _path_aliases(path: Path) -> set[str]:
    aliases = {str(path)}
    try:
        aliases.add(str(path.resolve()))
    except OSError:
        pass
    try:
        aliases.add(str(path.resolve().relative_to(Path.cwd())))
    except (OSError, ValueError):
        pass
    return aliases


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total
