"""Local web dashboard for inspecting Picochat runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import importlib.resources
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

from picochat import __version__
from picochat.compare import compare_runs
from picochat.data import DEFAULT_CHAT_INPUT, DEFAULT_EVAL_INPUT, preview_corpus_sources
from picochat.dataset_pack import (
    clone_dataset_pack,
    init_dataset_pack,
    load_dataset_pack,
    update_dataset_pack_tuning_paths,
)
from picochat.device import DEVICE_CHOICES
from picochat.eval_starter import generate_eval_starter
from picochat.benchmark_pack import (
    BENCHMARK_PROFILES,
    BENCHMARK_SKILL_ANSWER_STYLES,
    BENCHMARK_SOURCES,
    DEFAULT_BENCHMARK_EVAL_ROWS,
    DEFAULT_BENCHMARK_SFT_ROWS,
    generate_benchmark_tuning_pack,
)
from picochat.generate import GenerateConfig, generate_text_with_trace
from picochat.hf_import import HFImportConfig, HFSplitError, import_hf_dataset
from picochat.lora import DEFAULT_LORA_TARGETS, PEFT_MODES, parse_lora_targets
from picochat.model import SDPA_BACKENDS
from picochat.optim import LR_DECAYS, OPTIMIZER_TYPES
from picochat.precision import COMPILE_MODES, MATMUL_PRECISION_MODES, PRECISION_MODES
from picochat.preference_starter import PreferenceStarterConfig, generate_preference_starter
from picochat.run import LONG_RUN_GATE_PROFILES, TinyRunConfig
from picochat.run_preflight import assess_run_preflight
from picochat.scales import RUN_SCALES
from picochat.sft_starter import generate_sft_starter
from picochat.sft import SFT_PACKING_MODES, SFT_SAMPLING_MODES
from picochat.tokenizer import BPE_PRETOKENIZERS, DEFAULT_BPE_PRETOKENIZER, TOKENIZER_TYPES
from picochat.tuning_data import inspect_chat_eval_data, inspect_chat_sft_data

_RUN_JOBS: dict[str, dict] = {}
_RUN_JOBS_LOCK = threading.Lock()
# Cap on retained in-memory job records. Finished jobs stay discoverable on disk
# (web_run.log + sidecar), so evicting them from memory loses nothing.
_MAX_RUN_JOBS = 50
# Background `pico serve` processes, keyed by run name.
_SERVE_JOBS: dict[str, dict] = {}
_SERVE_JOBS_LOCK = threading.Lock()
# LRU cache of loaded HF inference engines (each holds a model in RAM) so
# Playground chat on a fine-tuned model does not reload weights every message.
_HF_ENGINES: dict[str, object] = {}
_HF_ENGINES_LOCK = threading.Lock()
_MAX_HF_ENGINES = 2


def _register_run_job(job_id: str, job: dict) -> None:
    """Track a launched job, evicting the oldest *finished* jobs past the cap.

    Running jobs are never evicted; finished ones remain available via disk
    discovery, so dropping them from memory is safe.
    """
    with _RUN_JOBS_LOCK:
        _RUN_JOBS[job_id] = job
        overflow = len(_RUN_JOBS) - _MAX_RUN_JOBS
        if overflow <= 0:
            return
        finished = sorted(
            (j for j in _RUN_JOBS.values()
             if (proc := j.get("process")) is not None and proc.poll() is not None),
            key=lambda j: j.get("started_at", 0.0),
        )
        for stale in finished[:overflow]:
            _RUN_JOBS.pop(stale["id"], None)


def _get_hf_engine(model_dir: str, base_only: bool = False):
    # Cache base and fine-tuned engines separately so the Playground's base/sft
    # toggle compares the stock model against the adapter without reloading.
    cache_key = f"{model_dir}::{'base' if base_only else 'sft'}"
    with _HF_ENGINES_LOCK:
        engine = _HF_ENGINES.pop(cache_key, None)
        if engine is None:
            from picochat.hf_infer import HFGenerator
            engine = HFGenerator(model_path=model_dir, device="cpu", base_only=base_only)
        _HF_ENGINES[cache_key] = engine  # reinsert as most-recently-used
        while len(_HF_ENGINES) > _MAX_HF_ENGINES:
            _HF_ENGINES.pop(next(iter(_HF_ENGINES)))
        return engine
# Server-sent-events log stream cadence. The cap bounds a single connection;
# the browser's EventSource auto-reconnects for runs longer than that.
_STREAM_INTERVAL_SECONDS = 1.0
_STREAM_MAX_TICKS = 3600
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_LARGE_IMPORT_ROWS = 100_000
CLIMBMIX_LARGE_IMPORT_DOCUMENT_SHARD_ROWS = 1000
RUN_PRESETS = {
    "smoke": {
        "label": "Smoke",
        "description": "Fast CPU sanity check for UI and data wiring.",
        "context_size": 128,
        "base_steps": 40,
        "sft_steps": 60,
        "base_batch_size": 4,
        "sft_batch_size": 4,
        "base_learning_rate": 3e-4,
        "sft_learning_rate": 1e-3,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 1,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "activation": "gelu",
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
        "base_grad_accum_steps": 1,
        "sft_grad_accum_steps": 1,
        "base_optimizer": "adamw",
        "sft_optimizer": "adamw",
        "base_muon_learning_rate": 0.02,
        "sft_muon_learning_rate": 0.02,
        "base_ema_decay": 0.0,
        "sft_ema_decay": 0.0,
        "device": "cpu",
        "sft_sampling": "uniform",
        "sft_packing": "separate",
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
        "base_learning_rate": 3e-4,
        "sft_learning_rate": 1e-3,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "activation": "gelu",
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
        "base_grad_accum_steps": 1,
        "sft_grad_accum_steps": 1,
        "base_optimizer": "adamw",
        "sft_optimizer": "adamw",
        "base_muon_learning_rate": 0.02,
        "sft_muon_learning_rate": 0.02,
        "base_ema_decay": 0.0,
        "sft_ema_decay": 0.0,
        "device": "cpu",
        "sft_sampling": "uniform",
        "sft_packing": "separate",
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
        "base_learning_rate": 3e-4,
        "sft_learning_rate": 3e-4,
        "n_embd": 96,
        "n_head": 4,
        "n_layer": 3,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "activation": "gelu",
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
        "base_grad_accum_steps": 2,
        "sft_grad_accum_steps": 1,
        "base_optimizer": "adamw",
        "sft_optimizer": "adamw",
        "base_muon_learning_rate": 0.02,
        "sft_muon_learning_rate": 0.02,
        "base_ema_decay": 0.0,
        "sft_ema_decay": 0.0,
        "device": "auto",
        "sft_sampling": "category_sqrt",
        "sft_packing": "separate",
        "base_early_stop_patience": 3,
        "sft_early_stop_patience": 4,
        "eval_max_new_tokens": 120,
    },
    "small": RUN_SCALES["small"].to_dict(),
    "medium": RUN_SCALES["medium"].to_dict(),
    "mps-local": {**RUN_SCALES["mps-local"].to_dict(), "device": "auto"},
    "climbmix-pilot": {**RUN_SCALES["climbmix-pilot"].to_dict(), "device": "auto"},
    "h100-pilot": {**RUN_SCALES["h100-pilot"].to_dict(), "device": "cuda"},
    "h100-100m": {**RUN_SCALES["h100-100m"].to_dict(), "device": "cuda"},
    "h100-100m-ddp8": {**RUN_SCALES["h100-100m-ddp8"].to_dict(), "device": "cuda", "ddp": True},
    "h200-1b-ddp8": {**RUN_SCALES["h200-1b-ddp8"].to_dict(), "device": "cuda", "ddp": True},
}


@dataclass(frozen=True)
class WebConfig:
    runs_dir: str = "runs"
    host: str = "127.0.0.1"
    port: int = 8765
    # When set, every /api/* request must present this token (X-Picochat-Token
    # header or `Authorization: Bearer`). Required automatically for any
    # non-loopback bind so the code-executing API is never silently exposed.
    auth_token: str | None = None


def _is_loopback_host(host: str) -> bool:
    """True for localhost-style hosts that are safe to serve without auth."""
    name = (host or "").strip().lower()
    if name in {"", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


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
            "domain_pass_rate": eval_summary.get("domain_pass_rate"),
            "refusal_pass_rate": eval_summary.get("refusal_pass_rate"),
            "base_val_loss": summary["base"]["final_val_loss"],
            "sft_val_loss": summary["sft"]["final_val_loss"],
            "num_parameters": summary["base"]["num_parameters"],
            "context_size": summary["config"]["context_size"],
            "truncated_examples": summary["sft"]["truncated_examples"],
            "skipped_long_examples": summary["sft"].get("skipped_long_examples", 0),
        })

    # Fine-tuned Hugging Face runs (output of `train hf-sft`) have no summary.json
    # but produce a final_model/ + hf_sft_report.json; surface them so they are
    # selectable for chat and serving.
    hf_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and not (path / "summary.json").exists() and (path / "hf_sft_report.json").exists()
    )
    for run_dir in hf_dirs:
        report = _read_json_if_exists(run_dir / "hf_sft_report.json") or {}
        rows.append({
            "name": run_dir.name,
            "path": str(run_dir),
            "kind": "hf-sft",
            "base_model": report.get("model"),
            "eval_score": "--",
            "pass_rate": None,
            "domain_pass_rate": None,
            "refusal_pass_rate": None,
            "base_val_loss": None,
            "sft_val_loss": report.get("best_val_loss"),
            "num_parameters": report.get("num_parameters"),
            "context_size": None,
            "truncated_examples": None,
            "skipped_long_examples": None,
        })
    return rows


def _hf_run_detail(run_dir: Path) -> dict:
    """Minimal detail payload for a fine-tuned HF run (no native summary)."""
    report = _read_json_if_exists(run_dir / "hf_sft_report.json") or {}
    return {
        "summary": {
            "kind": "hf-sft",
            "config": {"scale": "hf-sft", "base_model": report.get("model")},
            "hf_sft_report": report,
        },
        "corpus_preview": "",
        "corpus_manifest": None,
        "corpus_report": "",
        "tokenizer_detail": None,
        "base_report": None,
        "sft_report": None,
        "eval": None,
        "eval_reports": [],
        "preflight": None,
        "base_sample": "",
        "sft_sample": "",
        "reports": {},
        "artifact_inventory": {"by_path": {}},
        "run_timeline": [],
        "handoff_packet": None,
        "release_repair_plan": None,
        "run_passport": None,
    }


def load_run_detail(runs_dir: str | Path, run_name: str) -> dict:
    """Load summary, eval details, and samples for one run."""
    run_dir = _safe_child(Path(runs_dir), run_name)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        if (run_dir / "final_model").exists() or (run_dir / "hf_sft_report.json").exists():
            return _hf_run_detail(run_dir)
        raise FileNotFoundError(f"missing run summary: {summary_path}")

    summary = _read_json(summary_path)
    artifacts = summary.get("artifacts", {})
    corpus_path = _local_run_artifact_path(run_dir, summary, artifacts.get("corpus", run_dir / "corpus.txt"))
    corpus_manifest_path = _local_run_artifact_path(run_dir, summary, artifacts.get("corpus_manifest", run_dir / "corpus_manifest.json"))
    corpus_report_path = _local_run_artifact_path(run_dir, summary, artifacts.get("corpus_report", run_dir / "corpus_report.md"))
    tokenizer_path = _local_run_artifact_path(run_dir, summary, artifacts.get("tokenizer", run_dir / "tokenizer.json"))
    reports = _load_report_status(run_dir, summary)
    artifact_inventory = _load_artifact_inventory(run_dir, summary)
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
        "preflight": _preflight_from_run_dir(run_dir),
        "base_sample": _read_text_if_exists(run_dir / "base" / "sample.txt"),
        "sft_sample": _read_text_if_exists(run_dir / "sft" / "sample.txt"),
        "reports": reports,
        "artifact_inventory": artifact_inventory,
    }
    detail["run_timeline"] = _run_timeline_from_detail(summary, detail)
    detail["handoff_packet"] = _handoff_packet_from_detail(summary, detail)
    detail["release_repair_plan"] = _release_repair_plan_from_detail(summary, detail)
    detail["run_passport"] = _run_passport_from_detail(summary, detail)
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
    hf_model_dir = run_dir / "final_model"
    is_hf = hf_model_dir.exists() and not summary_path.exists()
    if not is_hf and not summary_path.exists():
        raise FileNotFoundError(f"missing run summary: {summary_path}")

    max_new_tokens = _bounded_int(payload.get("max_new_tokens", 80), 1, 240)
    temperature = _bounded_float(payload.get("temperature", 0.8), 0.0, 2.0)
    top_k = _bounded_int(payload.get("top_k", 20), 0, 500)
    top_p = _bounded_float(payload.get("top_p", 1.0), 0.01, 1.0)
    repetition_penalty = _bounded_float(payload.get("repetition_penalty", 1.0), 0.1, 3.0)
    seed = _bounded_int(payload.get("seed", 42), 0, 9999)
    prompt = str(payload.get("prompt", ""))[:4000]

    if is_hf:
        checkpoint = str(payload.get("checkpoint", "sft")).lower()
        if checkpoint not in {"base", "sft"}:
            checkpoint = "sft"
        result = _get_hf_engine(str(hf_model_dir), base_only=(checkpoint == "base")).generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None if top_k <= 0 else top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
    else:
        checkpoint = str(payload.get("checkpoint", "sft")).lower()
        if checkpoint not in {"base", "sft"}:
            raise ValueError("checkpoint must be 'base' or 'sft'")
        checkpoint_path = run_dir / checkpoint / "checkpoint"
        tokenizer_path = run_dir / "tokenizer.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"missing tokenizer: {tokenizer_path}")
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
            "picochat",
            "data",
            "preview",
            "--dataset-pack",
            report.dataset_pack,
        ),
    }


def hf_import_plan(payload: dict, runs_dir: str | Path = "runs") -> dict:
    """Import a Hugging Face dataset into a local Picochat dataset pack."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_input = _optional_string(payload.get("dataset")) or _optional_string(payload.get("dataset_url"))
    if not dataset_input:
        raise ValueError("dataset or dataset_url is required")

    dataset = _normalize_hf_dataset_id(dataset_input)
    source_dataset = dataset
    if dataset.lower() == "nvidia/nemotron-climbmix":
        dataset = CLIMBMIX_DATASET
    config_name = _optional_string(payload.get("config_name"))
    token = _optional_string(payload.get("token"))
    split = _optional_string(payload.get("split")) or "train"
    text_column = _optional_string(payload.get("text_column")) or "text"
    max_rows = _bounded_int(payload.get("max_rows", 1000), 1, 1_000_000)
    shard_count = _bounded_int(payload.get("shards", 1), 1, 6543)
    min_chars = _bounded_int(payload.get("min_chars", 20), 1, 10000)
    streaming = payload.get("streaming", True)
    force = payload.get("force", False)
    if not isinstance(streaming, bool):
        raise ValueError("streaming must be true or false")
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")

    out_dir_text = _optional_string(payload.get("out_dir"))
    out_dir = Path(out_dir_text) if out_dir_text else Path(runs_dir) / f"hf-{_slug(dataset)}-{max_rows}"
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"output folder already exists: {out_dir}. Enable force to overwrite import artifacts.")

    corpus_path = out_dir / "corpus.txt"
    documents_dir = out_dir / "documents"
    report_path = out_dir / "hf_import_report.json"
    data_files = tuple(f"shard_{index:05d}.parquet" for index in range(shard_count)) if dataset == CLIMBMIX_DATASET else ()
    document_shard_rows = _hf_import_document_shard_rows(payload, dataset, max_rows)
    import_report = import_hf_dataset(HFImportConfig(
        dataset=dataset,
        config_name=config_name,
        token=token,
        split=split,
        text_column=text_column,
        out_path=str(corpus_path),
        report_path=str(report_path),
        documents_dir=str(documents_dir),
        document_shard_rows=document_shard_rows,
        max_rows=max_rows,
        min_chars=min_chars,
        streaming=streaming,
        data_files=data_files,
    ))
    pack_report = init_dataset_pack(
        out_dir=out_dir,
        corpus_path=documents_dir,
        name=dataset,
        description=f"Hugging Face dataset import: {dataset}.",
        force=force,
    )
    preview = preview_corpus_sources(
        dataset_pack=pack_report.dataset_pack,
        preview_chars=700,
    )
    if data_files:
        command_parts = [
            "picochat",
            "data",
            "climbmix-import",
            "--out-dir",
            str(out_dir),
            "--shards",
            str(shard_count),
            "--max-rows",
            str(max_rows),
            "--min-chars",
            str(min_chars),
            "--document-shard-rows",
            str(document_shard_rows),
        ]
        if not streaming:
            command_parts.append("--no-streaming")
    else:
        command_parts = [
            "picochat",
            "data",
            "hf-import",
            "--dataset",
            dataset,
            "--split",
            split,
            "--text-column",
            text_column,
            "--out",
            str(corpus_path),
            "--documents-dir",
            str(documents_dir),
            "--max-rows",
            str(max_rows),
            "--min-chars",
            str(min_chars),
            "--document-shard-rows",
            str(document_shard_rows),
            "--pack-out",
            str(out_dir),
            "--pack-force",
        ]
        if config_name:
            command_parts.extend(["--config", config_name])
        if not streaming:
            command_parts.append("--no-streaming")
    return {
        "dataset_input": dataset_input,
        "dataset": dataset,
        "source_dataset": source_dataset,
        "config_name": config_name,
        "split": split,
        "text_column": text_column,
        "streaming": streaming,
        "shards": shard_count if data_files else None,
        "data_files": list(data_files),
        "force": force,
        "out_dir": str(out_dir),
        "corpus": import_report.out_path,
        "documents_dir": import_report.documents_dir,
        "report_path": import_report.report_path,
        "rows_seen": import_report.rows_seen,
        "rows_written": import_report.rows_written,
        "rows_skipped": import_report.rows_skipped,
        "characters_written": import_report.characters_written,
        "document_shard_rows": import_report.document_shard_rows,
        "document_files_written": import_report.document_files_written,
        "dataset_pack": pack_report.dataset_pack,
        "corpus_recipe": pack_report.corpus_recipe,
        "chat_input": pack_report.chat_input,
        "eval_input": pack_report.eval_input,
        "preview": preview.to_dict(),
        "command": _shell_command(*command_parts),
        "next_actions": [
            "Inspect the imported documents before trusting the dataset.",
            "Create SFT and eval starters from this pack, then edit them for real domain behavior.",
            "Launch a smoke run before spending time on a longer training run.",
        ],
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
                "picochat",
                "data",
                "preview",
                "--dataset-pack",
                dataset_pack,
            )
            if dataset_pack else None
        ),
    }


def sft_starter_plan(payload: dict) -> dict:
    """Generate a starter chat SFT JSONL file from a web request."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_pack = _optional_string(payload.get("dataset_pack"))
    input_path = _optional_string(payload.get("input_path"))
    out_path = _optional_string(payload.get("out_path"))
    if not out_path:
        raise ValueError("out_path is required")
    if not dataset_pack and not input_path:
        raise ValueError("dataset_pack or input_path is required")

    max_items = _bounded_int(payload.get("max_items", 32), 8, 300)
    seed = _bounded_int(payload.get("seed", 42), 0, 9999)
    force = payload.get("force", False)
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")
    promote_to_pack = payload.get("promote_to_pack", False)
    if not isinstance(promote_to_pack, bool):
        raise ValueError("promote_to_pack must be true or false")
    _reject_bundled_example(out_path, what="Generating chat SFT data")
    if promote_to_pack:
        _reject_bundled_example(dataset_pack, what="Promoting chat data into the pack")

    report = generate_sft_starter(
        input_path=input_path,
        dataset_pack=dataset_pack,
        out_path=out_path,
        max_items=max_items,
        seed=seed,
        force=force,
    )
    promoted_pack = update_dataset_pack_tuning_paths(dataset_pack, chat_input=out_path) if dataset_pack and promote_to_pack else None
    command_parts = [
        "picochat",
        "data",
        "sft-starter",
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
        "promoted_to_pack": promoted_pack is not None,
        "pack_chat_input": promoted_pack.chat_input if promoted_pack else None,
        "pack_eval_input": promoted_pack.eval_input if promoted_pack else None,
        "command": _shell_command(*command_parts),
        "report_path": str(Path(out_path).with_suffix(".md")),
        "next_actions": [
            "Open the generated chat JSONL and rewrite prompts into real user questions for this domain.",
            "Keep refusal and memorization-refusal rows before trusting a custom SLM.",
            "Run tuning inspection after editing so Picochat can grade the SFT file before training.",
        ],
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
    promote_to_pack = payload.get("promote_to_pack", False)
    if not isinstance(promote_to_pack, bool):
        raise ValueError("promote_to_pack must be true or false")
    _reject_bundled_example(out_path, what="Generating eval data")
    if promote_to_pack:
        _reject_bundled_example(dataset_pack, what="Promoting eval data into the pack")

    report = generate_eval_starter(
        input_path=input_path,
        dataset_pack=dataset_pack,
        out_path=out_path,
        max_items=max_items,
        seed=seed,
        force=force,
    )
    promoted_pack = update_dataset_pack_tuning_paths(dataset_pack, eval_input=out_path) if dataset_pack and promote_to_pack else None
    command_parts = [
        "picochat",
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
        "promoted_to_pack": promoted_pack is not None,
        "pack_chat_input": promoted_pack.chat_input if promoted_pack else None,
        "pack_eval_input": promoted_pack.eval_input if promoted_pack else None,
        "command": _shell_command(*command_parts),
        "report_path": str(Path(out_path).with_suffix(".md")),
        "next_actions": [
            "Open the generated eval JSONL and replace generic wording with domain-specific prompts.",
            "Keep refusal and memorization-probe rows before trusting a custom SLM score.",
            "Run tuning inspection after editing so Picochat can grade the eval file before training.",
        ],
    }


def benchmark_tuning_pack_plan(payload: dict) -> dict:
    """Generate a curated benchmark SFT/eval pair from a web request."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    dataset_pack = _optional_string(payload.get("dataset_pack"))
    if not dataset_pack:
        raise ValueError("dataset_pack is required")
    chat_out = _optional_string(payload.get("chat_out"))
    eval_out = _optional_string(payload.get("eval_out"))
    sft_rows = _bounded_int(payload.get("sft_rows", DEFAULT_BENCHMARK_SFT_ROWS), 32, 2000)
    eval_rows = _bounded_int(payload.get("eval_rows", DEFAULT_BENCHMARK_EVAL_ROWS), 16, 500)
    seed = _bounded_int(payload.get("seed", 42), 0, 9999)
    source = _optional_string(payload.get("source")) or "offline"
    if source not in BENCHMARK_SOURCES:
        raise ValueError(f"source must be one of {', '.join(BENCHMARK_SOURCES)}")
    profile = _optional_string(payload.get("profile")) or "full"
    if profile not in BENCHMARK_PROFILES:
        raise ValueError(f"profile must be one of {', '.join(BENCHMARK_PROFILES)}")
    skill_answer_style = _optional_string(payload.get("skill_answer_style")) or "direct"
    if skill_answer_style not in BENCHMARK_SKILL_ANSWER_STYLES:
        raise ValueError(f"skill_answer_style must be one of {', '.join(BENCHMARK_SKILL_ANSWER_STYLES)}")
    force = payload.get("force", False)
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")
    promote_to_pack = payload.get("promote_to_pack", True)
    if not isinstance(promote_to_pack, bool):
        raise ValueError("promote_to_pack must be true or false")
    # Benchmark always writes its chat/eval pair next to the pack (or to the
    # given out paths) and re-points the pack, so the pack itself is off-limits
    # when it is a shipped example.
    _reject_bundled_example(chat_out, what="Building a benchmark pack")
    _reject_bundled_example(eval_out, what="Building a benchmark pack")
    _reject_bundled_example(dataset_pack, what="Building a benchmark pack")

    report = generate_benchmark_tuning_pack(
        dataset_pack=dataset_pack,
        chat_out=chat_out,
        eval_out=eval_out,
        sft_rows=sft_rows,
        eval_rows=eval_rows,
        seed=seed,
        source=source,
        profile=profile,
        skill_answer_style=skill_answer_style,
        force=force,
        promote_to_pack=promote_to_pack,
    )
    command_parts = [
        "picochat",
        "data",
        "benchmark-pack",
        "--dataset-pack",
        dataset_pack,
        "--sft-rows",
        str(sft_rows),
        "--eval-rows",
        str(eval_rows),
        "--source",
        source,
        "--profile",
        profile,
        "--skill-answer-style",
        skill_answer_style,
        "--seed",
        str(seed),
    ]
    if chat_out:
        command_parts.extend(["--chat-out", chat_out])
    if eval_out:
        command_parts.extend(["--eval-out", eval_out])
    if force:
        command_parts.append("--force")
    if not promote_to_pack:
        command_parts.append("--no-promote")
    return {
        **report.to_dict(),
        "command": _shell_command(*command_parts),
        "chat_data": inspect_chat_sft_data(report.pack_chat_input or report.chat_output_path).to_dict(),
        "eval_data": inspect_chat_eval_data(report.pack_eval_input or report.eval_output_path).to_dict(),
        "next_actions": [
            "Preview the dataset pack again so the launcher reads the curated chat/eval files.",
            "Run a smoke or small-local experiment before scaling the base run.",
            "Use failed eval categories to add targeted non-eval SFT rows, not copied eval prompts.",
        ],
    }


def preference_starter_plan(payload: dict) -> dict:
    """Generate starter DPO preference pairs from chat SFT rows."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    input_path = _optional_string(payload.get("input_path"))
    out_path = _optional_string(payload.get("out_path"))
    if not input_path:
        raise ValueError("input_path is required")
    if not out_path:
        raise ValueError("out_path is required")
    max_rows = _bounded_int(payload.get("max_rows", 0), 0, 10_000)
    force = payload.get("force", False)
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")
    _reject_bundled_example(out_path, what="Generating preference pairs")

    report = generate_preference_starter(PreferenceStarterConfig(
        input_path=input_path,
        output_path=out_path,
        max_rows=max_rows,
        force=force,
    ))
    command_parts = [
        "picochat",
        "data",
        "preference-starter",
        "--input",
        input_path,
        "--out",
        out_path,
    ]
    if max_rows:
        command_parts.extend(["--max-rows", str(max_rows)])
    if force:
        command_parts.append("--force")
    return {
        **report,
        "force": force,
        "command": _shell_command(*command_parts),
        "next_actions": [
            "Use this only for DPO plumbing smoke tests unless a human or judge has reviewed the preferences.",
            "Point DPO PREFS at this file, run a short DPO stage, then compare eval and refusal behavior.",
            "For release alignment, replace synthetic rejected answers with curated preference pairs.",
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

    preflight_only = bool(payload.get("preflight_only", False))
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
    base_resume_from = _optional_string(payload.get("base_resume_from"))
    sft_resume_from = _optional_string(payload.get("sft_resume_from"))
    if sft_resume_from and not base_resume_from:
        raise ValueError("sft_resume_from requires base_resume_from so the base phase is not retrained first")
    for label, resume_path in (("base_resume_from", base_resume_from), ("sft_resume_from", sft_resume_from)):
        if resume_path and not Path(resume_path).is_dir():
            raise FileNotFoundError(f"{label} must point to a checkpoint directory: {resume_path}")
    is_resume_launch = bool(base_resume_from or sft_resume_from)
    if out_dir.exists() and any(out_dir.iterdir()) and not is_resume_launch and not preflight_only:
        raise FileExistsError(f"run output already exists: {out_dir}")

    preset_name = _optional_string(payload.get("preset")) or "smoke"
    preset = _run_preset(preset_name)
    context_size = _bounded_int(payload.get("context_size", preset["context_size"]), 8, 8192)
    base_steps = _bounded_int(payload.get("base_steps", preset["base_steps"]), 1, 100000)
    sft_steps = _bounded_int(payload.get("sft_steps", preset["sft_steps"]), 1, 10000)
    base_batch_size = _bounded_int(payload.get("base_batch_size", preset["base_batch_size"]), 1, 64)
    sft_batch_size = _bounded_int(payload.get("sft_batch_size", preset["sft_batch_size"]), 1, 64)
    seed = _bounded_int(payload.get("seed", 42), 0, 9999)
    eval_max_new_tokens = _bounded_int(payload.get("eval_max_new_tokens", preset["eval_max_new_tokens"]), 1, 320)
    n_embd = _bounded_int(payload.get("n_embd", preset["n_embd"]), 16, 4096)
    n_head = _bounded_int(payload.get("n_head", preset["n_head"]), 1, 128)
    n_kv_head = _bounded_int(payload.get("n_kv_head", preset.get("n_kv_head", n_head)), 1, 128)
    n_layer = _bounded_int(payload.get("n_layer", preset["n_layer"]), 1, 128)
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if n_head % n_kv_head != 0:
        raise ValueError("n_head must be divisible by n_kv_head")
    norm_type = str(payload.get("norm_type", preset.get("norm_type", "layernorm")))
    if norm_type not in {"layernorm", "rmsnorm"}:
        raise ValueError("norm_type must be layernorm or rmsnorm")
    position_encoding = str(payload.get("position_encoding", preset.get("position_encoding", "learned")))
    if position_encoding not in {"learned", "rope"}:
        raise ValueError("position_encoding must be learned or rope")
    if position_encoding == "rope" and (n_embd // n_head) % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension")
    activation = str(payload.get("activation", preset.get("activation", "gelu")))
    if activation not in {"gelu", "relu2", "leaky_relu2", "swiglu"}:
        raise ValueError("activation must be gelu, relu2, leaky_relu2, or swiglu")
    tie_embeddings = bool(payload.get("tie_embeddings", preset.get("tie_embeddings", False)))
    qk_norm = bool(payload.get("qk_norm", preset.get("qk_norm", False)))
    parallel_residual = bool(payload.get("parallel_residual", preset.get("parallel_residual", False)))
    xsa_last_n = _bounded_int(payload.get("xsa_last_n", preset.get("xsa_last_n", 0)), 0, 128)
    scaled_residual_init = bool(payload.get("scaled_residual_init", preset.get("scaled_residual_init", False)))
    precision = str(payload.get("precision", preset.get("precision", "float32")))
    if precision not in PRECISION_MODES:
        raise ValueError(f"precision must be one of {', '.join(PRECISION_MODES)}")
    matmul_precision = str(payload.get("matmul_precision", preset.get("matmul_precision", "default")))
    if matmul_precision not in MATMUL_PRECISION_MODES:
        raise ValueError(f"matmul_precision must be one of {', '.join(MATMUL_PRECISION_MODES)}")
    attn_backend = str(payload.get("attn_backend", preset.get("attn_backend", "auto")))
    if attn_backend not in SDPA_BACKENDS:
        raise ValueError(f"attn_backend must be one of {', '.join(SDPA_BACKENDS)}")
    torch_compile = bool(payload.get("torch_compile", preset.get("torch_compile", False)))
    torch_compile_mode = str(payload.get("torch_compile_mode", preset.get("torch_compile_mode", "default")))
    if torch_compile_mode not in COMPILE_MODES:
        raise ValueError(f"torch_compile_mode must be one of {', '.join(COMPILE_MODES)}")
    gradient_checkpointing = bool(payload.get("gradient_checkpointing", preset.get("gradient_checkpointing", False)))
    tensorboard_log_dir = _optional_string(payload.get("tensorboard_log_dir"))
    auto_lr_scaling = bool(payload.get("auto_lr_scaling", preset.get("auto_lr_scaling", False)))
    loss_spike_rollback = bool(payload.get("loss_spike_rollback", preset.get("loss_spike_rollback", False)))
    tokenizer_type = str(payload.get("tokenizer_type", preset.get("tokenizer_type", "char")))
    if tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(f"tokenizer_type must be one of {', '.join(TOKENIZER_TYPES)}")
    bpe_pretokenizer = str(payload.get("bpe_pretokenizer", preset.get("bpe_pretokenizer", DEFAULT_BPE_PRETOKENIZER)))
    if bpe_pretokenizer not in BPE_PRETOKENIZERS:
        raise ValueError(f"bpe_pretokenizer must be one of {', '.join(BPE_PRETOKENIZERS)}")
    tokenizer_vocab_size = _optional_int(
        payload.get("tokenizer_vocab_size", preset.get("tokenizer_vocab_size")),
        minimum=4,
        maximum=32768,
    )
    tokenizer_min_freq = _bounded_int(payload.get("tokenizer_min_freq", preset.get("tokenizer_min_freq", 1)), 1, 1000)
    base_learning_rate = _bounded_float(payload.get("base_learning_rate", preset.get("base_learning_rate", 3e-4)), 0.0, 1.0)
    sft_learning_rate = _bounded_float(payload.get("sft_learning_rate", preset.get("sft_learning_rate", 3e-4)), 0.0, 1.0)
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
    base_grad_accum_steps = _bounded_int(
        payload.get("base_grad_accum_steps", preset.get("base_grad_accum_steps", 1)),
        1,
        128,
    )
    base_dataset_mode = str(payload.get("base_dataset_mode", preset.get("base_dataset_mode", "memory")))
    if base_dataset_mode not in {"memory", "sharded", "packed"}:
        raise ValueError("base_dataset_mode must be memory, sharded, or packed")
    base_shard_token_size = _bounded_int(
        payload.get("base_shard_token_size", preset.get("base_shard_token_size", 1_000_000)),
        1_000,
        50_000_000,
    )
    base_shard_cache_size = _bounded_int(
        payload.get("base_shard_cache_size", preset.get("base_shard_cache_size", 2)),
        1,
        16,
    )
    sft_grad_accum_steps = _bounded_int(
        payload.get("sft_grad_accum_steps", preset.get("sft_grad_accum_steps", 1)),
        1,
        128,
    )
    base_optimizer = str(payload.get("base_optimizer", preset.get("base_optimizer", "adamw")))
    sft_optimizer = str(payload.get("sft_optimizer", preset.get("sft_optimizer", "adamw")))
    if base_optimizer not in OPTIMIZER_TYPES or sft_optimizer not in OPTIMIZER_TYPES:
        raise ValueError(f"optimizer must be one of {', '.join(OPTIMIZER_TYPES)}")
    base_muon_learning_rate = _bounded_float(
        payload.get("base_muon_learning_rate", preset.get("base_muon_learning_rate", 0.02)),
        0.000001,
        1.0,
    )
    sft_muon_learning_rate = _bounded_float(
        payload.get("sft_muon_learning_rate", preset.get("sft_muon_learning_rate", 0.02)),
        0.000001,
        1.0,
    )
    base_ema_decay = _bounded_float(payload.get("base_ema_decay", preset.get("base_ema_decay", 0.0)), 0.0, 0.9999)
    sft_ema_decay = _bounded_float(payload.get("sft_ema_decay", preset.get("sft_ema_decay", 0.0)), 0.0, 0.9999)
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
    sft_packing = str(payload.get("sft_packing", preset.get("sft_packing", "separate")))
    if sft_packing not in SFT_PACKING_MODES:
        raise ValueError(f"sft_packing must be one of {', '.join(SFT_PACKING_MODES)}")
    sft_peft = str(payload.get("sft_peft", preset.get("sft_peft", "none")))
    if sft_peft not in PEFT_MODES:
        raise ValueError(f"sft_peft must be one of {', '.join(PEFT_MODES)}")
    sft_lora_rank = _bounded_int(payload.get("sft_lora_rank", preset.get("sft_lora_rank", 8)), 1, 256)
    sft_lora_alpha = _bounded_float(payload.get("sft_lora_alpha", preset.get("sft_lora_alpha", 16.0)), 0.000001, 1024.0)
    sft_lora_dropout = _bounded_float(payload.get("sft_lora_dropout", preset.get("sft_lora_dropout", 0.0)), 0.0, 0.9999)
    sft_lora_targets = parse_lora_targets(
        payload.get("sft_lora_targets", preset.get("sft_lora_targets", DEFAULT_LORA_TARGETS))
    )
    if sft_peft == "none" and (
        sft_lora_rank != 8
        or sft_lora_alpha != 16.0
        or sft_lora_dropout != 0.0
        or sft_lora_targets != DEFAULT_LORA_TARGETS
    ):
        raise ValueError("LoRA options require sft_peft=lora")
    dpo_input = _optional_string(payload.get("dpo_input"))
    if dpo_input and not Path(dpo_input).is_file():
        raise FileNotFoundError(f"dpo_input must point to a preference JSONL file: {dpo_input}")
    dpo_steps = _bounded_int(payload.get("dpo_steps", preset.get("dpo_steps", 0)), 0, 10000)
    if dpo_input and dpo_steps <= 0:
        raise ValueError("dpo_input requires dpo_steps > 0")
    dpo_batch_size = _bounded_int(payload.get("dpo_batch_size", preset.get("dpo_batch_size", 4)), 1, 64)
    dpo_learning_rate = _bounded_float(payload.get("dpo_learning_rate", preset.get("dpo_learning_rate", 5e-6)), 0.000001, 1.0)
    dpo_beta = _bounded_float(payload.get("dpo_beta", preset.get("dpo_beta", 0.1)), 0.000001, 10.0)
    dpo_grad_accum_steps = _bounded_int(
        payload.get("dpo_grad_accum_steps", preset.get("dpo_grad_accum_steps", 1)),
        1,
        128,
    )
    dpo_lr_warmup_steps = _bounded_int(
        payload.get("dpo_lr_warmup_steps", preset.get("dpo_lr_warmup_steps", 0)),
        0,
        max(1, dpo_steps),
    )
    dpo_lr_decay = str(payload.get("dpo_lr_decay", preset.get("dpo_lr_decay", "none")))
    if dpo_lr_decay not in LR_DECAYS:
        raise ValueError(f"dpo_lr_decay must be one of {', '.join(LR_DECAYS)}")
    dpo_grad_clip = _bounded_float(payload.get("dpo_grad_clip", preset.get("dpo_grad_clip", 0.0)), 0.0, 100.0)
    dpo_early_stop_patience = _bounded_int(
        payload.get("dpo_early_stop_patience", preset.get("dpo_early_stop_patience", 4)),
        0,
        100,
    )
    dpo_eval_batches = _bounded_int(payload.get("dpo_eval_batches", preset.get("dpo_eval_batches", 10)), 0, 200)
    dpo_length_normalize = bool(payload.get("dpo_length_normalize", preset.get("dpo_length_normalize", False)))
    target_param_data_ratio = _bounded_float(
        payload.get("target_param_data_ratio", preset.get("target_param_data_ratio", 20.0)),
        1.0,
        200.0,
    )
    long_run_gate_profile = str(payload.get("long_run_gate_profile", preset.get("long_run_gate_profile", "research")))
    if long_run_gate_profile not in LONG_RUN_GATE_PROFILES:
        raise ValueError(f"long_run_gate_profile must be one of: {', '.join(LONG_RUN_GATE_PROFILES)}")
    device = str(payload.get("device", preset.get("device", "cpu"))).strip().lower()
    if device not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {', '.join(DEVICE_CHOICES)}")
    ddp = bool(payload.get("ddp", preset.get("ddp", False)))
    ddp_world_size = _bounded_int(payload.get("ddp_world_size", 8 if ddp else 1), 1, 128)
    allow_unsafe_long_run = bool(payload.get("allow_unsafe_long_run", False))
    if ddp and loss_spike_rollback:
        raise ValueError(
            "loss spike rollback is not supported with DDP because rollback decisions "
            "are rank-local; disable rollback before launching distributed training"
        )

    preflight_config = TinyRunConfig(
        out_dir=str(out_dir),
        scale=preset_name,
        dataset_pack=dataset_pack,
        context_size=context_size,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_layer=n_layer,
        norm_type=norm_type,
        position_encoding=position_encoding,
        activation=activation,
        tie_embeddings=tie_embeddings,
        qk_norm=qk_norm,
        parallel_residual=parallel_residual,
        xsa_last_n=xsa_last_n,
        scaled_residual_init=scaled_residual_init,
        attn_backend=attn_backend,
        base_steps=base_steps,
        sft_steps=sft_steps,
        base_batch_size=base_batch_size,
        sft_batch_size=sft_batch_size,
        base_learning_rate=base_learning_rate,
        sft_learning_rate=sft_learning_rate,
        seed=seed,
        device=device,
        precision=precision,
        matmul_precision=matmul_precision,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        gradient_checkpointing=gradient_checkpointing,
        tensorboard_log_dir=tensorboard_log_dir,
        ddp=ddp,
        ddp_world_size=ddp_world_size,
        eval_max_new_tokens=eval_max_new_tokens,
        min_quality_score=min_quality_score,
        split_mode="document",
        tokenizer_type=tokenizer_type,
        tokenizer_vocab_size=tokenizer_vocab_size,
        tokenizer_min_freq=tokenizer_min_freq,
        bpe_pretokenizer=bpe_pretokenizer,
        base_early_stop_patience=base_early_stop_patience,
        sft_early_stop_patience=sft_early_stop_patience,
        base_lr_warmup_steps=base_lr_warmup_steps,
        sft_lr_warmup_steps=sft_lr_warmup_steps,
        base_lr_decay=base_lr_decay,
        sft_lr_decay=sft_lr_decay,
        base_min_lr_ratio=base_min_lr_ratio,
        sft_min_lr_ratio=sft_min_lr_ratio,
        base_grad_clip=base_grad_clip,
        sft_grad_clip=sft_grad_clip,
        base_grad_accum_steps=base_grad_accum_steps,
        base_dataset_mode=base_dataset_mode,
        base_shard_token_size=base_shard_token_size,
        base_shard_cache_size=base_shard_cache_size,
        sft_grad_accum_steps=sft_grad_accum_steps,
        base_optimizer=base_optimizer,
        sft_optimizer=sft_optimizer,
        base_muon_learning_rate=base_muon_learning_rate,
        sft_muon_learning_rate=sft_muon_learning_rate,
        base_ema_decay=base_ema_decay,
        sft_ema_decay=sft_ema_decay,
        sft_sampling=sft_sampling,
        sft_packing=sft_packing,
        sft_peft=sft_peft,
        sft_lora_rank=sft_lora_rank,
        sft_lora_alpha=sft_lora_alpha,
        sft_lora_dropout=sft_lora_dropout,
        sft_lora_targets=sft_lora_targets,
        dpo_input=dpo_input,
        dpo_steps=dpo_steps,
        dpo_batch_size=dpo_batch_size,
        dpo_learning_rate=dpo_learning_rate,
        dpo_beta=dpo_beta,
        dpo_grad_accum_steps=dpo_grad_accum_steps,
        dpo_lr_warmup_steps=dpo_lr_warmup_steps,
        dpo_lr_decay=dpo_lr_decay,
        dpo_grad_clip=dpo_grad_clip,
        dpo_early_stop_patience=dpo_early_stop_patience,
        dpo_eval_batches=dpo_eval_batches,
        dpo_length_normalize=dpo_length_normalize,
        auto_lr_scaling=auto_lr_scaling,
        loss_spike_rollback=loss_spike_rollback,
        allow_unsafe_long_run=allow_unsafe_long_run,
        base_resume_from=base_resume_from,
        sft_resume_from=sft_resume_from,
        target_param_data_ratio=target_param_data_ratio,
        long_run_gate_profile=long_run_gate_profile,
    )
    launch_preflight = assess_run_preflight(preflight_config, launch_preview)
    if launch_preflight.status == "blocked" and not allow_unsafe_long_run and not preflight_only:
        blocking = ", ".join(check.name for check in launch_preflight.checks if check.status == "block")
        raise ValueError(
            "long-run preflight blocked this launch. "
            f"Blocking checks: {blocking}. "
            "Use a smoke run, add more data/tuning rows, reduce exposure, or explicitly allow unsafe diagnostic runs."
        )

    log_path = out_dir / "web_run.log"
    command = [
        sys.executable,
        "-m",
    ]
    if ddp:
        command.extend([
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ddp_world_size}",
            "-m",
        ])
    command.extend([
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
        "--n-kv-head",
        str(n_kv_head),
        "--n-layer",
        str(n_layer),
        "--norm-type",
        norm_type,
        "--position-encoding",
        position_encoding,
        "--activation",
        activation,
        "--xsa-last-n",
        str(xsa_last_n),
        "--attn-backend",
        attn_backend,
        "--base-steps",
        str(base_steps),
        "--sft-steps",
        str(sft_steps),
        "--base-batch-size",
        str(base_batch_size),
        "--sft-batch-size",
        str(sft_batch_size),
        "--base-learning-rate",
        str(base_learning_rate),
        "--sft-learning-rate",
        str(sft_learning_rate),
        "--seed",
        str(seed),
        "--eval-max-new-tokens",
        str(eval_max_new_tokens),
        "--precision",
        precision,
        "--matmul-precision",
        matmul_precision,
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-min-freq",
        str(tokenizer_min_freq),
        "--bpe-pretokenizer",
        bpe_pretokenizer,
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
        "--base-grad-accum-steps",
        str(base_grad_accum_steps),
        "--base-dataset-mode",
        base_dataset_mode,
        "--sft-grad-accum-steps",
        str(sft_grad_accum_steps),
        "--base-optimizer",
        base_optimizer,
        "--sft-optimizer",
        sft_optimizer,
        "--base-muon-learning-rate",
        str(base_muon_learning_rate),
        "--sft-muon-learning-rate",
        str(sft_muon_learning_rate),
        "--base-ema-decay",
        str(base_ema_decay),
        "--sft-ema-decay",
        str(sft_ema_decay),
        "--base-early-stop-patience",
        str(base_early_stop_patience),
        "--sft-early-stop-patience",
        str(sft_early_stop_patience),
        "--sft-sampling",
        sft_sampling,
        "--sft-packing",
        sft_packing,
        "--sft-peft",
        sft_peft,
        "--sft-lora-rank",
        str(sft_lora_rank),
        "--sft-lora-alpha",
        str(sft_lora_alpha),
        "--sft-lora-dropout",
        str(sft_lora_dropout),
        "--sft-lora-targets",
        ",".join(sft_lora_targets),
        "--target-param-data-ratio",
        str(target_param_data_ratio),
        "--long-run-gate-profile",
        long_run_gate_profile,
        "--split-mode",
        "document",
        "--min-score",
        str(min_quality_score),
        "--device",
        device,
    ])
    if preset_name in RUN_SCALES:
        command.extend(["--scale", preset_name])
    if tie_embeddings:
        command.append("--tie-embeddings")
    if qk_norm:
        command.append("--qk-norm")
    if parallel_residual:
        command.append("--parallel-residual")
    if scaled_residual_init:
        command.append("--scaled-residual-init")
    if torch_compile:
        command.append("--torch-compile")
        command.extend(["--torch-compile-mode", torch_compile_mode])
    if gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if tensorboard_log_dir:
        command.extend(["--tensorboard-log-dir", tensorboard_log_dir])
    if auto_lr_scaling:
        command.append("--auto-lr-scaling")
    if loss_spike_rollback:
        command.append("--loss-spike-rollback")
    if ddp:
        command.append("--ddp")
        command.extend(["--ddp-world-size", str(ddp_world_size)])
    if base_resume_from:
        command.extend(["--base-resume-from", base_resume_from])
    if sft_resume_from:
        command.extend(["--sft-resume-from", sft_resume_from])
    if tokenizer_vocab_size is not None:
        command.extend(["--tokenizer-vocab-size", str(tokenizer_vocab_size)])
    if base_dataset_mode == "sharded":
        command.extend(["--base-shard-token-size", str(base_shard_token_size)])
        command.extend(["--base-shard-cache-size", str(base_shard_cache_size)])
    if dpo_input:
        command.extend([
            "--dpo-input",
            dpo_input,
            "--dpo-steps",
            str(dpo_steps),
            "--dpo-batch-size",
            str(dpo_batch_size),
            "--dpo-learning-rate",
            str(dpo_learning_rate),
            "--dpo-beta",
            str(dpo_beta),
            "--dpo-grad-accum-steps",
            str(dpo_grad_accum_steps),
            "--dpo-lr-warmup-steps",
            str(dpo_lr_warmup_steps),
            "--dpo-lr-decay",
            dpo_lr_decay,
            "--dpo-grad-clip",
            str(dpo_grad_clip),
            "--dpo-early-stop-patience",
            str(dpo_early_stop_patience),
            "--dpo-eval-batches",
            str(dpo_eval_batches),
        ])
        if dpo_length_normalize:
            command.append("--dpo-length-normalize")
    if allow_unsafe_long_run:
        command.append("--allow-unsafe-long-run")
    launch_config = {
        "context_size": context_size,
        "base_steps": base_steps,
        "sft_steps": sft_steps,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_layer": n_layer,
        "norm_type": norm_type,
        "position_encoding": position_encoding,
        "activation": activation,
        "tie_embeddings": tie_embeddings,
        "qk_norm": qk_norm,
        "parallel_residual": parallel_residual,
        "xsa_last_n": xsa_last_n,
        "scaled_residual_init": scaled_residual_init,
        "attn_backend": attn_backend,
        "precision": precision,
        "matmul_precision": matmul_precision,
        "torch_compile": torch_compile,
        "torch_compile_mode": torch_compile_mode,
        "gradient_checkpointing": gradient_checkpointing,
        "tensorboard_log_dir": tensorboard_log_dir,
        "auto_lr_scaling": auto_lr_scaling,
        "loss_spike_rollback": loss_spike_rollback,
        "tokenizer_type": tokenizer_type,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "bpe_pretokenizer": bpe_pretokenizer,
        "base_learning_rate": base_learning_rate,
        "sft_learning_rate": sft_learning_rate,
        "base_lr_decay": base_lr_decay,
        "sft_lr_decay": sft_lr_decay,
        "base_grad_clip": base_grad_clip,
        "sft_grad_clip": sft_grad_clip,
        "base_grad_accum_steps": base_grad_accum_steps,
        "base_dataset_mode": base_dataset_mode,
        "base_shard_token_size": base_shard_token_size,
        "base_shard_cache_size": base_shard_cache_size,
        "sft_grad_accum_steps": sft_grad_accum_steps,
        "base_optimizer": base_optimizer,
        "sft_optimizer": sft_optimizer,
        "base_muon_learning_rate": base_muon_learning_rate,
        "sft_muon_learning_rate": sft_muon_learning_rate,
        "base_ema_decay": base_ema_decay,
        "sft_ema_decay": sft_ema_decay,
        "base_early_stop_patience": base_early_stop_patience,
        "sft_early_stop_patience": sft_early_stop_patience,
        "sft_sampling": sft_sampling,
        "sft_packing": sft_packing,
        "sft_peft": sft_peft,
        "sft_lora_rank": sft_lora_rank,
        "sft_lora_alpha": sft_lora_alpha,
        "sft_lora_dropout": sft_lora_dropout,
        "sft_lora_targets": list(sft_lora_targets),
        "dpo_input": dpo_input,
        "dpo_steps": dpo_steps,
        "dpo_batch_size": dpo_batch_size,
        "dpo_learning_rate": dpo_learning_rate,
        "dpo_beta": dpo_beta,
        "dpo_grad_accum_steps": dpo_grad_accum_steps,
        "dpo_lr_warmup_steps": dpo_lr_warmup_steps,
        "dpo_lr_decay": dpo_lr_decay,
        "dpo_grad_clip": dpo_grad_clip,
        "dpo_early_stop_patience": dpo_early_stop_patience,
        "dpo_eval_batches": dpo_eval_batches,
        "dpo_length_normalize": dpo_length_normalize,
        "target_param_data_ratio": target_param_data_ratio,
        "long_run_gate_profile": long_run_gate_profile,
        "device": device,
        "ddp": ddp,
        "ddp_world_size": ddp_world_size,
        "base_resume_from": base_resume_from,
        "sft_resume_from": sft_resume_from,
        "allow_unsafe_long_run": allow_unsafe_long_run,
    }
    if preflight_only:
        job = {
            "id": f"preflight-{uuid.uuid4().hex[:12]}",
            "run_name": run_name,
            "out_dir": str(out_dir),
            "dataset_pack": dataset_pack,
            "log_path": str(log_path),
            "command": _shell_command(*command),
            "started_at": time.time(),
            "pid": None,
            "state": "preflight",
            "returncode": 0 if launch_preflight.status != "blocked" else 2,
            "elapsed_seconds": 0.0,
            "summary_exists": False,
            "log_tail": "Preflight-only dry run. No subprocess was launched.",
            "progress": None,
            "can_cancel": False,
            "source": "preflight",
            "updated_at": time.time(),
            "preset": preset_name,
            "min_quality_score": min_quality_score,
            "launch_config": launch_config,
            "launch_readiness": launch_preview.readiness.to_dict(),
            "launch_tuning": {
                "chat": launch_preview.chat_data.to_dict(),
                "eval": launch_preview.eval_data.to_dict(),
            },
            "launch_preflight": launch_preflight.to_dict(),
        }
        return {"job": job, "jobs": [job]}

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"$ {_shell_command(*command)}\n\n", encoding="utf-8")
    log_file = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=_child_env(device=device, ddp=ddp),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            # Own session/process group so the whole tree (incl. DDP workers)
            # can be signalled on cancel, and so the run survives a server exit.
            start_new_session=True,
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
        "pid": process.pid,
        "preset": preset_name,
        "min_quality_score": min_quality_score,
        "launch_config": launch_config,
        "launch_readiness": launch_preview.readiness.to_dict(),
        "launch_tuning": {
            "chat": launch_preview.chat_data.to_dict(),
            "eval": launch_preview.eval_data.to_dict(),
        },
        "launch_preflight": launch_preflight.to_dict(),
    }
    _register_run_job(job_id, job)
    # Persist a sidecar so the job survives a server restart: status discovery
    # and cancel can find the pid even after _RUN_JOBS is gone.
    _write_job_record(out_dir, job)
    return run_status_plan(job_id, runs_dir)


def hf_sft_start_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Fine-tune an existing Hugging Face model on a pack's chat data."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    model = _optional_string(payload.get("model"))
    if not model:
        raise ValueError("model is required")

    dataset_pack = _optional_string(payload.get("dataset_pack"))
    chat_input = _optional_string(payload.get("input"))
    if dataset_pack:
        chat_input = load_dataset_pack(dataset_pack).chat_input
    if not chat_input:
        raise ValueError("dataset_pack or input is required")
    if not Path(chat_input).is_file():
        raise FileNotFoundError(f"missing chat data: {chat_input}")

    run_name = _slug(_optional_string(payload.get("run_name")) or f"hf-sft-{_slug(model)}")
    out_dir = _safe_child(Path(runs_dir), run_name)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"run output already exists: {out_dir}")

    max_steps = _bounded_int(payload.get("max_steps", 100), 1, 100000)
    learning_rate = _bounded_float(payload.get("learning_rate", 2e-5), 0.0, 1.0)
    peft = str(payload.get("peft", "none"))
    if peft not in {"none", "lora"}:
        raise ValueError("peft must be none or lora")
    quantize = str(payload.get("quantize", "none"))
    if quantize not in {"none", "4bit"}:
        raise ValueError("quantize must be none or 4bit")
    if quantize == "4bit" and peft != "lora":
        raise ValueError("QLoRA (4-bit) requires LoRA — enable the LoRA adapter option.")
    device = str(payload.get("device", "auto")).strip().lower()
    if device not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {', '.join(DEVICE_CHOICES)}")
    trust_remote_code = bool(payload.get("trust_remote_code", False))
    revision = _optional_string(payload.get("revision"))

    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "picochat.cli", "train", "hf-sft",
        "--model", model,
        "--input", chat_input,
        "--out-dir", str(out_dir),
        "--max-steps", str(max_steps),
        "--learning-rate", str(learning_rate),
        "--device", device,
        "--peft", peft,
        "--quantize", quantize,
    ]
    if revision:
        command.extend(["--revision", revision])
    if trust_remote_code:
        command.append("--trust-remote-code")

    log_path = out_dir / "web_run.log"
    log_path.write_text(f"$ {_shell_command(*command)}\n\n", encoding="utf-8")
    log_file = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=_child_env(device=device),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
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
        "pid": process.pid,
        "preset": "hf-sft",
        "min_quality_score": 0,
        "launch_config": {
            "kind": "hf-sft",
            "model": model,
            "input": chat_input,
            "max_steps": max_steps,
            "peft": peft,
            "device": device,
        },
    }
    _register_run_job(job_id, job)
    _write_job_record(out_dir, job)
    return run_status_plan(job_id, runs_dir)


def hf_dpo_start_plan(runs_dir: str | Path, payload: dict) -> dict:
    """DPO-align a fine-tuned HF run from the dashboard (second stage after hf-sft)."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    src_run = _optional_string(payload.get("run")) or _optional_string(payload.get("model"))
    if not src_run:
        raise ValueError("run is required (the fine-tuned run to align)")
    preference_input = _optional_string(payload.get("preference_input"))
    if not preference_input:
        raise ValueError("preference_input is required (a JSONL with user/chosen/rejected rows)")
    _reject_bundled_example(preference_input, what="DPO")
    src_model_dir = Path(runs_dir) / src_run / "final_model"
    if not src_model_dir.exists():
        raise FileNotFoundError(f"no fine-tuned model at {src_model_dir}; run a fine-tune first")
    run_name = _slug(_optional_string(payload.get("run_name")) or f"{src_run}-dpo")
    out_dir = Path(runs_dir) / run_name
    max_steps = _bounded_int(payload.get("max_steps", 50), 1, 100000)
    beta = _bounded_float(payload.get("beta", 0.1), 0.01, 1.0)
    learning_rate = _bounded_float(payload.get("learning_rate", 5e-6), 1e-7, 1e-3)
    device = str(payload.get("device", "auto")).strip().lower()
    if device not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {', '.join(DEVICE_CHOICES)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "picochat.cli", "train", "hf-dpo",
        "--model", str(src_model_dir),
        "--input", preference_input,
        "--out-dir", str(out_dir),
        "--max-steps", str(max_steps),
        "--beta", str(beta),
        "--learning-rate", str(learning_rate),
        "--device", device,
    ]
    log_path = out_dir / "web_run.log"
    log_path.write_text(f"$ {_shell_command(*command)}\n\n", encoding="utf-8")
    log_file = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command, cwd=Path.cwd(), env=_child_env(device=device),
            stdout=log_file, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
    finally:
        log_file.close()

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id, "run_name": run_name, "out_dir": str(out_dir),
        "log_path": str(log_path), "command": _shell_command(*command),
        "started_at": time.time(), "process": process, "pid": process.pid,
        "preset": "hf-dpo", "min_quality_score": 0,
        "launch_config": {"kind": "hf-dpo", "model": str(src_model_dir), "input": preference_input, "max_steps": max_steps, "beta": beta, "device": device},
    }
    _register_run_job(job_id, job)
    _write_job_record(out_dir, job)
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


def run_log_plan(
    runs_dir: str | Path = "runs",
    *,
    run_name: str | None = None,
    job_id: str | None = None,
    limit: int = 50_000,
) -> dict:
    """Return a bounded live log tail for an active or persisted web-launched run."""
    scope = _optional_string(job_id) or _optional_string(run_name)
    if not scope:
        raise ValueError("run or job is required")
    limit = _bounded_int(limit, 1_000, 200_000)
    status = run_status_plan(scope, runs_dir)
    job = status.get("job")
    if not job:
        raise ValueError(f"unknown run job: {scope}")
    log_path = Path(job.get("log_path") or "")
    log_tail = _read_log_tail(log_path, limit=limit) if log_path else ""
    return {
        "job_id": job.get("id"),
        "run_name": job.get("run_name"),
        "state": job.get("state"),
        "running": job.get("state") == "running",
        "log_path": str(log_path) if log_path else "",
        "log_tail": log_tail,
        "progress": job.get("progress"),
        "updated_at": max(time.time(), float(job.get("updated_at") or 0)),
    }


def cancel_run_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Terminate a running web-launched job, including any orphaned process tree."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    job_id = _optional_string(payload.get("job_id"))
    if not job_id:
        raise ValueError("job_id is required")
    with _RUN_JOBS_LOCK:
        job = _RUN_JOBS.get(job_id)
    if job is not None:
        _terminate_job(job)
        return run_status_plan(job_id, runs_dir)

    # Not in memory (e.g. after a server restart): cancel via the persisted pid.
    record = _find_job_record(runs_dir, job_id)
    if record is None:
        raise ValueError(f"unknown active run job: {job_id}")
    pid = record.get("pid")
    if not _pid_alive(pid):
        raise ValueError(f"run job is not running: {job_id}")
    _kill_process_group(pid, signal.SIGTERM)
    time.sleep(0.05)
    _write_returncode(Path(record["out_dir"]) / "web_returncode.txt", -int(signal.SIGTERM))
    return run_status_plan(job_id, runs_dir)


def _find_free_port(start: int = 8001, span: int = 200) -> int:
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _serve_job_status(job: dict) -> dict:
    return {
        "run": job["run"],
        "host": job["host"],
        "port": job["port"],
        "api_key": job.get("api_key"),
        "pid": job.get("pid"),
        "model_name": job.get("model_name", "picochat"),
        "log_path": job.get("log_path"),
        "state": "running",
        "uptime_seconds": round(time.time() - job["started_at"], 1),
    }


def serve_status_plan(runs_dir: str | Path = "runs", *, run: str | None = None) -> dict:
    """List background serving processes, pruning any that have exited."""
    with _SERVE_JOBS_LOCK:
        for name, job in list(_SERVE_JOBS.items()):
            if job["process"].poll() is not None:
                _SERVE_JOBS.pop(name, None)
        jobs = list(_SERVE_JOBS.values())
    servers = [_serve_job_status(job) for job in jobs]
    selected = next((s for s in servers if s["run"] == run), None) if run else (servers[-1] if servers else None)
    return {"servers": servers, "server": selected}


def serve_start_plan(runs_dir: str | Path, payload: dict, *, host: str = "127.0.0.1") -> dict:
    """Launch an OpenAI-compatible `pico serve` for a run's SFT checkpoint."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    run_name = _optional_string(payload.get("run"))
    if not run_name:
        raise ValueError("run is required")
    run_dir = _safe_child(Path(runs_dir), run_name)
    hf_model_dir = run_dir / "final_model"
    checkpoint = run_dir / "sft" / "checkpoint"
    tokenizer = run_dir / "tokenizer.json"
    is_hf = hf_model_dir.exists() and not checkpoint.exists()
    if not is_hf:
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing SFT checkpoint for {run_name}: {checkpoint}")
        if not tokenizer.exists():
            raise FileNotFoundError(f"missing tokenizer for {run_name}: {tokenizer}")

    with _SERVE_JOBS_LOCK:
        existing = _SERVE_JOBS.get(run_name)
        alive = existing is not None and existing["process"].poll() is None
    if alive:
        return serve_status_plan(runs_dir, run=run_name)

    port = _find_free_port()
    api_key = None if _is_loopback_host(host) else secrets.token_urlsafe(16)
    serve_args = (
        ["--hf-model", str(hf_model_dir)] if is_hf
        else ["--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer)]
    )
    command = [
        sys.executable, "-m", "picochat.cli", "serve",
        *serve_args,
        "--host", host,
        "--port", str(port),
        "--model-name", run_name,
    ]
    if api_key:
        command.extend(["--api-key", api_key])
    log_path = run_dir / "web_serve.log"
    log_file = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=_child_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log_file.close()
    job = {
        "run": run_name,
        "host": host,
        "port": port,
        "api_key": api_key,
        "pid": process.pid,
        "process": process,
        "started_at": time.time(),
        "log_path": str(log_path),
        "model_name": run_name,
    }
    with _SERVE_JOBS_LOCK:
        _SERVE_JOBS[run_name] = job
    return serve_status_plan(runs_dir, run=run_name)


def serve_stop_plan(payload: dict) -> dict:
    """Stop a background serving process for a run."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    run_name = _optional_string(payload.get("run"))
    if not run_name:
        raise ValueError("run is required")
    with _SERVE_JOBS_LOCK:
        job = _SERVE_JOBS.pop(run_name, None)
    if job is None:
        raise ValueError(f"no server running for run: {run_name}")
    _terminate_job(job)
    return {"stopped": True, "run": run_name}


_MODAL_SCRIPT = Path("scripts/modal_picochat_train.py")


def remote_status_plan() -> dict:
    """Report which remote providers the local machine can drive directly."""
    return {
        "modal_available": shutil.which("modal") is not None,
        "modal_script": _MODAL_SCRIPT.exists(),
    }


def remote_modal_start_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Launch a Picochat training recipe on Modal as a tracked job."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    if shutil.which("modal") is None:
        raise ValueError(
            "the `modal` CLI is not installed or not on PATH. "
            "Install it with `pip install modal`, then run `modal token new`."
        )
    if not _MODAL_SCRIPT.exists():
        raise FileNotFoundError(f"missing Modal launch script: {_MODAL_SCRIPT}")

    repo_url = _optional_string(payload.get("repo_url")) or "https://github.com/gowtham0992/picochat.git"
    branch = _optional_string(payload.get("branch")) or "develop"
    run_name = _slug(_optional_string(payload.get("run_name")) or "picochat-modal-v1")
    scale = _optional_string(payload.get("scale")) or "h100-100m"
    gpu = _optional_string(payload.get("gpu")) or "A100"
    hf_dataset = _optional_string(payload.get("hf_dataset")) or "karpathy/climbmix-400b-shuffle"
    hf_max_rows = _bounded_int(payload.get("hf_max_rows", 800_000), 1, 100_000_000)
    secret_name = _optional_string(payload.get("secret_name"))

    out_dir = _safe_child(Path(runs_dir), run_name)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"run output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "modal", "run", str(_MODAL_SCRIPT),
        "--repo-url", repo_url,
        "--branch", branch,
        "--run-name", run_name,
        "--scale", scale,
        "--gpu", gpu,
        "--hf-dataset", hf_dataset,
        "--hf-max-rows", str(hf_max_rows),
    ]
    if secret_name:
        command.extend(["--secret-name", secret_name])

    log_path = out_dir / "web_run.log"
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
            start_new_session=True,
        )
    finally:
        log_file.close()

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "run_name": run_name,
        "out_dir": str(out_dir),
        "dataset_pack": None,
        "log_path": str(log_path),
        "command": _shell_command(*command),
        "started_at": time.time(),
        "process": process,
        "pid": process.pid,
        "preset": "modal",
        "min_quality_score": 0,
        "launch_config": {
            "kind": "modal",
            "gpu": gpu,
            "scale": scale,
            "hf_dataset": hf_dataset,
            "run_name": run_name,
        },
    }
    _register_run_job(job_id, job)
    _write_job_record(out_dir, job)
    return run_status_plan(job_id, runs_dir)


def remote_modal_pull_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Download a finished Modal run from its volume into local runs/."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    if shutil.which("modal") is None:
        raise ValueError(
            "the `modal` CLI is not installed or not on PATH. "
            "Install it with `pip install modal`, then run `modal token new`."
        )
    run_name = _slug(_optional_string(payload.get("run")) or "")
    if not run_name:
        raise ValueError("run is required")
    volume = _optional_string(payload.get("volume")) or "picochat-runs"

    dest_root = Path(runs_dir)
    landed = _safe_child(dest_root, run_name)
    if landed.exists() and any(landed.iterdir()):
        raise FileExistsError(f"run already exists locally: {landed}")
    dest_root.mkdir(parents=True, exist_ok=True)
    track_dir = dest_root / ".pulls" / run_name
    track_dir.mkdir(parents=True, exist_ok=True)

    command = ["modal", "volume", "get", volume, run_name, str(dest_root.resolve())]
    log_path = track_dir / "web_run.log"
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
            start_new_session=True,
        )
    finally:
        log_file.close()

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "run_name": f"pull-{run_name}",
        "out_dir": str(track_dir),
        "dataset_pack": None,
        "log_path": str(log_path),
        "command": _shell_command(*command),
        "started_at": time.time(),
        "process": process,
        "pid": process.pid,
        "preset": "modal-pull",
        "min_quality_score": 0,
        "launch_config": {"kind": "modal-pull", "run": run_name, "volume": volume},
    }
    _register_run_job(job_id, job)
    return run_status_plan(job_id, runs_dir)


def export_hf_run_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Export a run's SFT checkpoint to a Hugging Face model folder + card."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    run_name = _optional_string(payload.get("run"))
    if not run_name:
        raise ValueError("run is required")
    run_dir = _safe_child(Path(runs_dir), run_name)
    checkpoint = run_dir / "sft" / "checkpoint"
    tokenizer = run_dir / "tokenizer.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing SFT checkpoint for {run_name}: {checkpoint}")
    if not tokenizer.exists():
        raise FileNotFoundError(f"missing tokenizer for {run_name}: {tokenizer}")

    out_dir = run_dir / "export-hf"
    from picochat.hf_export import HFExportConfig, export_hf_checkpoint
    report = export_hf_checkpoint(HFExportConfig(
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(tokenizer),
        out_dir=str(out_dir),
        model_name=run_name,
        base_model=False,
    ))
    return {
        "run": run_name,
        "out_dir": report.get("out_dir", str(out_dir)),
        "manifest": report.get("manifest"),
        "model_card": report.get("model_card"),
    }


def eval_run_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Re-run the transparent chat eval on a run's SFT checkpoint."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    run_name = _optional_string(payload.get("run"))
    if not run_name:
        raise ValueError("run is required")
    run_dir = _safe_child(Path(runs_dir), run_name)
    checkpoint = run_dir / "sft" / "checkpoint"
    tokenizer = run_dir / "tokenizer.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing SFT checkpoint for {run_name}: {checkpoint}")
    if not tokenizer.exists():
        raise FileNotFoundError(f"missing tokenizer for {run_name}: {tokenizer}")

    eval_input = _optional_string(payload.get("input"))
    if not eval_input:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            pack = _read_json(summary_path).get("config", {}).get("dataset_pack")
            if pack and Path(pack).exists():
                eval_input = load_dataset_pack(pack).eval_input
    if not eval_input or not Path(eval_input).is_file():
        raise ValueError(
            "could not resolve an eval set; pass `input`, or keep the run's dataset pack available."
        )

    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "picochat.cli", "eval", "chat",
        "--input", eval_input,
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(tokenizer),
        "--out-dir", str(out_dir),
        "--device", "cpu",
    ]
    log_path = out_dir / "web_run.log"
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
            start_new_session=True,
        )
    finally:
        log_file.close()

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "run_name": f"eval-{run_name}",
        "out_dir": str(out_dir),
        "dataset_pack": None,
        "log_path": str(log_path),
        "command": _shell_command(*command),
        "started_at": time.time(),
        "process": process,
        "pid": process.pid,
        "preset": "eval",
        "min_quality_score": 0,
        "launch_config": {"kind": "eval", "run": run_name, "input": eval_input},
    }
    _register_run_job(job_id, job)
    return run_status_plan(job_id, runs_dir)


def archive_run_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Move a completed run out of the active run bank without deleting it."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    run_names = _archive_run_names(payload)

    root = Path(runs_dir)
    run_dirs = []
    for run_name in run_names:
        run_dir = _safe_child(root, run_name)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"missing run folder: {run_dir}")
        active_job = _active_job_for_run(run_dir)
        if active_job:
            raise ValueError(f"cannot archive running run: {run_name}. Cancel or wait for it first.")
        run_dirs.append((run_name, run_dir))

    archive_root = root / f"archive-{date.today().isoformat()}"
    archive_root.mkdir(parents=True, exist_ok=True)
    archived_runs = []
    for run_name, run_dir in run_dirs:
        destination = _unique_archive_path(archive_root / run_dir.name)
        shutil.move(str(run_dir), str(destination))
        archived_runs.append({
            "run_name": run_name,
            "source": str(run_dir),
            "archive_path": str(destination),
            "summary_exists": (destination / "summary.json").exists(),
        })
    archived_names = {item["run_name"] for item in archived_runs}
    with _RUN_JOBS_LOCK:
        for job_id, job in list(_RUN_JOBS.items()):
            if job.get("run_name") in archived_names:
                _RUN_JOBS.pop(job_id, None)
    return {
        "archived": True,
        "run_name": archived_runs[0]["run_name"],
        "source": archived_runs[0]["source"],
        "archive_path": archived_runs[0]["archive_path"],
        "archive_root": str(archive_root),
        "archived_runs": archived_runs,
        "runs": discover_runs(root),
    }


def import_run_plan(runs_dir: str | Path, payload: dict) -> dict:
    """Copy a completed external run folder into the active run bank."""
    source = _optional_string(payload.get("source_path"))
    if not source:
        raise ValueError("source_path is required")
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"missing imported run folder: {source_path}")
    if not source_path.is_dir():
        raise ValueError("source_path must be a run directory")
    if not (source_path / "summary.json").exists():
        raise ValueError("imported run must contain summary.json")

    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_name = _slug(_optional_string(payload.get("run_name")) or source_path.name)
    destination = _safe_child(root, run_name)
    if destination.resolve() == source_path:
        return {
            "imported": False,
            "run_name": run_name,
            "source": str(source_path),
            "destination": str(destination),
            "message": "Run is already in the active run bank.",
            "runs": discover_runs(root),
        }
    if destination.exists():
        raise FileExistsError(f"run output already exists: {destination}")

    shutil.copytree(source_path, destination)
    return {
        "imported": True,
        "run_name": run_name,
        "source": str(source_path),
        "destination": str(destination),
        "message": "Imported run is now available for compare, reports, and chat.",
        "runs": discover_runs(root),
    }


def run_presets_plan() -> dict:
    return {"presets": RUN_PRESETS}


def leaderboard_plan(runs_dir: str | Path = "runs") -> dict:
    """Rank completed runs by benchmark eval into a leaderboard."""
    from picochat.leaderboard import build_benchmark_leaderboard
    root = Path(runs_dir)
    if not root.exists():
        return {"rows": [], "best_run": None}
    run_dirs = [str(path) for path in sorted(root.iterdir()) if (path / "summary.json").exists()]
    if not run_dirs:
        return {"rows": [], "best_run": None}
    try:
        return build_benchmark_leaderboard(run_dirs)
    except (ValueError, KeyError, OSError):
        return {"rows": [], "best_run": None}


def scale_plan_plan(payload: dict) -> dict:
    """Plan a training recipe (architecture + token budget) for a target size."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    from picochat.scale_planner import parse_count, plan_scale, render_scale_plan_markdown
    target = parse_count(_optional_string(payload.get("target_params")) or "100m")
    tokens_raw = _optional_string(payload.get("dataset_tokens"))
    dataset_tokens = parse_count(tokens_raw) if tokens_raw else None
    world_size = _bounded_int(payload.get("world_size", 1), 1, 128)
    plan = plan_scale(target_parameters=target, dataset_tokens=dataset_tokens, world_size=world_size)
    return {
        "markdown": render_scale_plan_markdown(plan),
        "estimated_parameters": plan.estimated_parameters,
        "n_layer": plan.n_layer,
        "n_embd": plan.n_embd,
        "n_head": plan.n_head,
        "context_size": plan.context_size,
    }


def registry_plan(runs_dir: str | Path = "runs") -> dict:
    """Build a model registry (release status of every run) from summaries."""
    from picochat.registry import build_model_registry, discover_run_dirs
    run_dirs = discover_run_dirs(runs_dir)
    if not run_dirs:
        return {"entries": []}
    try:
        return build_model_registry(run_dirs)
    except (ValueError, KeyError, OSError):
        return {"entries": []}


def serve_web(config: WebConfig) -> None:
    """Start the blocking local web server."""
    if not _is_loopback_host(config.host) and not config.auth_token:
        # The API can launch training subprocesses and write files. Never expose
        # it on a non-loopback interface without a token; mint one and require it.
        config = replace(config, auth_token=secrets.token_urlsafe(24))
        print("WARNING: binding to a non-loopback address — requiring an auth token.")
    # Mark any runs whose subprocess died while the server was down so they do
    # not linger as "running"; live runs resurface as reconnectable orphans.
    reconcile_orphan_jobs(config.runs_dir)
    handler = _make_handler(config)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    url = f"http://{config.host}:{config.port}"
    print(f"Picochat web UI: {url}")
    if config.auth_token:
        print(f"Auth token required. Open: {url}/?token={config.auth_token}")
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
            content_type = self._content_type_for_route(parsed.path)
            if content_type:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
            else:
                self.send_error(404, "Not found")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/") and not self._is_authorized():
                self._send_json(_unauthorized_payload(), status=401)
                return
            try:
                if parsed.path == "/healthz":
                    self._send_json(_health_payload())
                elif parsed.path == "/":
                    self._send_app_shell()
                elif parsed.path.startswith("/react/"):
                    self._send_react_asset(parsed.path)
                elif parsed.path == "/assets/picochat-symbol.svg":
                    self._send_asset("picochat-symbol.svg", "image/svg+xml; charset=utf-8")
                elif parsed.path == "/assets/picochat-symbol.png":
                    self._send_asset("picochat-symbol.png", "image/png")
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
                elif parsed.path == "/api/run/log/stream":
                    query = parse_qs(parsed.query)
                    self._stream_run_log(
                        run_name=query.get("run", [None])[0],
                        job_id=query.get("job", [None])[0],
                        limit=_bounded_int(query.get("limit", ["50000"])[0], 1_000, 200_000),
                    )
                elif parsed.path == "/api/run/log":
                    query = parse_qs(parsed.query)
                    run_name = query.get("run", [None])[0]
                    job_id = query.get("job", [None])[0]
                    limit = _bounded_int(query.get("limit", ["50000"])[0], 1_000, 200_000)
                    self._send_json(run_log_plan(config.runs_dir, run_name=run_name, job_id=job_id, limit=limit))
                elif parsed.path == "/api/run/presets":
                    self._send_json(run_presets_plan())
                elif parsed.path == "/api/serve/status":
                    self._send_json(serve_status_plan(config.runs_dir))
                elif parsed.path == "/api/remote/status":
                    self._send_json(remote_status_plan())
                elif parsed.path == "/api/leaderboard":
                    self._send_json(leaderboard_plan(config.runs_dir))
                elif parsed.path == "/api/registry":
                    self._send_json(registry_plan(config.runs_dir))
                else:
                    self.send_error(404, "Not found")
            except Exception as exc:
                self._send_json(_error_payload(exc), status=400)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not self._origin_ok():
                self.send_error(403, "Cross-origin request rejected")
                return
            if parsed.path.startswith("/api/") and not self._is_authorized():
                self._send_json(_unauthorized_payload(), status=401)
                return
            outcome = "ok"
            try:
                if parsed.path == "/api/generate":
                    self._send_json(generate_run_text(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/corpus/preview":
                    self._send_json(preview_corpus_plan(self._read_json_body()))
                elif parsed.path == "/api/dataset-pack/init":
                    self._send_json(init_dataset_pack_plan(self._read_json_body()))
                elif parsed.path == "/api/pack/clone":
                    self._send_json(clone_example_pack_plan(self._read_json_body()))
                elif parsed.path == "/api/hf/import":
                    self._send_json(hf_import_plan(self._read_json_body(), config.runs_dir))
                elif parsed.path == "/api/tuning/inspect":
                    self._send_json(inspect_tuning_plan(self._read_json_body()))
                elif parsed.path == "/api/sft/starter":
                    self._send_json(sft_starter_plan(self._read_json_body()))
                elif parsed.path == "/api/eval/starter":
                    self._send_json(eval_starter_plan(self._read_json_body()))
                elif parsed.path == "/api/preference/starter":
                    self._send_json(preference_starter_plan(self._read_json_body()))
                elif parsed.path == "/api/tuning/benchmark-pack":
                    self._send_json(benchmark_tuning_pack_plan(self._read_json_body()))
                elif parsed.path == "/api/pack/editor/load":
                    self._send_json(load_pack_editor_plan(self._read_json_body()))
                elif parsed.path == "/api/pack/editor/save":
                    self._send_json(save_pack_editor_plan(self._read_json_body()))
                elif parsed.path == "/api/run/start":
                    self._send_json(start_run_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/train/hf-sft":
                    self._send_json(hf_sft_start_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/train/hf-dpo":
                    self._send_json(hf_dpo_start_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/export/hf":
                    self._send_json(export_hf_run_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/eval/run":
                    self._send_json(eval_run_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/scale/plan":
                    self._send_json(scale_plan_plan(self._read_json_body()))
                elif parsed.path == "/api/run/cancel":
                    self._send_json(cancel_run_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/run/archive":
                    self._send_json(archive_run_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/run/import":
                    self._send_json(import_run_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/serve/start":
                    self._send_json(serve_start_plan(config.runs_dir, self._read_json_body(), host=config.host))
                elif parsed.path == "/api/serve/stop":
                    self._send_json(serve_stop_plan(self._read_json_body()))
                elif parsed.path == "/api/remote/modal/start":
                    self._send_json(remote_modal_start_plan(config.runs_dir, self._read_json_body()))
                elif parsed.path == "/api/remote/modal/pull":
                    self._send_json(remote_modal_pull_plan(config.runs_dir, self._read_json_body()))
                else:
                    outcome = "not_found"
                    self.send_error(404, "Not found")
            except Exception as exc:
                outcome = "error"
                self._send_json(_error_payload(exc), status=400)
            finally:
                if parsed.path.startswith("/api/"):
                    _append_audit(
                        config.runs_dir,
                        action=parsed.path,
                        actor=self.client_address[0] if self.client_address else "?",
                        outcome=outcome,
                        params=_safe_audit_params(getattr(self, "_cached_body", None)),
                    )

        def log_message(self, format: str, *args) -> None:
            return

        def _is_authorized(self) -> bool:
            token = config.auth_token
            if not token:
                return True
            provided = self.headers.get("X-Picochat-Token", "")
            if not provided:
                auth = self.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    provided = auth[len("Bearer "):]
            if not provided:
                # EventSource cannot set headers, so allow ?token= for streams.
                provided = parse_qs(urlparse(self.path).query).get("token", [""])[0]
            return bool(provided) and hmac.compare_digest(provided, token)

        def _origin_ok(self) -> bool:
            # Browsers send Origin on state-changing requests; rejecting any
            # cross-origin Origin blocks CSRF and DNS-rebinding against the
            # localhost API. Non-browser clients (no Origin) fall back to the
            # token check.
            origin = self.headers.get("Origin")
            if not origin:
                return True
            return urlparse(origin).netloc == self.headers.get("Host", "")

        def _content_type_for_route(self, path: str) -> str | None:
            if path == "/healthz":
                return "application/json; charset=utf-8"
            if path == "/":
                return "text/html; charset=utf-8"
            if path.startswith("/react/"):
                asset_name = path.removeprefix("/react/")
                if not asset_name:
                    asset_name = "index.html"
                if _asset_exists("react", *asset_name.split("/")):
                    return _content_type_for_asset(asset_name)
                return None
            if path == "/assets/picochat-symbol.svg":
                return "image/svg+xml; charset=utf-8"
            if path == "/assets/picochat-symbol.png":
                return "image/png"
            return None

        def _send_app_shell(self) -> None:
            if _asset_exists("react/index.html"):
                self._send_asset_path(("react", "index.html"), "text/html; charset=utf-8")
                return
            self.send_error(503, "Web UI not built. Run: npm ci && npm run frontend:build")

        def _send_react_asset(self, path: str) -> None:
            asset_name = path.removeprefix("/react/")
            if not asset_name:
                asset_name = "index.html"
            if ".." in Path(asset_name).parts:
                self.send_error(404, "Not found")
                return
            parts = tuple(part for part in asset_name.split("/") if part)
            if not parts:
                parts = ("index.html",)
            if not _asset_exists("react", *parts):
                # React owns the /react prefix. Unknown subpaths still load the app
                # shell so client-side deep links resolve instead of 404ing.
                self._send_asset_path(("react", "index.html"), "text/html; charset=utf-8")
                return
            self._send_asset_path(("react", *parts), _content_type_for_asset(asset_name))

        def _send_asset(self, name: str, content_type: str) -> None:
            self._send_asset_path((name,), content_type)

        def _send_asset_path(self, parts: tuple[str, ...], content_type: str) -> None:
            data = _read_asset(*parts)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def _stream_run_log(self, *, run_name, job_id, limit) -> None:
            """Server-sent-events stream of a run's log/status until it ends."""
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            for _ in range(_STREAM_MAX_TICKS):
                try:
                    payload = run_log_plan(config.runs_dir, run_name=run_name, job_id=job_id, limit=limit)
                except Exception as exc:  # unknown job, transient read error, etc.
                    self._write_sse_event({"error": str(exc), "running": False, "state": "unknown"})
                    return
                if not self._write_sse_event(payload):
                    return  # client disconnected
                if not payload.get("running"):
                    return  # terminal state; client closes the EventSource
                time.sleep(_STREAM_INTERVAL_SECONDS)

        def _write_sse_event(self, payload: dict) -> bool:
            try:
                self.wfile.write(b"data: ")
                self.wfile.write(json.dumps(payload).encode("utf-8"))
                self.wfile.write(b"\n\n")
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return False

        def _read_json_body(self) -> dict:
            cached = getattr(self, "_cached_body", None)
            if cached is not None:
                return cached
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            self._cached_body = payload
            return payload

    return PicoWebHandler


def _asset_exists(*parts: str) -> bool:
    try:
        path = importlib.resources.files("picochat.web_assets")
        for part in parts:
            path = path.joinpath(part)
        return path.is_file()
    except (FileNotFoundError, ModuleNotFoundError):
        return False


def _read_asset(*parts: str) -> bytes:
    path = importlib.resources.files("picochat.web_assets")
    for part in parts:
        path = path.joinpath(part)
    return path.read_bytes()


def _content_type_for_asset(name: str) -> str:
    if name.endswith(".js"):
        return "application/javascript; charset=utf-8"
    if name.endswith(".css"):
        return "text/css; charset=utf-8"
    if name.endswith(".html"):
        return "text/html; charset=utf-8"
    if name.endswith(".svg"):
        return "image/svg+xml; charset=utf-8"
    content_type, _ = mimetypes.guess_type(name)
    return content_type or "application/octet-stream"


def _safe_child(root: Path, name: str) -> Path:
    if not name:
        raise ValueError("run name is required")
    root = root.resolve()
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError("run path escapes runs directory")
    return path


def _active_job_for_run(run_dir: Path) -> dict | None:
    target = run_dir.resolve()
    with _RUN_JOBS_LOCK:
        jobs = list(_RUN_JOBS.values())
    for job in jobs:
        try:
            out_dir = Path(job["out_dir"]).resolve()
            process = job["process"]
        except (KeyError, OSError):
            continue
        if out_dir == target and process.poll() is None:
            return job
    return None


def _unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not choose unique archive path for {path}")


def _archive_run_names(payload: dict) -> list[str]:
    raw_run_names = payload.get("run_names")
    if raw_run_names is None:
        run_name = _optional_string(payload.get("run_name"))
        raw_run_names = [run_name] if run_name else []
    if not isinstance(raw_run_names, list):
        raise ValueError("run_names must be a list")

    run_names = []
    seen = set()
    for item in raw_run_names:
        run_name = _optional_string(item)
        if not run_name:
            continue
        if Path(run_name).name != run_name:
            raise ValueError("run_name must be an active top-level run")
        if run_name not in seen:
            run_names.append(run_name)
            seen.add(run_name)

    if not run_names:
        raise ValueError("run_name is required")
    return run_names


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounded_int(value: object, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _optional_int(value: object, minimum: int, maximum: int) -> int | None:
    if value is None or value == "":
        return None
    return _bounded_int(value, minimum, maximum)


def _hf_import_document_shard_rows(payload: dict, dataset: str, max_rows: int) -> int:
    explicit = _optional_int(payload.get("document_shard_rows"), 1, 100_000)
    if explicit is not None:
        return explicit
    if dataset == CLIMBMIX_DATASET and max_rows >= CLIMBMIX_LARGE_IMPORT_ROWS:
        return CLIMBMIX_LARGE_IMPORT_DOCUMENT_SHARD_ROWS
    return 1


def _bounded_float(value: object, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _shell_command(*parts: str) -> str:
    return " ".join(shlex.quote(part) for part in parts)


# Directory holding the read-only example packs that ship with Picochat. Data
# generation must never write into here, so the dashboard clones these to a
# writable workspace before generating chat/eval/benchmark/preference files.
_BUNDLED_EXAMPLES_DIR = (Path(__file__).resolve().parents[2] / "examples")


def _is_bundled_example_path(path: str | None) -> bool:
    """True when ``path`` resolves inside the shipped, read-only examples dir."""
    if not path:
        return False
    try:
        resolved = Path(path).resolve(strict=False)
        resolved.relative_to(_BUNDLED_EXAMPLES_DIR.resolve(strict=False))
        return True
    except (ValueError, OSError):
        return False


def _reject_bundled_example(path: str | None, *, what: str) -> None:
    """Guard mutating data-generation against the shipped example files."""
    if _is_bundled_example_path(path):
        raise ValueError(
            f"{what} would write into the read-only example pack ({path}). "
            "Make a working copy first (the dashboard does this automatically when "
            "you pick an example) and generate into that instead."
        )


def clone_example_pack_plan(payload: dict) -> dict:
    """Clone a (read-only) example pack into a writable workspace.

    Idempotent for already-writable packs: those are returned unchanged so the
    caller can invoke this unconditionally before entering the prepare flow.
    """
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    dataset_pack = _optional_string(payload.get("dataset_pack"))
    if not dataset_pack:
        raise ValueError("dataset_pack is required")
    if not _is_bundled_example_path(dataset_pack):
        return {"dataset_pack": dataset_pack, "cloned": False}

    stem = Path(dataset_pack).stem
    target = Path("packs") / f"{stem}-copy"
    suffix = 2
    while target.exists():
        target = Path("packs") / f"{stem}-copy-{suffix}"
        suffix += 1
    new_pack = clone_dataset_pack(dataset_pack, target)
    return {"dataset_pack": new_pack, "cloned": True, "source": dataset_pack}


def _normalize_hf_dataset_id(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("dataset id is required")
    if "://" not in text:
        return text.removeprefix("datasets/").strip("/")

    parsed = urlparse(text)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError("only huggingface.co dataset URLs are supported")
    if not parts:
        raise ValueError("dataset URL is missing a dataset id")
    if parts[0] == "datasets":
        parts = parts[1:]
    stop_words = {"tree", "blob", "resolve", "viewer", "discussions", "commit"}
    dataset_parts = []
    for part in parts:
        if part in stop_words:
            break
        dataset_parts.append(part)
    if not dataset_parts:
        raise ValueError("dataset URL is missing a dataset id")
    return "/".join(dataset_parts)


def _error_payload(error: Exception) -> dict:
    payload = {
        "error": str(error),
        "error_type": error.__class__.__name__,
    }
    if isinstance(error, HFSplitError):
        payload.update({
            "dataset": error.dataset,
            "requested_split": error.requested_split,
            "available_splits": error.available_splits,
        })
    return payload


def _unauthorized_payload() -> dict:
    return {
        "error": "unauthorized",
        "error_type": "Unauthorized",
        "message": "Missing or invalid auth token. Open the URL printed by `pico web` (it includes ?token=...).",
    }


def _health_payload() -> dict:
    with _RUN_JOBS_LOCK:
        active = sum(
            1 for job in _RUN_JOBS.values()
            if (proc := job.get("process")) is not None and proc.poll() is None
        )
    return {"status": "ok", "version": __version__, "active_jobs": active}


# Audit log: append-only record of who triggered which state-changing action.
# Whitelisted keys only — never persist tokens, HF credentials, or free text.
_AUDIT_PARAM_KEYS = (
    "run", "run_name", "run_names", "job_id", "dataset_pack",
    "checkpoint", "out_dir", "preset", "source_path",
)


def _safe_audit_params(body) -> dict:
    if not isinstance(body, dict):
        return {}
    return {key: body[key] for key in _AUDIT_PARAM_KEYS if key in body}


def _append_audit(runs_dir: str | Path, *, action: str, actor: str, outcome: str, params: dict) -> None:
    try:
        audit_dir = Path(runs_dir) / ".audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "actor": actor,
            "action": action,
            "outcome": outcome,
            "params": params,
        }
        with (audit_dir / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


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


def _child_env(*, device: str = "cpu", ddp: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{current}" if current else src_path
    env["PYTHONUNBUFFERED"] = "1"
    if ddp:
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("PICOCHAT_DDP_TIMEOUT_MINUTES", "120")
    if ddp or device == "cuda":
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def _run_job_status(job: dict) -> dict:
    process = job["process"]
    returncode = process.poll()
    state = "running" if returncode is None else "succeeded" if returncode == 0 else "failed"
    summary_path = Path(job["out_dir"]) / "summary.json"
    log_tail = _read_log_tail(Path(job["log_path"]))
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
        "log_tail": log_tail,
        "progress": _parse_run_progress(log_tail, state=state, summary_exists=summary_path.exists()),
        "can_cancel": state == "running",
        "source": "active",
        "updated_at": time.time(),
        "preset": job.get("preset"),
        "min_quality_score": job.get("min_quality_score", 0),
        "launch_config": job.get("launch_config"),
        "launch_readiness": job.get("launch_readiness"),
        "launch_tuning": job.get("launch_tuning"),
        "launch_preflight": job.get("launch_preflight"),
    }


def _job_in_runs_dir(job: dict, runs_root: Path) -> bool:
    try:
        out_dir = Path(job["out_dir"]).resolve()
    except (KeyError, OSError):
        return False
    if not out_dir.exists():
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
        summary_exists = summary_path.exists()
        record = _read_job_record(run_dir)
        pid = record.get("pid") if record else None
        unfinished = returncode is None and not summary_exists
        alive = bool(record) and unfinished and _pid_alive(pid)
        log_tail = _read_log_tail(log_path)
        if summary_exists:
            state = "succeeded"
        elif returncode not in (None, 0):
            state = "failed"
        elif alive:
            state = "running"
        elif record and unfinished:
            state = "interrupted"
        else:
            state = "stopped"
        updated_at = max(log_path.stat().st_mtime, summary_path.stat().st_mtime if summary_exists else 0)
        started_at = record.get("started_at") if record else None
        jobs.append({
            "id": record["id"] if record and record.get("id") else f"run-{_slug(run_dir.name)}",
            "run_name": run_dir.name,
            "out_dir": str(run_dir),
            "dataset_pack": (record.get("dataset_pack") if record else None) or _dataset_pack_from_summary(summary_path),
            "log_path": str(log_path),
            "command": (record.get("command") if record else None) or _command_from_log(log_path),
            "pid": pid if alive else None,
            "state": state,
            "returncode": returncode,
            "elapsed_seconds": round(time.time() - started_at, 1) if (alive and started_at) else None,
            "summary_exists": summary_exists,
            "log_tail": log_tail,
            "progress": _parse_run_progress(log_tail, state=state, summary_exists=summary_exists),
            "can_cancel": alive,
            "source": "orphan" if alive else "disk",
            "updated_at": updated_at,
            "preset": record.get("preset") if record else None,
            "min_quality_score": record.get("min_quality_score") if record else None,
            "launch_config": record.get("launch_config") if record else None,
            "launch_readiness": record.get("launch_readiness") if record else None,
            "launch_tuning": record.get("launch_tuning") if record else None,
            "launch_preflight": (record.get("launch_preflight") if record else None) or _preflight_from_run_dir(run_dir),
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


_JOB_RECORD_NAME = "web_job.json"
_JOB_RECORD_FIELDS = (
    "id", "run_name", "out_dir", "dataset_pack", "log_path", "command",
    "started_at", "preset", "min_quality_score", "launch_config",
    "launch_readiness", "launch_tuning", "launch_preflight",
)


def _write_job_record(out_dir: str | Path, job: dict) -> None:
    record = {field: job.get(field) for field in _JOB_RECORD_FIELDS}
    record["pid"] = job.get("pid")
    try:
        (Path(out_dir) / _JOB_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass


def _read_job_record(run_dir: Path) -> dict | None:
    path = Path(run_dir) / _JOB_RECORD_NAME
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _find_job_record(runs_dir: str | Path, job_id: str) -> dict | None:
    root = Path(runs_dir)
    if not root.exists():
        return None
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        record = _read_job_record(run_dir)
        if record and (record.get("id") == job_id or record.get("run_name") == job_id):
            return record
    return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _kill_process_group(pid: int | None, sig: int) -> None:
    """Best-effort signal to the child's whole process group (DDP-safe)."""
    if not pid:
        return
    try:
        os.killpg(os.getpgid(int(pid)), sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _terminate_job(job: dict) -> None:
    process = job.get("process")
    if process is not None and process.poll() is None:
        process.terminate()
        time.sleep(0.05)
    _kill_process_group(job.get("pid"), signal.SIGTERM)


def reconcile_orphan_jobs(runs_dir: str | Path = "runs") -> None:
    """At startup, mark sidecar jobs whose process died as interrupted.

    Safe to call only when no jobs are tracked in memory (e.g. fresh process
    start): live runs keep their sidecar and resurface as reconnectable orphans.
    """
    root = Path(runs_dir)
    if not root.exists():
        return
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        record = _read_job_record(run_dir)
        if not record:
            continue
        if (run_dir / "web_returncode.txt").exists() or (run_dir / "summary.json").exists():
            continue
        if not _pid_alive(record.get("pid")):
            _write_returncode(run_dir / "web_returncode.txt", -1)


def _dataset_pack_from_summary(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    try:
        return _read_json(summary_path).get("config", {}).get("dataset_pack")
    except (OSError, json.JSONDecodeError):
        return None


def _preflight_from_run_dir(run_dir: Path) -> dict | None:
    for path in (run_dir / "summary.json", run_dir / "preflight.json"):
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if path.name == "summary.json":
            preflight = payload.get("preflight")
            return preflight if isinstance(preflight, dict) else None
        return payload if isinstance(payload, dict) else None
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


def _parse_run_progress(log_text: str, state: str = "running", summary_exists: bool = False) -> dict:
    stage = {
        "id": "waiting",
        "label": "Waiting for log",
        "index": 0,
        "total": 0,
        "message": "The run process has started, but Picochat has not printed a pipeline stage yet.",
    }
    base = None
    sft = None
    eval_result = None
    for line in str(log_text or "").splitlines():
        stage_match = re.match(r"^\[(\d+)/(\d+)\]\s+(.+?)(?:\s+->.*)?$", line.strip())
        if stage_match:
            stage = _progress_stage(int(stage_match.group(1)), int(stage_match.group(2)), stage_match.group(3))
            continue
        base_match = re.match(
            r"^step\s+(\d+)/(\d+)\s+\|\s+train\s+([0-9.]+)\s+\|\s+val\s+([0-9.]+)\s+\|\s+val_bpb\s+([0-9.]+|--)\s+\|\s+(?:(--|[0-9.]+k?)\s+tok/s\s+\|\s+)?([0-9.]+)s",
            line.strip(),
        )
        if base_match:
            base = _progress_loss(base_match)
            stage = {**stage, "id": "base", "label": "Base training", "message": "Learning next-token prediction on the corpus."}
            continue
        sft_match = re.match(
            r"^sft step\s+(\d+)/(\d+)\s+\|\s+train\s+([0-9.]+)\s+\|\s+val\s+([0-9.]+)\s+\|\s+val_bpb\s+([0-9.]+|--)\s+\|\s+(?:(--|[0-9.]+k?)\s+tok/s\s+\|\s+)?([0-9.]+)s",
            line.strip(),
        )
        if sft_match:
            sft = _progress_loss(sft_match)
            stage = {**stage, "id": "sft", "label": "Chat SFT", "message": "Teaching the model chat format and preferred behavior."}
            continue
        done_match = re.match(r"^done:\s+(\d+)/(\d+)\s+passed\s+\(([0-9.]+)%\)", line.strip())
        if done_match:
            eval_result = {
                "passed": int(done_match.group(1)),
                "total": int(done_match.group(2)),
                "pass_rate": float(done_match.group(3)),
            }
            stage = {**stage, "id": "eval", "label": "Eval complete", "message": "Scoring behavior against the eval set."}

    if state == "failed":
        stage = {**stage, "id": "failed", "label": "Run failed", "message": "Inspect the raw log for the first stack trace or error."}
    elif summary_exists or state == "succeeded":
        stage = {**stage, "id": "complete", "label": "Run complete", "message": "Summary artifacts are ready for eval, report, compare, and chat."}

    return {
        "stage": stage,
        "base": base,
        "sft": sft,
        "eval": eval_result,
        "summary_exists": summary_exists,
    }


def _progress_stage(index: int, total: int, label: str) -> dict:
    normalized = label.lower()
    if normalized.startswith("build corpus"):
        stage_id = "dataset"
        message = "Building the corpus artifacts from the selected dataset pack."
    elif normalized.startswith("check data honesty"):
        stage_id = "honesty"
        message = "Checking for leakage, duplication, and suspicious overlap."
    elif "tokenizer" in normalized:
        stage_id = "tokenizer"
        message = "Training the tokenizer that turns text into token IDs."
    elif normalized.startswith("train base"):
        stage_id = "base"
        message = "Learning next-token prediction on the corpus."
    elif normalized.startswith("train chat"):
        stage_id = "sft"
        message = "Teaching the model chat format and preferred behavior."
    elif normalized.startswith("run sft fit"):
        stage_id = "sft_fit"
        message = "Checking whether chat SFT learned its own training rows."
    elif normalized.startswith("run chat eval"):
        stage_id = "eval"
        message = "Scoring behavior against the eval set."
    else:
        stage_id = _slug(label)
        message = "Running the next pipeline stage."
    return {
        "id": stage_id,
        "label": label.strip(),
        "index": index,
        "total": total,
        "message": message,
    }


def _progress_loss(match: re.Match) -> dict:
    current = int(match.group(1))
    total = int(match.group(2))
    return {
        "current": current,
        "total": total,
        "percent": round((current / total) * 100, 2) if total else 0.0,
        "train_loss": float(match.group(3)),
        "val_loss": float(match.group(4)),
        "val_bpb": _progress_float(match.group(5)),
        "tokens_per_sec": _progress_rate(match.group(6)),
        "seconds": float(match.group(7)),
    }


def _progress_float(raw: str | None) -> float | None:
    if raw in {None, "--"}:
        return None
    return float(raw)


def _progress_rate(raw: str | None) -> float | None:
    if raw in {None, "--"}:
        return None
    text = str(raw)
    if text.endswith("k"):
        return float(text[:-1]) * 1000.0
    return float(text)


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
        "pretokenizer": data.get("pretokenizer", "char"),
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
        "honesty": _local_run_artifact_path(run_dir, summary, artifacts.get("honesty_report", run_dir / "honesty" / "report.md")),
        "base": _local_run_artifact_path(run_dir, summary, artifacts.get("base_report", run_dir / "base" / "report.md")),
        "sft": _local_run_artifact_path(run_dir, summary, artifacts.get("sft_report", run_dir / "sft" / "report.md")),
        "eval": _local_run_artifact_path(run_dir, summary, artifacts.get("eval_report", run_dir / "eval" / "report.md")),
    }
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
        }
        for name, path in report_paths.items()
    }


def _run_timeline_from_detail(summary: dict, detail: dict) -> list[dict]:
    """Build an answer-first run ledger for the web UI from existing artifacts."""
    preflight = detail.get("preflight") or summary.get("preflight") or {}
    gate = summary.get("long_run_gate") or {}
    eval_summary = (detail.get("eval_reports") or [{}])[-1].get("report", {}).get("summary") if detail.get("eval_reports") else None
    eval_summary = eval_summary or summary.get("eval") or {}
    reports = detail.get("reports") or {}
    inventory = detail.get("artifact_inventory") or {}
    by_key = {
        item.get("key"): item
        for item in inventory.get("items", [])
        if item.get("key")
    }

    def artifact(key: str) -> dict:
        return by_key.get(key) or {"exists": False, "path": ""}

    def artifact_status(key: str, label: str, ready: str, missing: str, *, warn: bool = False) -> dict:
        item = artifact(key)
        exists = bool(item.get("exists"))
        return {
            "id": key,
            "label": label,
            "status": "done" if exists else ("warn" if warn else "pending"),
            "summary": ready if exists else missing,
            "evidence": item.get("path") or "",
        }

    preflight_status = str(preflight.get("status") or "missing").lower()
    if preflight_status in {"ready", "pass", "passed"}:
        preflight_ui_status = "done"
    elif preflight_status in {"blocked", "block", "fail", "failed"}:
        preflight_ui_status = "blocked"
    elif preflight_status in {"warn", "warning"}:
        preflight_ui_status = "warn"
    else:
        preflight_ui_status = "pending"
    preflight_blockers = preflight.get("blocking_checks") or []
    preflight_warnings = preflight.get("warning_checks") or []

    honesty = summary.get("honesty") or {}
    honesty_status = str(honesty.get("status") or "").lower()
    honesty_blocked = any([
        int(honesty.get("exact_prompt_leaks") or 0) > 0,
        int(honesty.get("corpus_prompt_hits") or 0) > 0,
        int(honesty.get("duplicate_eval_prompts") or 0) > 0,
    ])

    base_trace = detail.get("base_report") or {}
    base_losses = base_trace.get("losses") or []
    sft_trace = detail.get("sft_report") or {}
    sft_losses = sft_trace.get("losses") or []
    eval_examples = eval_summary.get("num_examples")
    gate_status = str(gate.get("status") or "not_run").lower()

    return [
        {
            "id": "preflight",
            "label": "Preflight",
            "status": preflight_ui_status,
            "summary": preflight.get("summary")
            or ("Run is launch-ready." if preflight_ui_status == "done" else "Run preflight before spending GPU time."),
            "evidence": artifact("preflight_report").get("path") or artifact("summary_json").get("path") or "",
            "detail": f"{len(preflight_blockers)} blockers / {len(preflight_warnings)} warnings",
        },
        artifact_status(
            "corpus_manifest",
            "Data Pack",
            "Corpus manifest and data lineage are present.",
            "Corpus manifest is missing; data lineage is not inspectable.",
        ),
        {
            "id": "honesty",
            "label": "Honesty",
            "status": "blocked" if honesty_blocked else ("done" if honesty_status in {"ready", "pass", "passed"} or reports.get("honesty", {}).get("exists") else "warn"),
            "summary": honesty.get("summary") or "Honesty report not found.",
            "evidence": reports.get("honesty", {}).get("path") or artifact("honesty_json").get("path") or "",
            "detail": f"exact {honesty.get('exact_prompt_leaks', 0)} / corpus hits {honesty.get('corpus_prompt_hits', 0)}",
        },
        artifact_status(
            "tokenizer",
            "Tokenizer",
            "Tokenizer artifact is present.",
            "Tokenizer artifact is missing.",
        ),
        {
            "id": "base",
            "label": "Base Train",
            "status": "done" if artifact("base_checkpoint").get("exists") else ("warn" if base_losses else "pending"),
            "summary": f"{len(base_losses)} logged loss points; checkpoint {'ready' if artifact('base_checkpoint').get('exists') else 'missing'}.",
            "evidence": reports.get("base", {}).get("path") or artifact("base_trace").get("path") or "",
        },
        {
            "id": "sft",
            "label": "Behavior SFT",
            "status": "done" if artifact("sft_checkpoint").get("exists") else ("warn" if sft_losses else "pending"),
            "summary": f"{len(sft_losses)} logged loss points; checkpoint {'ready' if artifact('sft_checkpoint').get('exists') else 'missing'}.",
            "evidence": reports.get("sft", {}).get("path") or artifact("sft_trace").get("path") or "",
        },
        {
            "id": "eval",
            "label": "Visible Eval",
            "status": "done" if eval_examples else "pending",
            "summary": f"{eval_summary.get('num_passed', '--')}/{eval_examples or '--'} visible eval rows passed.",
            "evidence": reports.get("eval", {}).get("path") or artifact("eval_json").get("path") or "",
        },
        {
            "id": "release_gate",
            "label": "Release Gate",
            "status": "done" if gate_status == "approved" else ("blocked" if gate_status == "blocked" else "warn"),
            "summary": gate.get("summary") or (gate.get("issues") or [{}])[0].get("message") or "Post-run release gate has not approved this run yet.",
            "evidence": reports.get("summary", {}).get("path") or artifact("summary_json").get("path") or "",
            "detail": gate.get("status") or "not run",
        },
    ]


def _handoff_packet_from_detail(summary: dict, detail: dict) -> dict:
    """Summarize the artifacts a downstream team or reviewer needs first."""
    gate = summary.get("long_run_gate") or {}
    gate_status = str(gate.get("status") or "not_run").lower()
    reports = detail.get("reports") or {}
    inventory = detail.get("artifact_inventory") or {}
    by_key = {
        item.get("key"): item
        for item in inventory.get("items", [])
        if item.get("key")
    }

    def record(key: str, label: str, purpose: str, *, required: bool = True) -> dict:
        item = by_key.get(key) or {}
        return {
            "key": key,
            "label": label,
            "purpose": purpose,
            "required": required,
            "exists": bool(item.get("exists")),
            "path": item.get("path") or "",
            "kind": item.get("kind") or "missing",
        }

    def record_any(keys: list[str], label: str, purpose: str, *, required: bool = True) -> dict:
        for key in keys:
            item = by_key.get(key) or {}
            if item.get("exists"):
                return {
                    "key": key,
                    "label": label,
                    "purpose": purpose,
                    "required": required,
                    "exists": True,
                    "path": item.get("path") or "",
                    "kind": item.get("kind") or "missing",
                }
        return record(keys[0], label, purpose, required=required)

    def report_record(key: str, label: str, purpose: str) -> dict:
        report = reports.get(key) or {}
        return {
            "key": f"{key}_report",
            "label": label,
            "purpose": purpose,
            "required": True,
            "exists": bool(report.get("exists")),
            "path": report.get("path") or "",
            "kind": "file" if report.get("exists") else "missing",
        }

    artifacts = [
        report_record("summary", "Run summary", "Human-readable training, SFT, eval, and gate overview."),
        report_record("honesty", "Honesty report", "Contamination and leakage evidence."),
        report_record("base", "Base report", "Base-training loss, BPB, and checkpoint context."),
        report_record("sft", "SFT report", "Behavior-tuning fit and held-out loss context."),
        report_record("eval", "Eval report", "Visible eval results and failure examples."),
        record_any(["base_best_checkpoint", "base_checkpoint"], "Base checkpoint", "Foundation model checkpoint for domain teams."),
        record("sft_checkpoint", "SFT checkpoint", "Behavior-tuned checkpoint for demo/chat handoff."),
        record("tokenizer", "Tokenizer", "Tokenizer required to load either checkpoint."),
        record("preflight_report", "Preflight report", "Launch budget and blocker evidence.", required=False),
    ]
    missing_required = [item for item in artifacts if item["required"] and not item["exists"]]
    gate_issues = gate.get("issues") or []
    if gate_status == "approved" and not missing_required:
        status = "ready"
        summary_text = "Release handoff packet is complete."
    elif gate_status == "blocked":
        status = "blocked"
        summary_text = gate_issues[0].get("message") if gate_issues else "Release gate blocked this run."
    elif missing_required:
        status = "blocked"
        summary_text = f"{len(missing_required)} required handoff artifact{' is' if len(missing_required) == 1 else 's are'} missing."
    else:
        status = "watch"
        summary_text = "Artifacts are present, but the release gate has not approved this run."

    next_actions: list[str] = []
    for item in missing_required:
        next_actions.append(f"Create or recover {item['label']}.")
    for issue in gate_issues[:4]:
        message = issue.get("message") if isinstance(issue, dict) else str(issue)
        if message:
            next_actions.append(message)
    if not next_actions:
        if status == "ready":
            next_actions.append("Share the summary, honesty report, eval report, tokenizer, and checkpoint paths with downstream users.")
        elif status == "watch":
            next_actions.append("Run or inspect the long-run release gate before making release claims.")

    return {
        "status": status,
        "summary": summary_text,
        "gate_status": gate.get("status") or "not_run",
        "ready_count": sum(1 for item in artifacts if item["exists"]),
        "required_count": sum(1 for item in artifacts if item["required"]),
        "missing_required": [item["label"] for item in missing_required],
        "next_actions": next_actions,
        "artifacts": artifacts,
    }


def _release_repair_plan_from_detail(summary: dict, detail: dict) -> dict:
    """Turn gate/preflight issues into a short repair queue for the UI."""
    gate = summary.get("long_run_gate") or {}
    preflight = detail.get("preflight") or summary.get("preflight") or {}
    gate_issues = gate.get("issues") or []
    preflight_blockers = preflight.get("blocking_checks") or []
    preflight_warnings = preflight.get("warning_checks") or []
    actions: list[dict] = []

    for check in preflight_blockers[:4]:
        name = check.get("name") if isinstance(check, dict) else str(check)
        summary_text = check.get("summary") if isinstance(check, dict) else ""
        actions.append({
            "source": "preflight",
            "severity": "block",
            "title": f"Fix preflight: {name}",
            "reason": summary_text or "Preflight blocked this launch.",
            "action": _preflight_repair_action(str(name)),
        })

    for issue in gate_issues[:6]:
        name = issue.get("name") if isinstance(issue, dict) else str(issue)
        severity = issue.get("severity", "warn") if isinstance(issue, dict) else "warn"
        message = issue.get("message") if isinstance(issue, dict) else str(issue)
        actions.append({
            "source": "release_gate",
            "severity": severity,
            "title": _gate_issue_title(str(name)),
            "reason": message,
            "action": _gate_issue_repair_action(str(name)),
        })

    if not actions and preflight_warnings:
        for check in preflight_warnings[:3]:
            name = check.get("name") if isinstance(check, dict) else str(check)
            summary_text = check.get("summary") if isinstance(check, dict) else ""
            actions.append({
                "source": "preflight",
                "severity": "warn",
                "title": f"Inspect preflight: {name}",
                "reason": summary_text or "Preflight warned on this launch.",
                "action": _preflight_repair_action(str(name)),
            })

    if not actions:
        gate_status = str(gate.get("status") or "not_run").lower()
        if gate_status == "approved":
            status = "ready"
            headline = "No release repair actions are open."
            actions.append({
                "source": "release_gate",
                "severity": "pass",
                "title": "Keep this as a reference",
                "reason": "The long-run gate approved this run.",
                "action": "Export or hand off the checkpoint with the summary, honesty, and eval reports.",
            })
        else:
            status = "watch"
            headline = "Run the release gate before claiming readiness."
            actions.append({
                "source": "release_gate",
                "severity": "warn",
                "title": "No post-run gate found",
                "reason": "The run has artifacts, but no approved release gate decision.",
                "action": "Run eval, external benchmarks if required, then inspect the long-run gate result.",
            })
    else:
        status = "blocked" if any(item["severity"] == "block" for item in actions) else "watch"
        headline = "Repair these before using the run as a release recipe." if status == "blocked" else "Inspect these warnings before promotion."

    return {
        "status": status,
        "headline": headline,
        "actions": actions,
    }


def _run_passport_from_detail(summary: dict, detail: dict) -> dict:
    """Build a copy-ready run receipt for reviews, handoffs, and release notes."""
    config = summary.get("config") or {}
    base = summary.get("base") or {}
    sft = summary.get("sft") or {}
    eval_summary = summary.get("eval") or {}
    honesty = summary.get("honesty") or {}
    gate = summary.get("long_run_gate") or {}
    handoff = detail.get("handoff_packet") or {}
    repair = detail.get("release_repair_plan") or {}
    timeline = detail.get("run_timeline") or []
    run_name = Path(str(config.get("out_dir") or "")).name or summary.get("run_name") or "selected run"
    if run_name in {"", "."}:
        run_name = "selected run"
    gate_status = str(gate.get("status") or handoff.get("gate_status") or "not_run")
    preflight_stage = next((item for item in timeline if item.get("id") == "preflight"), {})
    checkpoint = next(
        (item for item in handoff.get("artifacts", []) if item.get("label") == "Base checkpoint" and item.get("exists")),
        {},
    )
    tokenizer = next(
        (item for item in handoff.get("artifacts", []) if item.get("label") == "Tokenizer" and item.get("exists")),
        {},
    )
    reports_ready = sum(1 for item in detail.get("reports", {}).values() if item.get("exists"))
    eval_examples = eval_summary.get("num_examples")
    eval_passed = eval_summary.get("num_passed")
    eval_rate = eval_summary.get("pass_rate")
    planned_tokens = config.get("planned_base_tokens") or config.get("base_tokens") or summary.get("planned_base_tokens")
    target_ratio = config.get("target_param_data_ratio") or summary.get("target_param_data_ratio")
    params = base.get("num_parameters") or summary.get("num_parameters")
    headline = "Approved release candidate" if gate_status == "approved" and handoff.get("status") == "ready" else "Research run; inspect gates before release claims"
    if handoff.get("status") == "blocked":
        headline = "Blocked run; repair before promotion"
    facts = [
        ("Gate", gate_status),
        ("Handoff", handoff.get("status") or "unknown"),
        ("Preflight", preflight_stage.get("detail") or preflight_stage.get("status") or "not found"),
        ("Parameters", _passport_number(params)),
        ("Context", _passport_number(config.get("context_size"))),
        ("Base steps", _passport_number(config.get("base_steps"))),
        ("SFT steps", _passport_number(config.get("sft_steps"))),
        ("Planned tokens", _passport_number(planned_tokens)),
        ("Target ratio", _passport_number(target_ratio)),
        ("Base val loss", _passport_number(base.get("final_val_loss"))),
        ("SFT val loss", _passport_number(sft.get("final_val_loss"))),
        ("Eval", _passport_eval_text(eval_passed, eval_examples, eval_rate)),
        ("Honesty", honesty.get("summary") or honesty.get("status") or "not found"),
        ("Reports", f"{reports_ready}/5 ready"),
        ("Checkpoint", checkpoint.get("path") or "missing"),
        ("Tokenizer", tokenizer.get("path") or "missing"),
    ]
    primary_repair = (repair.get("actions") or [{}])[0]
    open_issue = primary_repair.get("reason") or handoff.get("summary") or headline
    next_action = primary_repair.get("action") or (handoff.get("next_actions") or ["Inspect the release gate."])[0]
    markdown_lines = [
        f"# Picochat Run Passport: {run_name}",
        "",
        f"**Status:** {headline}",
        f"**Gate:** {gate_status}",
        f"**Handoff:** {handoff.get('status') or 'unknown'}",
        "",
        "## Run Facts",
    ]
    for label, value in facts:
        markdown_lines.append(f"- **{label}:** {value}")
    markdown_lines.extend([
        "",
        "## Next Action",
        f"- Issue: {open_issue}",
        f"- {next_action}",
    ])
    return {
        "title": f"Picochat Run Passport: {run_name}",
        "headline": headline,
        "status": handoff.get("status") or "watch",
        "gate_status": gate_status,
        "facts": [{"label": label, "value": str(value)} for label, value in facts],
        "open_issue": open_issue,
        "next_action": next_action,
        "markdown": "\n".join(markdown_lines),
    }


def _passport_number(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value >= 1_000:
            return f"{value:,.0f}"
        return f"{value:.4g}"
    return str(value)


def _passport_eval_text(passed, total, rate) -> str:
    if passed is None and total is None and rate is None:
        return "not found"
    if rate is not None:
        return f"{_passport_number(passed)}/{_passport_number(total)} ({float(rate) * 100:.1f}%)"
    return f"{_passport_number(passed)}/{_passport_number(total)}"


def _gate_issue_title(name: str) -> str:
    if name.startswith("skill_release_sft_"):
        return "Repair skill SFT coverage"
    if name.startswith("skill_release_eval_"):
        return "Repair held-out skill eval"
    if name.startswith("skill_release_stage_"):
        return "Repair weak subskill stage"
    if name.startswith("external_eval"):
        return "Add or repair external benchmark"
    titles = {
        "preflight": "Respect preflight blockers",
        "data_honesty": "Repair data honesty",
        "release_data_honesty": "Remove release contamination",
        "sft_fit": "Improve SFT fit",
        "sft_heldout_fit": "Improve held-out SFT transfer",
        "eval_non_choice": "Improve free-form eval",
        "first_release_eval": "Improve release behavior eval",
        "choice_adjusted_accuracy": "Improve choice robustness",
        "external_eval_missing": "Add external benchmark",
        "refusal": "Repair unsafe refusal behavior",
        "over_refusal": "Repair benign non-refusal",
        "prompt_echo": "Stop prompt echoing",
        "unsupported_claims": "Reduce unsupported claims",
    }
    return titles.get(name, name.replace("_", " ").title())


def _gate_issue_repair_action(name: str) -> str:
    if name.startswith("skill_release_sft_"):
        skill = name.removeprefix("skill_release_sft_")
        return f"Regenerate or rebalance the release_skills SFT pack with more reviewed {skill} rows, then rerun SFT."
    if name.startswith("skill_release_eval_"):
        skill = name.removeprefix("skill_release_eval_")
        return f"Add held-out {skill} eval rows from templates/pools not used in SFT, then rerun eval."
    if name.startswith("skill_release_stage_"):
        return "Find the weak stage in the eval breakdown, add targeted SFT drills, and keep the stage held out in eval."
    if name.startswith("external_eval"):
        return "Attach a scoreable ARC/MMLU-style external benchmark report with enough rows for release evidence."
    actions = {
        "preflight": "Do not launch the paid run. Fix the named preflight blockers and rerun preflight first.",
        "data_honesty": "Remove leaked or duplicated eval prompts from corpus/SFT data, then rebuild the dataset pack.",
        "release_data_honesty": "Remove corpus-eval contamination and regenerate the honesty report before making release claims.",
        "sft_fit": "Inspect failed SFT-fit examples, add targeted behavior rows, rebalance categories, and rerun SFT.",
        "sft_heldout_fit": "Diversify the SFT curriculum so behavior transfers to held-out rows instead of memorizing train rows.",
        "eval_non_choice": "Add and train against free-form answer rows; do not rely on multiple-choice pass rate.",
        "first_release_eval": "Improve identity, refusal, and release-choice held-out rows before scaling the recipe.",
        "choice_adjusted_accuracy": "Increase likelihood-margin and paraphrased choice examples so accuracy clears random baseline.",
        "external_eval_missing": "Run at least one external benchmark and attach its JSON report before approval.",
        "refusal": "Separate unsafe refusal prompts from benign prompts, then add targeted refusal SFT/eval rows.",
        "over_refusal": "Add benign answerable prompts and penalize blanket refusals so safety does not pass by saying no to everything.",
        "prompt_echo": "Reduce echo-prone SFT rows, lower prompt-copy examples, and inspect failed eval replies before scaling.",
        "unsupported_claims": "Add support-grounded answers and eval checks for required phrases/entities.",
    }
    return actions.get(name, "Inspect the failing examples for this issue, add targeted data, rerun SFT/eval, then recheck the gate.")


def _preflight_repair_action(name: str) -> str:
    actions = {
        "corpus_model_fit": "Import more unique corpus data or reduce model size/steps so corpus replay is under the gate.",
        "base_exposure": "Increase corpus size or lower planned token budget to avoid excessive document replay.",
        "release_token_budget": "Set base_steps to meet the target token budget, or use a non-release gate profile.",
        "sft_category_balance": "Regenerate the SFT pack with required release categories represented.",
        "eval_skill_release_coverage": "Regenerate eval with held-out math/spelling/choice/refusal coverage.",
        "ddp_scale_launch": "Launch with the requested DDP world size or switch to a matching local scale.",
        "attention_backend_runtime": "Use a CUDA/BF16-compatible attention backend, or choose flash/math for the actual runtime.",
    }
    return actions.get(name, "Fix the preflight check, rerun preflight, and only launch once blockers clear.")


def _load_artifact_inventory(run_dir: Path, summary: dict) -> dict:
    artifacts = summary.get("artifacts", {})
    config = summary.get("config", {})
    out_dir = Path(config.get("out_dir") or run_dir)
    base_checkpoint = summary.get("base", {}).get("checkpoint", out_dir / "base" / "checkpoint")
    base_best_checkpoint = summary.get("base", {}).get("best_checkpoint", {}).get("path", out_dir / "base" / "best_checkpoint")
    sft_checkpoint = summary.get("sft", {}).get("checkpoint", out_dir / "sft" / "checkpoint")

    known_paths = {
        "dataset_pack": (config.get("dataset_pack"), False),
        "corpus_source": (config.get("corpus_recipe") or config.get("corpus_input"), False),
        "chat_input": (config.get("chat_input"), False),
        "eval_input": (config.get("eval_input"), False),
        "corpus": (artifacts.get("corpus", run_dir / "corpus.txt"), True),
        "corpus_manifest": (artifacts.get("corpus_manifest", run_dir / "corpus_manifest.json"), True),
        "corpus_report": (artifacts.get("corpus_report", run_dir / "corpus_report.md"), True),
        "preflight_report": (artifacts.get("preflight_report", run_dir / "preflight.md"), True),
        "honesty_json": (artifacts.get("honesty_json", run_dir / "honesty" / "honesty_report.json"), True),
        "honesty_report": (artifacts.get("honesty_report", run_dir / "honesty" / "report.md"), True),
        "tokenizer": (artifacts.get("tokenizer", run_dir / "tokenizer.json"), True),
        "summary_json": (run_dir / "summary.json", False),
        "summary_report": (run_dir / "summary.md", False),
        "base_checkpoint": (base_checkpoint, True),
        "base_best_checkpoint": (base_best_checkpoint, True),
        "base_trace": (out_dir / "base" / "train_report.json", True),
        "base_report": (artifacts.get("base_report", out_dir / "base" / "report.md"), True),
        "base_sample": (out_dir / "base" / "sample.txt", True),
        "sft_checkpoint": (sft_checkpoint, True),
        "sft_trace": (out_dir / "sft" / "sft_report.json", True),
        "sft_report": (artifacts.get("sft_report", out_dir / "sft" / "report.md"), True),
        "sft_sample": (out_dir / "sft" / "sample.txt", True),
        "eval_json": (out_dir / "eval" / "eval_report.json", True),
        "eval_report": (artifacts.get("eval_report", out_dir / "eval" / "report.md"), True),
    }
    items = [
        _artifact_record(
            key,
            _local_run_artifact_path(run_dir, summary, path) if localize else Path(path),
            aliases=[path] if localize else [],
        )
        for key, (path, localize) in known_paths.items()
        if path
    ]
    return {
        "items": items,
        "by_path": _artifact_records_by_path(items),
    }


def _local_run_artifact_path(run_dir: Path, summary: dict, path: str | Path) -> Path:
    """Resolve run-output artifact paths against the currently selected run folder."""
    source = Path(path)
    out_dir_value = summary.get("config", {}).get("out_dir")
    if not out_dir_value:
        return source

    out_dir = Path(out_dir_value)
    source_variants = [source]
    out_dir_variants = [out_dir]
    if not source.is_absolute():
        source_variants.append(Path.cwd() / source)
    if not out_dir.is_absolute():
        out_dir_variants.append(Path.cwd() / out_dir)

    for source_variant in source_variants:
        for out_dir_variant in out_dir_variants:
            try:
                return run_dir / source_variant.relative_to(out_dir_variant)
            except ValueError:
                continue
    return source


def _artifact_record(key: str, path: Path, aliases: list[str | Path] | None = None) -> dict:
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
        "aliases": [str(alias) for alias in aliases or []],
        "exists": exists,
        "kind": kind,
        "size_bytes": size_bytes,
    }


def _artifact_records_by_path(items: list[dict]) -> dict:
    by_path = {}
    for item in items:
        for raw_path in [item["path"], *item.get("aliases", [])]:
            for path in _path_aliases(Path(raw_path)):
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
