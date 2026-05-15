"""Pre-run sanity suites for long Picochat training jobs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

from picochat.batching import load_sharded_token_split
from picochat.checkpoint import save_checkpoint
from picochat.device import resolve_device
from picochat.generate import GenerateConfig, generate_text_with_trace
from picochat.hf_export import HFExportConfig, export_hf_checkpoint
from picochat.model import GPTConfig, TinyGPT
from picochat.precision import (
    autocast_context,
    configure_float32_matmul_precision,
    make_grad_scaler,
    maybe_compile_model,
    resolve_precision,
)
from picochat.tokenizer import CharTokenizer
from picochat.train import TrainConfig, train_base


@dataclass(frozen=True)
class PreH100SanityConfig:
    out_dir: str
    device: str = "cpu"
    precision: str = "auto"
    matmul_precision: str = "default"
    attn_backend: str = "auto"
    include_compile: bool = False


def run_preh100_sanity(config: PreH100SanityConfig) -> dict[str, Any]:
    """Run lightweight checks before spending H100 time on a long run."""
    out_dir = Path(config.out_dir)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    for name, check in [
        ("attention_backend", _check_attention_backend),
        ("precision_backward", _check_precision_backward),
        ("kv_cache_equivalence", _check_kv_cache_equivalence),
        ("resume_fingerprint_guard", _check_resume_fingerprint_guard),
        ("sharded_loader", _check_sharded_loader),
        ("hf_export", _check_hf_export),
        ("torch_compile", _check_torch_compile),
    ]:
        checks.append(_run_check(name, check, config, work_dir))

    failed = [check for check in checks if check["status"] == "fail"]
    report = {
        "suite": "pre-h100",
        "status": "failed" if failed else "passed",
        "out_dir": str(out_dir),
        "checks": checks,
    }
    json_path = out_dir / "preh100_sanity.json"
    markdown_path = out_dir / "preh100_sanity.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(preh100_sanity_markdown(report), encoding="utf-8")
    report["report_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def preh100_sanity_markdown(report: dict[str, Any]) -> str:
    """Render a concise Markdown report for a pre-H100 sanity run."""
    lines = [
        "# Picochat Pre-H100 Sanity",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Check | Status | Seconds | Details |",
        "| --- | --- | ---: | --- |",
    ]
    for check in report["checks"]:
        detail = check.get("detail") or check.get("error") or ""
        lines.append(
            f"| `{check['name']}` | {check['status']} | {check['seconds']:.3f} | {detail} |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_check(
    name: str,
    check: Callable[[PreH100SanityConfig, Path], dict[str, Any]],
    config: PreH100SanityConfig,
    work_dir: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = check(config, work_dir / name)
        status = result.pop("status", "pass")
        detail = result.pop("detail", "")
        return {
            "name": name,
            "status": status,
            "seconds": time.perf_counter() - start,
            "detail": detail,
            **result,
        }
    except Exception as exc:  # pragma: no cover - exercised through failure reports
        return {
            "name": name,
            "status": "fail",
            "seconds": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _check_precision_backward(config: PreH100SanityConfig, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    matmul_runtime = configure_float32_matmul_precision(config.matmul_precision)
    runtime = resolve_precision(config.precision, device)
    model = TinyGPT(_tiny_model_config(
        vocab_size=32,
        gradient_checkpointing=True,
        attn_backend=config.attn_backend,
    )).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(runtime)
    x = torch.randint(0, model.config.vocab_size, (2, 8), device=device)
    y = torch.randint(0, model.config.vocab_size, (2, 8), device=device)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(runtime):
        _, loss = model(x, y)
    if loss is None or not torch.isfinite(loss):
        raise AssertionError("loss is not finite")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "detail": (
            f"loss={float(loss.detach().cpu()):.4f}, precision={runtime.dtype_name}, "
            f"matmul={matmul_runtime['after'] or matmul_runtime['requested']}"
        ),
        "precision_runtime": runtime.to_dict(),
        "matmul_precision_runtime": matmul_runtime,
    }


def _check_attention_backend(config: PreH100SanityConfig, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model = TinyGPT(_tiny_model_config(vocab_size=32, attn_backend=config.attn_backend)).to(device)
    model.eval()
    x = torch.randint(0, model.config.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        logits, _ = model(x)
    if logits.shape != (1, 8, model.config.vocab_size):
        raise AssertionError("attention backend returned bad logits shape")
    return {
        "detail": f"attn_backend={config.attn_backend}, device={device.type}",
        "attn_backend": config.attn_backend,
    }


def _check_kv_cache_equivalence(config: PreH100SanityConfig, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path, checkpoint_path = _write_checkpoint_fixture(work_dir)
    tokenizer = CharTokenizer.load(tokenizer_path)
    model_config = GPTConfig(
        vocab_size=len(tokenizer),
        context_size=24,
        n_embd=32,
        n_head=4,
        n_layer=2,
        position_encoding="rope",
        attn_backend=config.attn_backend,
    )
    model = TinyGPT(model_config)
    model.eval()
    ids = torch.tensor([tokenizer.encode("hello pico", add_bos=True)], dtype=torch.long)
    with torch.no_grad():
        full_logits, _ = model(ids)
        cached_parts = []
        past_kv = None
        for token_index in range(ids.size(1)):
            logits, _, past_kv = model(
                ids[:, token_index: token_index + 1],
                past_kv=past_kv,
                use_cache=True,
            )
            cached_parts.append(logits)
        cached_logits = torch.cat(cached_parts, dim=1)
    max_diff = float((full_logits - cached_logits).abs().max().item())
    if max_diff > 1e-4:
        raise AssertionError(f"cached logits diverged from full logits: {max_diff}")

    cached = generate_text_with_trace(GenerateConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        prompt="hello",
        max_new_tokens=2,
        temperature=0,
        use_kv_cache=True,
        device=config.device,
    ))
    uncached = generate_text_with_trace(GenerateConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        prompt="hello",
        max_new_tokens=2,
        temperature=0,
        use_kv_cache=False,
        device=config.device,
    ))
    if cached["completion"] != uncached["completion"]:
        raise AssertionError("cached and uncached generation produced different completions")
    if not cached["used_kv_cache"]:
        raise AssertionError("generation did not use the KV cache")
    return {
        "detail": f"max_logit_diff={max_diff:.2e}, generated={cached['completion']!r}",
        "max_logit_diff": max_diff,
    }


def _check_resume_fingerprint_guard(config: PreH100SanityConfig, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_path, tokenizer_path = _write_training_fixture(work_dir)
    run_dir = work_dir / "run"
    train_config = TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(run_dir),
        context_size=8,
        batch_size=2,
        max_steps=1,
        learning_rate=1e-3,
        n_embd=32,
        n_head=4,
        n_layer=1,
        log_every=1,
        val_fraction=0.25,
        eval_batches=1,
        sample_tokens=2,
        seed=123,
        device=config.device,
        precision=config.precision,
        matmul_precision=config.matmul_precision,
        attn_backend=config.attn_backend,
    )
    report = train_base(train_config)
    checkpoint = report["checkpoint"]
    corpus_path.write_text(
        corpus_path.read_text(encoding="utf-8") + "\nchanged corpus after checkpoint\n",
        encoding="utf-8",
    )
    try:
        train_base(TrainConfig(
            **{
                **train_config.__dict__,
                "max_steps": 2,
                "resume_from": checkpoint,
            }
        ))
    except ValueError as exc:
        if "fingerprint" not in str(exc):
            raise
        return {
            "detail": "changed corpus rejected on resume",
            "checkpoint": checkpoint,
        }
    raise AssertionError("resume accepted a changed corpus fingerprint")


def _check_sharded_loader(config: PreH100SanityConfig, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_path, tokenizer_path = _write_training_fixture(work_dir)
    bundle = load_sharded_token_split(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        context_size=8,
        cache_dir=work_dir / "token_shards",
        val_fraction=0.25,
        seed=123,
        shard_token_size=12,
        shard_cache_size=2,
    )
    x, y = bundle.train_dataset[0]
    if x.shape != y.shape or x.numel() != 8:
        raise AssertionError("bad sharded batch shape")
    return {
        "detail": (
            f"shards={bundle.stats['num_shards']}, "
            f"train_sequences={bundle.stats['train_sequences']}"
        ),
        "stats": bundle.stats,
    }


def _check_hf_export(config: PreH100SanityConfig, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path, checkpoint_path = _write_checkpoint_fixture(work_dir)
    out_dir = work_dir / "hf"
    report = export_hf_checkpoint(HFExportConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        model_name="picochat-preh100-sanity",
        dataset_summary="Synthetic pre-H100 sanity data.",
        eval_summary="Smoke export only.",
    ))
    required = ["config", "weights", "tokenizer", "model_card", "serving_manifest"]
    missing = [
        name for name in required
        if not (out_dir / report["files"][name]).exists()
    ]
    if missing:
        raise AssertionError(f"missing export files: {missing}")
    return {
        "detail": f"files={len(report['files'])}, params={report['num_parameters']}",
        "export": report,
    }


def _check_torch_compile(config: PreH100SanityConfig, work_dir: Path) -> dict[str, Any]:
    if not config.include_compile:
        return {
            "status": "skip",
            "detail": "pass --include-compile to run torch.compile smoke test",
        }
    work_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model = TinyGPT(_tiny_model_config(vocab_size=32, attn_backend=config.attn_backend)).to(device)
    compiled, metadata = maybe_compile_model(model, enabled=True)
    x = torch.randint(0, model.config.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        logits, _ = compiled(x)
    if logits.shape != (1, 8, model.config.vocab_size):
        raise AssertionError("compiled model returned bad logits shape")
    return {
        "detail": "compiled forward pass succeeded",
        "compile": metadata,
    }


def _write_training_fixture(work_dir: Path) -> tuple[Path, Path]:
    text = (
        "Picochat sanity data checks resumes, shards, cache paths, and export paths.\n"
        "Small transparent language models need disciplined preflight gates.\n"
    ) * 4
    corpus_path = work_dir / "corpus.txt"
    tokenizer_path = work_dir / "tokenizer.json"
    corpus_path.write_text(text, encoding="utf-8")
    tokenizer = CharTokenizer.train([text])
    tokenizer.save(tokenizer_path)
    return corpus_path, tokenizer_path


def _write_checkpoint_fixture(work_dir: Path) -> tuple[Path, Path]:
    text = "hello picochat sanity"
    tokenizer_path = work_dir / "tokenizer.json"
    checkpoint_path = work_dir / "checkpoint"
    tokenizer = CharTokenizer.train([text])
    tokenizer.save(tokenizer_path)
    torch.manual_seed(7)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=24,
        n_embd=32,
        n_head=4,
        n_layer=2,
        position_encoding="rope",
        attn_backend="auto",
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)
    return tokenizer_path, checkpoint_path


def _tiny_model_config(
    vocab_size: int,
    *,
    gradient_checkpointing: bool = False,
    attn_backend: str = "auto",
) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size,
        context_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="relu2",
        logit_softcap=30.0,
        gradient_checkpointing=gradient_checkpointing,
        attn_backend=attn_backend,
    )
