"""Learning-rate range tests for expensive base training recipes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from picochat.batching import (
    DeviceBatchPrefetcher,
    load_packed_token_split,
    load_sharded_token_split,
    load_token_split,
    make_resumable_batcher,
)
from picochat.device import resolve_device
from picochat.distributed import (
    barrier_if_distributed,
    ddp_env_metadata,
    initialize_ddp,
    is_main_process,
    mean_scalar_if_distributed,
    no_sync_if_distributed,
    prepare_ddp_model,
)
from picochat.model import GPTConfig, TinyGPT
from picochat.optim import (
    create_optimizer,
    maybe_clip_grad_norm,
    set_optimizer_lr,
    validate_optim_controls,
)
from picochat.precision import (
    autocast_context,
    configure_float32_matmul_precision,
    make_grad_scaler,
    maybe_compile_model,
    resolve_precision,
)
from picochat.tokenizer import load_tokenizer
from picochat.train import _wait_for_generated_dataset_manifest


@dataclass(frozen=True)
class LRRangeConfig:
    corpus_path: str
    tokenizer_path: str
    out_dir: str
    context_size: int = 512
    batch_size: int = 8
    grad_accum_steps: int = 4
    steps: int = 100
    min_lr: float = 1e-6
    max_lr: float = 3e-4
    divergence_factor: float = 4.0
    smoothing_beta: float = 0.98
    n_embd: int = 128
    n_head: int = 4
    n_kv_head: int | None = None
    n_layer: int = 2
    dropout: float = 0.0
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    activation: str = "gelu"
    tie_embeddings: bool = False
    qk_norm: bool = False
    attn_backend: str = "auto"
    parallel_residual: bool = False
    xsa_last_n: int = 0
    linear_bias: bool = True
    scaled_residual_init: bool = False
    logit_softcap: float = 0.0
    gradient_checkpointing: bool = False
    seed: int = 42
    device: str = "cpu"
    precision: str = "float32"
    matmul_precision: str = "default"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    dataset_mode: str = "memory"
    split_mode: str = "window"
    val_fraction: float = 0.1
    corpus_manifest_path: str | None = None
    shard_token_size: int = 1_000_000
    shard_cache_size: int = 2
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    muon_learning_rate: float = 0.02
    grad_clip: float = 0.0
    ddp: bool = False
    log_every: int = 10


def run_lr_range(config: LRRangeConfig) -> dict:
    """Run a short exponential LR range test and write auditable artifacts."""
    _validate_lr_range(config)
    validate_optim_controls(
        max_steps=config.steps,
        lr_warmup_steps=0,
        lr_decay="none",
        min_lr_ratio=1.0,
        grad_clip=config.grad_clip,
        grad_accum_steps=config.grad_accum_steps,
        optimizer_type=config.optimizer,
        weight_decay=config.weight_decay,
        weight_decay_decay="none",
        muon_learning_rate=config.muon_learning_rate,
        muon_momentum_schedule="none",
        ema_decay=0.0,
        loss_spike_threshold=2.5,
        loss_spike_lr_decay=0.5,
        loss_spike_min_lr_scale=0.1,
        loss_spike_snapshot_every=10,
    )

    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    env_metadata = ddp_env_metadata(config.ddp)
    main_process = is_main_process(env_metadata)
    tokenizer = load_tokenizer(config.tokenizer_path)

    split = _load_split(config, out_dir=out_dir, main_process=main_process)
    ddp_metadata = initialize_ddp(device, enabled=config.ddp)
    main_process = is_main_process(ddp_metadata)
    train_batcher = make_resumable_batcher(
        split.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        pin_memory=device.type == "cuda",
        rank=int(ddp_metadata["rank"]),
        world_size=int(ddp_metadata["world_size"]),
    )

    model_config = GPTConfig(
        vocab_size=len(tokenizer),
        context_size=config.context_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        norm_type=config.norm_type,
        position_encoding=config.position_encoding,
        activation=config.activation,
        tie_embeddings=config.tie_embeddings,
        qk_norm=config.qk_norm,
        attn_backend=config.attn_backend,
        parallel_residual=config.parallel_residual,
        xsa_last_n=config.xsa_last_n,
        linear_bias=config.linear_bias,
        scaled_residual_init=config.scaled_residual_init,
        logit_softcap=config.logit_softcap,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    model = TinyGPT(model_config).to(device)
    matmul_precision_runtime = configure_float32_matmul_precision(config.matmul_precision)
    precision_runtime = resolve_precision(config.precision, device)
    compiled_model, compile_metadata = maybe_compile_model(
        model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    ddp_model, ddp_metadata = prepare_ddp_model(compiled_model, device, enabled=config.ddp)
    main_process = is_main_process(ddp_metadata)
    train_model = ddp_model
    optimizer = create_optimizer(
        model,
        optimizer_type=config.optimizer,
        learning_rate=config.min_lr,
        weight_decay=config.weight_decay,
        muon_learning_rate=config.muon_learning_rate,
    )
    scaler = make_grad_scaler(precision_runtime)
    train_batches = DeviceBatchPrefetcher(train_batcher, device)

    rows: list[dict[str, float | int | str | bool | None]] = []
    smooth_loss = None
    best_smooth_loss = float("inf")
    best_step = 0
    best_lr = config.min_lr
    stopped_reason = "max_steps"
    started = time.time()

    model.train()
    train_model.train()
    for step in range(1, config.steps + 1):
        lr = _lr_for_range_step(
            step=step,
            steps=config.steps,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
        )
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for micro_step in range(config.grad_accum_steps):
            x, y = next(train_batches)
            sync_gradients = micro_step == config.grad_accum_steps - 1
            with no_sync_if_distributed(ddp_model, enabled=config.ddp and not sync_gradients):
                with autocast_context(precision_runtime):
                    _, loss = train_model(x, y)
                assert loss is not None
                micro_losses.append(float(loss.item()))
                scaled_loss = loss / config.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = maybe_clip_grad_norm(model, config.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        local_loss = sum(micro_losses) / len(micro_losses)
        loss_value = mean_scalar_if_distributed(local_loss, device, ddp_metadata)
        if smooth_loss is None:
            smooth_loss = loss_value
        else:
            smooth_loss = config.smoothing_beta * smooth_loss + (1.0 - config.smoothing_beta) * loss_value
        smooth_corrected = smooth_loss
        if smooth_corrected < best_smooth_loss and math.isfinite(smooth_corrected):
            best_smooth_loss = smooth_corrected
            best_step = step
            best_lr = lr
        diverged = (
            step >= max(8, config.steps // 10)
            and math.isfinite(best_smooth_loss)
            and smooth_corrected > best_smooth_loss * config.divergence_factor
        )
        row = {
            "step": step,
            "lr": lr,
            "loss": loss_value,
            "smooth_loss": smooth_corrected,
            "grad_norm": None if grad_norm is None else float(grad_norm),
            "elapsed_sec": time.time() - started,
            "diverged": diverged,
        }
        rows.append(row)
        if main_process and (step == 1 or step == config.steps or step % max(1, config.log_every) == 0 or diverged):
            print(
                f"lr range step {step:04d}/{config.steps:04d} | "
                f"lr {lr:.3g} | loss {loss_value:.4f} | "
                f"smooth {smooth_corrected:.4f} | "
                f"{row['elapsed_sec']:.1f}s",
                flush=True,
            )
        if not math.isfinite(loss_value) or not math.isfinite(smooth_corrected):
            stopped_reason = "nonfinite_loss"
            break
        if diverged:
            stopped_reason = "diverged"
            break

    recommendation = _recommend_lr(rows, best_lr=best_lr, best_step=best_step, config=config)
    report = {
        "config": {
            **config.__dict__,
            "model_config": model_config.to_dict(),
            "matmul_precision_runtime": matmul_precision_runtime,
            "precision_runtime": precision_runtime.to_dict(),
            "compile": compile_metadata,
            "world_size": int(ddp_metadata.get("world_size", 1)),
        },
        "summary": {
            "steps_run": len(rows),
            "stop_reason": stopped_reason,
            "best_step": best_step,
            "best_lr": best_lr,
            "best_smooth_loss": best_smooth_loss,
            **recommendation,
        },
        "rows": rows,
    }
    if main_process:
        json_path = out_dir / "lr_range.json"
        markdown_path = out_dir / "lr_range.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown_path.write_text(_lr_range_markdown(report), encoding="utf-8")
        print(f"lr range report: {markdown_path}", flush=True)
    barrier_if_distributed(ddp_metadata)
    return report


def _validate_lr_range(config: LRRangeConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.min_lr <= 0 or config.max_lr <= 0:
        raise ValueError("min_lr and max_lr must be positive")
    if config.max_lr <= config.min_lr:
        raise ValueError("max_lr must be greater than min_lr")
    if config.divergence_factor <= 1.0:
        raise ValueError("divergence_factor must be greater than 1")
    if not 0.0 <= config.smoothing_beta < 1.0:
        raise ValueError("smoothing_beta must be in [0, 1)")
    if config.dataset_mode not in {"memory", "sharded", "packed"}:
        raise ValueError("dataset_mode must be 'memory', 'sharded', or 'packed'")


def _load_split(config: LRRangeConfig, *, out_dir: Path, main_process: bool):
    if config.dataset_mode in {"sharded", "packed"}:
        cache_dir = out_dir / (
            "packed_token_shards" if config.dataset_mode == "packed" else "token_shards"
        )
        manifest_path = cache_dir / (
            "packed_shards_manifest.json" if config.dataset_mode == "packed" else "token_shards_manifest.json"
        )
        loader = load_packed_token_split if config.dataset_mode == "packed" else load_sharded_token_split
        if main_process:
            print(
                f"lr range data: preparing {config.dataset_mode} token dataset "
                f"({config.shard_token_size:,} tokens/shard, cache={config.shard_cache_size})",
                flush=True,
            )
            return loader(
                corpus_path=config.corpus_path,
                tokenizer_path=config.tokenizer_path,
                context_size=config.context_size,
                cache_dir=cache_dir,
                val_fraction=config.val_fraction,
                seed=config.seed,
                shard_token_size=config.shard_token_size,
                shard_cache_size=config.shard_cache_size,
                corpus_manifest_path=config.corpus_manifest_path,
                progress=True,
                rebuild=True,
            )
        _wait_for_generated_dataset_manifest(
            manifest_path,
            description=f"{config.dataset_mode} LR range token dataset",
        )
        return loader(
            corpus_path=config.corpus_path,
            tokenizer_path=config.tokenizer_path,
            context_size=config.context_size,
            cache_dir=cache_dir,
            val_fraction=config.val_fraction,
            seed=config.seed,
            shard_token_size=config.shard_token_size,
            shard_cache_size=config.shard_cache_size,
            corpus_manifest_path=config.corpus_manifest_path,
            rebuild=False,
        )
    if main_process:
        print("lr range data: preparing in-memory token split", flush=True)
    return load_token_split(
        corpus_path=config.corpus_path,
        tokenizer_path=config.tokenizer_path,
        context_size=config.context_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
        split_mode=config.split_mode,
        corpus_manifest_path=config.corpus_manifest_path,
    )


def _lr_for_range_step(*, step: int, steps: int, min_lr: float, max_lr: float) -> float:
    if steps <= 1:
        return min_lr
    ratio = (step - 1) / (steps - 1)
    return min_lr * ((max_lr / min_lr) ** ratio)


def _recommend_lr(
    rows: list[dict[str, float | int | str | bool | None]],
    *,
    best_lr: float,
    best_step: int,
    config: LRRangeConfig,
) -> dict[str, float | int | str | None]:
    finite_rows = [
        row for row in rows
        if isinstance(row.get("smooth_loss"), float) and math.isfinite(float(row["smooth_loss"]))
    ]
    if not finite_rows:
        return {
            "recommended_lr": None,
            "safe_lr_ceiling": None,
            "recommendation_note": "No finite losses were recorded; lower min_lr and rerun.",
        }
    min_loss = min(float(row["smooth_loss"]) for row in finite_rows)
    safe_rows = [
        row for row in finite_rows
        if float(row["smooth_loss"]) <= min_loss * 1.2 and int(row["step"]) >= max(3, config.steps // 20)
    ]
    safe_lr_ceiling = float(safe_rows[-1]["lr"]) if safe_rows else best_lr
    recommended = max(config.min_lr, min(safe_lr_ceiling / 3.0, best_lr))
    note = (
        "Use the recommendation as an upper-bound probe, then confirm with a short exact-config smoke run."
    )
    if rows and bool(rows[-1].get("diverged")):
        note = "Divergence was detected; use the recommendation or lower, then confirm with a smoke run."
    elif best_step == len(rows):
        note = "Loss was still improving at max_lr; rerun with a higher max_lr or treat this as conservative."
    return {
        "recommended_lr": recommended,
        "safe_lr_ceiling": safe_lr_ceiling,
        "recommendation_note": note,
    }


def _lr_range_markdown(report: dict) -> str:
    summary = report["summary"]
    config = report["config"]
    rows = report["rows"]
    lines = [
        "# Picochat LR Range Test",
        "",
        "This is a short diagnostic for base-training LR choice. It is not a quality claim.",
        "",
        "## Summary",
        "",
        f"- Steps run: {summary['steps_run']}",
        f"- Stop reason: `{summary['stop_reason']}`",
        f"- Best smooth loss: {summary['best_smooth_loss']:.4f} at step {summary['best_step']} / lr {_fmt(summary['best_lr'])}",
        f"- Safe LR ceiling: {_fmt(summary['safe_lr_ceiling'])}",
        f"- Recommended LR: {_fmt(summary['recommended_lr'])}",
        f"- Note: {summary['recommendation_note']}",
        "",
        "## Config",
        "",
        f"- Device: `{config['device']}`",
        f"- Precision: `{config['precision_runtime']['dtype_name']}`",
        f"- Attention backend: `{config['model_config']['attn_backend']}`",
        f"- Dataset mode: `{config['dataset_mode']}`",
        f"- Context: {config['context_size']}",
        f"- Batch / grad accumulation: {config['batch_size']} / {config['grad_accum_steps']}",
        f"- World size: {config['world_size']}",
        "",
        "## Last Rows",
        "",
        "| Step | LR | Loss | Smooth Loss | Grad Norm | Diverged |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows[-12:]:
        lines.append(
            f"| {row['step']} | {_fmt(row['lr'])} | {float(row['loss']):.4f} | "
            f"{float(row['smooth_loss']):.4f} | {_fmt(row['grad_norm'])} | {row['diverged']} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.6g}"
