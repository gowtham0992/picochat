"""Base language-model training loop."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from picochat.batching import (
    load_sharded_token_split,
    load_token_split,
    make_dataloader,
    make_resumable_batcher,
)
from picochat.checkpoint import load_checkpoint, load_training_state, save_checkpoint
from picochat.device import resolve_device
from picochat.memorization import memorization_diagnostics
from picochat.model import GPTConfig, TinyGPT
from picochat.optim import (
    ExponentialMovingAverage,
    create_optimizer,
    learning_rate_for_step,
    maybe_clip_grad_norm,
    set_optimizer_lr,
    using_ema_weights,
    validate_optim_controls,
)
from picochat.precision import (
    autocast_context,
    make_grad_scaler,
    maybe_compile_model,
    resolve_precision,
)
from picochat.report import loss_diagnostics, optimization_stability, training_report_markdown
from picochat.resume import make_training_state, restore_training_state
from picochat.tokenizer import Tokenizer, load_tokenizer, token_byte_lengths


@dataclass(frozen=True)
class TrainConfig:
    corpus_path: str
    tokenizer_path: str
    out_dir: str
    context_size: int = 64
    batch_size: int = 16
    max_steps: int = 200
    learning_rate: float = 3e-4
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    activation: str = "gelu"
    seed: int = 42
    device: str = "cpu"
    log_every: int = 20
    val_fraction: float = 0.1
    eval_batches: int = 10
    sample_tokens: int = 120
    split_mode: str = "window"
    corpus_manifest_path: str | None = None
    dataset_mode: str = "memory"
    shard_token_size: int = 1_000_000
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    max_minutes: float | None = None
    canary_count: int = 0
    lr_warmup_steps: int = 0
    lr_decay: str = "none"
    min_lr_ratio: float = 1.0
    grad_clip: float = 0.0
    grad_accum_steps: int = 1
    optimizer: str = "adamw"
    muon_learning_rate: float = 0.02
    ema_decay: float = 0.0
    logit_softcap: float = 0.0
    precision: str = "float32"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    resume_from: str | None = None


@torch.no_grad()
def evaluate_loss(model: TinyGPT, loader, device: torch.device, max_batches: int) -> float:
    """Estimate loss over a limited number of validation batches."""
    return evaluate_metrics(model, loader, device, max_batches)["loss"]


@torch.no_grad()
def evaluate_metrics(
    model: TinyGPT,
    loader,
    device: torch.device,
    max_batches: int,
    token_bytes: torch.Tensor | None = None,
    precision_runtime=None,
) -> dict[str, float | None]:
    """Estimate loss and optional bits-per-byte over a limited number of batches."""
    model.eval()
    losses = []
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.long, device=device)
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with autocast_context(precision_runtime) if precision_runtime else nullcontext():
            logits, _ = model(x)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = y.view(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
        losses.append(float(loss.item()))
        if token_bytes is not None:
            loss_by_token = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=-100,
                reduction="none",
            )
            valid = flat_targets >= 0
            safe_targets = torch.where(valid, flat_targets, torch.zeros_like(flat_targets))
            byte_counts = torch.where(
                valid,
                token_bytes[safe_targets],
                torch.zeros_like(safe_targets, dtype=token_bytes.dtype),
            )
            counted = byte_counts > 0
            total_nats += (loss_by_token * counted).sum()
            total_bytes += byte_counts.sum()
    model.train()
    bpb = None
    if token_bytes is not None and int(total_bytes.item()) > 0:
        bpb = float(total_nats.item() / (0.6931471805599453 * int(total_bytes.item())))
    if not losses:
        return {"loss": float("nan"), "bpb": bpb}
    return {"loss": sum(losses) / len(losses), "bpb": bpb}


def train_base(config: TrainConfig) -> dict:
    """Train a tiny next-token model and save artifacts."""
    validate_optim_controls(
        max_steps=config.max_steps,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay=config.lr_decay,
        min_lr_ratio=config.min_lr_ratio,
        grad_clip=config.grad_clip,
        grad_accum_steps=config.grad_accum_steps,
        optimizer_type=config.optimizer,
        muon_learning_rate=config.muon_learning_rate,
        ema_decay=config.ema_decay,
    )
    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.tokenizer_path)
    if config.dataset_mode not in {"memory", "sharded"}:
        raise ValueError("dataset_mode must be 'memory' or 'sharded'")
    canary_values = _canary_values(config.seed, config.canary_count)
    if config.dataset_mode == "sharded":
        split = load_sharded_token_split(
            corpus_path=config.corpus_path,
            tokenizer_path=config.tokenizer_path,
            context_size=config.context_size,
            cache_dir=out_dir / "token_shards",
            val_fraction=config.val_fraction,
            seed=config.seed,
            shard_token_size=config.shard_token_size,
        )
    else:
        split = load_token_split(
            corpus_path=config.corpus_path,
            tokenizer_path=config.tokenizer_path,
            context_size=config.context_size,
            val_fraction=config.val_fraction,
            seed=config.seed,
            split_mode=config.split_mode,
            corpus_manifest_path=config.corpus_manifest_path,
            canary_values=canary_values,
        )
    train_batcher = make_resumable_batcher(
        split.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    train_eval_loader = make_dataloader(split.train_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    val_loader = make_dataloader(split.val_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)

    device = resolve_device(config.device)
    token_bytes = torch.tensor(token_byte_lengths(tokenizer), dtype=torch.long, device=device)
    model_config = GPTConfig(
        vocab_size=len(tokenizer),
        context_size=config.context_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        norm_type=config.norm_type,
        position_encoding=config.position_encoding,
        activation=config.activation,
        logit_softcap=config.logit_softcap,
    )
    resume_state = None
    resume_metadata = None
    if config.resume_from:
        model, resume_metadata = load_checkpoint(config.resume_from, map_location=device)
        if model.config.to_dict() != model_config.to_dict():
            raise ValueError("resume checkpoint model config does not match this train command")
        resume_state = load_training_state(config.resume_from, map_location=device)
    else:
        model = TinyGPT(model_config)
    model = model.to(device)
    precision_runtime = resolve_precision(config.precision, device)
    train_model, compile_metadata = maybe_compile_model(
        model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    scaler = make_grad_scaler(precision_runtime)
    optimizer = create_optimizer(
        model,
        optimizer_type=config.optimizer,
        learning_rate=config.learning_rate,
        muon_learning_rate=config.muon_learning_rate,
    )
    ema = ExponentialMovingAverage(model, config.ema_decay) if config.ema_decay > 0 else None
    effective_batch_size = config.batch_size * config.grad_accum_steps
    effective_tokens_per_step = effective_batch_size * config.context_size

    losses: list[dict[str, float | int]] = []
    start = time.time()
    elapsed_offset = 0.0
    last_loss = float("nan")
    final_step = 0
    stop_reason = "max_steps"
    best_metric = float("inf")
    best_checkpoint: dict[str, float | int | str] | None = None
    best_checkpoint_dir = out_dir / "best_checkpoint"
    evals_without_improvement = 0
    start_step = 1
    if resume_state is not None:
        restore_training_state(
            resume_state,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            batcher=train_batcher,
        )
        final_step = int(resume_state.get("step", resume_metadata.get("step", 0)))
        start_step = final_step + 1
        losses = list(resume_state.get("losses", []))
        if losses:
            last_loss = float(losses[-1].get("train_loss", last_loss))
        best_metric = float(resume_state.get("best_metric", best_metric))
        best_checkpoint = resume_state.get("best_checkpoint")
        evals_without_improvement = int(resume_state.get("evals_without_improvement", 0))
        elapsed_offset = float(resume_state.get("elapsed_sec", 0.0))

    model.train()
    train_model.train()
    for step in range(start_step, config.max_steps + 1):
        learning_rate = learning_rate_for_step(
            base_learning_rate=config.learning_rate,
            step=step,
            max_steps=config.max_steps,
            warmup_steps=config.lr_warmup_steps,
            decay=config.lr_decay,
            min_lr_ratio=config.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(config.grad_accum_steps):
            x, y = next(train_batcher)
            x = x.to(device)
            y = y.to(device)
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
        if ema is not None:
            ema.update(model)

        last_loss = sum(micro_losses) / len(micro_losses)
        final_step = step
        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            train_metrics = evaluate_metrics(
                train_model,
                train_eval_loader,
                device,
                max_batches=config.eval_batches,
                token_bytes=token_bytes,
                precision_runtime=precision_runtime,
            )
            val_metrics = evaluate_metrics(
                train_model,
                val_loader,
                device,
                max_batches=config.eval_batches,
                token_bytes=token_bytes,
                precision_runtime=precision_runtime,
            )
            elapsed = elapsed_offset + time.time() - start
            val_loss = float(val_metrics["loss"])
            val_bpb = val_metrics["bpb"]
            train_eval_loss = float(train_metrics["loss"])
            train_bpb = train_metrics["bpb"]
            ema_val_metrics = None
            if ema is not None:
                with using_ema_weights(model, ema):
                    ema_val_metrics = evaluate_metrics(
                        train_model,
                        val_loader,
                        device,
                        max_batches=config.eval_batches,
                        token_bytes=token_bytes,
                        precision_runtime=precision_runtime,
                    )
            losses.append({
                "step": step,
                "train_loss": last_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                "train_bpb": train_bpb,
                "val_bpb": val_bpb,
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "grad_accum_steps": config.grad_accum_steps,
                "effective_batch_size": effective_batch_size,
                "effective_tokens_per_step": effective_tokens_per_step,
                "elapsed_sec": elapsed,
                **({
                    "ema_val_loss": float(ema_val_metrics["loss"]),
                    "ema_val_bpb": ema_val_metrics["bpb"],
                    "ema_decay": config.ema_decay,
                } if ema_val_metrics is not None else {}),
            })
            if ema_val_metrics is not None:
                ema_val_loss = float(ema_val_metrics["loss"])
                ema_val_bpb = ema_val_metrics["bpb"]
                metric = ema_val_bpb if ema_val_bpb is not None else ema_val_loss
                checkpoint_weights = "ema"
                checkpoint_val_loss = ema_val_loss
                checkpoint_val_bpb = ema_val_bpb
            else:
                metric = val_bpb if val_bpb is not None else val_loss
                checkpoint_weights = "raw"
                checkpoint_val_loss = val_loss
                checkpoint_val_bpb = val_bpb
            if metric < best_metric - config.early_stop_min_delta:
                best_metric = metric
                evals_without_improvement = 0
                with using_ema_weights(model, ema):
                    save_checkpoint(
                        best_checkpoint_dir,
                        model,
                        step=step,
                        train_loss=last_loss,
                        extra_metadata={
                            "checkpoint_kind": "best_validation",
                            "weights": checkpoint_weights,
                            "val_loss": checkpoint_val_loss,
                            "val_bpb": checkpoint_val_bpb,
                            "raw_val_loss": val_loss,
                            "raw_val_bpb": val_bpb,
                            "ema_decay": config.ema_decay if ema is not None else None,
                            "ema_updates": ema.num_updates if ema is not None else 0,
                        },
                    )
                best_checkpoint = {
                    "path": str(best_checkpoint_dir),
                    "step": step,
                    "train_loss": last_loss,
                    "val_loss": checkpoint_val_loss,
                    "val_bpb": checkpoint_val_bpb,
                    "weights": checkpoint_weights,
                    "raw_val_loss": val_loss,
                    "raw_val_bpb": val_bpb,
                }
            else:
                evals_without_improvement += 1
            print(
                f"step {step:04d}/{config.max_steps:04d} | "
                f"train {last_loss:.4f} | val {val_loss:.4f} | "
                f"val_bpb {_format_optional(val_bpb)} | {elapsed:.1f}s"
            )
            save_checkpoint(
                out_dir / "resume_checkpoint",
                model,
                step=final_step,
                train_loss=last_loss,
                extra_metadata={
                    "checkpoint_kind": "resume",
                    "weights": "raw",
                    "best_checkpoint": best_checkpoint,
                    "resume_source": config.resume_from,
                },
                training_state=make_training_state(
                    step=final_step,
                    losses=losses,
                    best_metric=best_metric,
                    best_checkpoint=best_checkpoint,
                    evals_without_improvement=evals_without_improvement,
                    stop_reason=stop_reason,
                    elapsed_sec=elapsed,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    batcher=train_batcher,
                    device=device,
                ),
            )
            if (
                config.early_stop_patience > 0
                and evals_without_improvement >= config.early_stop_patience
            ):
                stop_reason = "early_stop"
                print(
                    f"early stop: validation did not improve for "
                    f"{config.early_stop_patience} evals"
                )
                break
        if config.max_minutes is not None and elapsed_offset + time.time() - start >= config.max_minutes * 60:
            stop_reason = "max_minutes"
            print(f"time stop: reached {config.max_minutes:.2f} minute budget")
            break

    checkpoint_dir = out_dir / "checkpoint"
    ema_checkpoint_dir = out_dir / "ema_checkpoint"
    elapsed_final = elapsed_offset + time.time() - start
    save_checkpoint(
        checkpoint_dir,
        model,
        step=final_step,
        train_loss=last_loss,
        extra_metadata={
            "checkpoint_kind": "final",
            "weights": "raw",
            "stop_reason": stop_reason,
            "best_checkpoint": best_checkpoint,
            "ema_checkpoint": str(ema_checkpoint_dir) if ema is not None else None,
        },
        training_state=make_training_state(
            step=final_step,
            losses=losses,
            best_metric=best_metric,
            best_checkpoint=best_checkpoint,
            evals_without_improvement=evals_without_improvement,
            stop_reason=stop_reason,
            elapsed_sec=elapsed_final,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            batcher=train_batcher,
            device=device,
        ),
    )
    if ema is not None:
        with using_ema_weights(model, ema):
            save_checkpoint(
                ema_checkpoint_dir,
                model,
                step=final_step,
                train_loss=last_loss,
                extra_metadata={
                    "checkpoint_kind": "final_ema",
                    "weights": "ema",
                    "stop_reason": stop_reason,
                    "ema_decay": config.ema_decay,
                    "ema_updates": ema.num_updates,
                    "raw_checkpoint": str(checkpoint_dir),
                    "best_checkpoint": best_checkpoint,
                },
            )
    if best_checkpoint is None:
        with using_ema_weights(model, ema):
            save_checkpoint(
                best_checkpoint_dir,
                model,
                step=final_step,
                train_loss=last_loss,
                extra_metadata={
                    "checkpoint_kind": "best_validation_fallback",
                    "weights": "ema" if ema is not None else "raw",
                    "ema_decay": config.ema_decay if ema is not None else None,
                    "ema_updates": ema.num_updates if ema is not None else 0,
                },
            )
        best_checkpoint = {
            "path": str(best_checkpoint_dir),
            "step": final_step,
            "train_loss": last_loss,
            "val_loss": losses[-1]["val_loss"] if losses else None,
            "val_bpb": losses[-1].get("val_bpb") if losses else None,
            "weights": "ema" if ema is not None else "raw",
        }

    sample = _generate_sample(model, tokenizer, device, config.sample_tokens, config.seed)
    canary_probe = ""
    if split.canary_values:
        canary_probe = _generate_sample(
            model,
            tokenizer,
            device,
            min(config.sample_tokens, 80),
            config.seed + 17,
            prompt_text="Memorization canary phrase:",
        )
    memorization_text = f"{sample}\n{canary_probe}" if canary_probe else sample

    report = {
        "config": {
            **config.__dict__,
            "requested_device": config.device,
            "device": device.type,
            "effective_batch_size": effective_batch_size,
            "effective_tokens_per_step": effective_tokens_per_step,
            "optimizer_metadata": optimizer.metadata,
            "precision_runtime": precision_runtime.to_dict(),
            "torch_compile_metadata": compile_metadata,
        },
        "dataset": {
            **split.stats,
        },
        "coverage": _coverage_report(split.stats, config, final_step),
        "model": {
            "config": model_config.to_dict(),
            "num_parameters": model.num_parameters(),
        },
        "losses": losses,
        "loss_diagnostics": loss_diagnostics(losses),
        "optimization_stability": optimization_stability(losses, config.grad_clip),
        "memorization": memorization_diagnostics(memorization_text, split.train_text, split.val_text),
        "sample": sample,
        "canary_probe": canary_probe,
        "checkpoint": str(checkpoint_dir),
        "resume_checkpoint": str(out_dir / "resume_checkpoint"),
        "ema_checkpoint": str(ema_checkpoint_dir) if ema is not None else None,
        "best_checkpoint": best_checkpoint,
        "stop_reason": stop_reason,
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(training_report_markdown(report), encoding="utf-8")
    (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
    if canary_probe:
        (out_dir / "canary_probe.txt").write_text(canary_probe, encoding="utf-8")
    return report


def _canary_values(seed: int, count: int) -> tuple[str, ...]:
    if count < 0:
        raise ValueError("canary_count must be non-negative")
    return tuple(f"pico-canary-{seed:04d}-{index:02d}" for index in range(count))


@torch.no_grad()
def _generate_sample(
    model: TinyGPT,
    tokenizer: Tokenizer,
    device: torch.device,
    sample_tokens: int,
    seed: int,
    prompt_text: str | None = None,
) -> str:
    model.eval()
    if prompt_text:
        prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
    else:
        prompt_ids = [tokenizer.bos_id]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        prompt,
        max_new_tokens=sample_tokens,
        temperature=0.8,
        top_k=20,
        seed=seed,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(generated[0].tolist())


def _coverage_report(dataset_stats: dict, config: TrainConfig, actual_steps: int) -> dict:
    tokens_per_step = config.batch_size * config.grad_accum_steps * config.context_size
    tokens_seen = actual_steps * tokens_per_step
    train_tokens = dataset_stats.get("train_tokens") or dataset_stats.get("num_tokens")
    total_tokens = dataset_stats.get("num_tokens")
    report = {
        "actual_steps": actual_steps,
        "planned_steps": config.max_steps,
        "micro_batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "tokens_per_step_estimate": tokens_per_step,
        "planned_training_tokens": config.max_steps * tokens_per_step,
        "actual_training_tokens": tokens_seen,
        "train_tokens": train_tokens,
        "dataset_tokens": total_tokens,
        "estimated_train_epochs": _safe_ratio(tokens_seen, train_tokens),
        "estimated_dataset_passes": _safe_ratio(tokens_seen, total_tokens),
    }
    report["warnings"] = _coverage_warnings(
        train_epochs=report["estimated_train_epochs"],
        dataset_passes=report["estimated_dataset_passes"],
    )
    return report


def _safe_ratio(numerator: int | float, denominator) -> float | None:
    try:
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator <= 0:
        return None
    return float(numerator) / denominator


def _coverage_warnings(
    *,
    train_epochs: float | None,
    dataset_passes: float | None,
) -> list[str]:
    warnings: list[str] = []
    if train_epochs is not None:
        if train_epochs >= 20:
            warnings.append(
                "High base exposure: planned tokens are >=20 train-set passes. "
                "Prefer more corpus or a shorter run unless validation BPB is still improving."
            )
        elif train_epochs >= 8:
            warnings.append(
                "Moderate base exposure: planned tokens are >=8 train-set passes. "
                "Inspect memorization and validation BPB before scaling further."
            )
    if dataset_passes is not None and dataset_passes >= 10:
        warnings.append(
            "The full corpus is recycled many times. Treat any score gain as suspect "
            "unless held-out document validation improves."
        )
    return warnings


def _format_optional(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"
