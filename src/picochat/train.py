"""Base language-model training loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from picochat.batching import load_token_split, make_dataloader
from picochat.checkpoint import save_checkpoint
from picochat.device import resolve_device
from picochat.memorization import memorization_diagnostics
from picochat.model import GPTConfig, TinyGPT
from picochat.optim import learning_rate_for_step, maybe_clip_grad_norm, set_optimizer_lr, validate_optim_controls
from picochat.report import loss_diagnostics, training_report_markdown
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
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    max_minutes: float | None = None
    canary_count: int = 0
    lr_warmup_steps: int = 0
    lr_decay: str = "none"
    min_lr_ratio: float = 1.0
    grad_clip: float = 0.0
    grad_accum_steps: int = 1


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
    )
    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.tokenizer_path)
    canary_values = _canary_values(config.seed, config.canary_count)
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
    train_loader = make_dataloader(split.train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    train_eval_loader = make_dataloader(split.train_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    val_loader = make_dataloader(split.val_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    data_iter = iter(train_loader)

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
    )
    model = TinyGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    effective_batch_size = config.batch_size * config.grad_accum_steps
    effective_tokens_per_step = effective_batch_size * config.context_size

    losses: list[dict[str, float | int]] = []
    start = time.time()
    last_loss = float("nan")
    final_step = 0
    stop_reason = "max_steps"
    best_metric = float("inf")
    best_checkpoint: dict[str, float | int | str] | None = None
    best_checkpoint_dir = out_dir / "best_checkpoint"
    evals_without_improvement = 0

    model.train()
    for step in range(1, config.max_steps + 1):
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
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            assert loss is not None
            micro_losses.append(float(loss.item()))
            (loss / config.grad_accum_steps).backward()
        grad_norm = maybe_clip_grad_norm(model, config.grad_clip)
        optimizer.step()

        last_loss = sum(micro_losses) / len(micro_losses)
        final_step = step
        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            train_metrics = evaluate_metrics(
                model,
                train_eval_loader,
                device,
                max_batches=config.eval_batches,
                token_bytes=token_bytes,
            )
            val_metrics = evaluate_metrics(
                model,
                val_loader,
                device,
                max_batches=config.eval_batches,
                token_bytes=token_bytes,
            )
            elapsed = time.time() - start
            val_loss = float(val_metrics["loss"])
            val_bpb = val_metrics["bpb"]
            train_eval_loss = float(train_metrics["loss"])
            train_bpb = train_metrics["bpb"]
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
            })
            metric = val_bpb if val_bpb is not None else val_loss
            if metric < best_metric - config.early_stop_min_delta:
                best_metric = metric
                evals_without_improvement = 0
                save_checkpoint(
                    best_checkpoint_dir,
                    model,
                    step=step,
                    train_loss=last_loss,
                    extra_metadata={
                        "checkpoint_kind": "best_validation",
                        "val_loss": val_loss,
                        "val_bpb": val_bpb,
                    },
                )
                best_checkpoint = {
                    "path": str(best_checkpoint_dir),
                    "step": step,
                    "train_loss": last_loss,
                    "val_loss": val_loss,
                    "val_bpb": val_bpb,
                }
            else:
                evals_without_improvement += 1
            print(
                f"step {step:04d}/{config.max_steps:04d} | "
                f"train {last_loss:.4f} | val {val_loss:.4f} | "
                f"val_bpb {_format_optional(val_bpb)} | {elapsed:.1f}s"
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
        if config.max_minutes is not None and time.time() - start >= config.max_minutes * 60:
            stop_reason = "max_minutes"
            print(f"time stop: reached {config.max_minutes:.2f} minute budget")
            break

    checkpoint_dir = out_dir / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        model,
        step=final_step,
        train_loss=last_loss,
        extra_metadata={
            "checkpoint_kind": "final",
            "stop_reason": stop_reason,
            "best_checkpoint": best_checkpoint,
        },
    )
    if best_checkpoint is None:
        save_checkpoint(
            best_checkpoint_dir,
            model,
            step=final_step,
            train_loss=last_loss,
            extra_metadata={"checkpoint_kind": "best_validation_fallback"},
        )
        best_checkpoint = {
            "path": str(best_checkpoint_dir),
            "step": final_step,
            "train_loss": last_loss,
            "val_loss": losses[-1]["val_loss"] if losses else None,
            "val_bpb": losses[-1].get("val_bpb") if losses else None,
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
        "memorization": memorization_diagnostics(memorization_text, split.train_text, split.val_text),
        "sample": sample,
        "canary_probe": canary_probe,
        "checkpoint": str(checkpoint_dir),
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
    return {
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


def _safe_ratio(numerator: int | float, denominator) -> float | None:
    try:
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator <= 0:
        return None
    return float(numerator) / denominator


def _format_optional(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"
