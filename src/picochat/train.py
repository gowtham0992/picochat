"""Base language-model training loop."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from picochat.batching import (
    DeviceBatchPrefetcher,
    load_sharded_token_split,
    load_token_split,
    make_dataloader,
    make_resumable_batcher,
    move_batch_to_device,
)
from picochat.checkpoint import load_checkpoint, load_training_state, save_checkpoint
from picochat.device import resolve_device
from picochat.distributed import (
    barrier_if_distributed,
    initialize_ddp,
    is_main_process,
    no_sync_if_distributed,
    prepare_ddp_model,
)
from picochat.memorization import memorization_diagnostics
from picochat.model import GPTConfig, TinyGPT
from picochat.optim import (
    ExponentialMovingAverage,
    create_optimizer,
    learning_rate_for_step,
    maybe_clip_grad_norm,
    muon_momentum_for_step,
    set_muon_momentum,
    set_optimizer_lr,
    set_optimizer_weight_decay,
    using_ema_weights,
    validate_optim_controls,
    weight_decay_for_step,
)
from picochat.precision import (
    autocast_context,
    configure_float32_matmul_precision,
    make_grad_scaler,
    maybe_compile_model,
    resolve_precision,
)
from picochat.progress import write_checkpoint_progress
from picochat.report import loss_diagnostics, optimization_stability, training_report_markdown
from picochat.resume import (
    file_sha256,
    make_training_fingerprint,
    make_training_state,
    restore_training_state,
    validate_training_fingerprint,
)
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
    shard_cache_size: int = 2
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
    weight_decay: float = 0.01
    weight_decay_decay: str = "none"
    muon_learning_rate: float = 0.02
    muon_momentum_schedule: str = "none"
    ema_decay: float = 0.0
    logit_softcap: float = 0.0
    precision: str = "float32"
    matmul_precision: str = "default"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    resume_from: str | None = None
    gradient_checkpointing: bool = False
    ddp: bool = False
    loss_spike_rollback: bool = False
    loss_spike_threshold: float = 2.5
    loss_spike_lr_decay: float = 0.5
    loss_spike_min_lr_scale: float = 0.1
    loss_spike_snapshot_every: int = 10


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
        x, y = move_batch_to_device(x, y, device)
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
    if config.ddp and config.loss_spike_rollback:
        raise ValueError(
            "loss_spike_rollback is not supported with DDP because rollback "
            "decisions are rank-local; disable rollback for distributed runs"
        )
    validate_optim_controls(
        max_steps=config.max_steps,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay=config.lr_decay,
        min_lr_ratio=config.min_lr_ratio,
        grad_clip=config.grad_clip,
        grad_accum_steps=config.grad_accum_steps,
        optimizer_type=config.optimizer,
        weight_decay=config.weight_decay,
        weight_decay_decay=config.weight_decay_decay,
        muon_learning_rate=config.muon_learning_rate,
        muon_momentum_schedule=config.muon_momentum_schedule,
        ema_decay=config.ema_decay,
        loss_spike_threshold=config.loss_spike_threshold,
        loss_spike_lr_decay=config.loss_spike_lr_decay,
        loss_spike_min_lr_scale=config.loss_spike_min_lr_scale,
        loss_spike_snapshot_every=config.loss_spike_snapshot_every,
    )
    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    ddp_metadata = initialize_ddp(device, enabled=config.ddp)
    main_process = is_main_process(ddp_metadata)
    tokenizer = load_tokenizer(config.tokenizer_path)
    if config.dataset_mode not in {"memory", "sharded"}:
        raise ValueError("dataset_mode must be 'memory' or 'sharded'")
    canary_values = _canary_values(config.seed, config.canary_count)
    if config.dataset_mode == "sharded":
        if main_process:
            print(
                "base data: preparing sharded token dataset "
                f"({config.shard_token_size:,} tokens/shard, cache={config.shard_cache_size})",
                flush=True,
            )
            split = load_sharded_token_split(
                corpus_path=config.corpus_path,
                tokenizer_path=config.tokenizer_path,
                context_size=config.context_size,
                cache_dir=out_dir / "token_shards",
                val_fraction=config.val_fraction,
                seed=config.seed,
                shard_token_size=config.shard_token_size,
                shard_cache_size=config.shard_cache_size,
                corpus_manifest_path=config.corpus_manifest_path,
                progress=True,
                rebuild=True,
            )
        barrier_if_distributed(ddp_metadata)
        if not main_process:
            split = load_sharded_token_split(
                corpus_path=config.corpus_path,
                tokenizer_path=config.tokenizer_path,
                context_size=config.context_size,
                cache_dir=out_dir / "token_shards",
                val_fraction=config.val_fraction,
                seed=config.seed,
                shard_token_size=config.shard_token_size,
                shard_cache_size=config.shard_cache_size,
                corpus_manifest_path=config.corpus_manifest_path,
                rebuild=False,
            )
    else:
        if main_process:
            print("base data: preparing in-memory token split", flush=True)
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
    if main_process:
        print(
            "base data: ready "
            f"train={len(split.train_dataset):,} val={len(split.val_dataset):,} "
            f"mode={split.stats.get('split_mode', config.dataset_mode)}",
            flush=True,
        )
    ddp_env = ddp_metadata
    train_batcher = make_resumable_batcher(
        split.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        pin_memory=device.type == "cuda",
        rank=int(ddp_env["rank"]),
        world_size=int(ddp_env["world_size"]),
    )
    batcher_metadata = train_batcher.state_dict()
    if main_process:
        print(
            "base data: batch sampling "
            f"mode={batcher_metadata.get('resolved_index_mode', 'unknown')} "
            f"batches/epoch={batcher_metadata.get('batches_per_epoch', 'unknown')}",
            flush=True,
        )
    pin_memory = device.type == "cuda"
    train_eval_loader = make_dataloader(
        split.train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        split.val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    token_bytes = torch.tensor(token_byte_lengths(tokenizer), dtype=torch.long, device=device)
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
        logit_softcap=config.logit_softcap,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    training_fingerprint = make_training_fingerprint({
        "kind": "base",
        "corpus_sha256": file_sha256(config.corpus_path),
        "tokenizer_sha256": file_sha256(config.tokenizer_path),
        "corpus_manifest_sha256": file_sha256(config.corpus_manifest_path),
        "model_config": model_config.to_dict(),
        "dataset_mode": config.dataset_mode,
        "split_mode": config.split_mode,
        "val_fraction": config.val_fraction,
        "seed": config.seed,
        "context_size": config.context_size,
        "shard_token_size": config.shard_token_size if config.dataset_mode == "sharded" else None,
        "shard_cache_size": config.shard_cache_size if config.dataset_mode == "sharded" else None,
    })
    resume_state = None
    resume_metadata = None
    if config.resume_from:
        model, resume_metadata = load_checkpoint(config.resume_from, map_location=device)
        if model.config.to_dict() != model_config.to_dict():
            raise ValueError("resume checkpoint model config does not match this train command")
        resume_state = load_training_state(config.resume_from, map_location=device)
        validate_training_fingerprint(resume_state, training_fingerprint)
    else:
        model = TinyGPT(model_config)
    model = model.to(device)
    matmul_precision_runtime = configure_float32_matmul_precision(config.matmul_precision)
    precision_runtime = resolve_precision(config.precision, device)
    ddp_model, ddp_metadata = prepare_ddp_model(model, device, enabled=config.ddp)
    main_process = is_main_process(ddp_metadata)
    train_model, compile_metadata = maybe_compile_model(
        ddp_model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    scaler = make_grad_scaler(precision_runtime)
    optimizer = create_optimizer(
        model,
        optimizer_type=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        muon_learning_rate=config.muon_learning_rate,
    )
    ema = ExponentialMovingAverage(model, config.ema_decay) if config.ema_decay > 0 else None
    world_size = int(ddp_metadata.get("world_size", 1))
    local_effective_batch_size = config.batch_size * config.grad_accum_steps
    effective_batch_size = local_effective_batch_size * world_size
    effective_tokens_per_step = effective_batch_size * config.context_size

    losses: list[dict[str, float | int]] = []
    start = time.time()
    elapsed_offset = 0.0
    last_log_wall_time = start
    last_log_step = 0
    last_loss = float("nan")
    final_step = 0
    stop_reason = "max_steps"
    best_metric = float("inf")
    best_checkpoint: dict[str, float | int | str] | None = None
    best_checkpoint_dir = out_dir / "best_checkpoint"
    evals_without_improvement = 0
    start_step = 1
    rollback_events: list[dict[str, float | int | None]] = []
    rollback_lr_scale = 1.0
    loss_spike_baseline: float | None = None
    rollback_state: dict | None = None
    rollback_snapshot_step = 0
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
        last_log_step = final_step
        losses = list(resume_state.get("losses", []))
        if losses:
            last_loss = float(losses[-1].get("train_loss", last_loss))
        best_metric = float(resume_state.get("best_metric", best_metric))
        best_checkpoint = resume_state.get("best_checkpoint")
        evals_without_improvement = int(resume_state.get("evals_without_improvement", 0))
        elapsed_offset = float(resume_state.get("elapsed_sec", 0.0))
        rollback_events = list(resume_state.get("rollback_events", []))
        rollback_lr_scale = float(resume_state.get("rollback_lr_scale", rollback_lr_scale))
        raw_baseline = resume_state.get("loss_spike_baseline")
        loss_spike_baseline = float(raw_baseline) if raw_baseline is not None else None
    if loss_spike_baseline is None and math.isfinite(last_loss) and last_loss > 0:
        loss_spike_baseline = last_loss

    train_batches = DeviceBatchPrefetcher(train_batcher, device)
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
        ) * rollback_lr_scale
        weight_decay = weight_decay_for_step(
            base_weight_decay=config.weight_decay,
            step=step,
            max_steps=config.max_steps,
            decay=config.weight_decay_decay,
        )
        muon_momentum = muon_momentum_for_step(
            schedule=config.muon_momentum_schedule,
            step=step,
            max_steps=config.max_steps,
        )
        set_optimizer_lr(optimizer, learning_rate)
        set_optimizer_weight_decay(optimizer, weight_decay)
        set_muon_momentum(optimizer, muon_momentum)
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
        if ema is not None:
            ema.update(model)

        last_loss = sum(micro_losses) / len(micro_losses)
        if (
            config.loss_spike_rollback
            and rollback_state is not None
            and loss_spike_baseline is not None
            and _is_loss_spike(last_loss, loss_spike_baseline, config.loss_spike_threshold)
        ):
            _restore_rollback_state(rollback_state, model, optimizer, scaler, ema)
            rollback_lr_scale = max(
                config.loss_spike_min_lr_scale,
                rollback_lr_scale * config.loss_spike_lr_decay,
            )
            spike_ratio = (
                last_loss / loss_spike_baseline
                if math.isfinite(last_loss) and loss_spike_baseline > 0
                else None
            )
            rollback_events.append({
                "step": step,
                "train_loss": last_loss,
                "baseline_loss": loss_spike_baseline,
                "spike_ratio": spike_ratio,
                "lr_scale_after": rollback_lr_scale,
                "restored_snapshot_step": rollback_snapshot_step,
            })
            if main_process:
                print(
                    f"loss spike rollback at step {step}: "
                    f"train {last_loss:.4f} vs baseline {loss_spike_baseline:.4f}; "
                    f"lr scale -> {rollback_lr_scale:.3f}"
                )
            last_loss = loss_spike_baseline
            continue
        if math.isfinite(last_loss) and last_loss > 0:
            loss_spike_baseline = _update_loss_spike_baseline(loss_spike_baseline, last_loss)
            if (
                config.loss_spike_rollback
                and (
                    rollback_state is None
                    or step - rollback_snapshot_step >= config.loss_spike_snapshot_every
                )
            ):
                rollback_state = _capture_rollback_state(model, optimizer, scaler, ema)
                rollback_snapshot_step = step
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
            now = time.time()
            elapsed = elapsed_offset + now - start
            throughput = _interval_throughput(
                step=step,
                last_log_step=last_log_step,
                now=now,
                last_log_wall_time=last_log_wall_time,
                tokens_per_step=effective_tokens_per_step,
            )
            last_log_wall_time = now
            last_log_step = step
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
                "weight_decay": weight_decay,
                **({"muon_momentum": muon_momentum} if muon_momentum is not None else {}),
                "grad_norm": grad_norm,
                "grad_accum_steps": config.grad_accum_steps,
                "local_effective_batch_size": local_effective_batch_size,
                "effective_batch_size": effective_batch_size,
                "effective_tokens_per_step": effective_tokens_per_step,
                **throughput,
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
                if main_process:
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
            barrier_if_distributed(ddp_metadata)
            if main_process:
                print(
                    f"step {step:04d}/{config.max_steps:04d} | "
                    f"train {last_loss:.4f} | val {val_loss:.4f} | "
                    f"val_bpb {_format_optional(val_bpb)} | "
                    f"{_format_rate(throughput['tokens_per_sec'])} tok/s | {elapsed:.1f}s"
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
                        batcher=train_batches,
                        device=device,
                        training_fingerprint=training_fingerprint,
                        extra_state={
                            "rollback_events": rollback_events,
                            "rollback_lr_scale": rollback_lr_scale,
                            "loss_spike_baseline": loss_spike_baseline,
                        },
                    ),
                )
                write_checkpoint_progress(
                    out_dir / "resume_checkpoint",
                    stage="base",
                    step=final_step,
                    max_steps=config.max_steps,
                    train_loss=last_loss,
                    losses=losses,
                    best_checkpoint=best_checkpoint,
                    stop_reason=stop_reason,
                    resume_from=config.resume_from,
                )
            barrier_if_distributed(ddp_metadata)
            if (
                config.early_stop_patience > 0
                and evals_without_improvement >= config.early_stop_patience
            ):
                stop_reason = "early_stop"
                if main_process:
                    print(
                        f"early stop: validation did not improve for "
                        f"{config.early_stop_patience} evals"
                    )
                break
        if config.max_minutes is not None and elapsed_offset + time.time() - start >= config.max_minutes * 60:
            stop_reason = "max_minutes"
            if main_process:
                print(f"time stop: reached {config.max_minutes:.2f} minute budget")
            break

    checkpoint_dir = out_dir / "checkpoint"
    ema_checkpoint_dir = out_dir / "ema_checkpoint"
    elapsed_final = elapsed_offset + time.time() - start
    if main_process:
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
                batcher=train_batches,
                device=device,
                training_fingerprint=training_fingerprint,
                extra_state={
                    "rollback_events": rollback_events,
                    "rollback_lr_scale": rollback_lr_scale,
                    "loss_spike_baseline": loss_spike_baseline,
                },
            ),
        )
    barrier_if_distributed(ddp_metadata)
    if ema is not None and main_process:
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
    barrier_if_distributed(ddp_metadata)
    if best_checkpoint is None:
        if main_process:
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
        barrier_if_distributed(ddp_metadata)
        best_checkpoint = {
            "path": str(best_checkpoint_dir),
            "step": final_step,
            "train_loss": last_loss,
            "val_loss": losses[-1]["val_loss"] if losses else None,
            "val_bpb": losses[-1].get("val_bpb") if losses else None,
            "weights": "ema" if ema is not None else "raw",
        }

    sample = ""
    canary_probe = ""
    memorization_report = {}
    if main_process:
        sample = _generate_sample(
            model,
            tokenizer,
            device,
            config.sample_tokens,
            config.seed,
            precision_runtime=precision_runtime,
        )
        if split.canary_values:
            canary_probe = _generate_sample(
                model,
                tokenizer,
                device,
                min(config.sample_tokens, 80),
                config.seed + 17,
                prompt_text="Memorization canary phrase:",
                precision_runtime=precision_runtime,
            )
        memorization_text = f"{sample}\n{canary_probe}" if canary_probe else sample
        memorization_report = memorization_diagnostics(memorization_text, split.train_text, split.val_text)

    report = {
        "config": {
            **config.__dict__,
            "requested_device": config.device,
            "device": device.type,
            "local_effective_batch_size": local_effective_batch_size,
            "effective_batch_size": effective_batch_size,
            "effective_tokens_per_step": effective_tokens_per_step,
            "optimizer_metadata": optimizer.metadata,
            "precision_runtime": precision_runtime.to_dict(),
            "matmul_precision_runtime": matmul_precision_runtime,
            "torch_compile_metadata": compile_metadata,
            "ddp_metadata": ddp_metadata,
            "batcher_metadata": batcher_metadata,
            "artifacts_written": main_process,
            "training_fingerprint": training_fingerprint,
        },
        "dataset": {
            **split.stats,
        },
        "coverage": _coverage_report(split.stats, config, final_step, world_size=world_size),
        "model": {
            "config": model_config.to_dict(),
            "num_parameters": model.num_parameters(),
        },
        "losses": losses,
        "loss_diagnostics": loss_diagnostics(losses),
        "optimization_stability": optimization_stability(losses, config.grad_clip),
        "throughput": _throughput_summary(losses),
        "rollback_events": rollback_events,
        "memorization": memorization_report,
        "sample": sample,
        "canary_probe": canary_probe,
        "checkpoint": str(checkpoint_dir),
        "resume_checkpoint": str(out_dir / "resume_checkpoint"),
        "ema_checkpoint": str(ema_checkpoint_dir) if ema is not None else None,
        "best_checkpoint": best_checkpoint,
        "stop_reason": stop_reason,
    }
    if main_process:
        (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "report.md").write_text(training_report_markdown(report), encoding="utf-8")
        (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
        if canary_probe:
            (out_dir / "canary_probe.txt").write_text(canary_probe, encoding="utf-8")
    barrier_if_distributed(ddp_metadata)
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
    precision_runtime=None,
) -> str:
    model.eval()
    if prompt_text:
        prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
    else:
        prompt_ids = [tokenizer.bos_id]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with autocast_context(precision_runtime) if precision_runtime is not None else nullcontext():
        generated = model.generate(
            prompt,
            max_new_tokens=sample_tokens,
            temperature=0.8,
            top_k=20,
            seed=seed,
            eos_id=tokenizer.eos_id,
        )
    return tokenizer.decode(generated[0].tolist())


def _coverage_report(
    dataset_stats: dict,
    config: TrainConfig,
    actual_steps: int,
    *,
    world_size: int = 1,
) -> dict:
    local_tokens_per_step = config.batch_size * config.grad_accum_steps * config.context_size
    tokens_per_step = local_tokens_per_step * world_size
    tokens_seen = actual_steps * tokens_per_step
    train_tokens = dataset_stats.get("train_tokens") or dataset_stats.get("num_tokens")
    total_tokens = dataset_stats.get("num_tokens")
    report = {
        "actual_steps": actual_steps,
        "planned_steps": config.max_steps,
        "micro_batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "world_size": world_size,
        "local_tokens_per_step_estimate": local_tokens_per_step,
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


def _format_rate(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def _interval_throughput(
    *,
    step: int,
    last_log_step: int,
    now: float,
    last_log_wall_time: float,
    tokens_per_step: int,
) -> dict[str, float]:
    interval_steps = max(1, step - last_log_step)
    interval_sec = max(1e-9, now - last_log_wall_time)
    interval_tokens = interval_steps * tokens_per_step
    return {
        "interval_steps": interval_steps,
        "interval_sec": interval_sec,
        "step_time_sec": interval_sec / interval_steps,
        "steps_per_sec": interval_steps / interval_sec,
        "tokens_per_sec": interval_tokens / interval_sec,
    }


def _throughput_summary(losses: list[dict]) -> dict[str, float | None]:
    rows = [row for row in losses if row.get("tokens_per_sec") is not None]
    if not rows:
        return {
            "avg_tokens_per_sec": None,
            "final_tokens_per_sec": None,
            "avg_step_time_sec": None,
            "final_step_time_sec": None,
        }
    return {
        "avg_tokens_per_sec": sum(float(row["tokens_per_sec"]) for row in rows) / len(rows),
        "final_tokens_per_sec": float(rows[-1]["tokens_per_sec"]),
        "avg_step_time_sec": sum(float(row["step_time_sec"]) for row in rows) / len(rows),
        "final_step_time_sec": float(rows[-1]["step_time_sec"]),
    }


def _is_loss_spike(loss: float, baseline: float, threshold: float) -> bool:
    if not math.isfinite(baseline) or baseline <= 0:
        return False
    if not math.isfinite(loss):
        return True
    return loss > baseline * threshold


def _update_loss_spike_baseline(
    baseline: float | None,
    loss: float,
    beta: float = 0.95,
) -> float:
    if baseline is None or not math.isfinite(baseline) or baseline <= 0:
        return loss
    return beta * baseline + (1.0 - beta) * loss


def _capture_rollback_state(model, optimizer, scaler, ema) -> dict:
    return {
        "model": _clone_state(model.state_dict()),
        "optimizer": _clone_state(optimizer.state_dict()),
        "scaler": _clone_state(scaler.state_dict() if hasattr(scaler, "state_dict") else {}),
        "ema": _clone_state(ema.state_dict()) if ema is not None else None,
    }


def _restore_rollback_state(state: dict, model, optimizer, scaler, ema) -> None:
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scaler") and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(state["scaler"])
    if state.get("ema") is not None:
        if ema is None:
            raise ValueError("rollback state contains EMA but this run has EMA disabled")
        ema.load_state_dict(state["ema"])


def _clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value
