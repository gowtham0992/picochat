"""Direct Preference Optimization for post-SFT Picochat checkpoints."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from picochat.chat import render_chat_prompt
from picochat.checkpoint import load_checkpoint, save_checkpoint
from picochat.device import resolve_device
from picochat.optim import (
    create_optimizer,
    learning_rate_for_step,
    maybe_clip_grad_norm,
    set_optimizer_lr,
    set_optimizer_weight_decay,
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
from picochat.telemetry import TensorBoardLogger
from picochat.tokenizer import Tokenizer, load_tokenizer


@dataclass(frozen=True)
class PreferenceExample:
    user: str
    chosen: str
    rejected: str
    category: str = "preference"
    group: str | None = None


@dataclass(frozen=True)
class PreferenceDatasetStats:
    source_rows: int
    num_examples: int
    context_size: int
    prompt_tokens: int
    chosen_tokens: int
    rejected_tokens: int
    skipped_long_examples: int
    skipped_long_category_counts: dict[str, int]
    category_counts: dict[str, int]
    num_groups: int


@dataclass(frozen=True)
class PreferenceSplit:
    train: torch.utils.data.Subset
    val: torch.utils.data.Subset
    method: str
    num_groups: int
    train_groups: int
    val_groups: int


@dataclass(frozen=True)
class DPOConfig:
    input_path: str
    tokenizer_path: str
    checkpoint_path: str
    out_dir: str
    reference_checkpoint_path: str | None = None
    batch_size: int = 4
    max_steps: int = 100
    learning_rate: float = 5e-6
    beta: float = 0.1
    seed: int = 42
    device: str = "cpu"
    log_every: int = 10
    val_fraction: float = 0.2
    eval_batches: int = 10
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    max_minutes: float | None = None
    lr_warmup_steps: int = 0
    lr_decay: str = "none"
    min_lr_ratio: float = 1.0
    grad_clip: float = 0.0
    grad_accum_steps: int = 1
    weight_decay: float = 0.01
    weight_decay_decay: str = "none"
    precision: str = "float32"
    matmul_precision: str = "default"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    length_normalize: bool = False
    tensorboard_log_dir: str | None = None


class PreferenceDataset(torch.utils.data.Dataset):
    """Preference rows with one prompt, one chosen answer, and one rejected answer."""

    def __init__(
        self,
        examples: list[PreferenceExample],
        tokenizer: Tokenizer,
        context_size: int,
    ) -> None:
        if context_size < 2:
            raise ValueError("context_size must be at least 2")
        self.context_size = context_size
        self.rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.categories: list[str] = []
        self.groups: list[str | None] = []
        prompt_tokens = 0
        chosen_tokens = 0
        rejected_tokens = 0
        skipped_long_examples = 0
        skipped_long_categories: Counter[str] = Counter()

        for example in examples:
            row = _preference_row(example, tokenizer=tokenizer, context_size=context_size)
            if row is None:
                skipped_long_examples += 1
                skipped_long_categories[example.category] += 1
                continue
            chosen_x, chosen_labels, rejected_x, rejected_labels, token_counts = row
            self.rows.append((chosen_x, chosen_labels, rejected_x, rejected_labels))
            self.categories.append(example.category)
            self.groups.append(example.group)
            prompt_tokens += token_counts["prompt_tokens"]
            chosen_tokens += token_counts["chosen_tokens"]
            rejected_tokens += token_counts["rejected_tokens"]

        if not self.rows:
            raise ValueError(
                "no usable preference examples fit inside the model context; "
                "increase --context-size in the checkpoint or shorten preference rows"
            )

        explicit_groups = {group for group in self.groups if group is not None}
        self._stats = PreferenceDatasetStats(
            source_rows=len(examples),
            num_examples=len(self.rows),
            context_size=context_size,
            prompt_tokens=prompt_tokens,
            chosen_tokens=chosen_tokens,
            rejected_tokens=rejected_tokens,
            skipped_long_examples=skipped_long_examples,
            skipped_long_category_counts=dict(sorted(skipped_long_categories.items())),
            category_counts=dict(sorted(Counter(self.categories).items())),
            num_groups=len(explicit_groups),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        return self.rows[index]

    def stats(self) -> PreferenceDatasetStats:
        return self._stats

    def group_key(self, index: int) -> str | None:
        if index < 0 or index >= len(self.groups):
            raise IndexError(index)
        return self.groups[index]

    def category_key(self, index: int) -> str:
        if index < 0 or index >= len(self.categories):
            raise IndexError(index)
        return self.categories[index]


def load_preference_examples(path: str | Path) -> list[PreferenceExample]:
    """Load preference rows from JSONL.

    Accepted fields:
    - prompt/user
    - chosen/preferred/winner
    - rejected/dispreferred/loser
    """
    examples: list[PreferenceExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        user = _first_string(record, ("user", "prompt"))
        chosen = _first_string(record, ("chosen", "preferred", "winner"))
        rejected = _first_string(record, ("rejected", "dispreferred", "loser"))
        if user is None:
            raise ValueError(f"line {line_number} must contain string user or prompt field")
        if chosen is None:
            raise ValueError(f"line {line_number} must contain string chosen/preferred/winner field")
        if rejected is None:
            raise ValueError(f"line {line_number} must contain string rejected/dispreferred/loser field")
        if chosen.strip() == rejected.strip():
            raise ValueError(f"line {line_number} chosen and rejected answers must differ")
        category = record.get("category", "preference")
        if not isinstance(category, str):
            raise ValueError(f"line {line_number} category field must be a string when present")
        group = _optional_string(
            record.get("group", record.get("group_id", record.get("template"))),
            line_number,
            "group",
        )
        examples.append(PreferenceExample(
            user=user,
            chosen=chosen,
            rejected=rejected,
            category=category,
            group=group,
        ))
    if not examples:
        raise ValueError("preference dataset is empty")
    return examples


def dpo_batch_metrics(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    beta: float,
    length_normalize: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the DPO loss and diagnostics for one already-device batch."""
    chosen_x, chosen_labels, rejected_x, rejected_labels = batch
    policy_chosen_logp = sequence_logprob(
        policy_model,
        chosen_x,
        chosen_labels,
        length_normalize=length_normalize,
    )
    policy_rejected_logp = sequence_logprob(
        policy_model,
        rejected_x,
        rejected_labels,
        length_normalize=length_normalize,
    )
    with torch.no_grad():
        reference_chosen_logp = sequence_logprob(
            reference_model,
            chosen_x,
            chosen_labels,
            length_normalize=length_normalize,
        )
        reference_rejected_logp = sequence_logprob(
            reference_model,
            rejected_x,
            rejected_labels,
            length_normalize=length_normalize,
        )

    policy_logratio = policy_chosen_logp - policy_rejected_logp
    reference_logratio = reference_chosen_logp - reference_rejected_logp
    logits = beta * (policy_logratio - reference_logratio)
    loss = -F.logsigmoid(logits).mean()
    return {
        "loss": loss,
        "accuracy": (logits > 0).float().mean(),
        "reward_margin": (policy_logratio - reference_logratio).mean(),
        "policy_chosen_logp": policy_chosen_logp.mean(),
        "policy_rejected_logp": policy_rejected_logp.mean(),
        "reference_chosen_logp": reference_chosen_logp.mean(),
        "reference_rejected_logp": reference_rejected_logp.mean(),
    }


def sequence_logprob(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Return per-row log probability over non-masked target labels."""
    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sums = (token_logps * mask).sum(dim=-1)
    if not length_normalize:
        return sums
    lengths = mask.sum(dim=-1).clamp_min(1)
    return sums / lengths


def evaluate_dpo(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    beta: float,
    max_batches: int,
    precision_runtime,
    length_normalize: bool = False,
) -> dict[str, float]:
    """Evaluate DPO loss and preference accuracy on a small loader."""
    was_training = policy_model.training
    policy_model.eval()
    reference_model.eval()
    totals: Counter[str] = Counter()
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            device_batch = _batch_to_device(batch, device)
            with autocast_context(precision_runtime):
                metrics = dpo_batch_metrics(
                    policy_model,
                    reference_model,
                    device_batch,
                    beta=beta,
                    length_normalize=length_normalize,
                )
            batch_size = int(device_batch[0].size(0))
            count += batch_size
            for key, value in metrics.items():
                totals[key] += float(value.item()) * batch_size
    if was_training:
        policy_model.train()
    if count == 0:
        raise ValueError("DPO eval loader produced no batches")
    return {key: value / count for key, value in totals.items()}


def train_dpo(config: DPOConfig) -> dict:
    """Apply DPO to an SFT checkpoint using prompt/chosen/rejected pairs."""
    if config.beta <= 0:
        raise ValueError("beta must be positive")
    validate_optim_controls(
        max_steps=config.max_steps,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay=config.lr_decay,
        min_lr_ratio=config.min_lr_ratio,
        grad_clip=config.grad_clip,
        grad_accum_steps=config.grad_accum_steps,
        optimizer_type="adamw",
        weight_decay=config.weight_decay,
        weight_decay_decay=config.weight_decay_decay,
    )
    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.tokenizer_path)
    device = resolve_device(config.device)
    policy_model, policy_metadata = load_checkpoint(config.checkpoint_path, map_location=device)
    reference_path = config.reference_checkpoint_path or config.checkpoint_path
    reference_model, reference_metadata = load_checkpoint(reference_path, map_location=device)
    if policy_model.config.to_dict() != reference_model.config.to_dict():
        raise ValueError("policy and reference checkpoints must use the same model config")
    if policy_model.config.vocab_size != len(tokenizer):
        raise ValueError("tokenizer vocabulary size does not match checkpoint")

    dataset = PreferenceDataset(
        load_preference_examples(config.input_path),
        tokenizer=tokenizer,
        context_size=policy_model.config.context_size,
    )
    split = split_preference_dataset(dataset, config.val_fraction, config.seed)
    train_loader = make_preference_dataloader(split.train, config.batch_size, shuffle=True, seed=config.seed)
    val_loader = make_preference_dataloader(split.val, config.batch_size, shuffle=False, seed=config.seed)

    policy_model = policy_model.to(device)
    reference_model = reference_model.to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    matmul_precision_runtime = configure_float32_matmul_precision(config.matmul_precision)
    precision_runtime = resolve_precision(config.precision, device)
    train_model, compile_metadata = maybe_compile_model(
        policy_model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    scaler = make_grad_scaler(precision_runtime)
    optimizer = create_optimizer(
        policy_model,
        optimizer_type="adamw",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    tensorboard = TensorBoardLogger(config.tensorboard_log_dir)

    losses: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_checkpoint: dict[str, float | int | str] | None = None
    evals_without_improvement = 0
    final_step = 0
    stop_reason = "max_steps"
    start = time.time()
    train_iter = _cycle(train_loader)

    policy_model.train()
    train_model.train()
    for step in range(1, config.max_steps + 1):
        learning_rate = learning_rate_for_step(
            base_learning_rate=config.learning_rate,
            step=step,
            max_steps=config.max_steps,
            warmup_steps=config.lr_warmup_steps,
            decay=config.lr_decay,
            min_lr_ratio=config.min_lr_ratio,
        )
        weight_decay = weight_decay_for_step(
            base_weight_decay=config.weight_decay,
            step=step,
            max_steps=config.max_steps,
            decay=config.weight_decay_decay,
        )
        set_optimizer_lr(optimizer, learning_rate)
        set_optimizer_weight_decay(optimizer, weight_decay)
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        micro_accuracies: list[float] = []
        micro_margins: list[float] = []
        for _ in range(config.grad_accum_steps):
            batch = _batch_to_device(next(train_iter), device)
            with autocast_context(precision_runtime):
                metrics = dpo_batch_metrics(
                    train_model,
                    reference_model,
                    batch,
                    beta=config.beta,
                    length_normalize=config.length_normalize,
                )
                scaled_loss = metrics["loss"] / config.grad_accum_steps
            micro_losses.append(float(metrics["loss"].item()))
            micro_accuracies.append(float(metrics["accuracy"].item()))
            micro_margins.append(float(metrics["reward_margin"].item()))
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = maybe_clip_grad_norm(policy_model, config.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        final_step = step
        train_loss = sum(micro_losses) / len(micro_losses)
        train_accuracy = sum(micro_accuracies) / len(micro_accuracies)
        train_margin = sum(micro_margins) / len(micro_margins)
        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            val_metrics = evaluate_dpo(
                train_model,
                reference_model,
                val_loader,
                device,
                beta=config.beta,
                max_batches=config.eval_batches,
                precision_runtime=precision_runtime,
                length_normalize=config.length_normalize,
            )
            elapsed = time.time() - start
            val_loss = float(val_metrics["loss"])
            losses.append({
                "step": step,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_reward_margin": train_margin,
                "val_loss": val_loss,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_reward_margin": float(val_metrics["reward_margin"]),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "grad_norm": grad_norm,
                "elapsed_sec": elapsed,
            })
            latest = losses[-1]
            tensorboard.scalars({
                "loss/train": latest["train_loss"],
                "loss/val": latest["val_loss"],
                "preference/train_accuracy": latest["train_accuracy"],
                "preference/val_accuracy": latest["val_accuracy"],
                "preference/train_reward_margin": latest["train_reward_margin"],
                "preference/val_reward_margin": latest["val_reward_margin"],
                "optim/learning_rate": latest["learning_rate"],
                "optim/weight_decay": latest["weight_decay"],
                "optim/grad_norm": latest["grad_norm"],
            }, step)
            print(
                f"dpo step {step:04d}/{config.max_steps:04d} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"val_acc {float(val_metrics['accuracy']) * 100:.1f}% | {elapsed:.1f}s"
            )
            if val_loss < best_loss - config.early_stop_min_delta:
                best_loss = val_loss
                evals_without_improvement = 0
                save_checkpoint(
                    out_dir / "best_checkpoint",
                    policy_model,
                    step=step,
                    train_loss=train_loss,
                    extra_metadata={
                        "checkpoint_kind": "best_dpo_validation",
                        "val_loss": val_loss,
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "reference_checkpoint": reference_path,
                    },
                )
                best_checkpoint = {
                    "path": str(out_dir / "best_checkpoint"),
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": float(val_metrics["accuracy"]),
                }
            else:
                evals_without_improvement += 1
            if (
                config.early_stop_patience > 0
                and evals_without_improvement >= config.early_stop_patience
            ):
                stop_reason = "early_stop"
                print(
                    f"dpo early stop: validation did not improve for "
                    f"{config.early_stop_patience} evals"
                )
                break
        if config.max_minutes is not None and time.time() - start >= config.max_minutes * 60:
            stop_reason = "max_minutes"
            print(f"dpo time stop: reached {config.max_minutes:.2f} minute budget")
            break

    checkpoint_dir = out_dir / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        policy_model,
        step=final_step,
        train_loss=losses[-1]["train_loss"] if losses else float("nan"),
        extra_metadata={
            "checkpoint_kind": "final_dpo",
            "stop_reason": stop_reason,
            "reference_checkpoint": reference_path,
            "best_checkpoint": best_checkpoint,
        },
    )
    if best_checkpoint is None:
        save_checkpoint(
            out_dir / "best_checkpoint",
            policy_model,
            step=final_step,
            train_loss=losses[-1]["train_loss"] if losses else float("nan"),
            extra_metadata={
                "checkpoint_kind": "best_dpo_fallback",
                "reference_checkpoint": reference_path,
            },
        )
        best_checkpoint = {
            "path": str(out_dir / "best_checkpoint"),
            "step": final_step,
            "train_loss": losses[-1]["train_loss"] if losses else None,
            "val_loss": losses[-1]["val_loss"] if losses else None,
            "val_accuracy": losses[-1]["val_accuracy"] if losses else None,
        }

    report = {
        "config": {
            **config.__dict__,
            "reference_checkpoint_path": reference_path,
            "device": device.type,
            "precision_runtime": precision_runtime.to_dict(),
            "matmul_precision_runtime": matmul_precision_runtime,
            "torch_compile_metadata": compile_metadata,
            "optimizer_metadata": optimizer.metadata,
        },
        "policy_checkpoint": {
            "path": config.checkpoint_path,
            "step": policy_metadata.get("step"),
            "train_loss": policy_metadata.get("train_loss"),
        },
        "reference_checkpoint": {
            "path": reference_path,
            "step": reference_metadata.get("step"),
            "train_loss": reference_metadata.get("train_loss"),
        },
        "dataset": {
            **dataset.stats().__dict__,
            "train_examples": len(split.train),
            "val_examples": len(split.val),
            "split_method": split.method,
            "train_groups": split.train_groups,
            "val_groups": split.val_groups,
            "train_category_counts": category_counts(split.train),
            "val_category_counts": category_counts(split.val),
        },
        "losses": losses,
        "checkpoint": str(checkpoint_dir),
        "best_checkpoint": best_checkpoint,
        "stop_reason": stop_reason,
    }
    (out_dir / "dpo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(dpo_report_markdown(report), encoding="utf-8")
    tensorboard.close()
    return report


def split_preference_dataset(
    dataset: PreferenceDataset,
    val_fraction: float,
    seed: int,
) -> PreferenceSplit:
    if len(dataset) == 1:
        subset = torch.utils.data.Subset(dataset, [0])
        return PreferenceSplit(
            train=subset,
            val=subset,
            method="single_example",
            num_groups=0,
            train_groups=0,
            val_groups=0,
        )
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    grouped = _grouped_split_indices(dataset, val_fraction, seed)
    if grouped is not None:
        train_indices, val_indices, num_groups, train_groups, val_groups = grouped
        return PreferenceSplit(
            train=torch.utils.data.Subset(dataset, train_indices),
            val=torch.utils.data.Subset(dataset, val_indices),
            method="group",
            num_groups=num_groups,
            train_groups=train_groups,
            val_groups=val_groups,
        )
    train_indices, val_indices = _random_split_indices(len(dataset), val_fraction, seed)
    return PreferenceSplit(
        train=torch.utils.data.Subset(dataset, train_indices),
        val=torch.utils.data.Subset(dataset, val_indices),
        method="random",
        num_groups=0,
        train_groups=0,
        val_groups=0,
    )


def make_preference_dataloader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def category_counts(dataset) -> dict[str, int]:
    counts = Counter(_category_key(dataset, index) for index in range(len(dataset)))
    return dict(sorted(counts.items()))


def dpo_report_markdown(report: dict) -> str:
    losses = report.get("losses", [])
    latest = losses[-1] if losses else {}
    dataset = report.get("dataset", {})
    lines = [
        "# Picochat DPO Report",
        "",
        f"- Checkpoint: `{report.get('checkpoint')}`",
        f"- Best checkpoint: `{(report.get('best_checkpoint') or {}).get('path')}`",
        f"- Stop reason: `{report.get('stop_reason')}`",
        f"- Train examples: `{dataset.get('train_examples')}`",
        f"- Validation examples: `{dataset.get('val_examples')}`",
        f"- Split method: `{dataset.get('split_method')}`",
        "",
        "## Latest Metrics",
        "",
        f"- Step: `{latest.get('step')}`",
        f"- Train loss: `{_format_optional(latest.get('train_loss'))}`",
        f"- Validation loss: `{_format_optional(latest.get('val_loss'))}`",
        f"- Validation preference accuracy: `{_format_percent(latest.get('val_accuracy'))}`",
        f"- Validation reward margin: `{_format_optional(latest.get('val_reward_margin'))}`",
        "",
        "DPO should be used after SFT with human or carefully curated preference pairs. "
        "It improves preference alignment; it does not replace base pretraining or evidence-based release gates.",
        "",
    ]
    return "\n".join(lines)


def _preference_row(
    example: PreferenceExample,
    *,
    tokenizer: Tokenizer,
    context_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None:
    prompt = render_chat_prompt([], example.user)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    chosen_answer_ids = tokenizer.encode(example.chosen, add_eos=True)
    rejected_answer_ids = tokenizer.encode(example.rejected, add_eos=True)
    max_ids = context_size + 1
    chosen_full = prompt_ids + chosen_answer_ids
    rejected_full = prompt_ids + rejected_answer_ids
    if len(chosen_full) > max_ids or len(rejected_full) > max_ids:
        return None
    chosen_x, chosen_labels = _row_from_full_ids(
        chosen_full,
        prompt_len=len(prompt_ids),
        pad_id=tokenizer.pad_id,
        context_size=context_size,
    )
    rejected_x, rejected_labels = _row_from_full_ids(
        rejected_full,
        prompt_len=len(prompt_ids),
        pad_id=tokenizer.pad_id,
        context_size=context_size,
    )
    return (
        chosen_x,
        chosen_labels,
        rejected_x,
        rejected_labels,
        {
            "prompt_tokens": len(prompt_ids),
            "chosen_tokens": int((chosen_labels != -100).sum().item()),
            "rejected_tokens": int((rejected_labels != -100).sum().item()),
        },
    )


def _row_from_full_ids(
    full_ids: list[int],
    *,
    prompt_len: int,
    pad_id: int,
    context_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(full_ids) < 2:
        raise ValueError("preference sequence must contain at least two tokens")
    x = full_ids[:-1]
    labels = [
        token_id if target_position >= prompt_len else -100
        for target_position, token_id in enumerate(full_ids[1:], start=1)
    ]
    if not any(label != -100 for label in labels):
        raise ValueError("preference sequence has no supervised answer tokens")
    pad_count = context_size - len(x)
    x = x + [pad_id] * pad_count
    labels = labels + [-100] * pad_count
    return torch.tensor(x, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _first_string(record: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str):
            return value
        if value is not None:
            raise ValueError(f"{field} field must be a string when present")
    return None


def _optional_string(value, line_number: int, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"line {line_number} {field} field must be a string when present")
    value = value.strip()
    return value or None


def _grouped_split_indices(
    dataset: PreferenceDataset,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], int, int, int] | None:
    groups: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        group = dataset.group_key(index)
        if group is None:
            return None
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        return None
    target_val = max(1, int(len(dataset) * val_fraction))
    generator = torch.Generator()
    generator.manual_seed(seed)
    group_names = list(groups)
    order = torch.randperm(len(group_names), generator=generator).tolist()
    val_groups: set[str] = set()
    val_indices: list[int] = []
    for position in order:
        group = group_names[position]
        if len(val_groups) >= len(groups) - 1:
            break
        val_groups.add(group)
        val_indices.extend(groups[group])
        if len(val_indices) >= target_val:
            break
    train_indices = [
        index
        for group, indices in groups.items()
        if group not in val_groups
        for index in indices
    ]
    if not train_indices or not val_indices:
        return None
    val_indices.sort()
    train_indices.sort()
    return (
        train_indices,
        val_indices,
        len(groups),
        len(groups) - len(val_groups),
        len(val_groups),
    )


def _random_split_indices(size: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    num_val = max(1, int(size * val_fraction))
    num_train = size - num_val
    if num_train < 1:
        num_train = 1
        num_val = size - 1
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    return indices[:num_train], indices[num_train: num_train + num_val]


def _category_key(dataset, index: int) -> str:
    if isinstance(dataset, torch.utils.data.Subset):
        return _category_key(dataset.dataset, dataset.indices[index])
    if hasattr(dataset, "category_key"):
        return dataset.category_key(index)
    return "preference"


def _batch_to_device(batch, device: torch.device):
    return tuple(tensor.to(device) for tensor in batch)


def _cycle(loader: torch.utils.data.DataLoader):
    while True:
        for batch in loader:
            yield batch


def _format_optional(value) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _format_percent(value) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100:.2f}%"
