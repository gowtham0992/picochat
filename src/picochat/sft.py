"""Supervised chat fine-tuning for Picochat."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch

from picochat.chat import render_chat_prompt
from picochat.checkpoint import load_checkpoint, save_checkpoint
from picochat.optim import learning_rate_for_step, maybe_clip_grad_norm, set_optimizer_lr, validate_optim_controls
from picochat.report import loss_diagnostics, sft_report_markdown
from picochat.tokenizer import Tokenizer, load_tokenizer, token_byte_lengths
from picochat.train import evaluate_metrics


SFT_SAMPLING_MODES = ("uniform", "category_balanced")


@dataclass(frozen=True)
class ChatExample:
    user: str
    assistant: str
    category: str = "chat"
    group: str | None = None


@dataclass(frozen=True)
class ChatDatasetStats:
    num_examples: int
    context_size: int
    supervised_tokens: int
    truncated_examples: int
    num_groups: int
    category_counts: dict[str, int]


@dataclass(frozen=True)
class ChatSplit:
    train: torch.utils.data.Subset
    val: torch.utils.data.Subset
    method: str
    num_groups: int
    train_groups: int
    val_groups: int


@dataclass(frozen=True)
class SFTConfig:
    input_path: str
    tokenizer_path: str
    checkpoint_path: str
    out_dir: str
    batch_size: int = 8
    max_steps: int = 100
    learning_rate: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    log_every: int = 10
    val_fraction: float = 0.2
    eval_batches: int = 10
    sample_prompt: str = "What is Picochat?"
    sample_tokens: int = 120
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    max_minutes: float | None = None
    lr_warmup_steps: int = 0
    lr_decay: str = "none"
    min_lr_ratio: float = 1.0
    grad_clip: float = 0.0
    sampling: str = "uniform"


class ChatSFTDataset(torch.utils.data.Dataset):
    """Fixed-length chat examples with loss only on assistant tokens."""

    def __init__(
        self,
        examples: list[ChatExample],
        tokenizer: Tokenizer,
        context_size: int,
    ):
        if context_size < 2:
            raise ValueError("context_size must be at least 2")

        self.context_size = context_size
        self.rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.groups: list[str | None] = []
        self.categories: list[str] = []
        supervised_tokens = 0
        truncated_examples = 0

        for example in examples:
            prompt = render_chat_prompt([], example.user)
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            answer_ids = tokenizer.encode(f" {example.assistant}", add_eos=True)
            max_ids = context_size + 1

            if len(prompt_ids) + len(answer_ids) > max_ids:
                truncated_examples += 1
                answer_keep = min(len(answer_ids), max_ids)
                prompt_keep = max_ids - answer_keep
                prompt_ids = prompt_ids[-prompt_keep:] if prompt_keep > 0 else []
                answer_ids = answer_ids[:answer_keep]

            full_ids = prompt_ids + answer_ids
            if len(full_ids) < 2:
                continue

            x = full_ids[:-1]
            labels = full_ids[1:]
            prompt_label_count = max(0, len(prompt_ids) - 1)
            labels[:prompt_label_count] = [-100] * min(prompt_label_count, len(labels))

            supervised = sum(1 for token_id in labels if token_id != -100)
            if supervised == 0:
                continue

            pad_count = context_size - len(x)
            x = x + [tokenizer.pad_id] * pad_count
            labels = labels + [-100] * pad_count

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))
            self.groups.append(example.group)
            self.categories.append(example.category)
            supervised_tokens += supervised

        if not self.rows:
            raise ValueError("no usable chat examples fit inside the model context")

        explicit_groups = {group for group in self.groups if group is not None}
        self._stats = ChatDatasetStats(
            num_examples=len(self.rows),
            context_size=context_size,
            supervised_tokens=supervised_tokens,
            truncated_examples=truncated_examples,
            num_groups=len(explicit_groups),
            category_counts=dict(sorted(Counter(self.categories).items())),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        return self.rows[index]

    def stats(self) -> ChatDatasetStats:
        return self._stats

    def group_key(self, index: int) -> str | None:
        if index < 0 or index >= len(self.groups):
            raise IndexError(index)
        return self.groups[index]

    def category_key(self, index: int) -> str:
        if index < 0 or index >= len(self.categories):
            raise IndexError(index)
        return self.categories[index]


def load_chat_examples(path: str | Path) -> list[ChatExample]:
    """Load one-turn chat examples from JSONL."""
    examples: list[ChatExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        user = record.get("user")
        assistant = record.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise ValueError(f"line {line_number} must contain string user and assistant fields")
        category = record.get("category", "chat")
        if not isinstance(category, str):
            raise ValueError(f"line {line_number} category field must be a string when present")
        group = _optional_string(
            record.get("group", record.get("group_id", record.get("template"))),
            line_number,
            "group",
        )
        examples.append(ChatExample(user=user, assistant=assistant, category=category, group=group))
    if not examples:
        raise ValueError("chat dataset is empty")
    return examples


def _optional_string(value, line_number: int, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"line {line_number} {field} field must be a string when present")
    value = value.strip()
    return value or None


def split_chat_dataset(
    dataset: ChatSFTDataset,
    val_fraction: float,
    seed: int,
) -> ChatSplit:
    """Deterministically split chat examples into train and validation subsets."""
    if len(dataset) == 1:
        subset = torch.utils.data.Subset(dataset, [0])
        return ChatSplit(
            train=subset,
            val=subset,
            method="single_example",
            num_groups=0,
            train_groups=0,
            val_groups=0,
        )
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    grouped_split = _grouped_split_indices(dataset, val_fraction, seed)
    if grouped_split is not None:
        train_indices, val_indices, num_groups, train_groups, val_groups = grouped_split
        return ChatSplit(
            train=torch.utils.data.Subset(dataset, train_indices),
            val=torch.utils.data.Subset(dataset, val_indices),
            method="group",
            num_groups=num_groups,
            train_groups=train_groups,
            val_groups=val_groups,
        )

    train_indices, val_indices = _random_split_indices(len(dataset), val_fraction, seed)
    return ChatSplit(
        train=torch.utils.data.Subset(dataset, train_indices),
        val=torch.utils.data.Subset(dataset, val_indices),
        method="random",
        num_groups=0,
        train_groups=0,
        val_groups=0,
    )


def _random_split_indices(
    size: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    num_val = max(1, int(size * val_fraction))
    num_train = size - num_val
    if num_train < 1:
        num_train = 1
        num_val = size - 1

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    return indices[:num_train], indices[num_train: num_train + num_val]


def _grouped_split_indices(
    dataset: ChatSFTDataset,
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


def make_chat_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    sampling: str = "uniform",
) -> torch.utils.data.DataLoader:
    """Create a deterministic dataloader for small chat datasets."""
    if sampling not in SFT_SAMPLING_MODES:
        raise ValueError(f"unsupported SFT sampling mode: {sampling}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    if sampling == "category_balanced":
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=category_balanced_weights(dataset),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def category_balanced_weights(dataset) -> torch.Tensor:
    """Return per-row weights that give each SFT category equal probability."""
    categories = [_category_key(dataset, index) for index in range(len(dataset))]
    if not categories:
        raise ValueError("cannot sample from an empty chat dataset")
    counts = Counter(categories)
    return torch.tensor([1.0 / counts[category] for category in categories], dtype=torch.double)


def category_counts(dataset) -> dict[str, int]:
    """Count SFT categories in a dataset or subset."""
    counts = Counter(_category_key(dataset, index) for index in range(len(dataset)))
    return dict(sorted(counts.items()))


def _category_key(dataset, index: int) -> str:
    if isinstance(dataset, torch.utils.data.Subset):
        return _category_key(dataset.dataset, dataset.indices[index])
    if hasattr(dataset, "category_key"):
        return dataset.category_key(index)
    return "chat"


def train_sft(config: SFTConfig) -> dict:
    """Fine-tune a base checkpoint on small chat examples."""
    if config.sampling not in SFT_SAMPLING_MODES:
        raise ValueError(f"sampling must be one of: {', '.join(SFT_SAMPLING_MODES)}")
    validate_optim_controls(
        max_steps=config.max_steps,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay=config.lr_decay,
        min_lr_ratio=config.min_lr_ratio,
        grad_clip=config.grad_clip,
    )
    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.tokenizer_path)
    model, metadata = load_checkpoint(config.checkpoint_path, map_location=config.device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("tokenizer vocabulary size does not match checkpoint")

    dataset = ChatSFTDataset(
        load_chat_examples(config.input_path),
        tokenizer=tokenizer,
        context_size=model.config.context_size,
    )
    split = split_chat_dataset(dataset, config.val_fraction, config.seed)
    train_dataset = split.train
    val_dataset = split.val
    train_loader = make_chat_dataloader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        seed=config.seed,
        sampling=config.sampling,
    )
    train_eval_loader = make_chat_dataloader(train_dataset, config.batch_size, shuffle=False, seed=config.seed)
    val_loader = make_chat_dataloader(val_dataset, config.batch_size, shuffle=False, seed=config.seed)
    data_iter = iter(train_loader)

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    token_bytes = torch.tensor(token_byte_lengths(tokenizer), dtype=torch.long, device=device)

    losses: list[dict[str, float | int]] = []
    start = time.time()
    last_loss = float("nan")
    best_loss = float("inf")
    best_checkpoint: dict[str, float | int | str] | None = None
    best_checkpoint_dir = out_dir / "best_checkpoint"
    evals_without_improvement = 0
    final_step = 0
    stop_reason = "max_steps"

    model.train()
    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        assert loss is not None

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
        loss.backward()
        grad_norm = maybe_clip_grad_norm(model, config.grad_clip)
        optimizer.step()

        last_loss = float(loss.item())
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
            val_loss = float(val_metrics["loss"])
            elapsed = time.time() - start
            losses.append({
                "step": step,
                "train_loss": last_loss,
                "train_eval_loss": float(train_metrics["loss"]),
                "val_loss": val_loss,
                "train_bpb": train_metrics["bpb"],
                "val_bpb": val_metrics["bpb"],
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "elapsed_sec": elapsed,
            })
            if val_loss < best_loss - config.early_stop_min_delta:
                best_loss = val_loss
                evals_without_improvement = 0
                save_checkpoint(
                    best_checkpoint_dir,
                    model,
                    step=step,
                    train_loss=last_loss,
                    extra_metadata={
                        "checkpoint_kind": "best_validation",
                        "val_loss": val_loss,
                        "val_bpb": val_metrics["bpb"],
                    },
                )
                best_checkpoint = {
                    "path": str(best_checkpoint_dir),
                    "step": step,
                    "train_loss": last_loss,
                    "val_loss": val_loss,
                    "val_bpb": val_metrics["bpb"],
                }
            else:
                evals_without_improvement += 1
            print(
                f"sft step {step:04d}/{config.max_steps:04d} | "
                f"train {last_loss:.4f} | val {val_loss:.4f} | "
                f"val_bpb {_format_optional(val_metrics['bpb'])} | {elapsed:.1f}s"
            )
            if (
                config.early_stop_patience > 0
                and evals_without_improvement >= config.early_stop_patience
            ):
                stop_reason = "early_stop"
                print(
                    f"sft early stop: validation did not improve for "
                    f"{config.early_stop_patience} evals"
                )
                break
        if config.max_minutes is not None and time.time() - start >= config.max_minutes * 60:
            stop_reason = "max_minutes"
            print(f"sft time stop: reached {config.max_minutes:.2f} minute budget")
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

    model.eval()
    prompt_text = render_chat_prompt([], config.sample_prompt)
    prompt = torch.tensor(
        [tokenizer.encode(prompt_text, add_bos=True)],
        dtype=torch.long,
        device=device,
    )
    generated = model.generate(
        prompt,
        max_new_tokens=config.sample_tokens,
        temperature=0.8,
        top_k=20,
        seed=config.seed,
        eos_id=tokenizer.eos_id,
    )
    sample = tokenizer.decode(generated[0].tolist())

    report = {
        "config": config.__dict__,
        "base_checkpoint": {
            "path": config.checkpoint_path,
            "step": metadata.get("step"),
            "train_loss": metadata.get("train_loss"),
        },
        "dataset": {
            **dataset.stats().__dict__,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "split_method": split.method,
            "train_groups": split.train_groups,
            "val_groups": split.val_groups,
            "category_counts": dataset.stats().category_counts,
            "train_category_counts": category_counts(train_dataset),
            "val_category_counts": category_counts(val_dataset),
            "sampling": config.sampling,
        },
        "coverage": _coverage_report(len(train_dataset), len(dataset), config, final_step),
        "model": {
            "config": model.config.to_dict(),
            "num_parameters": model.num_parameters(),
        },
        "losses": losses,
        "loss_diagnostics": loss_diagnostics(losses),
        "sample": sample,
        "checkpoint": str(checkpoint_dir),
        "best_checkpoint": best_checkpoint,
        "stop_reason": stop_reason,
    }
    (out_dir / "sft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(sft_report_markdown(report), encoding="utf-8")
    (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
    return report


def _coverage_report(train_examples: int, total_examples: int, config: SFTConfig, actual_steps: int) -> dict:
    examples_seen = actual_steps * config.batch_size
    return {
        "actual_steps": actual_steps,
        "planned_steps": config.max_steps,
        "examples_per_step_estimate": config.batch_size,
        "planned_example_updates": config.max_steps * config.batch_size,
        "actual_example_updates": examples_seen,
        "train_examples": train_examples,
        "total_examples": total_examples,
        "estimated_train_epochs": _safe_ratio(examples_seen, train_examples),
        "estimated_dataset_passes": _safe_ratio(examples_seen, total_examples),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _format_optional(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"
