"""Supervised chat fine-tuning for Picochat."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from picochat.batching import DeviceBatchPrefetcher, make_resumable_batcher
from picochat.chat import render_chat_prompt
from picochat.checkpoint import load_checkpoint, load_training_state, save_checkpoint
from picochat.device import resolve_device
from picochat.distributed import (
    barrier_if_distributed,
    ddp_env_metadata,
    is_main_process,
    mean_scalar_if_distributed,
    no_sync_if_distributed,
    prepare_ddp_model,
)
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
from picochat.report import loss_diagnostics, optimization_stability, sft_report_markdown
from picochat.resume import (
    file_sha256,
    make_training_fingerprint,
    make_training_state,
    restore_training_state,
    validate_training_fingerprint,
)
from picochat.tokenizer import Tokenizer, load_tokenizer, token_byte_lengths
from picochat.train import (
    _capture_rollback_state,
    _format_rate,
    _interval_throughput,
    _is_loss_spike,
    _restore_rollback_state,
    _throughput_summary,
    _update_loss_spike_baseline,
    evaluate_metrics,
)


SFT_SAMPLING_MODES = ("uniform", "category_sqrt", "category_balanced")
SFT_PACKING_MODES = ("separate", "bos_bestfit")


@dataclass(frozen=True)
class ChatExample:
    user: str
    assistant: str
    category: str = "chat"
    group: str | None = None


@dataclass(frozen=True)
class TokenizedChatExample:
    full_ids: list[int]
    prompt_len: int
    category: str
    group: str | None = None


@dataclass(frozen=True)
class ChatDatasetStats:
    source_rows: int
    num_examples: int
    context_size: int
    supervised_tokens: int
    masked_prompt_tokens: int
    truncated_examples: int
    skipped_long_examples: int
    skipped_long_category_counts: dict[str, int]
    num_groups: int
    category_counts: dict[str, int]
    packing: str = "separate"
    source_examples: int = 0
    num_sequences: int = 0
    packed_sequences: int = 0
    packed_tokens: int = 0
    padded_tokens: int = 0
    packing_efficiency: float = 0.0
    average_examples_per_sequence: float = 0.0
    mixed_category_sequences: int = 0


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
    grad_accum_steps: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    weight_decay_decay: str = "none"
    muon_learning_rate: float = 0.02
    muon_momentum_schedule: str = "none"
    ema_decay: float = 0.0
    packing: str = "separate"
    precision: str = "float32"
    matmul_precision: str = "default"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    resume_from: str | None = None
    ddp: bool = False
    loss_spike_rollback: bool = False
    loss_spike_threshold: float = 2.5
    loss_spike_lr_decay: float = 0.5
    loss_spike_min_lr_scale: float = 0.1
    loss_spike_snapshot_every: int = 10


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
        source_rows = len(examples)
        self.examples, skipped_long_examples, skipped_long_category_counts = _tokenize_chat_examples(
            examples,
            tokenizer=tokenizer,
            context_size=context_size,
        )
        self.rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.groups: list[str | None] = []
        self.categories: list[str] = []
        supervised_tokens = 0
        masked_prompt_tokens = 0
        padded_tokens = 0
        packed_tokens = 0

        for example in self.examples:
            x, labels, row_stats = _packed_row([example], tokenizer, context_size)
            self.rows.append((x, labels))
            self.groups.append(example.group)
            self.categories.append(example.category)
            supervised_tokens += row_stats["supervised_tokens"]
            masked_prompt_tokens += row_stats["masked_prompt_tokens"]
            padded_tokens += row_stats["padded_tokens"]
            packed_tokens += row_stats["packed_tokens"]

        if not self.rows:
            raise ValueError(
                "no usable chat examples fit inside the model context; "
                "increase --context-size or shorten the chat SFT rows"
            )

        explicit_groups = {group for group in self.groups if group is not None}
        self._stats = ChatDatasetStats(
            source_rows=source_rows,
            num_examples=len(self.examples),
            context_size=context_size,
            supervised_tokens=supervised_tokens,
            masked_prompt_tokens=masked_prompt_tokens,
            truncated_examples=0,
            skipped_long_examples=skipped_long_examples,
            skipped_long_category_counts=skipped_long_category_counts,
            num_groups=len(explicit_groups),
            category_counts=dict(sorted(Counter(self.categories).items())),
            packing="separate",
            source_examples=len(self.examples),
            num_sequences=len(self.rows),
            packed_sequences=len(self.rows),
            packed_tokens=packed_tokens,
            padded_tokens=padded_tokens,
            packing_efficiency=_safe_ratio(packed_tokens, len(self.rows) * context_size) or 0.0,
            average_examples_per_sequence=1.0,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        return self.rows[index]

    def stats(self) -> ChatDatasetStats:
        return self._stats

    def tokenized_example(self, index: int) -> TokenizedChatExample:
        if index < 0 or index >= len(self.examples):
            raise IndexError(index)
        return self.examples[index]

    def group_key(self, index: int) -> str | None:
        if index < 0 or index >= len(self.groups):
            raise IndexError(index)
        return self.groups[index]

    def category_key(self, index: int) -> str:
        if index < 0 or index >= len(self.categories):
            raise IndexError(index)
        return self.categories[index]


def _tokenize_chat_examples(
    examples: list[ChatExample],
    *,
    tokenizer: Tokenizer,
    context_size: int,
) -> tuple[list[TokenizedChatExample], int, dict[str, int]]:
    tokenized: list[TokenizedChatExample] = []
    skipped_long_examples = 0
    skipped_long_categories: Counter[str] = Counter()
    max_ids = context_size + 1
    for example in examples:
        prompt = render_chat_prompt([], example.user)
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        answer_ids = tokenizer.encode(example.assistant, add_eos=True)
        if len(prompt_ids) + len(answer_ids) > max_ids:
            skipped_long_examples += 1
            skipped_long_categories[example.category] += 1
            continue
        full_ids = prompt_ids + answer_ids
        if len(full_ids) < 2 or not answer_ids:
            continue
        tokenized.append(TokenizedChatExample(
            full_ids=full_ids,
            prompt_len=len(prompt_ids),
            category=example.category,
            group=example.group,
        ))
    return tokenized, skipped_long_examples, dict(sorted(skipped_long_categories.items()))


def _packed_row(
    examples: list[TokenizedChatExample],
    tokenizer: Tokenizer,
    context_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    packed_ids: list[int] = []
    supervised_target_positions: set[int] = set()
    masked_prompt_tokens = 0

    for example in examples:
        start = len(packed_ids)
        packed_ids.extend(example.full_ids)
        answer_start = start + example.prompt_len
        answer_stop = start + len(example.full_ids)
        supervised_target_positions.update(range(answer_start, answer_stop))
        masked_prompt_tokens += max(0, example.prompt_len - 1)
        if start > 0:
            masked_prompt_tokens += 1

    if len(packed_ids) > context_size + 1:
        raise ValueError("packed chat sequence exceeds context")
    if len(packed_ids) < 2:
        raise ValueError("packed chat sequence must contain at least two tokens")

    x = packed_ids[:-1]
    labels = [
        token_id if target_position in supervised_target_positions else -100
        for target_position, token_id in enumerate(packed_ids[1:], start=1)
    ]
    supervised_tokens = sum(1 for token_id in labels if token_id != -100)
    if supervised_tokens == 0:
        raise ValueError("packed chat sequence has no assistant tokens")

    pad_count = context_size - len(x)
    x = x + [tokenizer.pad_id] * pad_count
    labels = labels + [-100] * pad_count
    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        {
            "supervised_tokens": supervised_tokens,
            "masked_prompt_tokens": masked_prompt_tokens,
            "packed_tokens": context_size - pad_count,
            "padded_tokens": pad_count,
        },
    )


def _bestfit_pack_examples(
    examples: list[TokenizedChatExample],
    *,
    max_ids: int,
) -> list[list[TokenizedChatExample]]:
    packs: list[list[TokenizedChatExample]] = []
    used_ids: list[int] = []
    ordered = sorted(
        enumerate(examples),
        key=lambda item: (-len(item[1].full_ids), item[0]),
    )
    for _, example in ordered:
        length = len(example.full_ids)
        best_index = None
        best_remaining = max_ids + 1
        for index, used in enumerate(used_ids):
            remaining = max_ids - used
            after = remaining - length
            if after >= 0 and after < best_remaining:
                best_index = index
                best_remaining = after
        if best_index is None:
            packs.append([example])
            used_ids.append(length)
        else:
            packs[best_index].append(example)
            used_ids[best_index] += length
    return packs


def _bestfit_pack_examples_by_category(
    examples: list[TokenizedChatExample],
    *,
    max_ids: int,
) -> list[list[TokenizedChatExample]]:
    categories: dict[str, list[TokenizedChatExample]] = {}
    for example in examples:
        categories.setdefault(example.category, []).append(example)
    packs: list[list[TokenizedChatExample]] = []
    for category_examples in categories.values():
        packs.extend(_bestfit_pack_examples(category_examples, max_ids=max_ids))
    return packs


class PackedChatSFTDataset(torch.utils.data.Dataset):
    """Best-fit packed chat sequences with assistant-only loss masking."""

    def __init__(
        self,
        examples: list[TokenizedChatExample],
        tokenizer: Tokenizer,
        context_size: int,
    ):
        if context_size < 2:
            raise ValueError("context_size must be at least 2")
        if not examples:
            raise ValueError("cannot pack an empty chat dataset")

        self.context_size = context_size
        self.examples = list(examples)
        self.rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.groups: list[str | None] = []
        self.categories: list[str] = []
        self.sequence_source_counts: list[int] = []

        supervised_tokens = 0
        masked_prompt_tokens = 0
        padded_tokens = 0
        packed_tokens = 0
        mixed_category_sequences = 0

        for pack in _bestfit_pack_examples_by_category(self.examples, max_ids=context_size + 1):
            x, labels, row_stats = _packed_row(pack, tokenizer, context_size)
            categories = [example.category for example in pack]
            groups = {example.group for example in pack if example.group is not None}
            mixed_category = len(set(categories)) > 1

            self.rows.append((x, labels))
            self.groups.append(next(iter(groups)) if len(groups) == 1 else None)
            self.categories.append(categories[0] if not mixed_category else "mixed")
            self.sequence_source_counts.append(len(pack))

            supervised_tokens += row_stats["supervised_tokens"]
            masked_prompt_tokens += row_stats["masked_prompt_tokens"]
            padded_tokens += row_stats["padded_tokens"]
            packed_tokens += row_stats["packed_tokens"]
            mixed_category_sequences += int(mixed_category)

        if not self.rows:
            raise ValueError("no packed chat sequences were created")

        explicit_groups = {group for group in self.groups if group is not None}
        self._stats = ChatDatasetStats(
            source_rows=len(self.examples),
            num_examples=len(self.examples),
            context_size=context_size,
            supervised_tokens=supervised_tokens,
            masked_prompt_tokens=masked_prompt_tokens,
            truncated_examples=0,
            skipped_long_examples=0,
            skipped_long_category_counts={},
            num_groups=len(explicit_groups),
            category_counts=dict(sorted(Counter(example.category for example in self.examples).items())),
            packing="bos_bestfit",
            source_examples=len(self.examples),
            num_sequences=len(self.rows),
            packed_sequences=len(self.rows),
            packed_tokens=packed_tokens,
            padded_tokens=padded_tokens,
            packing_efficiency=_safe_ratio(packed_tokens, len(self.rows) * context_size) or 0.0,
            average_examples_per_sequence=_safe_ratio(len(self.examples), len(self.rows)) or 0.0,
            mixed_category_sequences=mixed_category_sequences,
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
    pin_memory: bool = False,
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
            pin_memory=pin_memory,
        )
    if sampling == "category_sqrt":
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=category_sqrt_weights(dataset),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            pin_memory=pin_memory,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        pin_memory=pin_memory,
    )


def category_balanced_weights(dataset) -> torch.Tensor:
    """Return per-row weights that give each SFT category equal probability."""
    categories = [_category_key(dataset, index) for index in range(len(dataset))]
    if not categories:
        raise ValueError("cannot sample from an empty chat dataset")
    counts = Counter(categories)
    return torch.tensor([1.0 / counts[category] for category in categories], dtype=torch.double)


def category_sqrt_weights(dataset) -> torch.Tensor:
    """Return per-row weights that softly boost rare SFT categories."""
    categories = [_category_key(dataset, index) for index in range(len(dataset))]
    if not categories:
        raise ValueError("cannot sample from an empty chat dataset")
    counts = Counter(categories)
    return torch.tensor(
        [1.0 / (counts[category] ** 0.5) for category in categories],
        dtype=torch.double,
    )


def category_counts(dataset) -> dict[str, int]:
    """Count SFT categories in a dataset or subset."""
    counts = Counter(_category_key(dataset, index) for index in range(len(dataset)))
    return dict(sorted(counts.items()))


def label_audit(dataset) -> dict:
    """Summarize SFT label coverage for a dataset or subset.

    This is a guard against silent assistant-mask failures. Every trainable
    sequence should contain at least one non-ignored label.
    """
    sequences = len(dataset)
    total_positions = 0
    supervised_tokens = 0
    zero_supervised_sequences = 0
    active_counts: list[int] = []
    category_sequences: Counter[str] = Counter()
    category_supervised: Counter[str] = Counter()
    category_zero_sequences: Counter[str] = Counter()
    for index in range(sequences):
        _, labels = dataset[index]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        active = int((labels != -100).sum().item())
        total = int(labels.numel())
        category = _category_key(dataset, index)
        total_positions += total
        supervised_tokens += active
        active_counts.append(active)
        category_sequences[category] += 1
        category_supervised[category] += active
        if active == 0:
            zero_supervised_sequences += 1
            category_zero_sequences[category] += 1
    ignored_tokens = total_positions - supervised_tokens
    return {
        "sequences": sequences,
        "total_label_positions": total_positions,
        "supervised_tokens": supervised_tokens,
        "ignored_tokens": ignored_tokens,
        "active_label_fraction": _safe_ratio(supervised_tokens, total_positions) or 0.0,
        "zero_supervised_sequences": zero_supervised_sequences,
        "min_supervised_tokens_per_sequence": min(active_counts) if active_counts else 0,
        "max_supervised_tokens_per_sequence": max(active_counts) if active_counts else 0,
        "avg_supervised_tokens_per_sequence": _safe_ratio(supervised_tokens, sequences) or 0.0,
        "category_sequences": dict(sorted(category_sequences.items())),
        "category_supervised_tokens": dict(sorted(category_supervised.items())),
        "category_zero_supervised_sequences": dict(sorted(category_zero_sequences.items())),
    }


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
    if config.packing not in SFT_PACKING_MODES:
        raise ValueError(f"packing must be one of: {', '.join(SFT_PACKING_MODES)}")
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

    tokenizer = load_tokenizer(config.tokenizer_path)
    device = resolve_device(config.device)
    checkpoint_source = config.resume_from or config.checkpoint_path
    model, metadata = load_checkpoint(checkpoint_source, map_location=device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("tokenizer vocabulary size does not match checkpoint")
    training_fingerprint = make_training_fingerprint({
        "kind": "sft",
        "input_sha256": file_sha256(config.input_path),
        "tokenizer_sha256": file_sha256(config.tokenizer_path),
        "model_config": model.config.to_dict(),
        "val_fraction": config.val_fraction,
        "seed": config.seed,
        "context_size": model.config.context_size,
        "sampling": config.sampling,
        "packing": config.packing,
    })
    resume_state = (
        load_training_state(config.resume_from, map_location=device)
        if config.resume_from
        else None
    )
    if resume_state is not None:
        validate_training_fingerprint(resume_state, training_fingerprint)

    source_dataset = ChatSFTDataset(
        load_chat_examples(config.input_path),
        tokenizer=tokenizer,
        context_size=model.config.context_size,
    )
    split = split_chat_dataset(source_dataset, config.val_fraction, config.seed)
    train_source_examples = len(split.train)
    val_source_examples = len(split.val)
    train_category_counts = category_counts(split.train)
    val_category_counts = category_counts(split.val)
    if config.packing == "bos_bestfit":
        dataset = PackedChatSFTDataset(
            [source_dataset.tokenized_example(index) for index in range(len(source_dataset))],
            tokenizer=tokenizer,
            context_size=model.config.context_size,
        )
        train_dataset = PackedChatSFTDataset(
            [source_dataset.tokenized_example(index) for index in split.train.indices],
            tokenizer=tokenizer,
            context_size=model.config.context_size,
        )
        val_dataset = PackedChatSFTDataset(
            [source_dataset.tokenized_example(index) for index in split.val.indices],
            tokenizer=tokenizer,
            context_size=model.config.context_size,
        )
    else:
        dataset = source_dataset
        train_dataset = split.train
        val_dataset = split.val
    label_audit_report = {
        "full": label_audit(dataset),
        "train": label_audit(train_dataset),
        "validation": label_audit(val_dataset),
        "skipped_long_examples": source_dataset.stats().skipped_long_examples,
        "skipped_long_category_counts": source_dataset.stats().skipped_long_category_counts,
    }
    if label_audit_report["train"]["zero_supervised_sequences"]:
        raise ValueError("SFT train split contains sequences with no supervised assistant labels")
    if label_audit_report["validation"]["zero_supervised_sequences"]:
        raise ValueError("SFT validation split contains sequences with no supervised assistant labels")
    train_weights = None
    if config.sampling == "category_balanced":
        train_weights = category_balanced_weights(train_dataset)
    elif config.sampling == "category_sqrt":
        train_weights = category_sqrt_weights(train_dataset)
    ddp_env = ddp_env_metadata(config.ddp)
    train_batcher = make_resumable_batcher(
        train_dataset,
        config.batch_size,
        shuffle=True,
        seed=config.seed,
        weights=train_weights,
        pin_memory=device.type == "cuda",
        rank=int(ddp_env["rank"]),
        world_size=int(ddp_env["world_size"]),
    )
    pin_memory = device.type == "cuda"
    train_eval_loader = make_chat_dataloader(
        train_dataset,
        config.batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    val_loader = make_chat_dataloader(
        val_dataset,
        config.batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )

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
    token_bytes = torch.tensor(token_byte_lengths(tokenizer), dtype=torch.long, device=device)
    world_size = int(ddp_metadata.get("world_size", 1))
    local_effective_batch_size = config.batch_size * config.grad_accum_steps
    effective_batch_size = local_effective_batch_size * world_size
    effective_tokens_per_step = effective_batch_size * model.config.context_size

    losses: list[dict[str, float | int]] = []
    start = time.time()
    elapsed_offset = 0.0
    last_log_wall_time = start
    last_log_step = 0
    last_loss = float("nan")
    best_loss = float("inf")
    best_checkpoint: dict[str, float | int | str] | None = None
    best_checkpoint_dir = out_dir / "best_checkpoint"
    evals_without_improvement = 0
    final_step = 0
    stop_reason = "max_steps"
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
        final_step = int(resume_state.get("step", metadata.get("step", 0)))
        start_step = final_step + 1
        last_log_step = final_step
        losses = list(resume_state.get("losses", []))
        if losses:
            last_loss = float(losses[-1].get("train_loss", last_loss))
        best_loss = float(resume_state.get("best_metric", best_loss))
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

        local_loss = sum(micro_losses) / len(micro_losses)
        last_loss = mean_scalar_if_distributed(local_loss, device, ddp_metadata)
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
                    f"sft loss spike rollback at step {step}: "
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
            val_loss = float(val_metrics["loss"])
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
                "train_eval_loss": float(train_metrics["loss"]),
                "val_loss": val_loss,
                "train_bpb": train_metrics["bpb"],
                "val_bpb": val_metrics["bpb"],
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
                checkpoint_val_loss = float(ema_val_metrics["loss"])
                checkpoint_val_bpb = ema_val_metrics["bpb"]
                checkpoint_weights = "ema"
            else:
                checkpoint_val_loss = val_loss
                checkpoint_val_bpb = val_metrics["bpb"]
                checkpoint_weights = "raw"
            if checkpoint_val_loss < best_loss - config.early_stop_min_delta:
                best_loss = checkpoint_val_loss
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
                                "raw_val_bpb": val_metrics["bpb"],
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
                    "raw_val_bpb": val_metrics["bpb"],
                }
            else:
                evals_without_improvement += 1
            barrier_if_distributed(ddp_metadata)
            if main_process:
                print(
                    f"sft step {step:04d}/{config.max_steps:04d} | "
                    f"train {last_loss:.4f} | val {val_loss:.4f} | "
                    f"val_bpb {_format_optional(val_metrics['bpb'])} | "
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
                        best_metric=best_loss,
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
                    stage="sft",
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
                        f"sft early stop: validation did not improve for "
                        f"{config.early_stop_patience} evals"
                    )
                break
        if config.max_minutes is not None and elapsed_offset + time.time() - start >= config.max_minutes * 60:
            stop_reason = "max_minutes"
            if main_process:
                print(f"sft time stop: reached {config.max_minutes:.2f} minute budget")
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
                best_metric=best_loss,
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
    if main_process:
        model.eval()
        prompt_text = render_chat_prompt([], config.sample_prompt)
        prompt = torch.tensor(
            [tokenizer.encode(prompt_text, add_bos=True)],
            dtype=torch.long,
            device=device,
        )
        with autocast_context(precision_runtime):
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
            "artifacts_written": main_process,
            "training_fingerprint": training_fingerprint,
        },
        "base_checkpoint": {
            "path": config.checkpoint_path,
            "step": metadata.get("step"),
            "train_loss": metadata.get("train_loss"),
        },
        "dataset": {
            **dataset.stats().__dict__,
            "source_rows": source_dataset.stats().source_rows,
            "skipped_long_examples": source_dataset.stats().skipped_long_examples,
            "skipped_long_category_counts": source_dataset.stats().skipped_long_category_counts,
            "truncated_examples": source_dataset.stats().truncated_examples,
            "num_groups": source_dataset.stats().num_groups,
            "train_examples": train_source_examples,
            "val_examples": val_source_examples,
            "train_source_examples": train_source_examples,
            "val_source_examples": val_source_examples,
            "train_sequences": len(train_dataset),
            "val_sequences": len(val_dataset),
            "train_packing_efficiency": (
                train_dataset.stats().packing_efficiency
                if hasattr(train_dataset, "stats")
                else None
            ),
            "val_packing_efficiency": (
                val_dataset.stats().packing_efficiency
                if hasattr(val_dataset, "stats")
                else None
            ),
            "split_method": split.method,
            "train_groups": split.train_groups,
            "val_groups": split.val_groups,
            "train_indices": list(split.train.indices),
            "val_indices": list(split.val.indices),
            "category_counts": dataset.stats().category_counts,
            "train_category_counts": train_category_counts,
            "val_category_counts": val_category_counts,
            "sampling": config.sampling,
            "packing": config.packing,
        },
        "coverage": _coverage_report(
            train_source_examples=train_source_examples,
            total_source_examples=dataset.stats().source_examples,
            train_sequences=len(train_dataset),
            total_sequences=len(dataset),
            config=config,
            actual_steps=final_step,
            world_size=world_size,
        ),
        "model": {
            "config": model.config.to_dict(),
            "num_parameters": model.num_parameters(),
        },
        "losses": losses,
        "loss_diagnostics": loss_diagnostics(losses),
        "optimization_stability": optimization_stability(losses, config.grad_clip),
        "label_audit": label_audit_report,
        "throughput": _throughput_summary(losses),
        "rollback_events": rollback_events,
        "sample": sample,
        "checkpoint": str(checkpoint_dir),
        "resume_checkpoint": str(out_dir / "resume_checkpoint"),
        "ema_checkpoint": str(ema_checkpoint_dir) if ema is not None else None,
        "best_checkpoint": best_checkpoint,
        "stop_reason": stop_reason,
    }
    if main_process:
        (out_dir / "sft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "sft_label_audit.json").write_text(
            json.dumps(label_audit_report, indent=2),
            encoding="utf-8",
        )
        (out_dir / "report.md").write_text(sft_report_markdown(report), encoding="utf-8")
        (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
    barrier_if_distributed(ddp_metadata)
    return report


def _coverage_report(
    *,
    train_source_examples: int,
    total_source_examples: int,
    train_sequences: int,
    total_sequences: int,
    config: SFTConfig,
    actual_steps: int,
    world_size: int = 1,
) -> dict:
    local_sequences_per_step = config.batch_size * config.grad_accum_steps
    sequences_per_step = local_sequences_per_step * world_size
    sequence_updates = actual_steps * sequences_per_step
    examples_per_sequence = _safe_ratio(train_source_examples, train_sequences) or 0.0
    examples_per_step = sequences_per_step * examples_per_sequence
    examples_seen = actual_steps * examples_per_step
    report = {
        "actual_steps": actual_steps,
        "planned_steps": config.max_steps,
        "micro_batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "world_size": world_size,
        "local_sequences_per_step_estimate": local_sequences_per_step,
        "sequences_per_step_estimate": sequences_per_step,
        "planned_sequence_updates": config.max_steps * sequences_per_step,
        "actual_sequence_updates": sequence_updates,
        "train_sequences": train_sequences,
        "total_sequences": total_sequences,
        "examples_per_sequence_estimate": examples_per_sequence,
        "examples_per_step_estimate": examples_per_step,
        "planned_example_updates": config.max_steps * examples_per_step,
        "actual_example_updates": examples_seen,
        "train_examples": train_source_examples,
        "total_examples": total_source_examples,
        "estimated_train_epochs": _safe_ratio(sequence_updates, train_sequences),
        "estimated_dataset_passes": _safe_ratio(examples_seen, total_source_examples),
    }
    train_epochs = report["estimated_train_epochs"]
    warnings: list[str] = []
    if train_epochs is not None:
        if train_epochs >= 80:
            warnings.append(
                "Very high SFT exposure: examples are replayed >=80 times. "
                "Expect memorization unless held-out eval and SFT fit both improve."
            )
        elif train_epochs >= 30:
            warnings.append(
                "High SFT exposure: examples are replayed >=30 times. "
                "Prefer more behavior rows before increasing steps further."
            )
        elif train_epochs >= 10:
            warnings.append(
                "Moderate SFT exposure: examples are replayed >=10 times. "
                "Watch SFT fit and held-out eval before increasing steps."
            )
    report["warnings"] = warnings
    return report


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _format_optional(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"
