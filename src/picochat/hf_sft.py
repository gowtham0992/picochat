"""Supervised fine-tuning for existing Hugging Face causal LMs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from picochat.chat import render_chat_prompt
from picochat.device import resolve_device
from picochat.optim import LR_DECAYS, learning_rate_for_step
from picochat.precision import (
    autocast_context,
    configure_float32_matmul_precision,
    make_grad_scaler,
    maybe_compile_model,
    resolve_precision,
)
from picochat.sft import ChatExample, load_chat_examples


@dataclass(frozen=True)
class HFSFTConfig:
    model: str
    input_path: str
    out_dir: str
    max_steps: int = 100
    batch_size: int = 1
    grad_accum_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    lr_warmup_steps: int = 0
    lr_decay: str = "cosine"
    min_lr_ratio: float = 0.1
    max_length: int = 1024
    val_fraction: float = 0.05
    eval_batches: int = 10
    log_every: int = 10
    seed: int = 42
    device: str = "auto"
    precision: str = "auto"
    matmul_precision: str = "default"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    revision: str | None = None


class HFChatDataset(Dataset):
    def __init__(
        self,
        examples: list[ChatExample],
        tokenizer,
        *,
        max_length: int,
    ) -> None:
        self.rows = [
            row
            for example in examples
            if (row := tokenize_hf_chat_example(example, tokenizer, max_length=max_length)) is not None
        ]
        if not self.rows:
            raise ValueError("all HF SFT examples were empty or too long after tokenization")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


def tokenize_hf_chat_example(
    example: ChatExample,
    tokenizer,
    *,
    max_length: int,
) -> dict[str, list[int]] | None:
    """Tokenize one Picochat chat row for assistant-only HF causal-LM SFT."""
    prompt_text, full_text = render_hf_chat_text(example, tokenizer)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids[:max_length]
    if len(full_ids) <= len(prompt_ids):
        return None
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        return None
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def render_hf_chat_text(example: ChatExample, tokenizer) -> tuple[str, str]:
    """Return prompt-only and prompt-plus-answer text for an HF tokenizer."""
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example.user},
                {"role": "assistant", "content": example.assistant},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return str(prompt), str(full)
    prompt = render_chat_prompt([], example.user)
    eos = getattr(tokenizer, "eos_token", None) or ""
    suffix = f" {example.assistant}{eos}"
    return prompt, prompt + suffix


def hf_chat_collate(rows: list[dict[str, torch.Tensor]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(row["input_ids"].numel()) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        length = int(row["input_ids"].numel())
        input_ids[index, :length] = row["input_ids"]
        attention_mask[index, :length] = row["attention_mask"]
        labels[index, :length] = row["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_hf_sft(config: HFSFTConfig) -> dict[str, Any]:
    """Fine-tune an existing Hugging Face causal LM on Picochat chat JSONL."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError('train hf-sft requires the optional dependency group: pip install -e ".[hf]"') from exc

    _validate_hf_sft_config(config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    matmul_precision_runtime = configure_float32_matmul_precision(config.matmul_precision)
    precision_runtime = resolve_precision(config.precision, device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("HF tokenizer must define pad_token_id or eos_token_id")

    model_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
    }
    if precision_runtime.dtype is not None and device.type in {"cuda", "mps"}:
        model_kwargs["torch_dtype"] = precision_runtime.dtype
    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if config.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("this HF model does not expose gradient_checkpointing_enable()")
        model.gradient_checkpointing_enable()
    model.to(device)
    train_model, compile_metadata = maybe_compile_model(
        model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )

    examples = load_chat_examples(config.input_path)
    train_examples, val_examples = _split_examples(examples, config.val_fraction, config.seed)
    train_dataset = HFChatDataset(train_examples, tokenizer, max_length=config.max_length)
    val_dataset = HFChatDataset(val_examples, tokenizer, max_length=config.max_length) if val_examples else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda rows: hf_chat_collate(rows, pad_token_id=pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda rows: hf_chat_collate(rows, pad_token_id=pad_token_id),
    ) if val_dataset is not None else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(precision_runtime)
    losses: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    start = time.time()
    loader_iter = iter(train_loader)

    train_model.train()
    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            batch = _move_batch(batch, device)
            with autocast_context(precision_runtime):
                output = train_model(**batch)
                loss = output.loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * config.grad_accum_steps

        lr = learning_rate_for_step(
            base_learning_rate=config.learning_rate,
            step=step,
            max_steps=config.max_steps,
            warmup_steps=config.lr_warmup_steps,
            decay=config.lr_decay,
            min_lr_ratio=config.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()

        should_log = step == 1 or step == config.max_steps or step % config.log_every == 0
        if should_log:
            val_loss = evaluate_hf_sft_loss(train_model, val_loader, device, precision_runtime, config.eval_batches)
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(out_dir / "best_model")
                tokenizer.save_pretrained(out_dir / "best_model")
            row = {
                "step": step,
                "train_loss": total_loss,
                "val_loss": val_loss if val_loss is not None else math.nan,
                "lr": lr,
                "elapsed_sec": time.time() - start,
            }
            losses.append(row)
            val_text = "n/a" if val_loss is None else f"{val_loss:.4f}"
            print(f"hf-sft step {step:04d}/{config.max_steps:04d} | train {total_loss:.4f} | val {val_text} | lr {lr:.2e}")

    model.save_pretrained(out_dir / "final_model")
    tokenizer.save_pretrained(out_dir / "final_model")
    report = {
        "model": config.model,
        "input": config.input_path,
        "out_dir": str(out_dir),
        "num_examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "tokenized_train_sequences": len(train_dataset),
        "tokenized_val_sequences": len(val_dataset) if val_dataset else 0,
        "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss,
        "final_train_loss": losses[-1]["train_loss"] if losses else None,
        "losses": losses,
        "precision_runtime": precision_runtime.to_dict(),
        "matmul_precision_runtime": matmul_precision_runtime,
        "compile": compile_metadata,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
    (out_dir / "hf_sft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_hf_sft_markdown(report), encoding="utf-8")
    return report


def evaluate_hf_sft_loss(
    model,
    val_loader: DataLoader | None,
    device: torch.device,
    precision_runtime,
    eval_batches: int,
) -> float | None:
    if val_loader is None or eval_batches <= 0:
        return None
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            if index >= eval_batches:
                break
            batch = _move_batch(batch, device)
            with autocast_context(precision_runtime):
                output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else None


def _split_examples(
    examples: list[ChatExample],
    val_fraction: float,
    seed: int,
) -> tuple[list[ChatExample], list[ChatExample]]:
    if len(examples) < 2 or val_fraction <= 0:
        return examples, []
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    val_count = min(len(examples) - 1, max(1, int(round(len(examples) * val_fraction))))
    val_ids = set(indices[:val_count])
    train = [example for index, example in enumerate(examples) if index not in val_ids]
    val = [example for index, example in enumerate(examples) if index in val_ids]
    return train, val


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _validate_hf_sft_config(config: HFSFTConfig) -> None:
    if config.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    if config.lr_decay not in LR_DECAYS:
        raise ValueError(f"lr_decay must be one of: {', '.join(LR_DECAYS)}")
    if not 0 <= config.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if config.max_length < 8:
        raise ValueError("max_length must be at least 8")
    if not 0 <= config.val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if config.eval_batches < 0:
        raise ValueError("eval_batches must be non-negative")
    if config.log_every < 1:
        raise ValueError("log_every must be at least 1")


def _hf_sft_markdown(report: dict[str, Any]) -> str:
    best = report.get("best_val_loss")
    best_text = "n/a" if best is None else f"{best:.4f}"
    lines = [
        "# HF SFT Report",
        "",
        f"- Source model: `{report['model']}`",
        f"- Input: `{report['input']}`",
        f"- Train examples: `{report['train_examples']}`",
        f"- Validation examples: `{report['val_examples']}`",
        f"- Tokenized train sequences: `{report['tokenized_train_sequences']}`",
        f"- Tokenized validation sequences: `{report['tokenized_val_sequences']}`",
        f"- Best validation loss: `{best_text}`",
        f"- Final model: `{Path(report['out_dir']) / 'final_model'}`",
    ]
    if best is not None:
        lines.append(f"- Best model: `{Path(report['out_dir']) / 'best_model'}`")
    lines.extend([
        "",
        "This path fine-tunes an existing Hugging Face causal LM. It does not train a Picochat-native base model from scratch.",
        "",
    ])
    return "\n".join(lines)
