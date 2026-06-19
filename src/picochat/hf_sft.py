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

from picochat.device import resolve_device
from picochat.optim import LR_DECAYS, learning_rate_for_step
from picochat.precision import (
    autocast_context,
    configure_float32_matmul_precision,
    make_grad_scaler,
    maybe_compile_model,
    resolve_precision,
)
from picochat.sft import ChatExample


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
    peft: str = "none"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    trust_remote_code: bool = False
    revision: str | None = None
    done_file: str | None = "done.txt"
    progress_file: str | None = None


@dataclass(frozen=True)
class HFConversationExample:
    messages: tuple[dict[str, str], ...]
    category: str = "chat"
    group: str | None = None


class HFChatDataset(Dataset):
    def __init__(
        self,
        examples: list[HFConversationExample],
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
    example: HFConversationExample | ChatExample,
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


def render_hf_chat_text(example: HFConversationExample | ChatExample, tokenizer) -> tuple[str, str]:
    """Return prompt-only and prompt-plus-answer text for an HF tokenizer."""
    conversation = _as_hf_conversation(example)
    if len(conversation.messages) < 2 or conversation.messages[-1]["role"] != "assistant":
        raise ValueError("HF SFT examples must end with the target assistant message")
    prompt_messages = list(conversation.messages[:-1])
    full_messages = list(conversation.messages)
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return str(prompt), str(full)
    prompt = render_plain_hf_prompt(prompt_messages)
    eos = getattr(tokenizer, "eos_token", None) or ""
    answer = full_messages[-1]["content"]
    return prompt, f"{prompt} {answer}{eos}"


def render_plain_hf_prompt(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message["role"].strip().lower()
        content = message["content"].strip()
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "tool":
            lines.append(f"Tool: {content}")
        else:
            lines.append(f"{role.title()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def load_hf_sft_examples(path: str | Path) -> list[HFConversationExample]:
    """Load HF SFT rows from one-turn Picochat JSONL or multi-turn messages JSONL."""
    examples: list[HFConversationExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        category = record.get("category", "chat")
        if not isinstance(category, str):
            raise ValueError(f"line {line_number} category field must be a string when present")
        group = _optional_record_string(record, "group", "group_id", "template")
        messages = _messages_from_record(record, line_number)
        examples.append(HFConversationExample(messages=tuple(messages), category=category, group=group))
    if not examples:
        raise ValueError("HF SFT dataset is empty")
    return examples


def _messages_from_record(record: dict[str, Any], line_number: int) -> list[dict[str, str]]:
    raw_messages = record.get("messages")
    if raw_messages is not None:
        if not isinstance(raw_messages, list):
            raise ValueError(f"line {line_number} messages must be a list when present")
        messages = [_normalize_message(message, line_number) for message in raw_messages]
    else:
        user = record.get("user")
        assistant = record.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise ValueError(f"line {line_number} must contain messages or string user and assistant fields")
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]

    system = record.get("system")
    tools = record.get("tools")
    system_parts = []
    if isinstance(system, str) and system.strip():
        system_parts.append(system.strip())
    if tools is not None:
        system_parts.append("Tools:\n" + json.dumps(tools, ensure_ascii=False, sort_keys=True))
    if system_parts and (not messages or messages[0]["role"] != "system"):
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"line {line_number} messages must end with the target assistant response")
    return messages


def _normalize_message(message: Any, line_number: int) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError(f"line {line_number} each message must be an object")
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError(f"line {line_number} each message must contain string role and content")
    normalized_role = role.strip().lower()
    if normalized_role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"line {line_number} unsupported message role: {role}")
    return {"role": normalized_role, "content": content}


def _optional_record_string(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{key} field must be a string when present")
        return value
    return None


def _as_hf_conversation(example: HFConversationExample | ChatExample) -> HFConversationExample:
    if isinstance(example, HFConversationExample):
        return example
    return HFConversationExample(messages=(
        {"role": "user", "content": example.user},
        {"role": "assistant", "content": example.assistant},
    ), category=example.category, group=example.group)


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
    _write_hf_sft_progress(config, out_dir, {
        "status": "loading",
        "step": 0,
        "max_steps": config.max_steps,
        "message": "Loading tokenizer and base model weights.",
    })

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
    model, peft_metadata = apply_hf_peft(model, config)
    model.to(device)
    train_model, compile_metadata = maybe_compile_model(
        model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )

    examples = load_hf_sft_examples(config.input_path)
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

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("HF SFT model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
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
            _write_hf_sft_progress(config, out_dir, {
                "status": "training",
                "step": step,
                "max_steps": config.max_steps,
                "train_loss": total_loss,
                "val_loss": val_loss,
                "lr": lr,
                "elapsed_sec": row["elapsed_sec"],
            })
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
        "peft": peft_metadata,
    }
    (out_dir / "hf_sft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_hf_sft_markdown(report), encoding="utf-8")
    _write_hf_sft_progress(config, out_dir, {
        "status": "done",
        "step": config.max_steps,
        "max_steps": config.max_steps,
        "best_val_loss": report["best_val_loss"],
        "final_train_loss": report["final_train_loss"],
        "elapsed_sec": time.time() - start,
    })
    if config.done_file:
        done_path = _resolve_hf_sft_done_path(out_dir, config.done_file)
        done_path.parent.mkdir(parents=True, exist_ok=True)
        done_path.write_text(json.dumps({
            "status": "done",
            "out_dir": str(out_dir),
            "final_model": str(out_dir / "final_model"),
            "best_model": str(out_dir / "best_model") if report["best_val_loss"] is not None else None,
        }, indent=2), encoding="utf-8")
    return report


def _write_hf_sft_progress(config: HFSFTConfig, out_dir: Path, payload: dict[str, Any]) -> None:
    if not config.progress_file:
        return
    path = _resolve_hf_sft_done_path(out_dir, config.progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _resolve_hf_sft_done_path(out_dir: Path, done_file: str | Path) -> Path:
    done_path = Path(done_file)
    if done_path.is_absolute() or done_path.parent != Path("."):
        return done_path
    return out_dir / done_path


def apply_hf_peft(model, config: HFSFTConfig) -> tuple[Any, dict[str, Any]]:
    if config.peft == "none":
        return model, {"mode": "none"}
    if config.peft != "lora":
        raise ValueError("peft must be one of: none, lora")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError('HF LoRA requires peft: pip install -e ".[hf]"') from exc
    targets = [target.strip() for target in config.lora_target_modules.split(",") if target.strip()]
    if not targets:
        raise ValueError("lora_target_modules must contain at least one module name")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    return model, {
        "mode": "lora",
        "rank": config.lora_rank,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "target_modules": targets,
    }


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
    examples: list[HFConversationExample],
    val_fraction: float,
    seed: int,
) -> tuple[list[HFConversationExample], list[HFConversationExample]]:
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
    if config.peft not in {"none", "lora"}:
        raise ValueError("peft must be one of: none, lora")
    if config.lora_rank < 1:
        raise ValueError("lora_rank must be at least 1")
    if config.lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    if config.lora_dropout < 0 or config.lora_dropout >= 1:
        raise ValueError("lora_dropout must be in [0, 1)")


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
