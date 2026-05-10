"""Supervised chat fine-tuning for Picochat."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch

from picochat.chat import render_chat_prompt
from picochat.checkpoint import load_checkpoint, save_checkpoint
from picochat.report import loss_diagnostics, sft_report_markdown
from picochat.tokenizer import CharTokenizer
from picochat.train import evaluate_loss


@dataclass(frozen=True)
class ChatExample:
    user: str
    assistant: str


@dataclass(frozen=True)
class ChatDatasetStats:
    num_examples: int
    context_size: int
    supervised_tokens: int
    truncated_examples: int


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


class ChatSFTDataset(torch.utils.data.Dataset):
    """Fixed-length chat examples with loss only on assistant tokens."""

    def __init__(
        self,
        examples: list[ChatExample],
        tokenizer: CharTokenizer,
        context_size: int,
    ):
        if context_size < 2:
            raise ValueError("context_size must be at least 2")

        self.context_size = context_size
        self.rows: list[tuple[torch.Tensor, torch.Tensor]] = []
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
            supervised_tokens += supervised

        if not self.rows:
            raise ValueError("no usable chat examples fit inside the model context")

        self._stats = ChatDatasetStats(
            num_examples=len(self.rows),
            context_size=context_size,
            supervised_tokens=supervised_tokens,
            truncated_examples=truncated_examples,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        return self.rows[index]

    def stats(self) -> ChatDatasetStats:
        return self._stats


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
        examples.append(ChatExample(user=user, assistant=assistant))
    if not examples:
        raise ValueError("chat dataset is empty")
    return examples


def split_chat_dataset(
    dataset: ChatSFTDataset,
    val_fraction: float,
    seed: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Deterministically split chat examples into train and validation subsets."""
    if len(dataset) == 1:
        return torch.utils.data.Subset(dataset, [0]), torch.utils.data.Subset(dataset, [0])
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    num_val = max(1, int(len(dataset) * val_fraction))
    num_train = len(dataset) - num_val
    if num_train < 1:
        num_train = 1
        num_val = len(dataset) - 1

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return (
        torch.utils.data.Subset(dataset, indices[:num_train]),
        torch.utils.data.Subset(dataset, indices[num_train: num_train + num_val]),
    )


def make_chat_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    """Create a deterministic dataloader for small chat datasets."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def train_sft(config: SFTConfig) -> dict:
    """Fine-tune a base checkpoint on small chat examples."""
    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer.load(config.tokenizer_path)
    model, metadata = load_checkpoint(config.checkpoint_path, map_location=config.device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("tokenizer vocabulary size does not match checkpoint")

    dataset = ChatSFTDataset(
        load_chat_examples(config.input_path),
        tokenizer=tokenizer,
        context_size=model.config.context_size,
    )
    train_dataset, val_dataset = split_chat_dataset(dataset, config.val_fraction, config.seed)
    train_loader = make_chat_dataloader(train_dataset, config.batch_size, shuffle=True, seed=config.seed)
    val_loader = make_chat_dataloader(val_dataset, config.batch_size, shuffle=False, seed=config.seed)
    data_iter = iter(train_loader)

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[dict[str, float | int]] = []
    start = time.time()
    last_loss = float("nan")
    best_loss = float("inf")
    best_checkpoint: dict[str, float | int | str] | None = None
    best_checkpoint_dir = out_dir / "best_checkpoint"

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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            val_loss = evaluate_loss(model, val_loader, device, max_batches=config.eval_batches)
            elapsed = time.time() - start
            losses.append({
                "step": step,
                "train_loss": last_loss,
                "val_loss": val_loss,
                "elapsed_sec": elapsed,
            })
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(best_checkpoint_dir, model, step=step, train_loss=last_loss)
                best_checkpoint = {
                    "path": str(best_checkpoint_dir),
                    "step": step,
                    "train_loss": last_loss,
                    "val_loss": val_loss,
                }
            print(
                f"sft step {step:04d}/{config.max_steps:04d} | "
                f"train {last_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s"
            )

    checkpoint_dir = out_dir / "checkpoint"
    save_checkpoint(checkpoint_dir, model, step=config.max_steps, train_loss=last_loss)

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
        },
        "model": {
            "config": model.config.to_dict(),
            "num_parameters": model.num_parameters(),
        },
        "losses": losses,
        "loss_diagnostics": loss_diagnostics(losses),
        "sample": sample,
        "checkpoint": str(checkpoint_dir),
        "best_checkpoint": best_checkpoint or {
            "path": str(checkpoint_dir),
            "step": config.max_steps,
            "train_loss": last_loss,
            "val_loss": None,
        },
    }
    (out_dir / "sft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(sft_report_markdown(report), encoding="utf-8")
    (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
    return report
