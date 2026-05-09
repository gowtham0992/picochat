"""Base language-model training loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch

from picochat.batching import load_token_dataset, make_dataloader, split_dataset
from picochat.checkpoint import save_checkpoint
from picochat.model import GPTConfig, TinyGPT
from picochat.report import training_report_markdown
from picochat.tokenizer import CharTokenizer


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
    seed: int = 42
    device: str = "cpu"
    log_every: int = 20
    val_fraction: float = 0.1
    eval_batches: int = 10
    sample_tokens: int = 120


@torch.no_grad()
def evaluate_loss(model: TinyGPT, loader, device: torch.device, max_batches: int) -> float:
    """Estimate loss over a limited number of validation batches."""
    model.eval()
    losses = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def train_base(config: TrainConfig) -> dict:
    """Train a tiny next-token model and save artifacts."""
    torch.manual_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer.load(config.tokenizer_path)
    dataset = load_token_dataset(
        config.corpus_path,
        config.tokenizer_path,
        context_size=config.context_size,
    )
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_loader = make_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    val_loader = make_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    data_iter = iter(train_loader)

    device = torch.device(config.device)
    model_config = GPTConfig(
        vocab_size=len(tokenizer),
        context_size=config.context_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
    )
    model = TinyGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[dict[str, float | int]] = []
    start = time.time()
    last_loss = float("nan")

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
            print(
                f"step {step:04d}/{config.max_steps:04d} | "
                f"train {last_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s"
            )

    checkpoint_dir = out_dir / "checkpoint"
    save_checkpoint(checkpoint_dir, model, step=config.max_steps, train_loss=last_loss)

    model.eval()
    prompt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
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
        "dataset": {
            **dataset.stats().__dict__,
            "train_sequences": len(train_dataset),
            "val_sequences": len(val_dataset),
        },
        "model": {
            "config": model_config.to_dict(),
            "num_parameters": model.num_parameters(),
        },
        "losses": losses,
        "sample": sample,
        "checkpoint": str(checkpoint_dir),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(training_report_markdown(report), encoding="utf-8")
    (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
    return report
