"""Batching utilities for next-token language-model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from picochat.tokenizer import CharTokenizer


@dataclass(frozen=True)
class TokenDatasetStats:
    num_tokens: int
    context_size: int
    num_sequences: int


class TokenWindowDataset(torch.utils.data.Dataset):
    """Fixed-size next-token windows over one long token stream."""

    def __init__(self, tokens: list[int], context_size: int):
        if context_size < 2:
            raise ValueError("context_size must be at least 2")
        if len(tokens) <= context_size:
            raise ValueError("token stream must be longer than context_size")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.tokens) - self.context_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk = self.tokens[index: index + self.context_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def stats(self) -> TokenDatasetStats:
        return TokenDatasetStats(
            num_tokens=len(self.tokens),
            context_size=self.context_size,
            num_sequences=len(self),
        )


def load_token_dataset(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    context_size: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> TokenWindowDataset:
    """Load text, encode it, and return fixed next-token windows."""
    tokenizer = CharTokenizer.load(tokenizer_path)
    text = Path(corpus_path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    return TokenWindowDataset(tokens, context_size=context_size)


def make_dataloader(
    dataset: TokenWindowDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create a deterministic PyTorch DataLoader for token windows."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=True,
    )


def split_dataset(
    dataset: TokenWindowDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Deterministically split token windows into train and validation subsets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    num_items = len(dataset)
    num_val = max(1, int(num_items * val_fraction))
    num_train = num_items - num_val
    if num_train < 1:
        raise ValueError("dataset is too small for a train/val split")

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator).tolist()
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )
