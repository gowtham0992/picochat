"""Batching utilities for next-token language-model training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from picochat.tokenizer import load_tokenizer


@dataclass(frozen=True)
class TokenDatasetStats:
    num_tokens: int
    context_size: int
    num_sequences: int


@dataclass(frozen=True)
class TokenSplitBundle:
    train_dataset: Any
    val_dataset: Any
    stats: dict[str, Any]
    train_text: str
    val_text: str


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
    tokenizer = load_tokenizer(tokenizer_path)
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
        drop_last=False,
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


def load_token_split(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    context_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
    split_mode: str = "window",
    corpus_manifest_path: str | Path | None = None,
) -> TokenSplitBundle:
    """Load train/validation token windows with either window or document splitting."""
    if split_mode not in {"window", "document"}:
        raise ValueError("split_mode must be 'window' or 'document'")
    if split_mode == "document" and corpus_manifest_path:
        bundle = _document_token_split(
            corpus_path=Path(corpus_path),
            tokenizer_path=Path(tokenizer_path),
            manifest_path=Path(corpus_manifest_path),
            context_size=context_size,
            val_fraction=val_fraction,
            seed=seed,
        )
        if bundle is not None:
            return bundle
    reason = "requested_window_split" if split_mode == "window" else "document_split_unavailable"
    return _window_token_split(
        corpus_path=Path(corpus_path),
        tokenizer_path=Path(tokenizer_path),
        context_size=context_size,
        val_fraction=val_fraction,
        seed=seed,
        split_reason=reason,
    )


def _window_token_split(
    corpus_path: Path,
    tokenizer_path: Path,
    context_size: int,
    val_fraction: float,
    seed: int,
    split_reason: str,
) -> TokenSplitBundle:
    dataset = load_token_dataset(corpus_path, tokenizer_path, context_size=context_size)
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=val_fraction, seed=seed)
    text = corpus_path.read_text(encoding="utf-8")
    stats = {
        **dataset.stats().__dict__,
        "split_mode": "window",
        "split_reason": split_reason,
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "num_documents": None,
        "train_documents": None,
        "val_documents": None,
        "val_document_paths": [],
    }
    return TokenSplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stats=stats,
        train_text=text,
        val_text=text,
    )


def _document_token_split(
    corpus_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
    context_size: int,
    val_fraction: float,
    seed: int,
) -> TokenSplitBundle | None:
    documents = _manifest_documents(manifest_path)
    if len(documents) < 2:
        return None
    corpus_text = corpus_path.read_text(encoding="utf-8")
    tokenizer = load_tokenizer(tokenizer_path)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(documents), generator=generator).tolist()
    num_val = max(1, int(len(documents) * val_fraction))
    num_train = len(documents) - num_val
    if num_train < 1:
        return None
    train_docs = [documents[index] for index in indices[:num_train]]
    val_docs = [documents[index] for index in indices[num_train: num_train + num_val]]
    train_text = "\n\n".join(_slice_document(corpus_text, document) for document in train_docs)
    val_text = "\n\n".join(_slice_document(corpus_text, document) for document in val_docs)

    train_tokens = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    val_tokens = tokenizer.encode(val_text, add_bos=True, add_eos=True)
    if len(train_tokens) <= context_size or len(val_tokens) <= context_size:
        return None

    train_dataset = TokenWindowDataset(train_tokens, context_size=context_size)
    val_dataset = TokenWindowDataset(val_tokens, context_size=context_size)
    full_tokens = tokenizer.encode(corpus_text, add_bos=True, add_eos=True)
    stats = {
        "num_tokens": len(full_tokens),
        "context_size": context_size,
        "num_sequences": len(train_dataset) + len(val_dataset),
        "split_mode": "document",
        "split_reason": "held_out_complete_documents",
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "num_documents": len(documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "val_document_paths": [document.get("path", "") for document in val_docs],
    }
    return TokenSplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stats=stats,
        train_text=train_text,
        val_text=val_text,
    )


def _manifest_documents(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        return []
    valid = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        if not isinstance(document.get("char_start"), int) or not isinstance(document.get("char_end"), int):
            continue
        valid.append(document)
    return valid


def _slice_document(corpus_text: str, document: dict[str, Any]) -> str:
    start = max(0, int(document["char_start"]))
    end = max(start, int(document["char_end"]))
    return corpus_text[start:end].strip()
