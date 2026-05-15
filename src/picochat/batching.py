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
    canary_values: tuple[str, ...] = ()


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


class ResumableBatcher:
    """Deterministic map-style batch iterator with explicit resume state."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        epoch: int = 0,
        batch_index: int = 0,
        weights: torch.Tensor | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(dataset) < 1:
            raise ValueError("dataset must not be empty")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.batch_index = batch_index
        self.weights = weights.double() if weights is not None else None
        self._indices_epoch: int | None = None
        self._indices: list[int] = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_index >= self.batches_per_epoch:
            self.epoch += 1
            self.batch_index = 0
            self._indices_epoch = None
        indices = self._epoch_indices()
        start = self.batch_index * self.batch_size
        end = min(start + self.batch_size, len(indices))
        self.batch_index += 1
        batch = [self.dataset[index] for index in indices[start:end]]
        return _collate_tensor_pairs(batch)

    @property
    def batches_per_epoch(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "weighted": self.weights is not None,
            "batches_per_epoch": self.batches_per_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("batch_size", self.batch_size)) != self.batch_size:
            raise ValueError("batcher state batch_size does not match this run")
        if bool(state.get("shuffle", self.shuffle)) != self.shuffle:
            raise ValueError("batcher state shuffle setting does not match this run")
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("batcher state seed does not match this run")
        if bool(state.get("weighted", self.weights is not None)) != (self.weights is not None):
            raise ValueError("batcher state weighting does not match this run")
        self.epoch = int(state.get("epoch", 0))
        self.batch_index = int(state.get("batch_index", 0))
        if self.batch_index < 0 or self.batch_index > self.batches_per_epoch:
            raise ValueError("batcher state batch_index is out of range")
        self._indices_epoch = None

    def _epoch_indices(self) -> list[int]:
        if self._indices_epoch == self.epoch:
            return self._indices
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            if self.weights is not None:
                self._indices = torch.multinomial(
                    self.weights,
                    num_samples=len(self.dataset),
                    replacement=True,
                    generator=generator,
                ).tolist()
            else:
                self._indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            self._indices = list(range(len(self.dataset)))
        self._indices_epoch = self.epoch
        return self._indices


def _collate_tensor_pairs(batch) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    return x, y


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


def make_resumable_batcher(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    weights: torch.Tensor | None = None,
) -> ResumableBatcher:
    """Create a deterministic train iterator that can save and load position."""
    return ResumableBatcher(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        weights=weights,
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
    canary_values: tuple[str, ...] | list[str] = (),
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
            canary_values=tuple(canary_values),
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
        canary_values=tuple(canary_values),
    )


def _window_token_split(
    corpus_path: Path,
    tokenizer_path: Path,
    context_size: int,
    val_fraction: float,
    seed: int,
    split_reason: str,
    canary_values: tuple[str, ...],
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
        "train_tokens": None,
        "val_tokens": None,
        "num_documents": None,
        "train_documents": None,
        "val_documents": None,
        "val_document_paths": [],
        "canary_values": [],
        "canaries_enabled": False,
        "canary_note": (
            "canaries require document split to guarantee train-only placement"
            if canary_values else "disabled"
        ),
    }
    return TokenSplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stats=stats,
        train_text=text,
        val_text=text,
        canary_values=(),
    )


def _document_token_split(
    corpus_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
    context_size: int,
    val_fraction: float,
    seed: int,
    canary_values: tuple[str, ...],
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
    train_doc_texts = [_slice_document(corpus_text, document) for document in train_docs]
    val_doc_texts = [_slice_document(corpus_text, document) for document in val_docs]
    train_text = "\n\n".join(train_doc_texts)
    val_text = "\n\n".join(val_doc_texts)
    canary_block = _canary_block(canary_values)
    if canary_block:
        train_doc_texts.append(canary_block)
        train_text = f"{train_text}\n\n{canary_block}" if train_text else canary_block

    train_tokens = _encode_documents(train_doc_texts, tokenizer)
    val_tokens = _encode_documents(val_doc_texts, tokenizer)
    if len(train_tokens) <= context_size or len(val_tokens) <= context_size:
        return None

    train_dataset = TokenWindowDataset(train_tokens, context_size=context_size)
    val_dataset = TokenWindowDataset(val_tokens, context_size=context_size)
    source_tokens = tokenizer.encode(corpus_text, add_bos=True, add_eos=True)
    stats = {
        "num_tokens": len(train_tokens) + len(val_tokens),
        "source_num_tokens": len(source_tokens),
        "context_size": context_size,
        "num_sequences": len(train_dataset) + len(val_dataset),
        "split_mode": "document",
        "split_reason": "held_out_complete_documents",
        "packing": "bos_eos_per_document",
        "document_boundary_tokens": True,
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "num_documents": len(documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "val_document_paths": [document.get("path", "") for document in val_docs],
        "canary_values": list(canary_values),
        "canaries_enabled": bool(canary_values),
        "canary_note": "train split only" if canary_values else "disabled",
    }
    return TokenSplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stats=stats,
        train_text=train_text,
        val_text=val_text,
        canary_values=canary_values,
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


def _encode_documents(documents: list[str], tokenizer) -> list[int]:
    tokens: list[int] = []
    for document in documents:
        text = document.strip()
        if text:
            tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return tokens


def _canary_block(canary_values: tuple[str, ...]) -> str:
    return "\n".join(
        f"Memorization canary phrase: {value}."
        for value in canary_values
    )
