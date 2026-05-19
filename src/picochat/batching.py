"""Batching utilities for next-token language-model training."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import bisect
import json
from pathlib import Path
from typing import Any

import torch

from picochat.resume import file_sha256
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


class ShardedTokenWindowDataset(torch.utils.data.Dataset):
    """Fixed-size token windows over token shards stored on disk."""

    def __init__(
        self,
        shards: list[dict[str, Any]],
        context_size: int,
        max_cached_shards: int = 2,
    ):
        if context_size < 2:
            raise ValueError("context_size must be at least 2")
        if max_cached_shards < 1:
            raise ValueError("max_cached_shards must be at least 1")
        usable_shards = [
            shard for shard in shards
            if int(shard.get("num_tokens", 0)) > context_size
        ]
        if not usable_shards:
            raise ValueError("no token shard is longer than context_size")
        self.shards = usable_shards
        self.context_size = context_size
        lengths = [int(shard["num_tokens"]) - context_size for shard in self.shards]
        self._window_counts = lengths
        self._prefix_lengths: list[int] = []
        total = 0
        for length in lengths:
            total += length
            self._prefix_lengths.append(total)
        self.max_cached_shards = max_cached_shards
        self._shard_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return self._prefix_lengths[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._prefix_lengths, index)
        previous = 0 if shard_index == 0 else self._prefix_lengths[shard_index - 1]
        local_index = index - previous
        tokens = self._load_shard(shard_index)
        chunk = tokens[local_index: local_index + self.context_size + 1]
        return chunk[:-1], chunk[1:]

    def random_batch_indices(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> list[int]:
        """Sample a random batch from one shard to avoid cross-shard cache thrash."""
        draw = int(torch.randint(
            low=0,
            high=self._prefix_lengths[-1],
            size=(1,),
            generator=generator,
        ).item())
        shard_index = bisect.bisect_right(self._prefix_lengths, draw)
        previous = 0 if shard_index == 0 else self._prefix_lengths[shard_index - 1]
        shard_windows = self._window_counts[shard_index]
        local_indices = torch.randint(
            low=0,
            high=shard_windows,
            size=(batch_size,),
            generator=generator,
        )
        return (local_indices + previous).tolist()

    def stats(self) -> TokenDatasetStats:
        return TokenDatasetStats(
            num_tokens=sum(int(shard["num_tokens"]) for shard in self.shards),
            context_size=self.context_size,
            num_sequences=len(self),
        )

    def _load_shard(self, shard_index: int) -> torch.Tensor:
        if shard_index in self._shard_cache:
            tokens = self._shard_cache.pop(shard_index)
            self._shard_cache[shard_index] = tokens
            return tokens
        tokens = torch.load(self.shards[shard_index]["path"], map_location="cpu", weights_only=True)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        self._shard_cache[shard_index] = tokens.long()
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return self._shard_cache[shard_index]


class PackedTokenRowDataset(torch.utils.data.Dataset):
    """Fixed BOS-bestfit packed rows stored on disk as (rows, context+1) tensors."""

    def __init__(
        self,
        shards: list[dict[str, Any]],
        context_size: int,
        max_cached_shards: int = 2,
    ):
        if context_size < 2:
            raise ValueError("context_size must be at least 2")
        if max_cached_shards < 1:
            raise ValueError("max_cached_shards must be at least 1")
        usable_shards = [
            shard for shard in shards
            if int(shard.get("num_rows", 0)) > 0
        ]
        if not usable_shards:
            raise ValueError("no packed row shards are available")
        self.shards = usable_shards
        self.context_size = context_size
        self._row_counts = [int(shard["num_rows"]) for shard in self.shards]
        self._prefix_lengths: list[int] = []
        total = 0
        for count in self._row_counts:
            total += count
            self._prefix_lengths.append(total)
        self.max_cached_shards = max_cached_shards
        self._shard_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return self._prefix_lengths[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._prefix_lengths, index)
        previous = 0 if shard_index == 0 else self._prefix_lengths[shard_index - 1]
        local_index = index - previous
        rows = self._load_shard(shard_index)
        row = rows[local_index]
        return row[:-1], row[1:]

    def random_batch_indices(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> list[int]:
        draw = int(torch.randint(
            low=0,
            high=self._prefix_lengths[-1],
            size=(1,),
            generator=generator,
        ).item())
        shard_index = bisect.bisect_right(self._prefix_lengths, draw)
        previous = 0 if shard_index == 0 else self._prefix_lengths[shard_index - 1]
        shard_rows = self._row_counts[shard_index]
        local_indices = torch.randint(
            low=0,
            high=shard_rows,
            size=(batch_size,),
            generator=generator,
        )
        return (local_indices + previous).tolist()

    def stats(self) -> TokenDatasetStats:
        return TokenDatasetStats(
            num_tokens=sum(int(shard["num_tokens"]) for shard in self.shards),
            context_size=self.context_size,
            num_sequences=len(self),
        )

    def _load_shard(self, shard_index: int) -> torch.Tensor:
        if shard_index in self._shard_cache:
            rows = self._shard_cache.pop(shard_index)
            self._shard_cache[shard_index] = rows
            return rows
        rows = torch.load(self.shards[shard_index]["path"], map_location="cpu", weights_only=True)
        if not isinstance(rows, torch.Tensor):
            rows = torch.tensor(rows, dtype=torch.long)
        rows = rows.long()
        expected_width = self.context_size + 1
        if rows.ndim != 2 or rows.size(1) != expected_width:
            raise ValueError(
                f"packed shard has shape {tuple(rows.shape)}, expected (*, {expected_width})"
            )
        self._shard_cache[shard_index] = rows
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return self._shard_cache[shard_index]


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
        index_mode: str = "auto",
        permutation_threshold: int = 5_000_000,
        pin_memory: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(dataset) < 1:
            raise ValueError("dataset must not be empty")
        if index_mode not in {"auto", "permutation", "random"}:
            raise ValueError("index_mode must be 'auto', 'permutation', or 'random'")
        if permutation_threshold < 1:
            raise ValueError("permutation_threshold must be at least 1")
        if weights is not None and index_mode == "random":
            raise ValueError("random index_mode does not support weighted sampling")
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.batch_index = batch_index
        self.weights = weights.double() if weights is not None else None
        self.index_mode = index_mode
        self.permutation_threshold = permutation_threshold
        self.pin_memory = pin_memory
        self.rank = rank
        self.world_size = world_size
        self._indices_epoch: int | None = None
        self._indices: list[int] = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_index >= self.batches_per_epoch:
            self.epoch += 1
            self.batch_index = 0
            self._indices_epoch = None
        if self._resolved_index_mode() == "random":
            return self._next_random_batch()
        indices = self._epoch_indices()
        start = (
            self.batch_index * self.batch_size * self.world_size
            + self.rank * self.batch_size
        )
        batch_indices = [
            indices[(start + offset) % len(indices)]
            for offset in range(self.batch_size)
        ]
        self.batch_index += 1
        batch = [self.dataset[index] for index in batch_indices]
        return _collate_tensor_pairs(batch, pin_memory=self.pin_memory)

    @property
    def batches_per_epoch(self) -> int:
        global_batch_size = self.batch_size * self.world_size
        return max(1, len(self.dataset) // global_batch_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "weighted": self.weights is not None,
            "batches_per_epoch": self.batches_per_epoch,
            "index_mode": self.index_mode,
            "resolved_index_mode": self._resolved_index_mode(),
            "permutation_threshold": self.permutation_threshold,
            "pin_memory": self.pin_memory,
            "rank": self.rank,
            "world_size": self.world_size,
            "global_batch_size": self.batch_size * self.world_size,
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
        observed_index_mode = str(state.get("index_mode", self.index_mode))
        if observed_index_mode != self.index_mode:
            raise ValueError("batcher state index_mode does not match this run")
        if bool(state.get("pin_memory", self.pin_memory)) != self.pin_memory:
            raise ValueError("batcher state pin_memory does not match this run")
        if int(state.get("world_size", self.world_size)) != self.world_size:
            raise ValueError("batcher state world_size does not match this run")
        self.epoch = int(state.get("epoch", 0))
        self.batch_index = int(state.get("batch_index", 0))
        if self.batch_index < 0 or self.batch_index > self.batches_per_epoch:
            raise ValueError("batcher state batch_index is out of range")
        self._indices_epoch = None

    def _resolved_index_mode(self) -> str:
        if not self.shuffle:
            return "permutation"
        if self.index_mode == "auto":
            if self.weights is None and len(self.dataset) > self.permutation_threshold:
                return "random"
            return "permutation"
        return self.index_mode

    def _next_random_batch(self):
        batch_size = min(self.batch_size, len(self.dataset))
        generator = torch.Generator()
        generator.manual_seed(
            self.seed
            + (self.epoch * 1_000_003)
            + (self.batch_index * self.world_size)
            + self.rank
        )
        if hasattr(self.dataset, "random_batch_indices"):
            indices = self.dataset.random_batch_indices(batch_size, generator)
        else:
            indices = torch.randint(
                low=0,
                high=len(self.dataset),
                size=(batch_size,),
                generator=generator,
            ).tolist()
        self.batch_index += 1
        batch = [self.dataset[index] for index in indices]
        return _collate_tensor_pairs(batch, pin_memory=self.pin_memory)

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


class DeviceBatchPrefetcher:
    """Move batches to device ahead of use while preserving resumable batch state."""

    def __init__(self, batcher, device: torch.device):
        self.batcher = batcher
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.use_cuda else None
        self._next_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._next_state: dict[str, Any] | None = None
        self._preload()

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._next_batch is None:
            raise StopIteration
        if self.use_cuda:
            assert self.stream is not None
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.stream)
            for tensor in self._next_batch:
                tensor.record_stream(current_stream)
        batch = self._next_batch
        self._preload()
        return batch

    def state_dict(self) -> dict[str, Any]:
        return self._next_state or self.batcher.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.batcher.load_state_dict(state)
        self._preload()

    def _preload(self) -> None:
        try:
            next_state = self.batcher.state_dict()
            x, y = next(self.batcher)
        except StopIteration:
            self._next_batch = None
            self._next_state = self.batcher.state_dict()
            return
        self._next_state = next_state
        if self.use_cuda:
            assert self.stream is not None
            with torch.cuda.stream(self.stream):
                self._next_batch = move_batch_to_device(x, y, self.device)
        else:
            self._next_batch = move_batch_to_device(x, y, self.device)


def move_batch_to_device(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return (
        x.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def _collate_tensor_pairs(
    batch,
    *,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    if pin_memory and torch.cuda.is_available():
        x = x.pin_memory()
        y = y.pin_memory()
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


def build_token_shards(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    shard_token_size: int = 1_000_000,
    add_bos: bool = True,
    add_eos: bool = True,
    read_chars: int = 1_000_000,
    corpus_manifest_path: str | Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Tokenize a corpus into disk shards without holding all tokens in memory."""
    if shard_token_size < 2:
        raise ValueError("shard_token_size must be at least 2")
    tokenizer = load_tokenizer(tokenizer_path)
    corpus_path = Path(corpus_path)
    manifest_path = Path(corpus_manifest_path) if corpus_manifest_path else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_token_shards(out_dir)

    shard_rows: list[dict[str, Any]] = []
    buffer: list[int] = []
    total_tokens = 0
    num_documents = 0
    document_boundary_tokens = False
    document_aligned_shards = False
    boundary_token_documents = 0
    oversized_documents = 0

    def append_tokens(ids: list[int]) -> None:
        nonlocal buffer, total_tokens
        cursor = 0
        while cursor < len(ids):
            remaining = shard_token_size - len(buffer)
            buffer.extend(ids[cursor: cursor + remaining])
            cursor += remaining
            if len(buffer) >= shard_token_size:
                flush()
        total_tokens += len(ids)

    def append_document_tokens(ids: list[int]) -> None:
        nonlocal buffer
        if buffer and len(ids) <= shard_token_size and len(buffer) + len(ids) > shard_token_size:
            flush()
        append_tokens(ids)

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        shard_index = len(shard_rows)
        shard_path = out_dir / f"tokens-{shard_index:06d}.pt"
        torch.save(torch.tensor(buffer, dtype=torch.long), shard_path)
        shard_rows.append({
            "index": shard_index,
            "path": str(shard_path),
            "num_tokens": len(buffer),
        })
        if progress and (len(shard_rows) == 1 or len(shard_rows) % 50 == 0):
            written_tokens = sum(int(row["num_tokens"]) for row in shard_rows)
            print(
                "base data: token shard build "
                f"shards={len(shard_rows):,} tokens={written_tokens:,}",
                flush=True,
            )
        buffer = []

    document_texts = _iter_manifest_document_texts(corpus_path, manifest_path)
    if document_texts is not None:
        document_aligned_shards = True
        for document in document_texts:
            text = document.strip()
            if not text:
                continue
            ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
            if (
                add_bos
                and add_eos
                and ids
                and ids[0] == tokenizer.bos_id
                and ids[-1] == tokenizer.eos_id
            ):
                boundary_token_documents += 1
            if len(ids) > shard_token_size:
                oversized_documents += 1
                document_aligned_shards = False
            append_document_tokens(ids)
            num_documents += 1
        document_boundary_tokens = num_documents > 0 and boundary_token_documents == num_documents
    else:
        if add_bos:
            append_tokens([tokenizer.bos_id])
        with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                chunk = handle.read(read_chars)
                if not chunk:
                    break
                append_tokens(tokenizer.encode(chunk, add_bos=False, add_eos=False))
        if add_eos:
            append_tokens([tokenizer.eos_id])
    flush()

    manifest = {
        "corpus_path": str(corpus_path),
        "tokenizer_path": str(tokenizer_path),
        "corpus_manifest_path": str(manifest_path) if manifest_path and document_boundary_tokens else None,
        "corpus_sha256": file_sha256(corpus_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "corpus_manifest_sha256": file_sha256(manifest_path) if manifest_path and document_boundary_tokens else None,
        "shard_token_size": shard_token_size,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "document_boundary_tokens": document_boundary_tokens,
        "document_aligned_shards": document_aligned_shards,
        "num_documents": num_documents if document_boundary_tokens else None,
        "boundary_token_documents": boundary_token_documents if document_texts is not None else None,
        "oversized_documents": oversized_documents if document_texts is not None else None,
        "num_tokens": total_tokens,
        "num_shards": len(shard_rows),
        "shards": shard_rows,
    }
    (out_dir / "token_shards_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _clear_stale_token_shards(out_dir: Path) -> None:
    """Remove prior generated token shards before writing a fresh manifest."""
    for stale_path in out_dir.glob("tokens-*.pt"):
        if stale_path.is_file():
            stale_path.unlink()
    manifest_path = out_dir / "token_shards_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def load_token_shards_manifest(
    cache_dir: str | Path,
    *,
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    shard_token_size: int,
    corpus_manifest_path: str | Path | None = None,
    add_bos: bool = True,
    add_eos: bool = True,
) -> dict[str, Any]:
    """Load an existing token-shard manifest and verify it matches this run."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "token_shards_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"token shard manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "corpus_path": str(Path(corpus_path)),
        "tokenizer_path": str(Path(tokenizer_path)),
        "corpus_manifest_path": str(Path(corpus_manifest_path)) if corpus_manifest_path else None,
        "corpus_sha256": file_sha256(corpus_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "shard_token_size": int(shard_token_size),
        "add_bos": bool(add_bos),
        "add_eos": bool(add_eos),
    }
    observed = {
        "corpus_path": str(manifest.get("corpus_path")),
        "tokenizer_path": str(manifest.get("tokenizer_path")),
        "corpus_manifest_path": (
            str(manifest.get("corpus_manifest_path"))
            if manifest.get("corpus_manifest_path")
            else None
        ),
        "corpus_sha256": manifest.get("corpus_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "corpus_manifest_sha256": manifest.get("corpus_manifest_sha256"),
        "shard_token_size": int(manifest.get("shard_token_size", 0)),
        "add_bos": bool(manifest.get("add_bos", True)),
        "add_eos": bool(manifest.get("add_eos", True)),
    }
    if observed != expected:
        raise ValueError(
            "token shard manifest does not match this run; rebuild the sharded dataset"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("token shard manifest contains no shards")
    for shard in shards:
        shard_path = Path(str(shard.get("path", "")))
        if not shard_path.exists():
            raise FileNotFoundError(f"token shard is missing: {shard_path}")
    return manifest


def load_sharded_token_split(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    context_size: int,
    cache_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    shard_token_size: int = 1_000_000,
    shard_cache_size: int = 2,
    rebuild: bool = True,
    corpus_manifest_path: str | Path | None = None,
    progress: bool = False,
) -> TokenSplitBundle:
    """Build token shards and return train/validation sharded window datasets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if rebuild:
        manifest = build_token_shards(
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            out_dir=cache_dir,
            shard_token_size=shard_token_size,
            corpus_manifest_path=corpus_manifest_path,
            progress=progress,
        )
    else:
        manifest = load_token_shards_manifest(
            cache_dir,
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            shard_token_size=shard_token_size,
            corpus_manifest_path=corpus_manifest_path,
        )
    shards = manifest["shards"]
    if len(shards) < 2:
        raise ValueError("sharded split requires at least two token shards")
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(shards), generator=generator).tolist()
    num_val = max(1, int(len(shards) * val_fraction))
    num_train = len(shards) - num_val
    if num_train < 1:
        raise ValueError("sharded split needs at least one train shard")
    train_shards = [shards[index] for index in indices[:num_train]]
    val_shards = [shards[index] for index in indices[num_train: num_train + num_val]]
    train_dataset = ShardedTokenWindowDataset(
        train_shards,
        context_size=context_size,
        max_cached_shards=shard_cache_size,
    )
    val_dataset = ShardedTokenWindowDataset(
        val_shards,
        context_size=context_size,
        max_cached_shards=shard_cache_size,
    )
    document_boundary_tokens = bool(manifest.get("document_boundary_tokens", False))
    document_aligned_shards = bool(manifest.get("document_aligned_shards", False))
    stats = {
        "num_tokens": train_dataset.stats().num_tokens + val_dataset.stats().num_tokens,
        "source_num_tokens": manifest["num_tokens"],
        "context_size": context_size,
        "num_sequences": len(train_dataset) + len(val_dataset),
        "split_mode": "sharded",
        "split_reason": "disk_token_shards",
        "packing": (
            "bos_eos_per_document_token_shards"
            if document_boundary_tokens
            else "streamed_token_shards"
        ),
        "document_boundary_tokens": document_boundary_tokens,
        "document_aligned_shards": document_aligned_shards,
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "train_tokens": train_dataset.stats().num_tokens,
        "val_tokens": val_dataset.stats().num_tokens,
        "num_documents": manifest.get("num_documents"),
        "train_documents": None,
        "val_documents": None,
        "val_document_paths": [],
        "num_shards": manifest["num_shards"],
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "shard_token_size": shard_token_size,
        "shard_cache_size": shard_cache_size,
        "shards_manifest": str(Path(cache_dir) / "token_shards_manifest.json"),
        "canary_values": [],
        "canaries_enabled": False,
        "canary_note": "disabled for sharded token split",
    }
    return TokenSplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stats=stats,
        train_text="",
        val_text="",
        canary_values=(),
    )


def build_packed_token_shards(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    context_size: int,
    shard_token_size: int = 1_000_000,
    add_bos: bool = True,
    add_eos: bool = True,
    corpus_manifest_path: str | Path | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    buffer_documents: int = 1024,
    progress: bool = False,
) -> dict[str, Any]:
    """Tokenize manifest documents into BOS-bestfit packed row shards."""
    if context_size < 2:
        raise ValueError("context_size must be at least 2")
    if shard_token_size < context_size + 1:
        raise ValueError("shard_token_size must be at least context_size + 1")
    if buffer_documents < 1:
        raise ValueError("buffer_documents must be at least 1")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    manifest_path = Path(corpus_manifest_path) if corpus_manifest_path else None
    if manifest_path is None:
        raise ValueError("packed base data requires corpus_manifest_path")
    documents = _manifest_documents(manifest_path)
    if len(documents) < 2:
        raise ValueError("packed base data requires at least two manifest documents")

    tokenizer = load_tokenizer(tokenizer_path)
    corpus_path = Path(corpus_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_packed_shards(out_dir)

    train_docs, val_docs = _split_manifest_documents(documents, val_fraction=val_fraction, seed=seed)
    rows_per_shard = max(1, shard_token_size // (context_size + 1))
    train_report = _write_packed_split(
        split_name="train",
        documents=train_docs,
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        out_dir=out_dir / "train",
        context_size=context_size,
        rows_per_shard=rows_per_shard,
        add_bos=add_bos,
        add_eos=add_eos,
        buffer_documents=buffer_documents,
        progress=progress,
    )
    val_report = _write_packed_split(
        split_name="val",
        documents=val_docs,
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        out_dir=out_dir / "val",
        context_size=context_size,
        rows_per_shard=rows_per_shard,
        add_bos=add_bos,
        add_eos=add_eos,
        buffer_documents=buffer_documents,
        progress=progress,
    )
    manifest = {
        "corpus_path": str(corpus_path),
        "tokenizer_path": str(Path(tokenizer_path)),
        "corpus_manifest_path": str(manifest_path),
        "corpus_sha256": file_sha256(corpus_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "corpus_manifest_sha256": file_sha256(manifest_path),
        "context_size": context_size,
        "shard_token_size": shard_token_size,
        "rows_per_shard": rows_per_shard,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "val_fraction": val_fraction,
        "seed": seed,
        "buffer_documents": buffer_documents,
        "packing": "bos_bestfit",
        "document_boundary_tokens": True,
        "document_split_before_packing": True,
        "document_aligned_shards": False,
        "num_documents": len(documents),
        "train": train_report,
        "val": val_report,
    }
    (out_dir / "packed_shards_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_packed_token_shards_manifest(
    cache_dir: str | Path,
    *,
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    context_size: int,
    shard_token_size: int,
    corpus_manifest_path: str | Path | None = None,
    add_bos: bool = True,
    add_eos: bool = True,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "packed_shards_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"packed shard manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "corpus_path": str(Path(corpus_path)),
        "tokenizer_path": str(Path(tokenizer_path)),
        "corpus_manifest_path": str(Path(corpus_manifest_path)) if corpus_manifest_path else None,
        "corpus_sha256": file_sha256(corpus_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "context_size": int(context_size),
        "shard_token_size": int(shard_token_size),
        "add_bos": bool(add_bos),
        "add_eos": bool(add_eos),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
    }
    observed = {
        "corpus_path": str(manifest.get("corpus_path")),
        "tokenizer_path": str(manifest.get("tokenizer_path")),
        "corpus_manifest_path": (
            str(manifest.get("corpus_manifest_path"))
            if manifest.get("corpus_manifest_path")
            else None
        ),
        "corpus_sha256": manifest.get("corpus_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "corpus_manifest_sha256": manifest.get("corpus_manifest_sha256"),
        "context_size": int(manifest.get("context_size", 0)),
        "shard_token_size": int(manifest.get("shard_token_size", 0)),
        "add_bos": bool(manifest.get("add_bos", True)),
        "add_eos": bool(manifest.get("add_eos", True)),
        "val_fraction": float(manifest.get("val_fraction", 0.0)),
        "seed": int(manifest.get("seed", 0)),
    }
    if observed != expected:
        raise ValueError(
            "packed shard manifest does not match this run; rebuild the packed dataset"
        )
    for split_name in ("train", "val"):
        split = manifest.get(split_name) or {}
        shards = split.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"packed shard manifest contains no {split_name} shards")
        for shard in shards:
            shard_path = Path(str(shard.get("path", "")))
            if not shard_path.exists():
                raise FileNotFoundError(f"packed token shard is missing: {shard_path}")
    return manifest


def load_packed_token_split(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    context_size: int,
    cache_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    shard_token_size: int = 1_000_000,
    shard_cache_size: int = 2,
    rebuild: bool = True,
    corpus_manifest_path: str | Path | None = None,
    progress: bool = False,
) -> TokenSplitBundle:
    """Build or load BOS-bestfit packed row shards for base training."""
    if rebuild:
        manifest = build_packed_token_shards(
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            out_dir=cache_dir,
            context_size=context_size,
            shard_token_size=shard_token_size,
            corpus_manifest_path=corpus_manifest_path,
            val_fraction=val_fraction,
            seed=seed,
            progress=progress,
        )
    else:
        manifest = load_packed_token_shards_manifest(
            cache_dir,
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            context_size=context_size,
            shard_token_size=shard_token_size,
            corpus_manifest_path=corpus_manifest_path,
            val_fraction=val_fraction,
            seed=seed,
        )
    train = manifest["train"]
    val = manifest["val"]
    train_dataset = PackedTokenRowDataset(
        train["shards"],
        context_size=context_size,
        max_cached_shards=shard_cache_size,
    )
    val_dataset = PackedTokenRowDataset(
        val["shards"],
        context_size=context_size,
        max_cached_shards=shard_cache_size,
    )
    train_tokens = int(train["num_rows"]) * context_size
    val_tokens = int(val["num_rows"]) * context_size
    stats = {
        "num_tokens": train_tokens + val_tokens,
        "source_num_tokens": int(train.get("source_tokens", 0)) + int(val.get("source_tokens", 0)),
        "context_size": context_size,
        "num_sequences": len(train_dataset) + len(val_dataset),
        "split_mode": "packed",
        "split_reason": "held_out_complete_documents_bos_bestfit",
        "packing": "bos_bestfit_base",
        "document_boundary_tokens": True,
        "document_split_before_packing": True,
        "document_aligned_shards": False,
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "num_documents": manifest.get("num_documents"),
        "train_documents": train.get("num_documents"),
        "val_documents": val.get("num_documents"),
        "val_document_paths": val.get("document_paths", [])[:100],
        "num_shards": int(train.get("num_shards", 0)) + int(val.get("num_shards", 0)),
        "train_shards": int(train.get("num_shards", 0)),
        "val_shards": int(val.get("num_shards", 0)),
        "shard_token_size": shard_token_size,
        "shard_cache_size": shard_cache_size,
        "rows_per_shard": manifest.get("rows_per_shard"),
        "shards_manifest": str(Path(cache_dir) / "packed_shards_manifest.json"),
        "packed_source_tokens": int(train.get("source_tokens", 0)) + int(val.get("source_tokens", 0)),
        "packed_train_dropped_tokens": train.get("dropped_tokens"),
        "packed_val_dropped_tokens": val.get("dropped_tokens"),
        "packing_efficiency": _safe_ratio(
            train_tokens + val_tokens,
            int(train.get("source_tokens", 0)) + int(val.get("source_tokens", 0)),
        ),
        "canary_values": [],
        "canaries_enabled": False,
        "canary_note": "disabled for packed base split",
    }
    return TokenSplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stats=stats,
        train_text="",
        val_text="",
        canary_values=(),
    )


def _clear_stale_packed_shards(out_dir: Path) -> None:
    for stale_path in out_dir.glob("**/packed-*.pt"):
        if stale_path.is_file():
            stale_path.unlink()
    manifest_path = out_dir / "packed_shards_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _split_manifest_documents(
    documents: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(documents), generator=generator).tolist()
    num_val = max(1, int(len(documents) * val_fraction))
    val_indices = set(indices[:num_val])
    train_docs = [document for index, document in enumerate(documents) if index not in val_indices]
    val_docs = [document for index, document in enumerate(documents) if index in val_indices]
    if not train_docs or not val_docs:
        raise ValueError("packed split needs at least one train and one validation document")
    return train_docs, val_docs


def _write_packed_split(
    *,
    split_name: str,
    documents: list[dict[str, Any]],
    corpus_path: Path,
    tokenizer,
    out_dir: Path,
    context_size: int,
    rows_per_shard: int,
    add_bos: bool,
    add_eos: bool,
    buffer_documents: int,
    progress: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    row_width = context_size + 1
    source_tokens = 0
    dropped_tokens = 0
    row_buffer: list[list[int]] = []
    shard_rows: list[dict[str, Any]] = []
    doc_buffer: list[list[int]] = []
    text_iter = iter(_iter_corpus_document_slices(corpus_path, documents))
    exhausted = False

    def flush_rows() -> None:
        nonlocal row_buffer
        if not row_buffer:
            return
        shard_index = len(shard_rows)
        shard_path = out_dir / f"packed-{shard_index:06d}.pt"
        rows = torch.tensor(row_buffer, dtype=torch.long)
        torch.save(rows, shard_path)
        shard_rows.append({
            "index": shard_index,
            "path": str(shard_path),
            "num_rows": int(rows.size(0)),
            "num_tokens": int(rows.numel()),
        })
        if progress and (len(shard_rows) == 1 or len(shard_rows) % 50 == 0):
            written_rows = sum(int(row["num_rows"]) for row in shard_rows)
            print(
                "base data: packed shard build "
                f"{split_name} shards={len(shard_rows):,} rows={written_rows:,}",
                flush=True,
            )
        row_buffer = []

    def refill() -> None:
        nonlocal exhausted, source_tokens, dropped_tokens
        while len(doc_buffer) < buffer_documents and not exhausted:
            try:
                text = next(text_iter)
            except StopIteration:
                exhausted = True
                return
            ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
            if len(ids) < 2:
                continue
            source_tokens += len(ids)
            while len(ids) > row_width:
                row_buffer.append(ids[:row_width])
                ids = ids[row_width:]
                if len(row_buffer) >= rows_per_shard:
                    flush_rows()
            if len(ids) >= 2:
                doc_buffer.append(ids)
            else:
                dropped_tokens += len(ids)

    refill()
    while doc_buffer or not exhausted:
        refill()
        if not doc_buffer:
            break
        row: list[int] = []
        while len(row) < row_width and (doc_buffer or not exhausted):
            refill()
            if not doc_buffer:
                break
            remaining = row_width - len(row)
            best_idx = -1
            best_len = 0
            for index, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = index
                    best_len = doc_len
            if best_idx >= 0:
                row.extend(doc_buffer.pop(best_idx))
            else:
                shortest_idx = min(range(len(doc_buffer)), key=lambda idx: len(doc_buffer[idx]))
                doc = doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])
                remainder = doc[remaining:]
                if len(remainder) >= 2:
                    doc_buffer.append(remainder)
                else:
                    dropped_tokens += len(remainder)
        if len(row) == row_width:
            row_buffer.append(row)
            if len(row_buffer) >= rows_per_shard:
                flush_rows()
        else:
            dropped_tokens += len(row)
            break
    flush_rows()
    if not shard_rows:
        raise ValueError(f"packed {split_name} split produced no rows")
    return {
        "num_documents": len(documents),
        "document_paths": [str(document.get("path", "")) for document in documents[:1000]],
        "source_tokens": source_tokens,
        "dropped_tokens": dropped_tokens,
        "num_rows": sum(int(row["num_rows"]) for row in shard_rows),
        "num_shards": len(shard_rows),
        "shards": shard_rows,
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _iter_manifest_document_texts(corpus_path: Path, manifest_path: Path | None):
    if manifest_path is None:
        return None
    documents = _manifest_documents(manifest_path)
    if not documents:
        return None
    return _iter_corpus_document_slices(corpus_path, documents)


def _iter_corpus_document_slices(corpus_path: Path, documents: list[dict[str, Any]]):
    """Yield manifest document text without loading the full corpus into memory."""
    ordered = sorted(documents, key=lambda document: int(document["char_start"]))

    def iterator():
        position = 0
        with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
            for document in ordered:
                start = max(0, int(document["char_start"]))
                end = max(start, int(document["char_end"]))
                if start < position:
                    raise ValueError("corpus manifest document ranges overlap or move backwards")
                _consume_chars(handle, start - position)
                position = start
                text = _read_chars(handle, end - start)
                position += len(text)
                text = text.strip()
                if text:
                    yield text

    return iterator()


def _consume_chars(handle, count: int, chunk_size: int = 1_000_000) -> None:
    remaining = count
    while remaining > 0:
        chunk = handle.read(min(chunk_size, remaining))
        if not chunk:
            return
        remaining -= len(chunk)


def _read_chars(handle, count: int, chunk_size: int = 1_000_000) -> str:
    remaining = count
    parts: list[str] = []
    while remaining > 0:
        chunk = handle.read(min(chunk_size, remaining))
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return "".join(parts)


def make_dataloader(
    dataset: TokenWindowDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    pin_memory: bool = False,
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
        pin_memory=pin_memory,
    )


def make_resumable_batcher(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    weights: torch.Tensor | None = None,
    index_mode: str = "auto",
    permutation_threshold: int = 5_000_000,
    pin_memory: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> ResumableBatcher:
    """Create a deterministic train iterator that can save and load position."""
    return ResumableBatcher(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        weights=weights,
        index_mode=index_mode,
        permutation_threshold=permutation_threshold,
        pin_memory=pin_memory,
        rank=rank,
        world_size=world_size,
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
