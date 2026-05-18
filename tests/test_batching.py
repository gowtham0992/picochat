import json
from pathlib import Path

import pytest
import torch

from picochat.batching import (
    DeviceBatchPrefetcher,
    PackedTokenRowDataset,
    ShardedTokenWindowDataset,
    TokenWindowDataset,
    build_packed_token_shards,
    build_token_shards,
    load_packed_token_shards_manifest,
    load_packed_token_split,
    load_sharded_token_split,
    load_token_shards_manifest,
    load_token_dataset,
    load_token_split,
    make_dataloader,
    make_resumable_batcher,
    split_dataset,
)
from picochat.tokenizer import CharTokenizer


def test_token_window_dataset_returns_shifted_targets():
    dataset = TokenWindowDataset([1, 2, 3, 4, 5], context_size=3)

    x, y = dataset[0]

    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]
    assert len(dataset) == 2


def test_token_window_dataset_rejects_short_stream():
    with pytest.raises(ValueError):
        TokenWindowDataset([1, 2, 3], context_size=3)


def test_load_token_dataset(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    corpus_path.write_text("hello world", encoding="utf-8")
    tokenizer = CharTokenizer.train(["hello world"])
    tokenizer.save(tokenizer_path)

    dataset = load_token_dataset(corpus_path, tokenizer_path, context_size=4)

    assert dataset.stats().num_tokens == len(tokenizer.encode("hello world", add_bos=True, add_eos=True))
    assert dataset[0][0].shape[0] == 4


def test_load_sharded_token_split_uses_disk_shards(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    cache_dir = tmp_path / "shards"
    text = "alpha beta gamma delta epsilon\n" * 30
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    split = load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=40,
    )
    x, y = split.train_dataset[0]

    assert split.stats["split_mode"] == "sharded"
    assert split.stats["num_shards"] > 1
    assert (cache_dir / "token_shards_manifest.json").exists()
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_load_packed_token_split_uses_document_holdout_and_bos_rows(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    cache_dir = tmp_path / "packed"
    docs = [
        "alpha packed document " * 4,
        "beta packed document " * 4,
        "gamma packed document " * 4,
        "delta packed document " * 4,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus_text}\n", encoding="utf-8")
    CharTokenizer.train([corpus_text]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    split = load_packed_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=18,
        corpus_manifest_path=manifest_path,
    )
    x, y = split.train_dataset[0]

    assert split.stats["split_mode"] == "packed"
    assert split.stats["split_reason"] == "held_out_complete_documents_bos_bestfit"
    assert split.stats["packing"] == "bos_bestfit_base"
    assert split.stats["document_boundary_tokens"] is True
    assert split.stats["train_documents"] == 3
    assert split.stats["val_documents"] == 1
    assert split.stats["train_shards"] > 0
    assert (cache_dir / "packed_shards_manifest.json").exists()
    assert isinstance(split.train_dataset, PackedTokenRowDataset)
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_packed_token_shards_do_not_drop_long_documents(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    cache_dir = tmp_path / "packed"
    docs = [
        "long document alpha " * 40,
        "long document beta " * 40,
        "long document gamma " * 40,
        "long document delta " * 40,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus_text}\n", encoding="utf-8")
    CharTokenizer.train([corpus_text]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    manifest = build_packed_token_shards(
        corpus_path,
        tokenizer_path,
        cache_dir,
        context_size=16,
        shard_token_size=68,
        corpus_manifest_path=manifest_path,
        val_fraction=0.25,
        seed=1,
    )

    source_tokens = manifest["train"]["source_tokens"] + manifest["val"]["source_tokens"]
    dropped_tokens = manifest["train"]["dropped_tokens"] + manifest["val"]["dropped_tokens"]
    assert source_tokens > 1000
    assert dropped_tokens < source_tokens * 0.05


def test_load_sharded_token_split_can_reuse_existing_manifest(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    cache_dir = tmp_path / "shards"
    text = "reuse token shards without every DDP rank rebuilding\n" * 30
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    built = load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=48,
        rebuild=True,
    )
    reused = load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=48,
        rebuild=False,
    )

    assert reused.stats["num_shards"] == built.stats["num_shards"]
    assert reused.stats["train_shards"] == built.stats["train_shards"]


def test_load_sharded_token_split_preserves_document_boundaries_from_manifest(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    cache_dir = tmp_path / "shards"
    docs = [
        "alpha document " * 8,
        "beta document " * 8,
        "gamma document " * 8,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus_text}\n", encoding="utf-8")
    CharTokenizer.train([corpus_text]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    split = load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.34,
        seed=1,
        shard_token_size=32,
        corpus_manifest_path=manifest_path,
    )

    tokenizer = CharTokenizer.load(tokenizer_path)
    shard_manifest = json.loads((cache_dir / "token_shards_manifest.json").read_text(encoding="utf-8"))
    tokens = []
    for shard in shard_manifest["shards"]:
        tokens.extend(torch.load(shard["path"], weights_only=True).tolist())
    assert split.stats["document_boundary_tokens"] is True
    assert split.stats["document_aligned_shards"] is True
    assert split.stats["packing"] == "bos_eos_per_document_token_shards"
    assert split.stats["num_documents"] == 3
    assert shard_manifest["corpus_manifest_path"] == str(manifest_path)
    assert tokens.count(tokenizer.bos_id) == 3
    assert tokens.count(tokenizer.eos_id) == 3


def test_manifest_token_shards_keep_documents_together_when_they_fit(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    cache_dir = tmp_path / "shards"
    docs = [
        "alpha source document " * 4,
        "beta source document " * 4,
        "gamma source document " * 4,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus_text}\n", encoding="utf-8")
    CharTokenizer.train([corpus_text]).save(tokenizer_path)
    tokenizer = CharTokenizer.load(tokenizer_path)
    tokenized_docs = [
        tokenizer.encode(doc.strip(), add_bos=True, add_eos=True)
        for doc in docs
    ]
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    manifest = build_token_shards(
        corpus_path,
        tokenizer_path,
        cache_dir,
        shard_token_size=max(len(ids) for ids in tokenized_docs) + 1,
        corpus_manifest_path=manifest_path,
    )

    assert manifest["document_aligned_shards"] is True
    assert manifest["num_shards"] == len(docs)
    for shard in manifest["shards"]:
        ids = torch.load(shard["path"], weights_only=True).tolist()
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id
        assert ids.count(tokenizer.bos_id) == 1
        assert ids.count(tokenizer.eos_id) == 1


def test_manifest_token_shards_stream_document_slices(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    cache_dir = tmp_path / "shards"
    docs = [
        "stream alpha " * 8,
        "stream beta " * 8,
        "stream gamma " * 8,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(corpus_text, encoding="utf-8")
    CharTokenizer.train([corpus_text]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    original_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if self == corpus_path:
            raise AssertionError("token shard builder should not read the full corpus")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    split = load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.34,
        seed=1,
        shard_token_size=32,
        corpus_manifest_path=manifest_path,
    )

    assert split.stats["document_boundary_tokens"] is True
    assert split.stats["num_documents"] == 3


def test_building_token_shards_removes_stale_generated_shards(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    cache_dir = tmp_path / "shards"
    text = "stale shards should not survive a rebuild\n" * 40
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=24,
        rebuild=True,
    )
    stale_path = cache_dir / "tokens-999999.pt"
    torch.save(torch.tensor([1, 2, 3]), stale_path)
    keep_path = cache_dir / "notes.txt"
    keep_path.write_text("operator note", encoding="utf-8")

    rebuilt = load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=48,
        rebuild=True,
    )

    manifest_shards = {
        Path(row["path"]).name
        for row in json.loads((cache_dir / "token_shards_manifest.json").read_text(encoding="utf-8"))["shards"]
    }
    disk_shards = {path.name for path in cache_dir.glob("tokens-*.pt")}
    assert stale_path.name not in disk_shards
    assert disk_shards == manifest_shards
    assert keep_path.read_text(encoding="utf-8") == "operator note"
    assert rebuilt.stats["num_shards"] == len(manifest_shards)


def test_load_token_shards_manifest_rejects_mismatched_run(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    cache_dir = tmp_path / "shards"
    text = "manifest guards stale token shards\n" * 30
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)
    load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=48,
    )

    with pytest.raises(ValueError, match="manifest does not match"):
        load_token_shards_manifest(
            cache_dir,
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            shard_token_size=64,
        )


def test_load_token_shards_manifest_rejects_changed_corpus_content(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    cache_dir = tmp_path / "shards"
    text = "manifest hashes stale corpus content\n" * 30
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)
    load_sharded_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=48,
    )

    corpus_path.write_text(text + "changed after shard build\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest does not match"):
        load_token_shards_manifest(
            cache_dir,
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            shard_token_size=48,
        )


def test_load_packed_token_shards_manifest_rejects_changed_tokenizer_content(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    cache_dir = tmp_path / "packed"
    docs = [
        "alpha packed guard " * 4,
        "beta packed guard " * 4,
        "gamma packed guard " * 4,
        "delta packed guard " * 4,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus_text}\n", encoding="utf-8")
    CharTokenizer.train([corpus_text]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")
    load_packed_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        cache_dir=cache_dir,
        val_fraction=0.25,
        seed=1,
        shard_token_size=18,
        corpus_manifest_path=manifest_path,
    )

    CharTokenizer.train([corpus_text + "new token vocabulary"]).save(tokenizer_path)

    with pytest.raises(ValueError, match="packed shard manifest does not match"):
        load_packed_token_shards_manifest(
            cache_dir,
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            context_size=8,
            shard_token_size=18,
            corpus_manifest_path=manifest_path,
            val_fraction=0.25,
            seed=1,
        )


def test_sharded_token_dataset_keeps_lru_shard_cache(tmp_path, monkeypatch):
    shard_rows = []
    for index in range(3):
        path = tmp_path / f"tokens-{index}.pt"
        torch.save(torch.arange(index * 10, index * 10 + 10), path)
        shard_rows.append({"index": index, "path": str(path), "num_tokens": 10})
    real_load = torch.load
    load_counts = {row["path"]: 0 for row in shard_rows}

    def counting_load(path, *args, **kwargs):
        load_counts[str(path)] += 1
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    dataset = ShardedTokenWindowDataset(
        shard_rows,
        context_size=2,
        max_cached_shards=2,
    )

    dataset[0]
    dataset[8]
    dataset[0]
    dataset[16]
    dataset[8]

    assert load_counts[shard_rows[0]["path"]] == 1
    assert load_counts[shard_rows[1]["path"]] == 2
    assert load_counts[shard_rows[2]["path"]] == 1


def test_make_dataloader_batches_examples():
    dataset = TokenWindowDataset(list(range(20)), context_size=4)
    loader = make_dataloader(dataset, batch_size=3, shuffle=False)

    x, y = next(iter(loader))

    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert x[0].tolist() == [0, 1, 2, 3]
    assert y[0].tolist() == [1, 2, 3, 4]


def test_resumable_batcher_restores_batch_position():
    dataset = TokenWindowDataset(list(range(20)), context_size=4)
    first = make_resumable_batcher(dataset, batch_size=3, shuffle=True, seed=7)
    next(first)
    state = first.state_dict()
    expected_x, expected_y = next(first)

    resumed = make_resumable_batcher(dataset, batch_size=3, shuffle=True, seed=7)
    resumed.load_state_dict(state)
    resumed_x, resumed_y = next(resumed)

    assert resumed_x.tolist() == expected_x.tolist()
    assert resumed_y.tolist() == expected_y.tolist()


def test_resumable_batcher_uses_random_sampling_for_large_auto_mode():
    class LargeSyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, index):
            return torch.tensor([index]), torch.tensor([index + 1])

    dataset = LargeSyntheticDataset()
    first = make_resumable_batcher(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=11,
        permutation_threshold=10,
    )

    first_x, first_y = next(first)
    state = first.state_dict()
    expected_x, expected_y = next(first)

    resumed = make_resumable_batcher(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=11,
        permutation_threshold=10,
    )
    resumed.load_state_dict(state)
    resumed_x, resumed_y = next(resumed)

    assert state["resolved_index_mode"] == "random"
    assert first._indices == []
    assert first_x.shape == (4, 1)
    assert first_y.tolist() == (first_x + 1).tolist()
    assert resumed_x.tolist() == expected_x.tolist()
    assert resumed_y.tolist() == expected_y.tolist()


def test_resumable_batcher_shards_permutation_batches_by_rank():
    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, index):
            return torch.tensor([index]), torch.tensor([index + 1])

    dataset = SyntheticDataset()
    rank0 = make_resumable_batcher(
        dataset,
        batch_size=4,
        shuffle=False,
        seed=11,
        rank=0,
        world_size=2,
    )
    rank1 = make_resumable_batcher(
        dataset,
        batch_size=4,
        shuffle=False,
        seed=11,
        rank=1,
        world_size=2,
    )

    x0, _ = next(rank0)
    x1, _ = next(rank1)

    assert x0.squeeze(-1).tolist() == [0, 1, 2, 3]
    assert x1.squeeze(-1).tolist() == [4, 5, 6, 7]
    assert set(x0.squeeze(-1).tolist()).isdisjoint(x1.squeeze(-1).tolist())


def test_resumable_batcher_random_mode_uses_rank_distinct_streams():
    class LargeSyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, index):
            return torch.tensor([index]), torch.tensor([index + 1])

    dataset = LargeSyntheticDataset()
    rank0 = make_resumable_batcher(
        dataset,
        batch_size=8,
        shuffle=True,
        seed=11,
        permutation_threshold=10,
        rank=0,
        world_size=2,
    )
    rank1 = make_resumable_batcher(
        dataset,
        batch_size=8,
        shuffle=True,
        seed=11,
        permutation_threshold=10,
        rank=1,
        world_size=2,
    )

    x0, _ = next(rank0)
    x1, _ = next(rank1)

    assert rank0.state_dict()["resolved_index_mode"] == "random"
    assert rank0.state_dict()["world_size"] == 2
    assert x0.tolist() != x1.tolist()


def test_resumable_batcher_resume_rejects_world_size_change():
    dataset = TokenWindowDataset(list(range(30)), context_size=4)
    first = make_resumable_batcher(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=7,
        rank=0,
        world_size=2,
    )
    state = first.state_dict()
    resumed = make_resumable_batcher(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=7,
        rank=0,
        world_size=1,
    )

    with pytest.raises(ValueError, match="world_size"):
        resumed.load_state_dict(state)


def test_resumable_batcher_ddp_resume_state_is_rank_independent():
    dataset = TokenWindowDataset(list(range(80)), context_size=4)
    rank0 = make_resumable_batcher(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=7,
        rank=0,
        world_size=2,
    )
    rank1 = make_resumable_batcher(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=7,
        rank=1,
        world_size=2,
    )

    next(rank0)
    next(rank1)
    shared_state = rank0.state_dict()
    expected_rank1_x, expected_rank1_y = next(rank1)

    resumed_rank1 = make_resumable_batcher(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=7,
        rank=1,
        world_size=2,
    )
    resumed_rank1.load_state_dict(shared_state)
    observed_rank1_x, observed_rank1_y = next(resumed_rank1)

    assert shared_state["rank"] == 0
    assert resumed_rank1.state_dict()["rank"] == 1
    assert observed_rank1_x.tolist() == expected_rank1_x.tolist()
    assert observed_rank1_y.tolist() == expected_rank1_y.tolist()


def test_sharded_random_batches_stay_within_one_shard(tmp_path):
    shards = []
    for index, start in enumerate((0, 100)):
        path = tmp_path / f"tokens-{index:06d}.pt"
        tokens = torch.arange(start, start + 24, dtype=torch.long)
        torch.save(tokens, path)
        shards.append({"index": index, "path": str(path), "num_tokens": len(tokens)})
    dataset = ShardedTokenWindowDataset(shards, context_size=4)
    generator = torch.Generator().manual_seed(123)

    for _ in range(10):
        indices = dataset.random_batch_indices(batch_size=8, generator=generator)
        assert len(indices) == 8
        assert max(indices) < len(dataset)
        assert min(indices) >= 0
        assert all(index < 20 for index in indices) or all(index >= 20 for index in indices)


def test_device_batch_prefetcher_preserves_resume_position():
    dataset = TokenWindowDataset(list(range(30)), context_size=4)
    batcher = make_resumable_batcher(dataset, batch_size=3, shuffle=True, seed=9)
    prefetcher = DeviceBatchPrefetcher(batcher, torch.device("cpu"))

    first_x, first_y = next(prefetcher)
    state = prefetcher.state_dict()
    expected_x, expected_y = next(prefetcher)

    resumed_batcher = make_resumable_batcher(dataset, batch_size=3, shuffle=True, seed=9)
    resumed_prefetcher = DeviceBatchPrefetcher(resumed_batcher, torch.device("cpu"))
    resumed_prefetcher.load_state_dict(state)
    resumed_x, resumed_y = next(resumed_prefetcher)

    assert first_y.tolist() == (first_x + 1).tolist()
    assert resumed_x.tolist() == expected_x.tolist()
    assert resumed_y.tolist() == expected_y.tolist()


def test_split_dataset_is_deterministic():
    dataset = TokenWindowDataset(list(range(30)), context_size=4)

    train_a, val_a = split_dataset(dataset, val_fraction=0.2, seed=123)
    train_b, val_b = split_dataset(dataset, val_fraction=0.2, seed=123)

    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert len(train_a) + len(val_a) == len(dataset)
    assert len(val_a) >= 1


def test_load_token_split_can_hold_out_complete_documents(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    docs = [
        "alpha " * 20,
        "beta " * 20,
        "gamma " * 20,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus_text}\n", encoding="utf-8")
    CharTokenizer.train([corpus_text]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    split = load_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        val_fraction=0.34,
        seed=1,
        split_mode="document",
        corpus_manifest_path=manifest_path,
    )

    assert split.stats["split_mode"] == "document"
    assert split.stats["packing"] == "bos_eos_per_document"
    assert split.stats["train_documents"] == 2
    assert split.stats["val_documents"] == 1
    assert split.val_text.strip()
    assert split.val_text not in split.train_text
    train_tokens = split.train_dataset.tokens.tolist()
    val_tokens = split.val_dataset.tokens.tolist()
    tokenizer = CharTokenizer.load(tokenizer_path)
    assert train_tokens.count(tokenizer.bos_id) == 2
    assert val_tokens.count(tokenizer.bos_id) == 1


def test_load_token_split_places_canaries_only_in_train_text(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    docs = [
        "alpha story " * 20,
        "beta story " * 20,
        "gamma story " * 20,
    ]
    corpus_text = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus_text}\n", encoding="utf-8")
    CharTokenizer.train([corpus_text, "pico-canary-0042-00"]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    split = load_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        val_fraction=0.34,
        seed=1,
        split_mode="document",
        corpus_manifest_path=manifest_path,
        canary_values=("pico-canary-0042-00",),
    )

    assert split.canary_values == ("pico-canary-0042-00",)
    assert "pico-canary-0042-00" in split.train_text
    assert "pico-canary-0042-00" not in split.val_text
    assert split.stats["canaries_enabled"] is True


def test_load_token_split_falls_back_to_window_without_document_manifest(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    text = "hello picochat " * 20
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    split = load_token_split(
        corpus_path,
        tokenizer_path,
        context_size=8,
        split_mode="document",
        corpus_manifest_path=tmp_path / "missing.json",
    )

    assert split.stats["split_mode"] == "window"
    assert split.stats["split_reason"] == "document_split_unavailable"
