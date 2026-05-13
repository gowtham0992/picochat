import json

import pytest

from picochat.batching import TokenWindowDataset, load_token_dataset, load_token_split, make_dataloader, split_dataset
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


def test_make_dataloader_batches_examples():
    dataset = TokenWindowDataset(list(range(20)), context_size=4)
    loader = make_dataloader(dataset, batch_size=3, shuffle=False)

    x, y = next(iter(loader))

    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert x[0].tolist() == [0, 1, 2, 3]
    assert y[0].tolist() == [1, 2, 3, 4]


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
