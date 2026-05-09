import pytest

from picochat.batching import TokenWindowDataset, load_token_dataset, make_dataloader, split_dataset
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
