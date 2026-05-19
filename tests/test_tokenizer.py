import pytest

from picochat.tokenizer import (
    BPETokenizer,
    ByteTokenizer,
    CharTokenizer,
    HuggingFaceBPETokenizer,
    SPECIAL_TOKENS,
    load_tokenizer,
    token_byte_lengths,
    train_tokenizer,
)


def test_train_encode_decode_roundtrip():
    tokenizer = CharTokenizer.train(["hello", "world"])

    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hello"


def test_unknown_character_uses_unk():
    tokenizer = CharTokenizer.train(["abc"])

    ids = tokenizer.encode("az")

    assert ids == [tokenizer.token_to_id["a"], tokenizer.unk_id]


def test_vocab_size_limit_keeps_special_tokens():
    tokenizer = CharTokenizer.train(["abcde"], vocab_size=len(SPECIAL_TOKENS) + 2)

    assert len(tokenizer) == len(SPECIAL_TOKENS) + 2
    for token in SPECIAL_TOKENS:
        assert token in tokenizer.token_to_id


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = CharTokenizer.train(["tiny chat"])

    tokenizer.save(path)
    loaded = CharTokenizer.load(path)

    assert loaded.token_to_id == tokenizer.token_to_id
    assert loaded.decode(loaded.encode("tiny")) == "tiny"


def test_load_tokenizer_dispatches_by_type(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = CharTokenizer.train(["dispatch"])
    tokenizer.save(path)

    loaded = load_tokenizer(path)

    assert isinstance(loaded, CharTokenizer)
    assert loaded.decode(loaded.encode("dispatch")) == "dispatch"


def test_byte_tokenizer_roundtrip_ascii_and_unicode(tmp_path):
    path = tmp_path / "byte-tokenizer.json"
    tokenizer = ByteTokenizer.train(["hello café"])

    ids = tokenizer.encode("hello café", add_bos=True, add_eos=True)
    tokenizer.save(path)
    loaded = load_tokenizer(path)

    assert len(tokenizer) == len(SPECIAL_TOKENS) + 256
    assert tokenizer.stats().tokenizer_type == "byte"
    assert loaded.decode(ids) == "hello café"


def test_train_tokenizer_factory_supports_byte():
    tokenizer = train_tokenizer("byte", ["anything"])

    assert isinstance(tokenizer, ByteTokenizer)
    assert tokenizer.decode(tokenizer.encode("Picochat")) == "Picochat"


def test_bpe_tokenizer_learns_merges_and_roundtrips():
    text = "low lower lowest low low"
    tokenizer = BPETokenizer.train([text], vocab_size=len(SPECIAL_TOKENS) + 20, min_freq=2)

    ids = tokenizer.encode("low lowest", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "low lowest"
    assert tokenizer.stats().tokenizer_type == "bpe"
    assert tokenizer.pretokenizer == "regex"
    assert tokenizer.merges
    assert len(tokenizer.encode("low low")) < len(CharTokenizer.train([text]).encode("low low"))


def test_bpe_regex_pretokenizer_prevents_cross_boundary_merges():
    text = "cat! cat! cat!"
    regex_tokenizer = BPETokenizer.train([text], vocab_size=64, min_freq=2, pretokenizer="regex")
    legacy_tokenizer = BPETokenizer.train([text], vocab_size=64, min_freq=2, pretokenizer="char")

    assert regex_tokenizer.decode(regex_tokenizer.encode(text)) == text
    assert legacy_tokenizer.decode(legacy_tokenizer.encode(text)) == text
    assert "cat!" not in regex_tokenizer.token_to_id
    assert "cat!" in legacy_tokenizer.token_to_id


def test_bpe_save_load_roundtrip(tmp_path):
    path = tmp_path / "bpe-tokenizer.json"
    tokenizer = BPETokenizer.train(["picochat picochat learns"], vocab_size=32, min_freq=2)

    tokenizer.save(path)
    loaded = load_tokenizer(path)

    assert isinstance(loaded, BPETokenizer)
    assert loaded.merges == tokenizer.merges
    assert loaded.pretokenizer == tokenizer.pretokenizer
    assert loaded.decode(loaded.encode("picochat learns")) == "picochat learns"


def test_bpe_loads_legacy_tokenizer_as_char_pretokenized(tmp_path):
    path = tmp_path / "legacy-bpe-tokenizer.json"
    tokenizer = BPETokenizer.train(["legacy legacy"], vocab_size=32, min_freq=2, pretokenizer="char")
    tokenizer.save(path)
    data = path.read_text(encoding="utf-8")
    path.write_text(data.replace(',\n  "pretokenizer": "char"', ""), encoding="utf-8")

    loaded = load_tokenizer(path)

    assert isinstance(loaded, BPETokenizer)
    assert loaded.pretokenizer == "char"
    assert loaded.decode(loaded.encode("legacy")) == "legacy"


def test_train_tokenizer_factory_supports_bpe():
    tokenizer = train_tokenizer("bpe", ["tiny tiny stories"], vocab_size=32, min_freq=2)

    assert isinstance(tokenizer, BPETokenizer)
    assert tokenizer.pretokenizer == "regex"
    assert tokenizer.decode(tokenizer.encode("tiny stories")) == "tiny stories"


def test_hf_bpe_tokenizer_roundtrips_and_saves(tmp_path):
    pytest.importorskip("tokenizers")
    path = tmp_path / "hf-bpe-tokenizer.json"
    tokenizer = HuggingFaceBPETokenizer.train(
        ["picochat trains fast tokenizers\nhello café\n" * 20],
        vocab_size=300,
        min_freq=1,
    )

    ids = tokenizer.encode("hello café", add_bos=True, add_eos=True)
    tokenizer.save(path)
    loaded = load_tokenizer(path)

    assert isinstance(loaded, HuggingFaceBPETokenizer)
    assert tokenizer.stats().tokenizer_type == "hf_bpe"
    assert loaded.decode(ids) == "hello café"
    assert loaded.pretokenizer == "regex"
    assert "hf_tokenizer" in path.read_text(encoding="utf-8")


def test_train_tokenizer_factory_supports_hf_bpe():
    pytest.importorskip("tokenizers")
    tokenizer = train_tokenizer(
        "hf_bpe",
        ["tiny tiny stories\n" * 20],
        vocab_size=300,
        min_freq=1,
    )

    assert isinstance(tokenizer, HuggingFaceBPETokenizer)
    assert tokenizer.decode(tokenizer.encode("tiny stories")) == "tiny stories"


def test_byte_tokenizer_rejects_custom_vocab_size():
    with pytest.raises(ValueError, match="fixed vocab"):
        ByteTokenizer.train(["hello"], vocab_size=100)


def test_token_byte_lengths_count_text_bytes_not_specials():
    char_tokenizer = CharTokenizer.train(["éa"])
    byte_tokenizer = ByteTokenizer.train(["éa"])
    bpe_tokenizer = BPETokenizer.train(["hello hello"], vocab_size=20, min_freq=2)

    char_lengths = token_byte_lengths(char_tokenizer)
    byte_lengths = token_byte_lengths(byte_tokenizer)
    bpe_lengths = token_byte_lengths(bpe_tokenizer)

    assert char_lengths[char_tokenizer.bos_id] == 0
    assert char_lengths[char_tokenizer.token_to_id["é"]] == 2
    assert byte_lengths[byte_tokenizer.bos_id] == 0
    assert byte_lengths[byte_tokenizer.token_to_id["<byte:c3>"]] == 1
    assert bpe_lengths[bpe_tokenizer.bos_id] == 0
    assert any(length > 1 for length in bpe_lengths)


def test_hf_bpe_token_byte_lengths_count_text_bytes_not_specials():
    pytest.importorskip("tokenizers")
    tokenizer = HuggingFaceBPETokenizer.train(["hello hello café\n" * 20], vocab_size=300)
    lengths = token_byte_lengths(tokenizer)

    assert lengths[tokenizer.bos_id] == 0
    assert any(length > 1 for length in lengths)
