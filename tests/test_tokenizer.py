import pytest

from picochat.tokenizer import (
    ByteTokenizer,
    CharTokenizer,
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


def test_byte_tokenizer_rejects_custom_vocab_size():
    with pytest.raises(ValueError, match="fixed vocab"):
        ByteTokenizer.train(["hello"], vocab_size=100)


def test_token_byte_lengths_count_text_bytes_not_specials():
    char_tokenizer = CharTokenizer.train(["éa"])
    byte_tokenizer = ByteTokenizer.train(["éa"])

    char_lengths = token_byte_lengths(char_tokenizer)
    byte_lengths = token_byte_lengths(byte_tokenizer)

    assert char_lengths[char_tokenizer.bos_id] == 0
    assert char_lengths[char_tokenizer.token_to_id["é"]] == 2
    assert byte_lengths[byte_tokenizer.bos_id] == 0
    assert byte_lengths[byte_tokenizer.token_to_id["<byte:c3>"]] == 1
