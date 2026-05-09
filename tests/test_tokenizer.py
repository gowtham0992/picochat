from picochat.tokenizer import CharTokenizer, SPECIAL_TOKENS


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

