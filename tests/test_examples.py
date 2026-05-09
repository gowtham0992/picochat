from pathlib import Path

from picochat.data import inspect_path
from picochat.eval import load_chat_eval_items
from picochat.sft import load_chat_examples
from picochat.tokenizer import CharTokenizer


def test_tiny_corpus_is_usable():
    corpus_path = Path("examples/tiny_corpus.txt")

    stats = inspect_path(corpus_path)
    tokenizer = CharTokenizer.train([corpus_path.read_text(encoding="utf-8")])

    assert stats.num_documents == 1
    assert stats.num_characters > 500
    assert len(tokenizer) > 20
    assert tokenizer.decode(tokenizer.encode("Picochat")) == "Picochat"


def test_tiny_chat_examples_are_valid_jsonl():
    examples = load_chat_examples(Path("examples/tiny_chat.jsonl"))

    assert len(examples) >= 10
    assert any("honest" in example.assistant for example in examples)
    assert any(example.assistant.startswith("No.") for example in examples)


def test_tiny_eval_examples_are_valid_jsonl():
    items = load_chat_eval_items(Path("examples/tiny_eval.jsonl"))

    assert len(items) >= 6
    assert any("honest" in item.must_include for item in items)
    assert any(item.must_include_any for item in items)
