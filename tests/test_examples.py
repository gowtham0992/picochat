from pathlib import Path

from picochat.data import inspect_path
from picochat.dataset_pack import load_dataset_pack
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


def test_tiny_dataset_pack_points_at_example_files():
    pack = load_dataset_pack(Path("examples/tiny_dataset_pack.json"))

    assert pack.corpus_recipe == "examples/corpus_recipe.json"
    assert pack.chat_input == "examples/tiny_chat.jsonl"
    assert pack.eval_input == "examples/tiny_eval.jsonl"


def test_tinystories_chat_examples_are_valid_jsonl():
    examples = load_chat_examples(Path("examples/tinystories_chat.jsonl"))

    assert len(examples) >= 70
    assert any("tiny story model" in example.assistant for example in examples)
    assert any("I do not know" in example.assistant for example in examples)
    assert any("puppy" in example.user.lower() for example in examples)
    assert any("robot" in example.user.lower() for example in examples)
    assert any("medical" in example.assistant.lower() for example in examples)


def test_tinystories_chat_v2_examples_are_valid_jsonl():
    examples = load_chat_examples(Path("examples/tinystories_chat_v2.jsonl"))

    assert len(examples) >= 300
    assert any("tiny story model" in example.assistant for example in examples)
    assert any("I do not know" in example.assistant for example in examples)
    assert any("puppy" in example.user.lower() for example in examples)
    assert all("Example " not in example.user for example in examples)
    assert all("Version " not in example.user for example in examples)


def test_tinystories_eval_examples_are_valid_jsonl():
    items = load_chat_eval_items(Path("examples/tinystories_eval.jsonl"))

    assert len(items) >= 10
    assert any(item.category == "memorization_probe" for item in items)
    assert any(not item.answerable for item in items)
    assert any("needle" in item.must_not_include for item in items)


def test_tinystories_eval_v2_examples_are_valid_jsonl():
    items = load_chat_eval_items(Path("examples/tinystories_eval_v2.jsonl"))

    assert len(items) >= 40
    assert any(item.category == "required_words" for item in items)
    assert any(item.category == "memorization_probe" for item in items)
    assert any(not item.answerable for item in items)
    assert any("needle" in item.must_not_include for item in items)


def test_tinystories_chat_v3_examples_are_valid_jsonl():
    examples = load_chat_examples(Path("examples/tinystories_chat_v3.jsonl"))

    assert len(examples) >= 450
    assert all(example.group for example in examples)
    assert any(example.category == "required_words" for example in examples)
    assert any(example.category == "continuation" for example in examples)
    assert any("exact words" in example.user for example in examples)
    assert any("I do not know" in example.assistant for example in examples)


def test_tinystories_eval_v3_examples_are_valid_jsonl():
    items = load_chat_eval_items(Path("examples/tinystories_eval_v3.jsonl"))

    assert len(items) >= 40
    assert any(item.category == "story_generation" for item in items)
    assert any(item.category == "required_words" for item in items)
    assert any(item.category == "memorization_probe" for item in items)
    assert any(not item.answerable for item in items)
    assert any("needle" in item.must_not_include for item in items)


def test_tinystories_chat_v4_examples_are_valid_jsonl():
    examples = load_chat_examples(Path("examples/tinystories_chat_v4.jsonl"))

    assert len(examples) >= 650
    assert all(example.group for example in examples)
    assert any(example.group == "story-template-0" for example in examples)
    assert any(example.group == "required-template-0" for example in examples)
    assert any("Required subject" in example.user for example in examples)
    assert any("I do not know" in example.assistant for example in examples)


def test_tinystories_eval_v4_examples_are_valid_jsonl():
    items = load_chat_eval_items(Path("examples/tinystories_eval_v4.jsonl"))

    assert len(items) >= 40
    assert any(item.category == "story_generation" for item in items)
    assert any(item.category == "required_words" for item in items)
    assert any(item.category == "memorization_probe" for item in items)
    assert any(not item.answerable for item in items)
    assert any("needle" in item.must_not_include for item in items)


def test_tinystories_chat_v5_examples_are_valid_jsonl():
    examples = load_chat_examples(Path("examples/tinystories_chat_v5.jsonl"))

    assert len(examples) >= 900
    assert all(example.group for example in examples)
    assert any(example.group == "pc-story-template-0" for example in examples)
    assert any(example.category == "prompt_conditioned_words" for example in examples)
    assert any(example.assistant.startswith("Subject:") for example in examples)
    assert any(example.assistant.startswith("Words:") for example in examples)
    assert any("I do not know" in example.assistant for example in examples)


def test_tinystories_eval_v5_examples_are_valid_jsonl():
    items = load_chat_eval_items(Path("examples/tinystories_eval_v5.jsonl"))

    assert len(items) >= 50
    assert any(item.category == "prompt_conditioned_story" for item in items)
    assert any(item.category == "prompt_conditioned_words" for item in items)
    assert any(item.split == "prompt_conditioned" for item in items)
    assert any(item.split == "transfer" for item in items)
    assert any(item.split == "safety" for item in items)
    assert any(not item.answerable for item in items)
    assert any("needle" in item.must_not_include for item in items)


def test_tinystories_dataset_pack_points_at_local_import_and_examples():
    pack = load_dataset_pack(Path("examples/tinystories_dataset_pack.json"))

    assert pack.corpus_input == "examples/../runs/tinystories-1k/documents"
    assert pack.chat_input == "examples/tinystories_chat.jsonl"
    assert pack.eval_input == "examples/tinystories_eval.jsonl"


def test_tinystories_dataset_pack_v2_points_at_expanded_examples():
    pack = load_dataset_pack(Path("examples/tinystories_dataset_pack_v2.json"))

    assert pack.corpus_input == "examples/../runs/tinystories-1k/documents"
    assert pack.chat_input == "examples/tinystories_chat_v2.jsonl"
    assert pack.eval_input == "examples/tinystories_eval_v2.jsonl"


def test_tinystories_dataset_pack_v3_points_at_grouped_examples():
    pack = load_dataset_pack(Path("examples/tinystories_dataset_pack_v3.json"))

    assert pack.corpus_input == "examples/../runs/tinystories-1k/documents"
    assert pack.chat_input == "examples/tinystories_chat_v3.jsonl"
    assert pack.eval_input == "examples/tinystories_eval_v3.jsonl"


def test_tinystories_dataset_pack_v4_points_at_template_grouped_examples():
    pack = load_dataset_pack(Path("examples/tinystories_dataset_pack_v4.json"))

    assert pack.corpus_input == "examples/../runs/tinystories-1k/documents"
    assert pack.chat_input == "examples/tinystories_chat_v4.jsonl"
    assert pack.eval_input == "examples/tinystories_eval_v4.jsonl"


def test_tinystories_dataset_pack_v5_points_at_prompt_conditioned_examples():
    pack = load_dataset_pack(Path("examples/tinystories_dataset_pack_v5.json"))

    assert pack.corpus_input == "examples/../runs/tinystories-1k/documents"
    assert pack.chat_input == "examples/tinystories_chat_v5.jsonl"
    assert pack.eval_input == "examples/tinystories_eval_v5.jsonl"


def test_tinystories_dataset_pack_v5_10k_points_at_larger_import():
    pack = load_dataset_pack(Path("examples/tinystories_dataset_pack_v5_10k.json"))

    assert pack.corpus_input == "examples/../runs/tinystories-10k/documents"
    assert pack.chat_input == "examples/tinystories_chat_v5.jsonl"
    assert pack.eval_input == "examples/tinystories_eval_v5.jsonl"
