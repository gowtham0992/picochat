from types import SimpleNamespace

from picochat.hf_sft import render_hf_chat_text, tokenize_hf_chat_example
from picochat.sft import ChatExample


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(char) % 97 + 1 for char in text])


class TemplateTokenizer(FakeTokenizer):
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize
        rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def test_hf_sft_masks_user_prompt_tokens():
    tokenizer = FakeTokenizer()
    row = tokenize_hf_chat_example(
        ChatExample(user="hi", assistant="ok"),
        tokenizer,
        max_length=128,
    )

    assert row is not None
    assert row["input_ids"]
    assert row["attention_mask"] == [1] * len(row["input_ids"])
    assert -100 in row["labels"]
    assert any(label != -100 for label in row["labels"])
    first_supervised = next(index for index, label in enumerate(row["labels"]) if label != -100)
    assert all(label == -100 for label in row["labels"][:first_supervised])


def test_hf_sft_uses_chat_template_when_available():
    prompt, full = render_hf_chat_text(
        ChatExample(user="Question", assistant="Answer"),
        TemplateTokenizer(),
    )

    assert prompt == "<user>Question<assistant>"
    assert full == "<user>Question<assistant>Answer"
