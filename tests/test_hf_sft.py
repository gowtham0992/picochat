from types import SimpleNamespace
import json

from picochat.hf_sft import HFConversationExample, load_hf_sft_examples, render_hf_chat_text, tokenize_hf_chat_example
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


def test_hf_sft_loads_multiturn_messages_and_tool_context(tmp_path):
    input_path = tmp_path / "tool.jsonl"
    input_path.write_text(json.dumps({
        "system": "Use tools carefully.",
        "tools": [{"name": "search_schedule"}],
        "messages": [
            {"role": "user", "content": "Find my meeting."},
            {"role": "assistant", "content": "I will search."},
            {"role": "tool", "content": "Standup at 9 AM."},
            {"role": "assistant", "content": "The meeting is at 9 AM."},
        ],
        "category": "tool_calling",
    }) + "\n", encoding="utf-8")

    examples = load_hf_sft_examples(input_path)

    assert len(examples) == 1
    assert examples[0].category == "tool_calling"
    assert examples[0].messages[0]["role"] == "system"
    assert "search_schedule" in examples[0].messages[0]["content"]
    assert examples[0].messages[-1] == {"role": "assistant", "content": "The meeting is at 9 AM."}


def test_hf_sft_multiturn_masks_only_final_target():
    tokenizer = TemplateTokenizer()
    examples = [
        {
            "role": "system",
            "content": "Use tools.",
        },
        {
            "role": "user",
            "content": "Find my meeting.",
        },
        {
            "role": "assistant",
            "content": "I will search.",
        },
        {
            "role": "tool",
            "content": "Standup at 9 AM.",
        },
        {
            "role": "assistant",
            "content": "The meeting is at 9 AM.",
        },
    ]
    example = HFConversationExample(messages=tuple(examples), category="tool")
    row = tokenize_hf_chat_example(example, tokenizer, max_length=512)

    assert row is not None
    supervised = [label for label in row["labels"] if label != -100]
    assert supervised
    prompt, full = render_hf_chat_text(example, tokenizer)
    assert "I will search." in prompt
    assert "The meeting is at 9 AM." not in prompt
    assert "The meeting is at 9 AM." in full
