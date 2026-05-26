import json

import pytest

from picochat.preference_starter import (
    PreferenceStarterConfig,
    generate_preference_starter,
    preference_starter_markdown,
)


def test_generate_preference_starter_from_chat_rows(tmp_path):
    chat = tmp_path / "chat.jsonl"
    out = tmp_path / "preferences.jsonl"
    chat.write_text(
        json.dumps({"user": "Who are you?", "assistant": "I am Picochat.", "category": "identity"}) + "\n"
        + json.dumps({"user": "Add 2 and 3.", "assistant": "5", "category": "math"}) + "\n",
        encoding="utf-8",
    )

    report = generate_preference_starter(PreferenceStarterConfig(
        input_path=str(chat),
        output_path=str(out),
    ))

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert report["num_examples"] == 2
    assert rows[0]["chosen"] == "I am Picochat."
    assert rows[0]["rejected"] == "Who are you?"
    assert rows[0]["category"] == "identity_preference"
    assert rows[1]["rejected_type"] == "empty_answer"
    assert "Starter preference rows" in report["warning"]
    assert "# Preference Starter Pack" in preference_starter_markdown(report)
    assert (tmp_path / "preferences.jsonl.report.md").exists()


def test_preference_starter_refuses_overwrite_without_force(tmp_path):
    chat = tmp_path / "chat.jsonl"
    out = tmp_path / "preferences.jsonl"
    chat.write_text(json.dumps({"user": "u", "assistant": "a"}) + "\n", encoding="utf-8")
    out.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate_preference_starter(PreferenceStarterConfig(
            input_path=str(chat),
            output_path=str(out),
        ))
