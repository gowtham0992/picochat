import json

from picochat.tuning_data import inspect_chat_eval_data, inspect_chat_sft_data


def test_inspect_chat_sft_data_reports_ready_file(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    rows = [
        {
            "user": f"Question {index}?",
            "assistant": f"Answer {index}.",
            "category": "story_generation" if index < 4 else "refusal",
        }
        for index in range(8)
    ]
    chat_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = inspect_chat_sft_data(chat_path)

    assert report.status == "ready"
    assert report.num_examples == 8
    assert report.num_rows == 8
    assert report.invalid_rows == 0
    assert report.preview[0]["user"] == "Question 0?"
    assert report.preview[0]["category"] == "story_generation"
    assert report.categories == {"refusal": 4, "story_generation": 4}
    assert report.average_assistant_chars > 0


def test_inspect_chat_sft_data_blocks_invalid_rows(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    chat_path.write_text(
        "\n".join([
            json.dumps({"user": "valid", "assistant": "ok"}),
            json.dumps({"user": "missing assistant"}),
            "{not json",
        ]),
        encoding="utf-8",
    )

    report = inspect_chat_sft_data(chat_path)

    assert report.status == "blocked"
    assert report.num_examples == 1
    assert report.invalid_rows == 2
    assert report.issues[0].line == 2
    assert "assistant" in report.issues[0].message
    assert "invalid JSON" in report.issues[1].message


def test_inspect_chat_sft_data_rejects_invalid_category(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    chat_path.write_text(
        json.dumps({"user": "valid", "assistant": "ok", "category": ""}),
        encoding="utf-8",
    )

    report = inspect_chat_sft_data(chat_path)

    assert report.status == "blocked"
    assert report.invalid_rows == 1
    assert "category" in report.issues[0].message


def test_inspect_chat_eval_data_reports_rules_and_categories(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    rows = [
        {
            "user": "What is Picochat?",
            "category": "project",
            "split": "knowledge",
            "must_include": ["Picochat"],
        },
        {
            "user": "Should it make things up?",
            "answerable": False,
            "category": "honesty",
            "split": "safety",
            "must_include_any": [["No", "avoid unsupported claims"]],
        },
        {
            "user": "What should not appear?",
            "category": "negative",
            "must_not_include": ["yes"],
        },
        {
            "user": "Expected shorthand?",
            "expected": "expected phrase",
        },
    ]
    eval_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = inspect_chat_eval_data(eval_path)

    assert report.status == "ready"
    assert report.num_items == 4
    assert report.answerable_items == 3
    assert report.unanswerable_items == 1
    assert report.must_include_rules == 2
    assert report.must_include_any_groups == 1
    assert report.must_not_include_rules == 1
    assert report.categories["honesty"] == 1
    assert report.splits["knowledge"] == 1
    assert report.splits["safety"] == 1


def test_inspect_chat_eval_data_validates_choice_fields(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({
        "user": "Pick one. A. red B. blue",
        "must_include": ["B"],
        "choice_labels": ["A", "B"],
        "correct_choice": "B",
    }), encoding="utf-8")

    report = inspect_chat_eval_data(eval_path)

    assert report.status == "caution"
    assert report.preview[0]["choice_labels"] == ["A", "B"]
    assert report.preview[0]["correct_choice"] == "B"


def test_inspect_chat_eval_data_blocks_missing_rules(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"user": "No scoring rules?"}), encoding="utf-8")

    report = inspect_chat_eval_data(eval_path)

    assert report.status == "blocked"
    assert report.num_items == 1
    assert report.summary == "Eval items need visible pass/fail rules."


def test_inspect_tuning_data_reports_missing_file(tmp_path):
    report = inspect_chat_sft_data(tmp_path / "missing.jsonl")

    assert report.status == "blocked"
    assert report.invalid_rows == 1
    assert report.issues[0].message == "file does not exist"
