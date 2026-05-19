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
    assert report.category_entropy > 0
    assert report.category_entropy_normalized == 1.0
    assert report.assistant_length_distribution["count"] == 8
    assert report.answer_styles == {"direct": 8}
    assert report.curriculum_label == "mixed_sft"
    assert report.curriculum_breakdown == {"behavior": 4, "domain": 4}


def test_inspect_chat_sft_data_reports_quality_signals(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    rows = [
        {
            "user": "Solve 2 + 2. Return only the number.",
            "assistant": "4",
            "category": "bench_math",
            "group": "train-math-001",
        },
        {
            "user": "Solve 2 plus 2. Return only the number.",
            "assistant": "4",
            "category": "bench_math",
            "group": "train-math-002",
        },
        {
            "user": "Spell cat backward.",
            "assistant": "tac",
            "category": "bench_spelling",
            "group": "train-spelling-001",
        },
        {
            "user": "Spell cat backward.",
            "assistant": "tac",
            "category": "bench_spelling",
            "group": "train-spelling-002",
        },
    ] * 2
    chat_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = inspect_chat_sft_data(chat_path)

    assert report.curriculum_label == "skill_sft"
    assert report.curriculum_breakdown == {"skill": 8}
    assert report.duplicate_user_prompts == 5
    assert report.duplicate_user_samples
    assert report.near_duplicate_user_pairs >= 1
    assert report.template_families == {"train-math": 4, "train-spelling": 4}
    assert any("skill SFT" in warning for warning in report.quality_warnings)


def test_inspect_chat_sft_data_reports_answer_styles(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    rows = [
        {"user": "What is 2 + 2?", "assistant": "4", "category": "bench_math"},
        {
            "user": "What is 3 + 4?",
            "assistant": "Scratchpad:\n- Compute: 3 + 4 = 7\nFinal answer: 7",
            "category": "bench_math",
            "answer_style": "scratchpad",
        },
        {
            "user": "What is 5 + 6?",
            "assistant": "Scratchpad:\n- Compute: 5 + 6 = 11\nFinal answer: 11",
            "category": "bench_math",
        },
    ]
    chat_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = inspect_chat_sft_data(chat_path)

    assert report.answer_styles == {"direct": 1, "scratchpad": 2}


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
    assert report.heldout_categories["honesty"] == 1
    assert report.category_entropy > 0
    assert report.answer_length_distribution["count"] >= 3
    assert report.curriculum_label == "domain_eval"


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


def test_inspect_chat_eval_data_reports_quality_signals(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    rows = [
        {
            "user": "Choose the answer. A. red B. blue",
            "must_include": ["B"],
            "choice_labels": ["A", "B"],
            "correct_choice": "B",
            "category": "bench_choice_color",
            "split": "benchmark",
            "group": "heldout-choice-001",
        },
        {
            "user": "Choose the answer. A. red B. blue!",
            "must_include": ["B"],
            "choice_labels": ["A", "B"],
            "correct_choice": "B",
            "category": "bench_choice_color",
            "split": "benchmark",
            "group": "heldout-choice-002",
        },
        {
            "user": "Choose the answer. A. red B. blue",
            "must_include": ["B"],
            "choice_labels": ["A", "B"],
            "correct_choice": "B",
            "category": "bench_choice_color",
            "split": "benchmark",
            "group": "heldout-choice-003",
        },
        {
            "user": "What unsupported private fact should be refused?",
            "answerable": False,
            "must_include_any": [["I do not know", "not enough information"]],
            "category": "refusal",
            "split": "adversarial",
            "group": "heldout-refusal-001",
        },
    ]
    eval_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = inspect_chat_eval_data(eval_path)

    assert report.curriculum_label == "mixed_behavior_skill_eval"
    assert report.curriculum_breakdown == {"behavior": 1, "skill": 3}
    assert report.duplicate_user_prompts == 1
    assert report.near_duplicate_user_pairs >= 1
    assert report.heldout_categories == {"bench_choice_color": 3, "refusal": 1}
    assert report.template_families == {"heldout-choice": 3, "heldout-refusal": 1}


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
