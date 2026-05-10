import json

from picochat.honesty import inspect_data_honesty, write_data_honesty_report


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_inspect_data_honesty_blocks_exact_eval_prompt_leak(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    corpus_path = tmp_path / "corpus.txt"
    write_jsonl(chat_path, [{
        "user": "Write a tiny story about a turtle.",
        "assistant": "Once there was a turtle.",
    }])
    write_jsonl(eval_path, [{
        "user": "Write a tiny story about a turtle.",
        "must_include": ["turtle"],
        "category": "story_generation",
    }])
    corpus_path.write_text("A clean training story.", encoding="utf-8")

    report = inspect_data_honesty(chat_path, eval_path, corpus_path)

    assert report.status == "blocked"
    assert report.exact_prompt_leaks == 1
    assert report.findings[0].kind == "exact_sft_prompt_leak"


def test_inspect_data_honesty_warns_on_near_prompt_and_corpus_hit(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    corpus_path = tmp_path / "corpus.txt"
    write_jsonl(chat_path, [{
        "user": "Write a tiny story about a turtle who learns to share.",
        "assistant": "A turtle shared a shell.",
    }])
    write_jsonl(eval_path, [{
        "user": "Write a tiny story about a turtle that learns to share.",
        "must_include": ["turtle"],
        "category": "story_generation",
    }])
    corpus_path.write_text(
        "Write a tiny story about a turtle that learns to share.\nA leaked prompt.",
        encoding="utf-8",
    )

    report = inspect_data_honesty(chat_path, eval_path, corpus_path)

    assert report.status == "caution"
    assert report.near_prompt_leaks == 1
    assert report.corpus_prompt_hits == 1
    assert {finding.kind for finding in report.findings} == {
        "near_sft_prompt_leak",
        "eval_prompt_in_corpus",
    }


def test_write_data_honesty_report_writes_json_and_markdown(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(chat_path, [{"user": "hello", "assistant": "hi"}])
    write_jsonl(eval_path, [{"user": "What is a tiny story?", "must_include": ["story"]}])
    report = inspect_data_honesty(chat_path, eval_path)

    json_path, markdown_path = write_data_honesty_report(report, tmp_path / "honesty")

    payload = json.loads((tmp_path / "honesty" / "honesty_report.json").read_text(encoding="utf-8"))
    assert json_path.endswith("honesty_report.json")
    assert markdown_path.endswith("report.md")
    assert payload["status"] == "ready"
    assert "# Picochat Data Honesty Report" in (tmp_path / "honesty" / "report.md").read_text(encoding="utf-8")
