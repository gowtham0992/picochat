import json

from picochat.honesty import inspect_data_honesty, write_data_honesty_report


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def matrix_pair(report, name):
    pairs = {
        pair["name"]: pair
        for pair in report.contamination_matrix["pairs"]
    }
    return pairs[name]


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
    assert matrix_pair(report, "sft_vs_eval")["exact_text_hits"] == 1


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
    assert matrix_pair(report, "base_corpus_vs_eval")["risk"] == "high"


def test_inspect_data_honesty_reports_contamination_matrix(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    corpus_path = tmp_path / "corpus.txt"
    support_phrase = "lavender rope anchor below the quiet ridge marker"
    write_jsonl(chat_path, [{
        "user": "Explain the ridge safety cue.",
        "assistant": f"Use the {support_phrase} before moving.",
        "category": "domain_safety",
    }])
    write_jsonl(eval_path, [{
        "user": "Which ridge safety cue should be checked first?",
        "must_include": [support_phrase],
        "category": "domain_eval",
    }])
    corpus_path.write_text(
        f"Field note: Use the {support_phrase} before moving.",
        encoding="utf-8",
    )

    report = inspect_data_honesty(chat_path, eval_path, corpus_path)

    assert report.contamination_matrix["ngram_size"] == 8
    assert matrix_pair(report, "base_corpus_vs_sft")["checked"] is True
    assert matrix_pair(report, "base_corpus_vs_sft")["exact_text_hits"] == 1
    assert matrix_pair(report, "base_corpus_vs_eval")["exact_text_hits"] == 1
    assert matrix_pair(report, "sft_vs_eval")["exact_text_hits"] == 1
    assert matrix_pair(report, "sft_vs_eval")["max_ngram_overlap_rate"] == 1.0
    assert matrix_pair(report, "generated_vs_sft")["checked"] is False


def test_inspect_data_honesty_checks_generated_nearest_neighbors(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    corpus_path = tmp_path / "corpus.txt"
    generated = "Use the lavender rope anchor below the quiet ridge marker before moving."
    write_jsonl(chat_path, [{
        "user": "Explain the ridge safety cue.",
        "assistant": generated,
        "category": "domain_safety",
    }])
    write_jsonl(eval_path, [{
        "user": "Name a safe climbing habit.",
        "must_include": ["check the equipment"],
        "category": "domain_eval",
    }])
    corpus_path.write_text("A clean field guide without the generated sentence.", encoding="utf-8")

    report = inspect_data_honesty(
        chat_path,
        eval_path,
        corpus_path,
        generated_texts=[generated],
        ngram_size=4,
    )
    generated_pair = matrix_pair(report, "generated_vs_sft")
    sample = generated_pair["nearest_neighbors"][0]

    assert generated_pair["checked"] is True
    assert generated_pair["risk"] == "high"
    assert generated_pair["exact_text_hits"] == 1
    assert generated_pair["max_ngram_overlap_rate"] == 1.0
    assert sample["target_source"] == "generated_answers"
    assert sample["reference_kind"] == "answer"
    assert sample["reference_line"] == 1


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
    assert "contamination_matrix" in payload
    markdown = (tmp_path / "honesty" / "report.md").read_text(encoding="utf-8")
    assert "# Picochat Data Honesty Report" in markdown
    assert "## Contamination Matrix" in markdown
