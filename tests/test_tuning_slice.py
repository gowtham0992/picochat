import json
from pathlib import Path

import pytest

from picochat.dataset_pack import load_dataset_pack
from picochat.tuning_slice import parse_category_patterns, slice_tuning_pack


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_slice_tuning_pack_filters_categories_without_crossing_splits(tmp_path):
    corpus = tmp_path / "corpus.txt"
    chat = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    out_dir = tmp_path / "behavior"
    corpus.write_text("base corpus\n", encoding="utf-8")
    write_jsonl(chat, [
        {"user": "who are you?", "assistant": "I am Picochat.", "category": "identity"},
        {"user": "unknown?", "assistant": "I do not know.", "category": "refusal"},
        {"user": "2+2?", "assistant": "4", "category": "bench_math_addition"},
    ])
    write_jsonl(eval_path, [
        {"user": "name?", "must_include": ["Picochat"], "category": "identity"},
        {"user": "secret?", "must_include": ["I do not know"], "category": "refusal", "answerable": False},
        {"user": "3+4?", "must_include": ["7"], "category": "bench_math_addition"},
    ])
    pack_path.write_text(json.dumps({
        "name": "source",
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    report = slice_tuning_pack(
        pack_path,
        out_dir,
        include_categories=("identity", "refusal"),
    )

    assert report.chat_rows_out == 2
    assert report.eval_rows_out == 2
    assert report.chat_categories == {"identity": 1, "refusal": 1}
    assert report.eval_categories == {"identity": 1, "refusal": 1}
    sliced_pack = load_dataset_pack(report.dataset_pack)
    assert sliced_pack.corpus_input is not None
    assert sliced_pack.corpus_input.endswith("corpus.txt")
    assert Path(sliced_pack.corpus_input).resolve() == corpus.resolve()
    assert (out_dir / "tuning_slice.md").exists()
    assert "does not move rows between train and eval" in (out_dir / "tuning_slice.md").read_text(encoding="utf-8")
    chat_rows = [json.loads(line) for line in (out_dir / "chat_slice.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_rows = [json.loads(line) for line in (out_dir / "eval_slice.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["user"] for row in chat_rows] == ["who are you?", "unknown?"]
    assert [row["user"] for row in eval_rows] == ["name?", "secret?"]


def test_slice_tuning_pack_supports_category_globs_and_refuses_empty_slice(tmp_path):
    corpus = tmp_path / "corpus.txt"
    chat = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    corpus.write_text("base corpus\n", encoding="utf-8")
    write_jsonl(chat, [
        {"user": "language?", "assistant": "A", "category": "bench_choice_language"},
        {"user": "science?", "assistant": "B", "category": "bench_choice_science"},
        {"user": "identity?", "assistant": "Picochat", "category": "identity"},
    ])
    write_jsonl(eval_path, [
        {"user": "language?", "must_include": ["A"], "category": "bench_choice_language"},
        {"user": "science?", "must_include": ["B"], "category": "bench_choice_science"},
        {"user": "identity?", "must_include": ["Picochat"], "category": "identity"},
    ])
    pack_path.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    report = slice_tuning_pack(
        pack_path,
        tmp_path / "choice",
        include_categories=("bench_choice_*",),
    )

    assert report.chat_categories == {"bench_choice_language": 1, "bench_choice_science": 1}
    assert report.eval_categories == {"bench_choice_language": 1, "bench_choice_science": 1}
    with pytest.raises(ValueError, match="zero chat rows"):
        slice_tuning_pack(
            pack_path,
            tmp_path / "empty",
            include_categories=("missing_*",),
        )


def test_parse_category_patterns_trims_empty_parts():
    assert parse_category_patterns(" identity, refusal ,,") == ("identity", "refusal")
