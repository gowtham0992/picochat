import json

import pytest

from picochat.eval_starter import generate_eval_starter


def test_generate_eval_starter_writes_ladder_rows(tmp_path):
    source = tmp_path / "docs"
    source.mkdir()
    (source / "lesson.txt").write_text(
        "Pico Cafe roasts careful beans for morning guests. "
        "Mira Chen founded Pico Cafe beside the train station. "
        "The cafe teaches new baristas to weigh water and beans.",
        encoding="utf-8",
    )
    out_path = tmp_path / "eval.jsonl"

    report = generate_eval_starter(source, out_path, max_items=8, seed=1)

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert report.num_rows == len(rows)
    assert (tmp_path / "eval.md").exists()
    assert {"heldout", "adversarial", "memorization", "smoke"} <= {row["level"] for row in rows}
    assert any(row.get("require_corpus_support") for row in rows)
    assert any(row["answerable"] is False for row in rows)


def test_generate_eval_starter_refuses_overwrite_without_force(tmp_path):
    source = tmp_path / "lesson.txt"
    source.write_text("Pico Cafe roasts careful beans for morning guests.", encoding="utf-8")
    out_path = tmp_path / "eval.jsonl"
    out_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate_eval_starter(source, out_path)
