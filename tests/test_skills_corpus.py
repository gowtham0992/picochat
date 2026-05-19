import json

import pytest

from picochat.data import inspect_documents
from picochat.skills_corpus import HELDOUT_WORDS, generate_skills_corpus


def test_generate_skills_corpus_writes_train_only_drills_and_recipe(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text("base corpus text\n", encoding="utf-8")
    out = tmp_path / "skills.txt"
    recipe = tmp_path / "recipe.json"

    report = generate_skills_corpus(
        out,
        math_rows=10,
        spelling_rows=12,
        choice_rows=4,
        seed=7,
        base_corpus=base,
        recipe_out=recipe,
    )

    text = out.read_text(encoding="utf-8")
    recipe_payload = json.loads(recipe.read_text(encoding="utf-8"))
    assert report.total_rows == 26
    assert report.categories == {
        "skills_choice": 4,
        "skills_math": 10,
        "skills_spelling": 12,
    }
    assert "Arithmetic drill" in text or "Math fact" in text
    assert "Character drill" in text or "Spelling fact" in text
    assert "Nora had" not in text
    assert "blue marbles" not in text
    assert "How many marbles are in the box" not in text
    assert all(word not in text for word in HELDOUT_WORDS)
    assert recipe_payload["sources"][0]["label"] == "base"
    assert recipe_payload["sources"][1]["label"] == "micro_skills"
    assert (tmp_path / "skills.report.md").exists()


def test_generate_skills_corpus_keeps_duplicate_line_rate_low(tmp_path):
    out = tmp_path / "skills.txt"

    generate_skills_corpus(out, math_rows=1000, spelling_rows=1000, choice_rows=200, seed=7)

    stats = inspect_documents([out.read_text(encoding="utf-8")])

    assert stats.duplicate_line_rate < 0.15


def test_generate_skills_corpus_can_write_recipe_with_shards(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text("base corpus text\n", encoding="utf-8")
    out = tmp_path / "skills.txt"
    docs = tmp_path / "skills_docs"
    recipe = tmp_path / "recipe.json"

    report = generate_skills_corpus(
        out,
        math_rows=10,
        spelling_rows=10,
        choice_rows=5,
        seed=7,
        base_corpus=base,
        recipe_out=recipe,
        documents_dir=docs,
        rows_per_shard=6,
    )

    recipe_payload = json.loads(recipe.read_text(encoding="utf-8"))
    shard_paths = sorted(docs.glob("shard-*.txt"))

    assert report.documents_dir == str(docs)
    assert report.num_shards == 5
    assert report.rows_per_shard == 6
    assert len(shard_paths) == 5
    assert recipe_payload["sources"][1]["path"] == "skills_docs"


def test_generate_skills_corpus_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "skills.txt"
    generate_skills_corpus(out, math_rows=1, spelling_rows=0, choice_rows=0)

    with pytest.raises(FileExistsError):
        generate_skills_corpus(out, math_rows=1, spelling_rows=0, choice_rows=0)
