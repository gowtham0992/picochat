import json

import pytest

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
    assert all(word not in text for word in HELDOUT_WORDS)
    assert recipe_payload["sources"][0]["label"] == "base"
    assert recipe_payload["sources"][1]["label"] == "micro_skills"
    assert (tmp_path / "skills.report.md").exists()


def test_generate_skills_corpus_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "skills.txt"
    generate_skills_corpus(out, math_rows=1, spelling_rows=0, choice_rows=0)

    with pytest.raises(FileExistsError):
        generate_skills_corpus(out, math_rows=1, spelling_rows=0, choice_rows=0)
