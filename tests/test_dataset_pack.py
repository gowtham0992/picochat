import json

import pytest

from picochat.dataset_pack import init_dataset_pack, load_dataset_pack


def test_load_dataset_pack_resolves_relative_paths(tmp_path):
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    pack_path = pack_dir / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "name": "lesson-pack",
        "description": "A tiny lesson pack.",
        "corpus": {"recipe": "corpus_recipe.json"},
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    pack = load_dataset_pack(pack_path)

    assert pack.path == str(pack_path)
    assert pack.name == "lesson-pack"
    assert pack.description == "A tiny lesson pack."
    assert pack.corpus_input is None
    assert pack.corpus_recipe == str(pack_dir / "corpus_recipe.json")
    assert pack.chat_input == str(pack_dir / "chat.jsonl")
    assert pack.eval_input == str(pack_dir / "eval.jsonl")


def test_load_dataset_pack_accepts_corpus_string_shorthand(tmp_path):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat_input": "chat.jsonl",
        "eval_input": "eval.jsonl",
    }), encoding="utf-8")

    pack = load_dataset_pack(pack_path)

    assert pack.name == "dataset_pack"
    assert pack.corpus_input == str(tmp_path / "corpus.txt")
    assert pack.corpus_recipe is None


def test_load_dataset_pack_rejects_ambiguous_corpus(tmp_path):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": {
            "input": "corpus.txt",
            "recipe": "corpus_recipe.json",
        },
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_dataset_pack(pack_path)


def test_load_dataset_pack_rejects_missing_chat_or_eval(tmp_path):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="eval"):
        load_dataset_pack(pack_path)


def test_init_dataset_pack_creates_starter_files(tmp_path):
    corpus_dir = tmp_path / "docs"
    out_dir = tmp_path / "pack"
    corpus_dir.mkdir()

    report = init_dataset_pack(out_dir, corpus_dir, name="lesson-pack", description="Lessons.")

    assert report.dataset_pack == str(out_dir / "dataset_pack.json")
    assert (out_dir / "dataset_pack.json").exists()
    assert (out_dir / "corpus_recipe.json").exists()
    assert (out_dir / "chat.jsonl").exists()
    assert (out_dir / "eval.jsonl").exists()
    assert len(report.created) == 4
    pack = load_dataset_pack(out_dir / "dataset_pack.json")
    assert pack.name == "lesson-pack"
    assert pack.description == "Lessons."
    assert pack.chat_input == str(out_dir / "chat.jsonl")
    recipe = json.loads((out_dir / "corpus_recipe.json").read_text(encoding="utf-8"))
    assert recipe["sources"][0]["path"] == "../docs"
    assert "user" in (out_dir / "chat.jsonl").read_text(encoding="utf-8")
    assert "must_include" in (out_dir / "eval.jsonl").read_text(encoding="utf-8")


def test_init_dataset_pack_refuses_to_overwrite_without_force(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    out_dir = tmp_path / "pack"
    corpus_path.write_text("text", encoding="utf-8")
    init_dataset_pack(out_dir, corpus_path, name="first")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        init_dataset_pack(out_dir, corpus_path, name="second")

    report = init_dataset_pack(out_dir, corpus_path, name="second", force=True)
    pack = load_dataset_pack(out_dir / "dataset_pack.json")

    assert report.overwritten
    assert pack.name == "second"
