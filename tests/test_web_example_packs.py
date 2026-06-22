"""Bundled example packs are read-only: the dashboard clones before generating."""

import json
from pathlib import Path

import pytest

import picochat.web as web
from picochat.web import (
    benchmark_tuning_pack_plan,
    clone_example_pack_plan,
    eval_starter_plan,
    preference_starter_plan,
    sft_starter_plan,
)


def _make_pack(root: Path) -> Path:
    """A minimal but valid dataset pack with corpus recipe, chat, and eval."""
    (root / "corpus.txt").write_text("hello world\n", encoding="utf-8")
    (root / "recipe.json").write_text(
        json.dumps({"name": "c", "sources": [{"path": "corpus.txt", "label": "demo"}]}),
        encoding="utf-8",
    )
    (root / "chat.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}) + "\n",
        encoding="utf-8",
    )
    (root / "eval.jsonl").write_text(
        json.dumps({"prompt": "hi", "answer": "yo"}) + "\n",
        encoding="utf-8",
    )
    pack = root / "dataset_pack.json"
    pack.write_text(
        json.dumps({"name": "demo", "corpus": {"recipe": "recipe.json"}, "chat": "chat.jsonl", "eval": "eval.jsonl"}),
        encoding="utf-8",
    )
    return pack


def test_clone_example_pack_copies_and_leaves_source_untouched(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    pack = _make_pack(examples)
    monkeypatch.setattr(web, "_BUNDLED_EXAMPLES_DIR", examples)
    monkeypatch.chdir(tmp_path)
    original = pack.read_text(encoding="utf-8")

    result = clone_example_pack_plan({"dataset_pack": str(pack)})

    assert result["cloned"] is True
    new_pack = Path(result["dataset_pack"])
    assert new_pack.exists() and new_pack != pack
    assert (new_pack.parent / "chat.jsonl").exists()
    assert (new_pack.parent / "eval.jsonl").exists()
    # The shipped example is never modified.
    assert pack.read_text(encoding="utf-8") == original


def test_clone_writable_pack_is_noop(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    monkeypatch.setattr(web, "_BUNDLED_EXAMPLES_DIR", examples)
    mine = tmp_path / "mine"
    mine.mkdir()
    pack = _make_pack(mine)

    result = clone_example_pack_plan({"dataset_pack": str(pack)})
    assert result["cloned"] is False
    assert result["dataset_pack"] == str(pack)


def test_mutating_handlers_reject_bundled_examples(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    pack = _make_pack(examples)
    monkeypatch.setattr(web, "_BUNDLED_EXAMPLES_DIR", examples)
    out = str(examples / "chat_generated.jsonl")

    with pytest.raises(ValueError, match="read-only example pack"):
        sft_starter_plan({"dataset_pack": str(pack), "out_path": out, "promote_to_pack": True})
    with pytest.raises(ValueError, match="read-only example pack"):
        eval_starter_plan({"dataset_pack": str(pack), "out_path": out, "promote_to_pack": True})
    with pytest.raises(ValueError, match="read-only example pack"):
        benchmark_tuning_pack_plan({"dataset_pack": str(pack)})
    with pytest.raises(ValueError, match="read-only example pack"):
        preference_starter_plan({"input_path": str(examples / "chat.jsonl"), "out_path": out})
