import json

import pytest

from picochat import benchmark_pack
from picochat.benchmark_pack import BenchmarkSourceError, generate_benchmark_tuning_pack
from picochat.dataset_pack import load_dataset_pack
from picochat.honesty import inspect_data_honesty
from picochat.tuning_data import inspect_chat_eval_data, inspect_chat_sft_data


def write_pack(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains small local language models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "name": "test-pack",
        "corpus": str(corpus),
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")
    return pack


def test_generate_benchmark_tuning_pack_promotes_heldout_files(tmp_path):
    pack = write_pack(tmp_path)

    report = generate_benchmark_tuning_pack(pack, sft_rows=64, eval_rows=24, seed=7, promote_to_pack=True)

    chat_path = tmp_path / "chat_benchmark.jsonl"
    eval_path = tmp_path / "eval_benchmark.jsonl"
    assert chat_path.exists()
    assert eval_path.exists()
    assert (tmp_path / "benchmark_tuning_pack.md").exists()
    assert report.sft_rows == 64
    assert report.eval_rows == 24
    assert report.source_status == "offline"
    assert report.contamination["status"] == "ready"
    assert report.promoted_to_pack is True
    promoted = load_dataset_pack(pack)
    assert promoted.chat_input == str(chat_path)
    assert promoted.eval_input == str(eval_path)

    chat_rows = [json.loads(line) for line in chat_path.read_text(encoding="utf-8").splitlines()]
    eval_rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()]
    assert inspect_chat_sft_data(chat_path).status == "ready"
    assert inspect_chat_eval_data(eval_path).status == "ready"
    assert inspect_data_honesty(chat_path, eval_path).status == "ready"
    assert any(row["category"].startswith("bench_choice") for row in chat_rows)
    assert any(row.get("correct_choice") for row in eval_rows)
    assert {row["user"] for row in chat_rows}.isdisjoint({row["user"] for row in eval_rows})
    assert max(len(row["user"]) + len(row["assistant"]) for row in chat_rows) <= benchmark_pack.SFT_CHAR_BUDGET


def test_generate_benchmark_tuning_pack_refuses_overwrite_without_force(tmp_path):
    pack = write_pack(tmp_path)
    generate_benchmark_tuning_pack(pack, sft_rows=32, eval_rows=16)

    with pytest.raises(FileExistsError):
        generate_benchmark_tuning_pack(pack, sft_rows=32, eval_rows=16)


def test_generate_benchmark_tuning_pack_auto_falls_back_to_offline(tmp_path, monkeypatch):
    pack = write_pack(tmp_path)

    def fail_sft(count, seed):
        raise BenchmarkSourceError("hf unavailable")

    def fail_eval(count, seed):
        raise BenchmarkSourceError("hf unavailable")

    monkeypatch.setattr(benchmark_pack, "_build_hf_sft_result", fail_sft)
    monkeypatch.setattr(benchmark_pack, "_build_hf_eval_result", fail_eval)

    report = generate_benchmark_tuning_pack(
        pack,
        sft_rows=32,
        eval_rows=16,
        source="auto",
        force=True,
    )

    assert report.source_status == "offline_fallback"
    assert report.fallback_reason == "hf unavailable"
    assert (tmp_path / "chat_benchmark.jsonl").exists()
