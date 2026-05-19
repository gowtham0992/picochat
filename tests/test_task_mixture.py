import json
from collections import Counter

from picochat.dataset_pack import load_dataset_pack
from picochat.honesty import inspect_data_honesty
from picochat.task_mixture import generate_task_mixture_pack
from picochat.tuning_data import inspect_chat_eval_data, inspect_chat_sft_data


def write_pack(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains honest small language models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(
        json.dumps({
            "name": "task-pack-test",
            "corpus": str(corpus),
            "chat": "chat.jsonl",
            "eval": "eval.jsonl",
        }),
        encoding="utf-8",
    )
    return pack


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generate_capability_task_mixture_promotes_pack(tmp_path):
    pack = write_pack(tmp_path)

    report = generate_task_mixture_pack(
        pack,
        sft_rows=120,
        eval_rows=40,
        profile="capability",
        force=True,
    )

    chat_path = tmp_path / "chat_task_mixture_capability.jsonl"
    eval_path = tmp_path / "eval_task_mixture_capability.jsonl"
    chat_rows = read_jsonl(chat_path)
    eval_rows = read_jsonl(eval_path)
    chat_categories = Counter(row["category"] for row in chat_rows)
    eval_categories = Counter(row["category"] for row in eval_rows)

    assert report.profile == "capability"
    assert report.sft_rows == 120
    assert report.eval_rows == 40
    assert report.chat_component_counts == {"behavior_anchor": 34, "weak_skills": 86}
    assert report.eval_component_counts == {"behavior_anchor": 11, "weak_skills": 29}
    assert report.contamination["status"] in {"ready", "caution"}
    assert report.contamination["exact_prompt_overlaps"] == 0
    assert report.promoted_to_pack is True
    assert (tmp_path / "task_mixture_capability.md").exists()
    promoted = load_dataset_pack(pack)
    assert promoted.chat_input == str(chat_path)
    assert promoted.eval_input == str(eval_path)

    assert inspect_chat_sft_data(chat_path).status == "ready"
    assert inspect_chat_eval_data(eval_path).status == "ready"
    assert inspect_data_honesty(chat_path, eval_path).status in {"ready", "caution"}
    assert all(row["mixture_profile"] == "capability" for row in chat_rows + eval_rows)
    assert {row["user"] for row in chat_rows}.isdisjoint({row["user"] for row in eval_rows})
    assert chat_categories["identity"] > 0
    assert chat_categories["refusal"] > 0
    assert eval_categories["identity"] > 0
    assert eval_categories["refusal"] > 0
    assert any(category.startswith("bench_math_") for category in chat_categories)
    assert any(category.startswith("bench_spelling_") for category in chat_categories)
    assert any(row["mixture_component"] == "weak_skills" for row in chat_rows)
    assert any(row["mixture_component"] == "behavior_anchor" for row in chat_rows)


def test_balanced_task_mixture_keeps_component_provenance_without_promote(tmp_path):
    pack = write_pack(tmp_path)
    out_dir = tmp_path / "mixtures"

    report = generate_task_mixture_pack(
        pack,
        out_dir=out_dir,
        sft_rows=96,
        eval_rows=32,
        profile="balanced",
        skill_answer_style="direct",
        force=True,
        promote_to_pack=False,
    )

    chat_rows = read_jsonl(out_dir / "chat_task_mixture_balanced.jsonl")
    eval_rows = read_jsonl(out_dir / "eval_task_mixture_balanced.jsonl")
    chat_components = Counter(row["mixture_component"] for row in chat_rows)
    eval_components = Counter(row["mixture_component"] for row in eval_rows)

    assert report.promoted_to_pack is False
    assert load_dataset_pack(pack).chat_input.endswith("chat.jsonl")
    assert set(chat_components) == {"benchmark", "release_behavior", "weak_skills"}
    assert set(eval_components) == {"benchmark", "release_behavior", "weak_skills"}
    assert sum(chat_components.values()) == 96
    assert sum(eval_components.values()) == 32
    assert {row["user"] for row in chat_rows}.isdisjoint({row["user"] for row in eval_rows})
    assert all(row["mixture_benchmark_profile"] for row in chat_rows + eval_rows)
    report_text = (out_dir / "task_mixture_balanced.md").read_text(encoding="utf-8")
    assert "| release_behavior |" in report_text
    assert "| weak_skills |" in report_text
    assert "| benchmark |" in report_text
