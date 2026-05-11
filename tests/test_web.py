import json
import shutil
from pathlib import Path

import pytest

from picochat.web import (
    archive_run_plan,
    cancel_run_plan,
    discover_runs,
    eval_starter_plan,
    generate_run_text,
    hf_import_plan,
    init_dataset_pack_plan,
    inspect_tuning_plan,
    load_pack_editor_plan,
    load_run_detail,
    load_run_report,
    preview_corpus_plan,
    _parse_run_progress,
    run_presets_plan,
    run_status_plan,
    save_pack_editor_plan,
    sft_starter_plan,
    start_run_plan,
    _RUN_JOBS,
    _RUN_JOBS_LOCK,
    _error_payload,
)
from picochat.hf_import import HFSplitError


def write_run(root, name):
    run_dir = root / name
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "base").mkdir()
    (run_dir / "base" / "checkpoint").mkdir()
    (run_dir / "sft").mkdir()
    (run_dir / "sft" / "checkpoint").mkdir()
    summary = {
        "config": {
            "out_dir": str(run_dir),
            "context_size": 128,
        },
        "artifacts": {
            "corpus": str(run_dir / "corpus.txt"),
            "tokenizer": str(run_dir / "tokenizer.json"),
            "honesty_report": str(run_dir / "honesty" / "report.md"),
            "honesty_json": str(run_dir / "honesty" / "honesty_report.json"),
        },
        "base": {
            "final_val_loss": 2.0,
            "num_parameters": 1234,
        },
        "sft": {
            "final_val_loss": 4.0,
            "truncated_examples": 0,
        },
        "eval": {
            "num_examples": 2,
            "num_passed": 1,
            "num_failed": 1,
            "pass_rate": 0.5,
        },
        "honesty": {
            "status": "ready",
            "summary": "No obvious eval leakage was detected.",
            "exact_prompt_leaks": 0,
            "near_prompt_leaks": 0,
            "corpus_prompt_hits": 0,
            "duplicate_eval_prompts": 0,
            "max_sft_prompt_similarity": 0.0,
        },
    }
    tokenizer = {
        "type": "char",
        "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>"],
        "token_to_id": {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "h": 4,
            "i": 5,
        },
    }
    train_report = {
        "losses": [{"step": 1, "train_loss": 3.0, "val_loss": 2.0}],
    }
    sft_report = {
        "dataset": {
            "num_examples": 1,
            "supervised_tokens": 5,
        },
        "losses": [{"step": 1, "train_loss": 3.0, "val_loss": 4.0}],
    }
    eval_report = {
        "summary": summary["eval"],
        "examples": [{
            "user": "hi",
            "reply": "hello",
            "must_include": [],
            "must_include_any": [],
            "must_not_include": [],
            "missing": [],
            "missing_any": [],
            "found_forbidden": [],
            "passed": True,
        }],
    }
    (run_dir / "corpus.txt").write_text("User: hi\nAssistant: hello", encoding="utf-8")
    (run_dir / "corpus_manifest.json").write_text(json.dumps({
        "files": [{
            "path": str(run_dir / "source.txt"),
            "extension": ".txt",
            "num_characters": 24,
            "num_lines": 2,
            "included": True,
            "reason": "included",
        }],
        "warnings": [],
    }), encoding="utf-8")
    (run_dir / "corpus_report.md").write_text("# Corpus", encoding="utf-8")
    (run_dir / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "summary.md").write_text("# Summary", encoding="utf-8")
    (run_dir / "honesty").mkdir()
    (run_dir / "honesty" / "honesty_report.json").write_text(json.dumps(summary["honesty"]), encoding="utf-8")
    (run_dir / "honesty" / "report.md").write_text("# Honesty", encoding="utf-8")
    (run_dir / "base" / "train_report.json").write_text(json.dumps(train_report), encoding="utf-8")
    (run_dir / "base" / "report.md").write_text("# Base", encoding="utf-8")
    (run_dir / "sft" / "sft_report.json").write_text(json.dumps(sft_report), encoding="utf-8")
    (run_dir / "sft" / "report.md").write_text("# SFT", encoding="utf-8")
    (run_dir / "eval" / "eval_report.json").write_text(json.dumps(eval_report), encoding="utf-8")
    (run_dir / "eval" / "report.md").write_text("# Eval", encoding="utf-8")
    (run_dir / "base" / "sample.txt").write_text("base sample", encoding="utf-8")
    (run_dir / "sft" / "sample.txt").write_text("sft sample", encoding="utf-8")
    return run_dir


def test_discover_runs_returns_dashboard_rows(tmp_path):
    write_run(tmp_path, "tiny-a")

    rows = discover_runs(tmp_path)

    assert rows == [{
        "name": "tiny-a",
        "path": str(tmp_path / "tiny-a"),
        "eval_score": "1/2",
        "pass_rate": 0.5,
        "base_val_loss": 2.0,
        "sft_val_loss": 4.0,
        "num_parameters": 1234,
        "context_size": 128,
        "truncated_examples": 0,
    }]


def test_load_run_detail_reads_eval_reports_and_samples(tmp_path):
    write_run(tmp_path, "tiny-a")

    detail = load_run_detail(tmp_path, "tiny-a")

    assert detail["summary"]["eval"]["num_passed"] == 1
    assert detail["eval"]["summary"]["num_examples"] == 2
    assert detail["eval_reports"][0]["name"] == "eval"
    assert detail["base_report"]["losses"][0]["step"] == 1
    assert detail["sft_report"]["dataset"]["supervised_tokens"] == 5
    assert "User: hi" in detail["corpus_preview"]
    assert detail["corpus_manifest"]["files"][0]["reason"] == "included"
    assert detail["corpus_report"] == "# Corpus"
    assert detail["tokenizer_detail"]["vocab_size"] == 6
    assert detail["tokenizer_detail"]["token_to_id"]["h"] == 4
    assert detail["tokenizer_detail"]["sample_tokens"] == ["h", "i"]
    assert detail["base_sample"] == "base sample"
    assert detail["sft_sample"] == "sft sample"
    assert detail["reports"]["summary"]["exists"] is True
    assert detail["reports"]["honesty"]["exists"] is True
    assert detail["reports"]["base"]["exists"] is True
    assert detail["reports"]["sft"]["exists"] is True
    assert detail["reports"]["eval"]["exists"] is True
    corpus_status = detail["artifact_inventory"]["by_path"][str(tmp_path / "tiny-a" / "corpus.txt")]
    checkpoint_status = detail["artifact_inventory"]["by_path"][str(tmp_path / "tiny-a" / "base" / "checkpoint")]
    assert corpus_status["exists"] is True
    assert corpus_status["kind"] == "file"
    assert corpus_status["size_bytes"] > 0
    assert checkpoint_status["exists"] is True
    assert checkpoint_status["kind"] == "directory"


def test_artifact_inventory_indexes_relative_and_absolute_paths(tmp_path, monkeypatch):
    run_dir = write_run(tmp_path / "runs", "tiny-a")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config"]["out_dir"] = "runs/tiny-a"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    detail = load_run_detail("runs", "tiny-a")

    by_path = detail["artifact_inventory"]["by_path"]
    relative_manifest = by_path["runs/tiny-a/corpus_manifest.json"]
    absolute_manifest = by_path[str(run_dir / "corpus_manifest.json")]
    assert relative_manifest["exists"] is True
    assert absolute_manifest["exists"] is True
    assert relative_manifest["key"] == "corpus_manifest"
    assert absolute_manifest["key"] == "corpus_manifest"


def test_load_run_report_returns_markdown(tmp_path):
    write_run(tmp_path, "tiny-a")

    report = load_run_report(tmp_path, "tiny-a", "base")

    assert report["run"] == "tiny-a"
    assert report["report"] == "base"
    assert report["markdown"] == "# Base"


def test_load_run_detail_localizes_copied_run_artifacts(tmp_path):
    original = write_run(tmp_path / "runs", "original")
    copied_root = tmp_path / "copied"
    copied = copied_root / "renamed"
    shutil.copytree(original, copied)
    (original / "base" / "report.md").write_text("# Original Base", encoding="utf-8")
    (copied / "base" / "report.md").write_text("# Copied Base", encoding="utf-8")
    (copied / "corpus.txt").write_text("copied corpus", encoding="utf-8")

    detail = load_run_detail(copied_root, "renamed")
    report = load_run_report(copied_root, "renamed", "base")

    assert detail["corpus_preview"] == "copied corpus"
    assert detail["reports"]["base"]["path"] == str(copied / "base" / "report.md")
    assert report["markdown"] == "# Copied Base"
    by_path = detail["artifact_inventory"]["by_path"]
    assert by_path[str(original / "base" / "report.md")]["path"] == str(copied / "base" / "report.md")


def test_generate_run_text_uses_selected_checkpoint(tmp_path, monkeypatch):
    write_run(tmp_path, "tiny-a")
    captured = {}

    def fake_generate(config):
        captured["config"] = config
        return {
            "text": "User: hi\nAssistant: hello",
            "completion": "hello",
            "generated_tokens": [{"token": "h", "id": 4, "probability": 0.5, "logprob": -0.6931}],
            "prompt_tokens": 4,
            "total_tokens": 5,
        }

    monkeypatch.setattr("picochat.web.generate_text_with_trace", fake_generate)

    result = generate_run_text(tmp_path, {
        "run": "tiny-a",
        "checkpoint": "sft",
        "prompt": "User: hi\nAssistant:",
        "max_new_tokens": 12,
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "seed": 7,
    })

    assert result["checkpoint"] == "sft"
    assert result["completion"] == "hello"
    assert result["generated_tokens"][0]["token"] == "h"
    assert captured["config"].checkpoint_path == str(tmp_path / "tiny-a" / "sft" / "checkpoint")
    assert captured["config"].tokenizer_path == str(tmp_path / "tiny-a" / "tokenizer.json")
    assert captured["config"].top_k is None
    assert captured["config"].top_p == 0.9
    assert captured["config"].repetition_penalty == 1.2
    assert captured["config"].seed == 7


def test_preview_corpus_plan_returns_source_decisions(tmp_path):
    source_path = tmp_path / "lesson.txt"
    recipe_path = tmp_path / "corpus_recipe.json"
    source_path.write_text("lesson text", encoding="utf-8")
    recipe_path.write_text(json.dumps({
        "sources": [
            {"path": "lesson.txt", "label": "lesson"},
            {"path": "missing.txt", "label": "missing"},
        ],
    }), encoding="utf-8")

    report = preview_corpus_plan({
        "recipe_path": str(recipe_path),
        "chat_input": "domain/chat.jsonl",
        "eval_input": "domain/eval.jsonl",
        "preview_chars": 6,
    })

    assert report["recipe_path"] == str(recipe_path)
    assert report["preview"] == "lesson"
    assert report["files"][0]["label"] == "lesson"
    assert report["files"][0]["included"] is True
    assert report["files"][1]["reason"] == "missing_source"
    assert report["training_command"]["chat_input"] == "domain/chat.jsonl"
    assert report["training_command"]["eval_input"] == "domain/eval.jsonl"
    assert "--chat-input domain/chat.jsonl" in report["training_command"]["command"]
    assert report["chat_data"]["status"] == "blocked"
    assert report["eval_data"]["status"] == "blocked"
    assert not (tmp_path / "corpus_manifest.json").exists()


def test_preview_corpus_plan_accepts_dataset_pack(tmp_path):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson text", encoding="utf-8")
    chat_path.write_text(
        "\n".join(json.dumps({"user": f"q{index}", "assistant": f"a{index}"}) for index in range(8)),
        encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join([
            json.dumps({"user": "q1", "must_include": ["a1"]}),
            json.dumps({"user": "q2", "must_include": ["a2"]}),
            json.dumps({"user": "q3", "must_not_include": ["bad"]}),
            json.dumps({"user": "q4", "answerable": False, "must_include_any": [["unknown"]]}),
        ]),
        encoding="utf-8",
    )
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    report = preview_corpus_plan({
        "dataset_pack": str(pack_path),
        "preview_chars": 6,
    })

    assert report["dataset_pack"] == str(pack_path)
    assert report["input_path"] == str(source_path)
    assert report["training_command"]["dataset_pack"] == str(pack_path)
    assert "--dataset-pack" in report["training_command"]["command"]
    assert report["chat_data"]["status"] == "ready"
    assert report["eval_data"]["status"] == "ready"


def test_preview_corpus_plan_accepts_min_quality_score(tmp_path):
    source_path = tmp_path / "lesson.txt"
    source_path.write_text("tiny", encoding="utf-8")

    report = preview_corpus_plan({
        "input_path": str(source_path),
        "preview_chars": 20,
        "min_quality_score": 80,
    })

    assert report["min_quality_score"] == 80
    assert report["preview"] == ""
    assert report["files"][0]["included"] is False
    assert report["files"][0]["reason"] == "below_min_score"
    assert report["files"][0]["quality_score"] < 80


def test_init_dataset_pack_plan_creates_starter_files(tmp_path):
    corpus_dir = tmp_path / "docs"
    pack_dir = tmp_path / "my_pack"
    corpus_dir.mkdir()

    report = init_dataset_pack_plan({
        "name": "ui-pack",
        "description": "Created from the workbench.",
        "corpus_path": str(corpus_dir),
        "out_dir": str(pack_dir),
    })

    assert report["name"] == "ui-pack"
    assert report["dataset_pack"] == str(pack_dir / "dataset_pack.json")
    assert report["corpus_recipe"] == str(pack_dir / "corpus_recipe.json")
    assert report["chat_input"] == str(pack_dir / "chat.jsonl")
    assert report["eval_input"] == str(pack_dir / "eval.jsonl")
    assert report["overwritten"] == []
    assert set(report["created"]) == {
        str(pack_dir / "dataset_pack.json"),
        str(pack_dir / "corpus_recipe.json"),
        str(pack_dir / "chat.jsonl"),
        str(pack_dir / "eval.jsonl"),
    }
    assert "--dataset-pack" in report["preview_command"]
    pack = json.loads((pack_dir / "dataset_pack.json").read_text(encoding="utf-8"))
    assert pack["name"] == "ui-pack"
    assert pack["chat"] == "chat.jsonl"


def test_init_dataset_pack_plan_refuses_overwrite_without_force(tmp_path):
    corpus_path = tmp_path / "lesson.txt"
    pack_dir = tmp_path / "pack"
    corpus_path.write_text("lesson", encoding="utf-8")
    payload = {
        "name": "overwrite-pack",
        "corpus_path": str(corpus_path),
        "out_dir": str(pack_dir),
    }
    init_dataset_pack_plan(payload)

    with pytest.raises(FileExistsError):
        init_dataset_pack_plan(payload)

    report = init_dataset_pack_plan({**payload, "force": True})

    assert str(pack_dir / "dataset_pack.json") in report["overwritten"]


def test_hf_import_plan_accepts_dataset_url_and_creates_pack(tmp_path, monkeypatch):
    from picochat.hf_import import HFImportReport

    captured = {}

    def fake_import(config):
        captured["config"] = config
        out_path = tmp_path / "hf-out" / "corpus.txt"
        docs_dir = tmp_path / "hf-out" / "documents"
        docs_dir.mkdir(parents=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("first story\n\nsecond story\n", encoding="utf-8")
        (docs_dir / "row-000000.txt").write_text("first story\n", encoding="utf-8")
        (docs_dir / "row-000001.txt").write_text("second story\n", encoding="utf-8")
        return HFImportReport(
            dataset=config.dataset,
            config_name=config.config_name,
            split=config.split,
            text_column=config.text_column,
            streaming=config.streaming,
            max_rows=config.max_rows,
            min_chars=config.min_chars,
            out_path=str(out_path),
            report_path=str(tmp_path / "hf-out" / "hf_import_report.json"),
            documents_dir=str(docs_dir),
            rows_seen=2,
            rows_written=2,
            rows_skipped=0,
            characters_written=25,
            rows=(),
        )

    monkeypatch.setattr("picochat.web.import_hf_dataset", fake_import)

    report = hf_import_plan({
        "dataset_url": "https://huggingface.co/datasets/demo/stories/tree/main",
        "split": "train[:1%]",
        "text_column": "body",
        "out_dir": str(tmp_path / "hf-out"),
        "max_rows": 2,
        "min_chars": 5,
    }, runs_dir=tmp_path)

    assert captured["config"].dataset == "demo/stories"
    assert captured["config"].split == "train[:1%]"
    assert captured["config"].text_column == "body"
    assert report["dataset"] == "demo/stories"
    assert report["rows_written"] == 2
    assert report["dataset_pack"] == str(tmp_path / "hf-out" / "dataset_pack.json")
    assert report["preview"]["dataset_pack"] == report["dataset_pack"]
    assert "--dataset demo/stories" in report["command"]


def test_hf_import_plan_refuses_existing_output_without_force(tmp_path):
    out_dir = tmp_path / "hf-out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output folder already exists"):
        hf_import_plan({
            "dataset": "demo/stories",
            "out_dir": str(out_dir),
        }, runs_dir=tmp_path)


def test_error_payload_includes_hf_split_options():
    payload = _error_payload(HFSplitError(
        dataset="SWE-bench/SWE-bench_Verified",
        requested_split="train",
        available_splits=["test"],
    ))

    assert payload["error_type"] == "HFSplitError"
    assert payload["requested_split"] == "train"
    assert payload["available_splits"] == ["test"]


def test_inspect_tuning_plan_accepts_ready_dataset_pack(tmp_path):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson", encoding="utf-8")
    chat_path.write_text(
        "\n".join(
            json.dumps({"user": f"question {index}", "assistant": f"answer {index}"})
            for index in range(8)
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join([
            json.dumps({"user": "q1", "must_include": ["a1"]}),
            json.dumps({"user": "q2", "must_include": ["a2"]}),
            json.dumps({"user": "q3", "must_not_include": ["bad"]}),
            json.dumps({"user": "q4", "answerable": False, "must_include_any": [["unknown"]]}),
        ]),
        encoding="utf-8",
    )
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    report = inspect_tuning_plan({"dataset_pack": str(pack_path)})

    assert report["status"] == "ready"
    assert report["training_ready"] is True
    assert report["can_train"] is True
    assert report["chat_input"] == str(chat_path)
    assert report["eval_input"] == str(eval_path)
    assert report["chat_data"]["num_examples"] == 8
    assert report["eval_data"]["num_items"] == 4
    assert "--dataset-pack" in report["preview_command"]


def test_inspect_tuning_plan_marks_starter_pack_as_caution(tmp_path):
    corpus_path = tmp_path / "lesson.txt"
    pack_dir = tmp_path / "pack"
    corpus_path.write_text("lesson", encoding="utf-8")
    init_report = init_dataset_pack_plan({
        "name": "starter-pack",
        "corpus_path": str(corpus_path),
        "out_dir": str(pack_dir),
    })

    report = inspect_tuning_plan({"dataset_pack": init_report["dataset_pack"]})

    assert report["status"] == "caution"
    assert report["training_ready"] is False
    assert report["can_train"] is True
    assert report["chat_data"]["status"] == "caution"
    assert report["eval_data"]["status"] == "caution"
    assert any(action.startswith("Improve Chat SFT") for action in report["next_actions"])


def test_inspect_tuning_plan_rejects_pack_with_overrides(tmp_path):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be combined"):
        inspect_tuning_plan({
            "dataset_pack": str(pack_path),
            "chat_input": "other_chat.jsonl",
        })


def test_eval_starter_plan_generates_domain_eval_rows(tmp_path):
    source_path = tmp_path / "lesson.txt"
    out_path = tmp_path / "domain_eval.jsonl"
    source_path.write_text(
        "Picochat turns local text into tiny training runs. The workbench checks data quality before training.\n"
        "A careful eval includes answerable questions, refusal checks, and memorization probes.",
        encoding="utf-8",
    )

    report = eval_starter_plan({
        "input_path": str(source_path),
        "out_path": str(out_path),
        "max_items": 8,
        "seed": 7,
    })

    assert report["output_path"] == str(out_path)
    assert report["num_rows"] >= 4
    assert report["categories"]["memorization_probe"] == 1
    assert "--input" in report["command"]
    assert "--max-items 8" in report["command"]
    assert out_path.exists()
    assert out_path.with_suffix(".md").exists()


def test_eval_starter_plan_accepts_dataset_pack(tmp_path):
    source_path = tmp_path / "lesson.txt"
    pack_path = tmp_path / "dataset_pack.json"
    out_path = tmp_path / "eval.jsonl"
    source_path.write_text(
        "Coffee shop training data describes espresso, filters, and pastry pairings. "
        "The menu changes slowly and should not be treated as live news.",
        encoding="utf-8",
    )
    pack_path.write_text(json.dumps({"corpus": "lesson.txt", "chat": "chat.jsonl", "eval": "eval.jsonl"}), encoding="utf-8")

    report = eval_starter_plan({
        "dataset_pack": str(pack_path),
        "out_path": str(out_path),
        "force": True,
    })

    assert report["dataset_pack"] == str(pack_path)
    assert report["input_path"] == str(source_path)
    assert "--dataset-pack" in report["command"]
    assert report["force"] is True


def test_sft_starter_plan_generates_ready_chat_rows(tmp_path):
    source_path = tmp_path / "lesson.txt"
    out_path = tmp_path / "domain_chat.jsonl"
    source_path.write_text(
        "Picochat turns local text into tiny training runs. The workbench checks data quality before training.\n"
        "A careful assistant says when the provided domain material does not contain the answer.\n"
        "Domain training should include answerable examples, refusal examples, and memorization refusals.",
        encoding="utf-8",
    )

    report = sft_starter_plan({
        "input_path": str(source_path),
        "out_path": str(out_path),
        "max_items": 10,
        "seed": 7,
    })

    assert report["output_path"] == str(out_path)
    assert report["num_rows"] >= 8
    assert report["categories"]["refusal_memorization"] == 1
    assert "--input" in report["command"]
    assert "--max-items 10" in report["command"]
    assert out_path.exists()
    assert out_path.with_suffix(".md").exists()
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert all("user" in row and "assistant" in row for row in rows)


def test_sft_starter_plan_accepts_dataset_pack(tmp_path):
    source_path = tmp_path / "lesson.txt"
    pack_path = tmp_path / "dataset_pack.json"
    out_path = tmp_path / "chat.jsonl"
    source_path.write_text(
        "Coffee shop training data describes espresso, filters, and pastry pairings. "
        "The menu changes slowly and should not be treated as live news.",
        encoding="utf-8",
    )
    pack_path.write_text(json.dumps({"corpus": "lesson.txt", "chat": "chat.jsonl", "eval": "eval.jsonl"}), encoding="utf-8")

    report = sft_starter_plan({
        "dataset_pack": str(pack_path),
        "out_path": str(out_path),
        "force": True,
    })

    assert report["dataset_pack"] == str(pack_path)
    assert report["input_path"] == str(source_path)
    assert "--dataset-pack" in report["command"]
    assert report["force"] is True


def test_starter_plans_accept_recipe_backed_dataset_pack(tmp_path):
    docs_dir = tmp_path / "docs"
    pack_dir = tmp_path / "pack"
    docs_dir.mkdir()
    (docs_dir / "lesson.txt").write_text(
        "Coffee shop training data describes espresso, filters, and pastry pairings. "
        "The menu changes slowly and should not be treated as live news.\n"
        "A careful assistant should answer from the provided material and refuse missing facts.",
        encoding="utf-8",
    )
    init_report = init_dataset_pack_plan({
        "name": "recipe-pack",
        "corpus_path": str(docs_dir),
        "out_dir": str(pack_dir),
    })

    sft_report = sft_starter_plan({
        "dataset_pack": init_report["dataset_pack"],
        "out_path": str(pack_dir / "chat_generated.jsonl"),
        "force": True,
    })
    eval_report = eval_starter_plan({
        "dataset_pack": init_report["dataset_pack"],
        "out_path": str(pack_dir / "eval_generated.jsonl"),
        "force": True,
    })

    assert sft_report["dataset_pack"] == init_report["dataset_pack"]
    assert eval_report["dataset_pack"] == init_report["dataset_pack"]
    assert sft_report["input_path"] == init_report["corpus_recipe"]
    assert eval_report["input_path"] == init_report["corpus_recipe"]
    assert sft_report["num_documents"] == 1
    assert eval_report["num_documents"] == 1


def test_starter_plans_can_promote_generated_files_to_dataset_pack(tmp_path):
    source_path = tmp_path / "lesson.txt"
    pack_path = tmp_path / "dataset_pack.json"
    chat_path = tmp_path / "chat_generated.jsonl"
    eval_path = tmp_path / "eval_generated.jsonl"
    source_path.write_text(
        "Coffee shop training data describes espresso, filters, and pastry pairings. "
        "The menu changes slowly and should not be treated as live news.",
        encoding="utf-8",
    )
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    sft_report = sft_starter_plan({
        "dataset_pack": str(pack_path),
        "out_path": str(chat_path),
        "force": True,
        "promote_to_pack": True,
    })
    eval_report = eval_starter_plan({
        "dataset_pack": str(pack_path),
        "out_path": str(eval_path),
        "force": True,
        "promote_to_pack": True,
    })

    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    assert payload["chat"] == "chat_generated.jsonl"
    assert payload["eval"] == "eval_generated.jsonl"
    assert sft_report["promoted_to_pack"] is True
    assert eval_report["promoted_to_pack"] is True
    assert sft_report["pack_chat_input"] == str(chat_path)
    assert eval_report["pack_eval_input"] == str(eval_path)


def test_pack_editor_loads_and_saves_pack_jsonl(tmp_path):
    corpus_path = tmp_path / "lesson.txt"
    pack_dir = tmp_path / "pack"
    corpus_path.write_text("lesson", encoding="utf-8")
    init_report = init_dataset_pack_plan({
        "name": "editable-pack",
        "corpus_path": str(corpus_path),
        "out_dir": str(pack_dir),
    })

    loaded = load_pack_editor_plan({"dataset_pack": init_report["dataset_pack"]})

    assert loaded["dataset_pack"] == init_report["dataset_pack"]
    assert "Replace this with a real user question" in loaded["chat_text"]
    assert '"category": "refusal"' in loaded["chat_text"]
    assert loaded["chat_lines"] == 3

    chat_rows = [
        {"user": f"question {index}", "assistant": f"answer {index}"}
        for index in range(8)
    ]
    eval_rows = [
        {"user": "q1", "must_include": ["a1"]},
        {"user": "q2", "must_include": ["a2"]},
        {"user": "q3", "must_not_include": ["bad"]},
        {"user": "q4", "answerable": False, "must_include_any": [["unknown"]]},
    ]
    saved = save_pack_editor_plan({
        "dataset_pack": init_report["dataset_pack"],
        "chat_text": "\n".join(json.dumps(row) for row in chat_rows),
        "eval_text": "\n".join(json.dumps(row) for row in eval_rows),
    })

    assert saved["saved"] is True
    assert saved["status"] == "ready"
    assert saved["chat_data"]["num_examples"] == 8
    assert saved["eval_data"]["num_items"] == 4
    assert (pack_dir / "chat.jsonl").read_text(encoding="utf-8").endswith("\n")


def test_pack_editor_rejects_invalid_jsonl_without_overwriting(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="chat_text line 1"):
        save_pack_editor_plan({
            "chat_input": str(chat_path),
            "eval_input": str(eval_path),
            "chat_text": "{not-json",
            "eval_text": eval_path.read_text(encoding="utf-8"),
        })

    assert chat_path.read_text(encoding="utf-8") == json.dumps({"user": "hi", "assistant": "hello"})


def test_start_run_plan_launches_background_cli(tmp_path, monkeypatch):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson", encoding="utf-8")
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}), encoding="utf-8")
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")
    captured = {}

    class FakeProcess:
        pid = 4321

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr("picochat.web.subprocess.Popen", FakeProcess)

    status = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "ui run",
        "context_size": 32,
        "base_steps": 2,
        "sft_steps": 3,
        "seed": 9,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 1,
        "base_learning_rate": 0.0002,
        "sft_learning_rate": 0.0004,
        "preset": "smoke",
        "tokenizer_type": "bpe",
        "min_quality_score": 0,
    })

    job = status["job"]
    assert job["state"] == "running"
    assert job["run_name"] == "ui-run"
    assert job["preset"] == "smoke"
    assert job["min_quality_score"] == 0
    assert "--dataset-pack" in captured["command"]
    assert "--min-score" in captured["command"]
    assert "--tokenizer-type" in captured["command"]
    assert "--base-learning-rate" in captured["command"]
    assert "--sft-learning-rate" in captured["command"]
    assert "--base-early-stop-patience" in captured["command"]
    assert "--sft-early-stop-patience" in captured["command"]
    assert "--sft-sampling" in captured["command"]
    assert "bpe" in captured["command"]
    assert "0.0002" in captured["command"]
    assert "0.0004" in captured["command"]
    assert str(pack_path) in captured["command"]
    assert status["job"]["launch_config"]["base_learning_rate"] == 0.0002
    assert status["job"]["launch_config"]["sft_learning_rate"] == 0.0004
    assert status["job"]["launch_config"]["base_early_stop_patience"] == 4
    assert status["job"]["launch_config"]["sft_early_stop_patience"] == 4
    assert captured["kwargs"]["cwd"].name == "picochat"
    assert captured["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert (tmp_path / "runs" / "ui-run" / "web_run.log").exists()
    assert run_status_plan(job["id"], tmp_path / "runs")["job"]["pid"] == 4321


def test_run_progress_parser_extracts_training_and_eval_steps():
    progress = _parse_run_progress(
        "\n".join([
            "[4/6] train base model",
            "step 0030/0100 | train 3.0000 | val 3.5000 | val_bpb 1.9000 | 2.5s",
            "[5/6] train chat SFT",
            "sft step 0040/0200 | train 2.0000 | val 2.5000 | val_bpb 1.4000 | 4.0s",
            "[6/6] run chat eval",
            "done: 7/10 passed (70.00%)",
        ]),
        state="succeeded",
        summary_exists=True,
    )

    assert progress["stage"]["id"] == "complete"
    assert progress["base"]["current"] == 30
    assert progress["base"]["percent"] == 30.0
    assert progress["sft"]["current"] == 40
    assert progress["sft"]["val_bpb"] == 1.4
    assert progress["eval"]["passed"] == 7
    assert progress["eval"]["pass_rate"] == 70.0


def test_run_status_discovers_completed_web_runs_from_disk(tmp_path):
    run_dir = tmp_path / "runs" / "disk-run"
    run_dir.mkdir(parents=True)
    (run_dir / "web_run.log").write_text("$ python -m picochat.cli run tiny\ncompleted\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({
        "config": {"dataset_pack": "pack/dataset_pack.json"},
    }), encoding="utf-8")

    status = run_status_plan(runs_dir=tmp_path / "runs")

    assert status["job"]["run_name"] == "disk-run"
    assert status["job"]["state"] == "succeeded"
    assert status["job"]["source"] == "disk"
    assert status["job"]["summary_exists"] is True
    assert status["job"]["dataset_pack"] == "pack/dataset_pack.json"
    assert status["job"]["command"] == "python -m picochat.cli run tiny"


def test_archive_run_plan_moves_run_out_of_active_bank(tmp_path):
    run_dir = write_run(tmp_path / "runs", "tiny-a")

    report = archive_run_plan(tmp_path / "runs", {"run_name": "tiny-a"})

    archive_path = Path(report["archive_path"])
    assert report["archived"] is True
    assert report["run_name"] == "tiny-a"
    assert not run_dir.exists()
    assert (archive_path / "summary.json").exists()
    assert archive_path.parent.name.startswith("archive-")
    assert report["runs"] == []
    assert discover_runs(tmp_path / "runs") == []


def test_archive_run_plan_moves_multiple_runs(tmp_path):
    run_a = write_run(tmp_path / "runs", "tiny-a")
    run_b = write_run(tmp_path / "runs", "tiny-b")

    report = archive_run_plan(tmp_path / "runs", {"run_names": ["tiny-a", "tiny-b", "tiny-a"]})

    archived_names = [item["run_name"] for item in report["archived_runs"]]
    assert archived_names == ["tiny-a", "tiny-b"]
    assert not run_a.exists()
    assert not run_b.exists()
    assert Path(report["archive_root"]).name.startswith("archive-")
    for item in report["archived_runs"]:
        assert (Path(item["archive_path"]) / "summary.json").exists()
    assert discover_runs(tmp_path / "runs") == []


def test_archive_run_plan_refuses_running_job(tmp_path):
    run_dir = write_run(tmp_path / "runs", "tiny-a")

    class FakeProcess:
        def poll(self):
            return None

    with _RUN_JOBS_LOCK:
        _RUN_JOBS["archive-test"] = {
            "out_dir": str(run_dir),
            "process": FakeProcess(),
        }
    try:
        with pytest.raises(ValueError, match="cannot archive running run"):
            archive_run_plan(tmp_path / "runs", {"run_name": "tiny-a"})
    finally:
        with _RUN_JOBS_LOCK:
            _RUN_JOBS.pop("archive-test", None)

    assert run_dir.exists()


def test_archive_run_plan_refuses_nested_run_name(tmp_path):
    write_run(tmp_path / "runs" / "archive-2026-05-11", "tiny-a")

    with pytest.raises(ValueError, match="top-level run"):
        archive_run_plan(tmp_path / "runs", {"run_name": "archive-2026-05-11/tiny-a"})


def test_start_run_plan_blocks_unready_dataset_pack(tmp_path):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "missing.txt",
        "chat": "missing_chat.jsonl",
        "eval": "missing_eval.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="corpus readiness blocked"):
        start_run_plan(tmp_path / "runs", {
            "dataset_pack": str(pack_path),
            "run_name": "bad-run",
        })


def test_cancel_run_plan_terminates_active_job(tmp_path, monkeypatch):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson text " * 20, encoding="utf-8")
    chat_path.write_text(
        "\n".join(
            json.dumps({"user": f"question {index}", "assistant": f"answer {index}"})
            for index in range(8)
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join([
            json.dumps({"user": "q1", "must_include": ["a1"]}),
            json.dumps({"user": "q2", "must_include": ["a2"]}),
            json.dumps({"user": "q3", "must_not_include": ["bad"]}),
            json.dumps({"user": "q4", "answerable": False, "must_include_any": [["unknown"]]}),
        ]),
        encoding="utf-8",
    )
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    class FakeProcess:
        pid = 9876

        def __init__(self, *args, **kwargs):
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

    fake_processes = []

    def fake_popen(*args, **kwargs):
        process = FakeProcess(*args, **kwargs)
        fake_processes.append(process)
        return process

    monkeypatch.setattr("picochat.web.subprocess.Popen", fake_popen)
    started = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "cancel-me",
        "base_steps": 1,
        "sft_steps": 1,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 1,
    })

    cancelled = cancel_run_plan(tmp_path / "runs", {"job_id": started["job"]["id"]})

    assert fake_processes[0].terminated is True
    assert cancelled["job"]["state"] == "failed"
    assert cancelled["job"]["can_cancel"] is False


def test_run_presets_are_exposed_for_web_launcher():
    presets = run_presets_plan()["presets"]

    assert presets["smoke"]["base_steps"] < presets["tiny"]["base_steps"]
    assert presets["smoke"]["sft_learning_rate"] == 1e-3
    assert presets["small-local"]["sft_learning_rate"] == 3e-4
    assert presets["small-local"]["n_layer"] >= presets["tiny"]["n_layer"]


def test_start_run_plan_rejects_unknown_preset(tmp_path):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")
    (tmp_path / "lesson.txt").write_text("lesson text", encoding="utf-8")
    (tmp_path / "chat.jsonl").write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(json.dumps({"user": "hi", "must_include": ["hello"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="preset must be one of"):
        start_run_plan(tmp_path / "runs", {
            "dataset_pack": str(pack_path),
            "run_name": "bad-preset",
            "preset": "giant",
        })
