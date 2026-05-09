import json

from picochat.web import (
    discover_runs,
    generate_run_text,
    load_run_detail,
    load_run_report,
    preview_corpus_plan,
)


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
        "seed": 7,
    })

    assert result["checkpoint"] == "sft"
    assert result["completion"] == "hello"
    assert result["generated_tokens"][0]["token"] == "h"
    assert captured["config"].checkpoint_path == str(tmp_path / "tiny-a" / "sft" / "checkpoint")
    assert captured["config"].tokenizer_path == str(tmp_path / "tiny-a" / "tokenizer.json")
    assert captured["config"].top_k is None
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
