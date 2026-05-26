import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from picochat.web import (
    archive_run_plan,
    benchmark_tuning_pack_plan,
    cancel_run_plan,
    discover_runs,
    eval_starter_plan,
    generate_run_text,
    hf_import_plan,
    init_dataset_pack_plan,
    import_run_plan,
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
from picochat.honesty import inspect_data_honesty


def test_web_css_variables_are_defined():
    css = Path("src/picochat/web_assets/style.css").read_text(encoding="utf-8")
    used = set(re.findall(r"var\((--[a-z0-9-]+)", css))
    defined = set(re.findall(r"(?m)^\s*(--[a-z0-9-]+)\s*:", css))

    assert used <= defined


def test_web_scale_lane_exposes_ddp8_recipe():
    html = Path("src/picochat/web_assets/index.html").read_text(encoding="utf-8")
    js = Path("src/picochat/web_assets/app.js").read_text(encoding="utf-8")

    assert 'id="launch-n-embd" type="number" min="16" max="4096"' in html
    assert 'id="launch-n-layer" type="number" min="1" max="128"' in html
    assert 'id="launch-base-resume-from"' in html
    assert 'id="launch-sft-resume-from"' in html
    assert 'id="launch-sft-peft"' in html
    assert 'id="launch-sft-lora-targets"' in html
    assert 'id="launch-dpo-input"' in html
    assert 'id="launch-dpo-length-normalize"' in html
    assert 'option value="packed"' in html
    assert 'option value="external_flash"' in html
    assert 'option value="fa3"' in html
    assert 'option value="release_skills"' in html
    assert 'option value="skill_release"' in html
    assert 'id="scale-attn-backend"' in html
    assert "COMMANDS GENERATED" in js
    assert "SCALE_ATTN_DEFAULTS" in js
    assert 'option value="h200-1b-ddp8"' in html
    assert 'option value="h100-100m-ddp8"' in html
    assert '"h200-1b-ddp8"' in js
    assert '"h100-100m-ddp8"' in js
    assert '"release_skills"' in js
    assert '"--sft-peft"' in js
    assert '"--sft-lora-targets"' in js
    assert '"--dpo-input"' in js
    assert '"--dpo-length-normalize"' in js
    assert "const DDP_SCALE_PRESETS" in js
    assert '"torchrun"' in js
    assert "--nproc_per_node=${ddpWorldSize}" in js
    assert "ddp_world_size" in js
    assert '"--ddp-world-size"' in js
    assert '"--ddp"' in js
    assert '"--base-resume-from"' in js
    assert '"--sft-resume-from"' in js
    assert '"OMP_NUM_THREADS=1"' in js
    assert '"PICOCHAT_DDP_TIMEOUT_MINUTES=120"' in js
    assert '"TORCH_NCCL_ASYNC_ERROR_HANDLING=1"' in js
    assert '"PYTORCH_ALLOC_CONF=expandable_segments:True"' in js
    assert "python3.10-dev build-essential" in js
    assert '"bundle"' in js
    assert '"inspect-bundle"' in js
    assert '"--logs-dir"' in js
    assert "base data: token shard build" in js
    assert "Packed base data holds out complete source documents" in js
    assert "preserves BOS/EOS document boundaries" in js
    assert 'id="scale-remote-dryrun-command"' in html
    assert "REMOTE DDP DRY RUN" in js
    assert "release token-budget gates should block 100-step runs" in js
    assert "sanity, preflight, and dry run required before train" in js
    assert '"--base-steps"' in js
    assert '"--sft-steps"' in js


def test_web_ui_exposes_release_readiness_and_preflight_dry_run_controls():
    html = Path("src/picochat/web_assets/index.html").read_text(encoding="utf-8")
    js = Path("src/picochat/web_assets/app.js").read_text(encoding="utf-8")
    css = Path("src/picochat/web_assets/style.css").read_text(encoding="utf-8")

    assert "preflight-run-button" in html
    assert "run-release-panel" in html
    assert 'id="run-release-panel" class="release-readiness"' in html
    assert 'id="run-release-panel" class="release-readiness learn-only"' not in html
    assert "scale-remote-dryrun-command" in html
    assert "preflight_only" in js
    assert "RELEASE READINESS" in js
    assert "requiresGpuLaunchConfirmation" in js
    assert "Confirm paid GPU launch" in js
    assert "loss-axis-label" in js
    assert "loss-tick-label" in js
    assert "lower is better across tokenizers" in js
    assert "ARCHIVE MARKED" in html
    assert "ARCHIVE?" in js
    assert ".release-grid" in css
    assert ".loss-chart" in css
    assert ".loss-grid" in css


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
            "skipped_long_examples": 0,
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
        "domain_pass_rate": None,
        "refusal_pass_rate": None,
        "base_val_loss": 2.0,
        "sft_val_loss": 4.0,
        "num_parameters": 1234,
        "context_size": 128,
        "truncated_examples": 0,
        "skipped_long_examples": 0,
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
            document_shard_rows=config.document_shard_rows,
            document_files_written=2,
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
    assert captured["config"].document_shard_rows == 1
    assert "--document-shard-rows 1" in report["command"]


def test_hf_import_plan_auto_shards_large_climbmix_imports(tmp_path, monkeypatch):
    from picochat.hf_import import HFImportReport

    captured = {}

    def fake_import(config):
        captured["config"] = config
        return HFImportReport(
            dataset=config.dataset,
            config_name=config.config_name,
            split=config.split,
            text_column=config.text_column,
            streaming=config.streaming,
            max_rows=config.max_rows,
            min_chars=config.min_chars,
            out_path=config.out_path,
            report_path=config.report_path,
            documents_dir=config.documents_dir,
            document_shard_rows=config.document_shard_rows,
            document_files_written=800,
            rows_seen=800000,
            rows_written=796000,
            rows_skipped=4000,
            characters_written=123,
            rows=(),
            data_files=tuple(config.data_files),
        )

    monkeypatch.setattr("picochat.web.import_hf_dataset", fake_import)
    monkeypatch.setattr(
        "picochat.web.init_dataset_pack",
        lambda **_kwargs: SimpleNamespace(
            dataset_pack=str(tmp_path / "climbmix" / "dataset_pack.json"),
            corpus_recipe=str(tmp_path / "climbmix" / "corpus_recipe.json"),
            chat_input=str(tmp_path / "climbmix" / "chat.jsonl"),
            eval_input=str(tmp_path / "climbmix" / "eval.jsonl"),
        ),
    )
    monkeypatch.setattr(
        "picochat.web.preview_corpus_sources",
        lambda **_kwargs: SimpleNamespace(to_dict=lambda: {"dataset_pack": str(tmp_path / "climbmix" / "dataset_pack.json")}),
    )

    report = hf_import_plan({
        "dataset_url": "nvidia/Nemotron-ClimbMix",
        "out_dir": str(tmp_path / "climbmix"),
        "max_rows": 800000,
        "shards": 170,
        "min_chars": 100,
        "force": True,
    }, runs_dir=tmp_path)

    assert captured["config"].dataset == "karpathy/climbmix-400b-shuffle"
    assert captured["config"].max_rows == 800000
    assert captured["config"].document_shard_rows == 1000
    assert len(captured["config"].data_files) == 170
    assert report["document_shard_rows"] == 1000
    assert report["document_files_written"] == 800
    assert "--document-shard-rows 1000" in report["command"]


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


def test_benchmark_tuning_pack_plan_generates_and_promotes_curriculum(tmp_path):
    source_path = tmp_path / "lesson.txt"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("Picochat base data lives here.", encoding="utf-8")
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    report = benchmark_tuning_pack_plan({
        "dataset_pack": str(pack_path),
        "sft_rows": 64,
        "eval_rows": 24,
        "source": "offline",
        "seed": 3,
        "force": True,
        "promote_to_pack": True,
    })

    assert report["dataset_pack"] == str(pack_path)
    assert report["promoted_to_pack"] is True
    assert report["chat_data"]["status"] == "ready"
    assert report["eval_data"]["status"] == "ready"
    assert report["chat_categories"]
    assert report["eval_categories"]
    assert report["source_status"] == "offline"
    assert report["profile"] == "full"
    assert report["skill_answer_style"] == "direct"
    assert report["contamination"]["status"] == "ready"
    assert "benchmark-pack" in report["command"]
    assert "--source offline" in report["command"]
    assert "--profile full" in report["command"]
    assert "--skill-answer-style direct" in report["command"]


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


def test_generated_sft_and_eval_starters_do_not_exactly_overlap(tmp_path):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat_generated.jsonl"
    eval_path = tmp_path / "eval_generated.jsonl"
    source_path.write_text(
        "\n".join(
            f"Training document sentence {index} explains careful domain concept {index} "
            f"with enough supporting words for starter generation."
            for index in range(80)
        ),
        encoding="utf-8",
    )

    sft_starter_plan({
        "input_path": str(source_path),
        "out_path": str(chat_path),
        "force": True,
        "seed": 42,
    })
    eval_starter_plan({
        "input_path": str(source_path),
        "out_path": str(eval_path),
        "force": True,
        "seed": 42,
    })

    honesty = inspect_data_honesty(chat_path, eval_path, source_path)
    assert honesty.exact_prompt_leaks == 0
    assert all(finding.kind != "exact_sft_prompt_leak" for finding in honesty.findings)


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
    dpo_path = tmp_path / "preferences.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson", encoding="utf-8")
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}), encoding="utf-8")
    dpo_path.write_text(json.dumps({
        "prompt": "say hi",
        "chosen": "hi",
        "rejected": "bye",
    }) + "\n", encoding="utf-8")
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
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("PICOCHAT_DDP_TIMEOUT_MINUTES", raising=False)

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
        "base_optimizer": "muon",
        "sft_optimizer": "adamw",
        "base_muon_learning_rate": 0.01,
        "base_ema_decay": 0.5,
        "base_dataset_mode": "sharded",
        "base_shard_token_size": 64000,
        "base_shard_cache_size": 3,
        "sft_packing": "bos_bestfit",
        "sft_peft": "lora",
        "sft_lora_rank": 4,
        "sft_lora_alpha": 8.0,
        "sft_lora_dropout": 0.05,
        "sft_lora_targets": "attn_qkv",
        "dpo_input": str(dpo_path),
        "dpo_steps": 5,
        "dpo_batch_size": 2,
        "dpo_learning_rate": 0.000005,
        "dpo_beta": 0.2,
        "dpo_grad_accum_steps": 3,
        "dpo_lr_warmup_steps": 1,
        "dpo_lr_decay": "cosine",
        "dpo_grad_clip": 1.0,
        "dpo_early_stop_patience": 2,
        "dpo_eval_batches": 4,
        "dpo_length_normalize": True,
        "long_run_gate_profile": "first_release",
        "preset": "smoke",
        "tokenizer_type": "bpe",
        "min_quality_score": 0,
    })

    job = status["job"]
    assert job["state"] == "running"
    assert job["run_name"] == "ui-run"
    assert job["preset"] == "smoke"
    assert job["min_quality_score"] == 0
    assert job["launch_preflight"]["status"] in {"ready", "warn"}
    assert job["launch_preflight"]["budget"]["estimated_parameters"] > 0
    assert job["launch_preflight"]["budget"]["recommended_base_steps"] >= 1
    assert "--dataset-pack" in captured["command"]
    assert "--min-score" in captured["command"]
    assert "--tokenizer-type" in captured["command"]
    assert "--base-learning-rate" in captured["command"]
    assert "--sft-learning-rate" in captured["command"]
    assert "--base-optimizer" in captured["command"]
    assert "--base-ema-decay" in captured["command"]
    assert "--base-dataset-mode" in captured["command"]
    assert "--base-shard-token-size" in captured["command"]
    assert "--base-shard-cache-size" in captured["command"]
    assert "--base-early-stop-patience" in captured["command"]
    assert "--sft-early-stop-patience" in captured["command"]
    assert "--sft-sampling" in captured["command"]
    assert "--sft-packing" in captured["command"]
    assert "--sft-peft" in captured["command"]
    assert "--sft-lora-rank" in captured["command"]
    assert "--sft-lora-targets" in captured["command"]
    assert "--dpo-input" in captured["command"]
    assert "--dpo-length-normalize" in captured["command"]
    assert "--target-param-data-ratio" in captured["command"]
    assert "--long-run-gate-profile" in captured["command"]
    assert "bpe" in captured["command"]
    assert "0.0002" in captured["command"]
    assert "0.0004" in captured["command"]
    assert str(pack_path) in captured["command"]
    assert status["job"]["launch_config"]["base_learning_rate"] == 0.0002
    assert status["job"]["launch_config"]["sft_learning_rate"] == 0.0004
    assert status["job"]["launch_config"]["base_optimizer"] == "muon"
    assert status["job"]["launch_config"]["base_ema_decay"] == 0.5
    assert status["job"]["launch_config"]["base_dataset_mode"] == "sharded"
    assert status["job"]["launch_config"]["base_shard_token_size"] == 64000
    assert status["job"]["launch_config"]["base_shard_cache_size"] == 3
    assert status["job"]["launch_config"]["sft_packing"] == "bos_bestfit"
    assert status["job"]["launch_config"]["sft_peft"] == "lora"
    assert status["job"]["launch_config"]["sft_lora_rank"] == 4
    assert status["job"]["launch_config"]["sft_lora_alpha"] == 8.0
    assert status["job"]["launch_config"]["sft_lora_dropout"] == 0.05
    assert status["job"]["launch_config"]["sft_lora_targets"] == ["attn_qkv"]
    assert status["job"]["launch_config"]["dpo_input"] == str(dpo_path)
    assert status["job"]["launch_config"]["dpo_steps"] == 5
    assert status["job"]["launch_config"]["dpo_batch_size"] == 2
    assert status["job"]["launch_config"]["dpo_learning_rate"] == 0.000005
    assert status["job"]["launch_config"]["dpo_beta"] == 0.2
    assert status["job"]["launch_config"]["dpo_grad_accum_steps"] == 3
    assert status["job"]["launch_config"]["dpo_lr_warmup_steps"] == 1
    assert status["job"]["launch_config"]["dpo_lr_decay"] == "cosine"
    assert status["job"]["launch_config"]["dpo_grad_clip"] == 1.0
    assert status["job"]["launch_config"]["dpo_early_stop_patience"] == 2
    assert status["job"]["launch_config"]["dpo_eval_batches"] == 4
    assert status["job"]["launch_config"]["dpo_length_normalize"] is True
    assert status["job"]["launch_config"]["long_run_gate_profile"] == "first_release"
    assert status["job"]["launch_config"]["base_early_stop_patience"] == 4
    assert status["job"]["launch_config"]["sft_early_stop_patience"] == 4
    assert captured["kwargs"]["cwd"].name == "picochat"
    assert captured["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert (tmp_path / "runs" / "ui-run" / "web_run.log").exists()
    assert run_status_plan(job["id"], tmp_path / "runs")["job"]["pid"] == 4321


def test_start_run_plan_preflight_only_does_not_launch_or_write_run_dir(tmp_path, monkeypatch):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson text " * 40, encoding="utf-8")
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}), encoding="utf-8")
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    def fail_popen(*args, **kwargs):
        raise AssertionError("preflight-only must not launch a subprocess")

    monkeypatch.setattr("picochat.web.subprocess.Popen", fail_popen)

    status = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "dry-run",
        "preset": "smoke",
        "context_size": 32,
        "base_steps": 2,
        "sft_steps": 2,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 1,
        "preflight_only": True,
    })

    job = status["job"]
    assert job["state"] == "preflight"
    assert job["source"] == "preflight"
    assert job["launch_preflight"]["budget"]["estimated_parameters"] > 0
    assert "--dataset-pack" in job["command"]
    assert not (tmp_path / "runs" / "dry-run").exists()


def test_start_run_plan_preflight_only_returns_blocked_preflight_without_launch(tmp_path):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson text about local models\n" * 200, encoding="utf-8")
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}), encoding="utf-8")
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    status = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "unsafe-dry-run",
        "preset": "small-local",
        "tokenizer_type": "bpe",
        "tokenizer_vocab_size": 1024,
        "context_size": 512,
        "n_embd": 192,
        "n_head": 6,
        "n_layer": 6,
        "base_steps": 5000,
        "sft_steps": 1000,
        "base_batch_size": 8,
        "sft_batch_size": 8,
        "preflight_only": True,
    })

    assert status["job"]["state"] == "preflight"
    assert status["job"]["returncode"] == 2
    assert status["job"]["launch_preflight"]["status"] == "blocked"
    assert not (tmp_path / "runs" / "unsafe-dry-run").exists()


def test_start_run_plan_accepts_resume_paths_for_existing_run_dir(tmp_path, monkeypatch):
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
    run_dir = tmp_path / "runs" / "resume-run"
    base_resume = run_dir / "base" / "resume_checkpoint"
    sft_resume = run_dir / "sft" / "resume_checkpoint"
    base_resume.mkdir(parents=True)
    sft_resume.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("partial run", encoding="utf-8")
    captured = {}

    class FakeProcess:
        pid = 9876

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr("picochat.web.subprocess.Popen", FakeProcess)

    status = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "resume run",
        "preset": "smoke",
        "base_resume_from": str(base_resume),
        "sft_resume_from": str(sft_resume),
    })

    assert status["job"]["state"] == "running"
    assert "--base-resume-from" in captured["command"]
    assert "--sft-resume-from" in captured["command"]
    assert str(base_resume) in captured["command"]
    assert str(sft_resume) in captured["command"]
    assert status["job"]["launch_config"]["base_resume_from"] == str(base_resume)
    assert status["job"]["launch_config"]["sft_resume_from"] == str(sft_resume)


def test_start_run_plan_rejects_sft_resume_without_base_resume(tmp_path):
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
    sft_resume = tmp_path / "runs" / "resume-run" / "sft" / "resume_checkpoint"
    sft_resume.mkdir(parents=True)

    with pytest.raises(ValueError, match="sft_resume_from requires base_resume_from"):
        start_run_plan(tmp_path / "runs", {
            "dataset_pack": str(pack_path),
            "run_name": "resume run",
            "preset": "smoke",
            "sft_resume_from": str(sft_resume),
        })


def test_start_run_plan_preserves_h100_100m_ddp8_preset(tmp_path, monkeypatch):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson\n" * 100, encoding="utf-8")
    chat_path.write_text(json.dumps({"user": "who", "assistant": "Picochat"}) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "who", "must_include": ["Picochat"]}) + "\n", encoding="utf-8")
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
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("PICOCHAT_DDP_TIMEOUT_MINUTES", raising=False)

    started = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "ddp8",
        "preset": "h100-100m-ddp8",
        "allow_unsafe_long_run": True,
    })

    command = captured["command"]
    assert "torch.distributed.run" in command
    assert "--nproc_per_node=8" in command
    assert "-m" in command
    assert "picochat.cli" in command
    assert "--ddp" in command
    assert "--ddp-world-size" in command
    assert "8" in command
    assert "--n-embd" in command
    assert "768" in command
    assert "--n-layer" in command
    assert "16" in command
    assert "--loss-spike-rollback" not in command
    assert started["job"]["launch_config"]["n_embd"] == 768
    assert started["job"]["launch_config"]["n_layer"] == 16
    assert started["job"]["launch_config"]["ddp"] is True
    assert started["job"]["launch_config"]["ddp_world_size"] == 8
    assert started["job"]["launch_config"]["loss_spike_rollback"] is False
    assert captured["kwargs"]["env"]["OMP_NUM_THREADS"] == "1"
    assert captured["kwargs"]["env"]["PICOCHAT_DDP_TIMEOUT_MINUTES"] == "120"
    assert captured["kwargs"]["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_start_run_plan_rejects_ddp_loss_spike_rollback_even_when_unsafe_allowed(tmp_path):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson\n" * 100, encoding="utf-8")
    chat_path.write_text(json.dumps({"user": "who", "assistant": "Picochat"}) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "who", "must_include": ["Picochat"]}) + "\n", encoding="utf-8")
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="loss spike rollback is not supported with DDP"):
        start_run_plan(tmp_path / "runs", {
            "dataset_pack": str(pack_path),
            "run_name": "bad-ddp-rollback",
            "preset": "h100-100m-ddp8",
            "loss_spike_rollback": True,
            "allow_unsafe_long_run": True,
        })


def test_run_progress_parser_extracts_training_and_eval_steps():
    progress = _parse_run_progress(
        "\n".join([
            "[4/7] train base model",
            "step 0030/0100 | train 3.0000 | val 3.5000 | val_bpb 1.9000 | 12.5k tok/s | 2.5s",
            "[5/7] train chat SFT",
            "sft step 0040/0200 | train 2.0000 | val 2.5000 | val_bpb 1.4000 | 4.0s",
            "[6/7] run SFT fit diagnostic",
            "[7/7] run chat eval",
            "done: 7/10 passed (70.00%)",
        ]),
        state="succeeded",
        summary_exists=True,
    )

    assert progress["stage"]["id"] == "complete"
    assert progress["base"]["current"] == 30
    assert progress["base"]["percent"] == 30.0
    assert progress["base"]["tokens_per_sec"] == 12500.0
    assert progress["sft"]["current"] == 40
    assert progress["sft"]["val_bpb"] == 1.4
    assert progress["sft"]["tokens_per_sec"] is None
    assert progress["eval"]["passed"] == 7
    assert progress["eval"]["pass_rate"] == 70.0


def test_run_progress_parser_understands_sft_fit_stage():
    progress = _parse_run_progress("[6/7] run SFT fit diagnostic")

    assert progress["stage"]["id"] == "sft_fit"
    assert progress["stage"]["index"] == 6
    assert progress["stage"]["total"] == 7


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


def test_archive_run_plan_moves_failed_no_summary_run(tmp_path):
    run_dir = tmp_path / "runs" / "failed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "web_run.log").write_text("Traceback: failed before summary\n", encoding="utf-8")
    (run_dir / "web_returncode.txt").write_text("1\n", encoding="utf-8")

    report = archive_run_plan(tmp_path / "runs", {"run_name": "failed-run"})

    archive_path = Path(report["archive_path"])
    assert report["archived"] is True
    assert not run_dir.exists()
    assert (archive_path / "web_run.log").exists()
    assert report["archived_runs"][0]["summary_exists"] is False
    assert run_status_plan(runs_dir=tmp_path / "runs")["jobs"] == []


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


def test_archive_run_plan_removes_completed_memory_job(tmp_path):
    run_dir = write_run(tmp_path / "runs", "tiny-a")

    class FakeProcess:
        pid = 1234

        def poll(self):
            return 0

    with _RUN_JOBS_LOCK:
        _RUN_JOBS["archive-complete"] = {
            "id": "archive-complete",
            "run_name": "tiny-a",
            "out_dir": str(run_dir),
            "dataset_pack": "pack.json",
            "log_path": str(run_dir / "web_run.log"),
            "command": "python -m picochat.cli run tiny",
            "started_at": 0,
            "process": FakeProcess(),
        }
    try:
        archive_run_plan(tmp_path / "runs", {"run_name": "tiny-a"})
        with _RUN_JOBS_LOCK:
            assert "archive-complete" not in _RUN_JOBS
        status = run_status_plan(runs_dir=tmp_path / "runs")
        assert all(job["run_name"] != "tiny-a" for job in status["jobs"])
    finally:
        with _RUN_JOBS_LOCK:
            _RUN_JOBS.pop("archive-complete", None)


def test_archive_run_plan_refuses_nested_run_name(tmp_path):
    write_run(tmp_path / "runs" / "archive-2026-05-11", "tiny-a")

    with pytest.raises(ValueError, match="top-level run"):
        archive_run_plan(tmp_path / "runs", {"run_name": "archive-2026-05-11/tiny-a"})


def test_import_run_plan_copies_external_completed_run(tmp_path):
    source = write_run(tmp_path / "colab" / "runs", "gpu-run")

    report = import_run_plan(tmp_path / "runs", {
        "source_path": str(source),
        "run_name": "gpu-run-imported",
    })

    destination = Path(report["destination"])
    assert report["imported"] is True
    assert destination.parent == tmp_path / "runs"
    assert (destination / "summary.json").exists()
    assert report["runs"][0]["name"] == "gpu-run-imported"


def test_import_run_plan_rejects_folder_without_summary(tmp_path):
    source = tmp_path / "not-a-run"
    source.mkdir()

    with pytest.raises(ValueError, match="summary.json"):
        import_run_plan(tmp_path / "runs", {"source_path": str(source)})


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


def test_start_run_plan_blocks_unsafe_long_run(tmp_path):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("lesson text about local models\n" * 200, encoding="utf-8")
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}), encoding="utf-8")
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="long-run preflight blocked"):
        start_run_plan(tmp_path / "runs", {
            "dataset_pack": str(pack_path),
            "run_name": "unsafe-run",
            "preset": "small-local",
            "tokenizer_type": "bpe",
            "tokenizer_vocab_size": 1024,
            "context_size": 512,
            "n_embd": 192,
            "n_head": 6,
            "n_layer": 6,
            "base_steps": 5000,
            "sft_steps": 1000,
            "base_batch_size": 8,
            "sft_batch_size": 8,
        })

    assert not (tmp_path / "runs" / "unsafe-run").exists()


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
        "norm_type": "rmsnorm",
        "position_encoding": "rope",
        "activation": "relu2",
    })

    cancelled = cancel_run_plan(tmp_path / "runs", {"job_id": started["job"]["id"]})

    assert fake_processes[0].terminated is True
    assert cancelled["job"]["state"] == "failed"
    assert cancelled["job"]["can_cancel"] is False
    assert "--norm-type rmsnorm" in started["job"]["command"]
    assert "--position-encoding rope" in started["job"]["command"]
    assert "--activation relu2" in started["job"]["command"]


def test_start_run_plan_accepts_swiglu_activation(tmp_path, monkeypatch):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")
    (tmp_path / "lesson.txt").write_text("lesson text", encoding="utf-8")
    (tmp_path / "chat.jsonl").write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")

    class FakeProcess:
        pid = 1234
        returncode = None

        def poll(self):
            return self.returncode

    monkeypatch.setattr("picochat.web.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    started = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "swiglu",
        "scale": "smoke",
        "base_steps": 1,
        "sft_steps": 1,
        "activation": "swiglu",
        "n_kv_head": 2,
        "bpe_pretokenizer": "regex",
        "tie_embeddings": True,
        "qk_norm": True,
        "parallel_residual": True,
        "xsa_last_n": 2,
        "scaled_residual_init": True,
        "precision": "bf16",
        "matmul_precision": "high",
        "attn_backend": "math",
        "base_dataset_mode": "packed",
        "torch_compile": True,
        "torch_compile_mode": "reduce-overhead",
        "gradient_checkpointing": True,
        "auto_lr_scaling": True,
        "loss_spike_rollback": True,
        "long_run_gate_profile": "first_release",
    })

    assert "--activation swiglu" in started["job"]["command"]
    assert "--n-kv-head 2" in started["job"]["command"]
    assert "--bpe-pretokenizer regex" in started["job"]["command"]
    assert "--tie-embeddings" in started["job"]["command"]
    assert "--qk-norm" in started["job"]["command"]
    assert "--parallel-residual" in started["job"]["command"]
    assert "--xsa-last-n 2" in started["job"]["command"]
    assert "--scaled-residual-init" in started["job"]["command"]
    assert "--precision bf16" in started["job"]["command"]
    assert "--matmul-precision high" in started["job"]["command"]
    assert "--attn-backend math" in started["job"]["command"]
    assert "--base-dataset-mode packed" in started["job"]["command"]
    assert "--torch-compile" in started["job"]["command"]
    assert "--torch-compile-mode reduce-overhead" in started["job"]["command"]
    assert "--gradient-checkpointing" in started["job"]["command"]
    assert "--auto-lr-scaling" in started["job"]["command"]
    assert "--loss-spike-rollback" in started["job"]["command"]
    assert "--long-run-gate-profile first_release" in started["job"]["command"]
    assert started["job"]["launch_config"]["n_kv_head"] == 2
    assert started["job"]["launch_config"]["bpe_pretokenizer"] == "regex"
    assert started["job"]["launch_config"]["tie_embeddings"] is True
    assert started["job"]["launch_config"]["qk_norm"] is True
    assert started["job"]["launch_config"]["parallel_residual"] is True
    assert started["job"]["launch_config"]["xsa_last_n"] == 2
    assert started["job"]["launch_config"]["scaled_residual_init"] is True
    assert started["job"]["launch_config"]["precision"] == "bf16"
    assert started["job"]["launch_config"]["matmul_precision"] == "high"
    assert started["job"]["launch_config"]["attn_backend"] == "math"
    assert started["job"]["launch_config"]["base_dataset_mode"] == "packed"
    assert started["job"]["launch_config"]["torch_compile"] is True
    assert started["job"]["launch_config"]["torch_compile_mode"] == "reduce-overhead"
    assert started["job"]["launch_config"]["gradient_checkpointing"] is True
    assert started["job"]["launch_config"]["auto_lr_scaling"] is True
    assert started["job"]["launch_config"]["loss_spike_rollback"] is True
    assert started["job"]["launch_config"]["long_run_gate_profile"] == "first_release"


def test_start_run_plan_accepts_leaky_relu2_activation(tmp_path, monkeypatch):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")
    (tmp_path / "lesson.txt").write_text("lesson text", encoding="utf-8")
    (tmp_path / "chat.jsonl").write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")

    class FakeProcess:
        pid = 1234
        returncode = None

        def poll(self):
            return self.returncode

    monkeypatch.setattr("picochat.web.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    started = start_run_plan(tmp_path / "runs", {
        "dataset_pack": str(pack_path),
        "run_name": "leaky",
        "scale": "smoke",
        "base_steps": 1,
        "sft_steps": 1,
        "activation": "leaky_relu2",
    })

    assert "--activation leaky_relu2" in started["job"]["command"]


def test_start_run_plan_rejects_invalid_runtime_knob(tmp_path):
    pack_path = tmp_path / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": "lesson.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")
    (tmp_path / "lesson.txt").write_text("lesson text", encoding="utf-8")
    (tmp_path / "chat.jsonl").write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attn_backend"):
        start_run_plan(tmp_path / "runs", {
            "dataset_pack": str(pack_path),
            "run_name": "bad-runtime",
            "base_steps": 1,
            "sft_steps": 1,
            "attn_backend": "imaginary",
        })

    with pytest.raises(ValueError, match="long_run_gate_profile"):
        start_run_plan(tmp_path / "runs", {
            "dataset_pack": str(pack_path),
            "run_name": "bad-gate-profile",
            "base_steps": 1,
            "sft_steps": 1,
            "long_run_gate_profile": "marketing",
        })


def test_run_presets_are_exposed_for_web_launcher():
    presets = run_presets_plan()["presets"]

    assert presets["smoke"]["base_steps"] < presets["tiny"]["base_steps"]
    assert presets["smoke"]["sft_learning_rate"] == 1e-3
    assert presets["small-local"]["sft_learning_rate"] == 3e-4
    assert presets["small-local"]["n_layer"] >= presets["tiny"]["n_layer"]
    assert presets["h100-100m"]["n_embd"] == 768
    assert presets["h100-100m"]["base_steps"] == 33000
    assert presets["h100-100m"]["long_run_gate_profile"] == "skill_release"
    assert presets["h100-100m-ddp8"]["base_steps"] == 4100
    assert presets["h100-100m-ddp8"]["sft_steps"] == 24
    assert presets["h100-100m-ddp8"]["ddp"] is True
    assert presets["h100-100m-ddp8"]["loss_spike_rollback"] is False
    assert presets["h100-100m-ddp8"]["long_run_gate_profile"] == "skill_release"
    assert presets["h200-1b-ddp8"]["n_embd"] == 2048
    assert presets["h200-1b-ddp8"]["context_size"] == 2048
    assert presets["h200-1b-ddp8"]["base_grad_accum_steps"] == 8
    assert presets["h200-1b-ddp8"]["attn_backend"] == "fa3"
    assert presets["h200-1b-ddp8"]["scaled_residual_init"] is True
    assert presets["h200-1b-ddp8"]["ddp"] is True
    assert presets["h200-1b-ddp8"]["long_run_gate_profile"] == "skill_release"


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
