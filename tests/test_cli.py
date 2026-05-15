from picochat.cli import main


def test_cli_version(capsys):
    exit_code = main(["--version"])

    assert exit_code == 0
    assert "picochat" in capsys.readouterr().out


def test_cli_sanity_preh100(tmp_path, capsys, monkeypatch):
    def fake_run(config):
        assert config.out_dir == str(tmp_path / "sanity")
        assert config.precision == "float32"
        assert config.matmul_precision == "high"
        assert config.attn_backend == "math"
        assert config.include_compile is True
        return {
            "status": "passed",
            "report_path": str(tmp_path / "sanity" / "preh100_sanity.json"),
            "markdown_path": str(tmp_path / "sanity" / "preh100_sanity.md"),
            "checks": [
                {"name": "precision_backward", "status": "pass", "detail": "ok"},
            ],
        }

    monkeypatch.setattr("picochat.cli.run_preh100_sanity", fake_run)

    exit_code = main([
        "sanity",
        "preh100",
        "--out-dir",
        str(tmp_path / "sanity"),
        "--precision",
        "float32",
        "--matmul-precision",
        "high",
        "--attn-backend",
        "math",
        "--include-compile",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "sanity: passed" in output
    assert "precision_backward: pass ok" in output


def test_cli_run_tiny_multiseed(tmp_path, capsys, monkeypatch):
    def fake_run(config, n_seeds):
        assert config.out_dir == str(tmp_path / "multi")
        assert config.seed == 7
        assert n_seeds == 3
        return {
            "aggregate": {
                "eval_pass_rate": {
                    "n": 3,
                    "mean": 0.25,
                    "std": 0.05,
                },
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny_multiseed", fake_run)

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "multi"),
        "--n-seeds",
        "3",
        "--seed",
        "7",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "multi-seed tiny run: n=3 eval pass mean 25.00% std 5.00%" in output


def test_cli_train_sft_sweep_uses_dataset_pack(tmp_path, capsys, monkeypatch):
    import json

    corpus = tmp_path / "corpus.txt"
    chat = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack = tmp_path / "dataset_pack.json"
    corpus.write_text("base corpus\n", encoding="utf-8")
    chat.write_text(json.dumps({"user": "who", "assistant": "Picochat"}) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "who", "must_include": ["Picochat"]}) + "\n", encoding="utf-8")
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    def fake_run(config):
        assert config.input_path == str(chat)
        assert config.eval_input_path == str(eval_path)
        assert config.support_corpus_path == str(corpus)
        assert config.tokenizer_path == "tok.json"
        assert config.checkpoint_path == "base"
        assert config.eval_log_every == 50
        assert config.matmul_precision == "default"
        return {
            "best_sft_fit": {
                "candidate": "uniform-lr1em04-steps1",
                "sft_fit_pass_rate": 0.9,
            },
            "best_eval": {
                "candidate": "uniform-lr1em04-steps1",
                "eval_pass_rate": 0.4,
            },
        }

    monkeypatch.setattr("picochat.cli.run_sft_sweep", fake_run)

    exit_code = main([
        "train",
        "sft-sweep",
        "--dataset-pack",
        str(pack),
        "--tokenizer",
        "tok.json",
        "--checkpoint",
        "base",
        "--out-dir",
        str(tmp_path / "sweep"),
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "sft sweep report:" in output
    assert "best sft fit: uniform-lr1em04-steps1 (90.00%)" in output
    assert "best eval: uniform-lr1em04-steps1 (40.00%)" in output


def test_cli_tokenizer_train(tmp_path, capsys):
    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("hello picochat", encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
    ])

    assert exit_code == 0
    assert tokenizer_path.exists()
    output = capsys.readouterr().out
    assert "trained tokenizer" in output
    assert "type: char" in output
    assert "vocab_size" in output


def test_cli_tokenizer_train_byte(tmp_path, capsys):
    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("hello café", encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
        "--type",
        "byte",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "type: byte" in output
    assert "vocab_size: 260" in output


def test_cli_tokenizer_train_bpe(tmp_path, capsys):
    import json

    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("picochat picochat learns small stories", encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
        "--type",
        "bpe",
        "--vocab-size",
        "32",
        "--min-freq",
        "2",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert "type: bpe" in output
    assert data["type"] == "bpe"
    assert data["merges"]


def test_cli_tokenizer_train_hf_bpe(tmp_path, capsys):
    import json
    import pytest

    pytest.importorskip("tokenizers")
    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("picochat trains fast tokenizers\n" * 20, encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
        "--type",
        "hf_bpe",
        "--vocab-size",
        "300",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert "type: hf_bpe" in output
    assert data["type"] == "hf_bpe"
    assert data["backend"] == "huggingface_tokenizers"


def test_cli_data_inspect(tmp_path, capsys):
    data_path = tmp_path / "data.txt"
    data_path.write_text("hello\npicochat\n", encoding="utf-8")

    exit_code = main(["data", "inspect", "--input", str(data_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "num_documents" in output
    assert "num_characters" in output


def test_cli_data_build(tmp_path, capsys):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "corpus.txt"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha", encoding="utf-8")

    exit_code = main(["data", "build", "--input", str(input_dir), "--out", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert (output_path.parent / "corpus_manifest.json").exists()
    output = capsys.readouterr().out
    assert "built corpus" in output
    assert "manifest:" in output


def test_cli_data_build_from_recipe(tmp_path, capsys):
    import json

    source_path = tmp_path / "lesson.txt"
    recipe_path = tmp_path / "corpus.recipe.json"
    output_path = tmp_path / "corpus.txt"
    source_path.write_text("lesson text", encoding="utf-8")
    recipe_path.write_text(json.dumps({
        "sources": [
            {"path": "lesson.txt", "label": "lesson"},
        ],
    }), encoding="utf-8")

    exit_code = main(["data", "build", "--recipe", str(recipe_path), "--out", str(output_path)])

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "lesson text\n"
    manifest = json.loads((output_path.parent / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recipe_path"] == str(recipe_path)
    assert manifest["files"][0]["label"] == "lesson"
    output = capsys.readouterr().out
    assert "built corpus" in output


def test_cli_run_tiny_preflight_only(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    chat = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack = tmp_path / "dataset_pack.json"
    corpus.write_text("Picochat trains tiny local models.\n" * 20, encoding="utf-8")
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "run"),
        "--dataset-pack",
        str(pack),
        "--preflight-only",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Picochat Run Preflight" in output
    assert "Recommended base steps" in output
    assert "assistant_only_masking" in output


def test_cli_data_skills_corpus(tmp_path, capsys):
    base_path = tmp_path / "base.txt"
    out_path = tmp_path / "skills.txt"
    docs_path = tmp_path / "skills_docs"
    recipe_path = tmp_path / "skills_recipe.json"
    base_path.write_text("base text", encoding="utf-8")

    exit_code = main([
        "data",
        "skills-corpus",
        "--out",
        str(out_path),
        "--math-rows",
        "3",
        "--spelling-rows",
        "3",
        "--choice-rows",
        "2",
        "--base-corpus",
        str(base_path),
        "--recipe-out",
        str(recipe_path),
        "--documents-dir",
        str(docs_path),
        "--rows-per-shard",
        "4",
    ])

    assert exit_code == 0
    assert out_path.exists()
    assert len(list(docs_path.glob("shard-*.txt"))) == 2
    assert recipe_path.exists()
    output = capsys.readouterr().out
    assert "skills corpus:" in output
    assert "documents_dir:" in output
    assert "shards: 2" in output
    assert "skills_math" in output
    assert "recipe:" in output


def test_cli_data_benchmark_pack(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains tiny local models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "corpus": str(corpus),
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "benchmark-pack",
        "--dataset-pack",
        str(pack),
        "--sft-rows",
        "32",
        "--eval-rows",
        "16",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "benchmark chat SFT:" in output
    assert "benchmark eval:" in output
    assert "source_status: offline" in output
    assert "profile: full" in output
    assert "skill_answer_style: direct" in output
    assert "contamination: ready" in output
    assert "promoted_to_pack: True" in output
    assert (tmp_path / "chat_benchmark.jsonl").exists()
    assert (tmp_path / "eval_benchmark.jsonl").exists()


def test_cli_data_slice_pack(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains tiny local models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        json.dumps({"user": "who", "assistant": "Picochat", "category": "identity"})
        + "\n"
        + json.dumps({"user": "add", "assistant": "4", "category": "bench_math_addition"})
        + "\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"user": "who", "must_include": ["Picochat"], "category": "identity"})
        + "\n"
        + json.dumps({"user": "add", "must_include": ["4"], "category": "bench_math_addition"})
        + "\n",
        encoding="utf-8",
    )
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "slice-pack",
        "--dataset-pack",
        str(pack),
        "--out-dir",
        str(tmp_path / "identity-pack"),
        "--include-categories",
        "identity",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "sliced dataset pack:" in output
    assert "chat rows: 1/2" in output
    assert "eval rows: 1/2" in output
    assert "- identity: 1" in output
    assert (tmp_path / "identity-pack" / "dataset_pack.json").exists()
    assert (tmp_path / "identity-pack" / "tuning_slice.md").exists()


def test_cli_data_stage_pack(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains tiny local models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        json.dumps({"user": "who", "assistant": "Picochat", "category": "identity"})
        + "\n"
        + json.dumps({"user": "unknown", "assistant": "I do not know.", "category": "refusal"})
        + "\n"
        + json.dumps({"user": "choice", "assistant": "A", "category": "bench_choice_language"})
        + "\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"user": "who", "must_include": ["Picochat"], "category": "identity"})
        + "\n"
        + json.dumps({"user": "unknown", "must_include": ["I do not know"], "category": "refusal"})
        + "\n"
        + json.dumps({"user": "choice", "must_include": ["A"], "category": "bench_choice_language"})
        + "\n",
        encoding="utf-8",
    )
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "stage-pack",
        "--dataset-pack",
        str(pack),
        "--out-dir",
        str(tmp_path / "staged"),
        "--stages",
        "behavior,choice",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "staged tuning pack:" in output
    assert "- behavior: chat 2/3, eval 2/3" in output
    assert "- choice: chat 1/3, eval 1/3" in output
    assert (tmp_path / "staged" / "staged_tuning_pack.md").exists()
    assert (tmp_path / "staged" / "behavior" / "dataset_pack.json").exists()
    assert (tmp_path / "staged" / "choice" / "dataset_pack.json").exists()


def test_cli_data_preview_from_recipe(tmp_path, capsys):
    import json

    source_path = tmp_path / "lesson.txt"
    recipe_path = tmp_path / "corpus.recipe.json"
    source_path.write_text("lesson text", encoding="utf-8")
    recipe_path.write_text(json.dumps({
        "sources": [
            {"path": "lesson.txt", "label": "lesson"},
        ],
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "preview",
        "--recipe",
        str(recipe_path),
        "--preview-chars",
        "6",
        "--chat-input",
        "domain/chat.jsonl",
        "--eval-input",
        "domain/eval.jsonl",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "source plan:" in output
    assert "readiness:" in output
    assert "readiness checks:" in output
    assert "training budget:" in output
    assert "suggested_context_size" in output
    assert "suggested run:" in output
    assert "run tiny" in output
    assert "chat_input: domain/chat.jsonl" in output
    assert "eval_input: domain/eval.jsonl" in output
    assert "--chat-input domain/chat.jsonl" in output
    assert "chat/eval data:" in output
    assert "chat_sft: blocked" in output
    assert "eval: blocked" in output
    assert "include" in output
    assert "score=" in output
    assert "label=lesson" in output
    assert "lesson" in output
    assert not (tmp_path / "corpus_manifest.json").exists()


def test_cli_data_preview_limits_large_source_plan(tmp_path, capsys):
    for index in range(30):
        (tmp_path / f"doc-{index:02d}.txt").write_text(f"document {index} text", encoding="utf-8")

    exit_code = main([
        "data",
        "preview",
        "--input",
        str(tmp_path),
        "--preview-chars",
        "0",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "25 more" not in output
    assert "5 more source file(s) omitted" in output
    assert "include" in output
    assert "score=" in output
    assert "doc-00.txt" in output
    assert "doc-29.txt" not in output
    assert not (tmp_path / "corpus_manifest.json").exists()


def test_cli_data_preview_from_dataset_pack(capsys):
    exit_code = main([
        "data",
        "preview",
        "--dataset-pack",
        "examples/tiny_dataset_pack.json",
        "--preview-chars",
        "6",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "dataset_pack: examples/tiny_dataset_pack.json" in output
    assert "--dataset-pack examples/tiny_dataset_pack.json" in output
    assert "chat_sft: ready" in output
    assert "eval: ready" in output


def test_cli_data_init_pack_creates_starter_pack(tmp_path, capsys):
    corpus_dir = tmp_path / "docs"
    out_dir = tmp_path / "pack"
    corpus_dir.mkdir()

    exit_code = main([
        "data",
        "init-pack",
        "--name",
        "lesson-pack",
        "--corpus",
        str(corpus_dir),
        "--out",
        str(out_dir),
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "initialized dataset pack:" in output
    assert "data preview --dataset-pack" in output
    assert (out_dir / "dataset_pack.json").exists()
    assert (out_dir / "corpus_recipe.json").exists()
    assert (out_dir / "chat.jsonl").exists()
    assert (out_dir / "eval.jsonl").exists()


def test_cli_data_eval_starter(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.txt"
    out_path = tmp_path / "eval.jsonl"
    corpus_path.write_text(
        "Pico Cafe roasts careful beans for morning guests. "
        "Mira Chen founded Pico Cafe beside the train station.",
        encoding="utf-8",
    )

    exit_code = main([
        "data",
        "eval-starter",
        "--input",
        str(corpus_path),
        "--out",
        str(out_path),
        "--max-items",
        "6",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "eval starter:" in output
    assert out_path.exists()
    assert (tmp_path / "eval.md").exists()


def test_cli_data_hf_import_uses_importer(tmp_path, capsys, monkeypatch):
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
            report_path=config.report_path or str(tmp_path / "hf_import_report.json"),
            documents_dir=config.documents_dir or str(tmp_path / "documents"),
            rows_seen=3,
            rows_written=2,
            rows_skipped=1,
            characters_written=42,
            rows=(),
        )

    monkeypatch.setattr("picochat.cli.import_hf_dataset", fake_import)

    exit_code = main([
        "data",
        "hf-import",
        "--dataset",
        "demo/dataset",
        "--config",
        "plain",
        "--split",
        "train",
        "--text-column",
        "body",
        "--out",
        str(tmp_path / "corpus.txt"),
        "--documents-dir",
        str(tmp_path / "docs"),
        "--max-rows",
        "3",
        "--min-chars",
        "5",
        "--no-streaming",
    ])

    assert exit_code == 0
    assert captured["config"].dataset == "demo/dataset"
    assert captured["config"].config_name == "plain"
    assert captured["config"].text_column == "body"
    assert captured["config"].streaming is False
    assert captured["config"].documents_dir == str(tmp_path / "docs")
    output = capsys.readouterr().out
    assert "imported dataset: demo/dataset" in output
    assert "rows_written: 2" in output
    assert f"documents_dir: {tmp_path / 'docs'}" in output


def test_cli_demo_uses_default_pipeline(tmp_path, capsys, monkeypatch):
    captured = {}

    def fake_run_tiny(config):
        captured["config"] = config
        return {
            "eval": {
                "num_passed": 6,
                "num_examples": 6,
                "pass_rate": 1.0,
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run_tiny)

    exit_code = main(["demo", "--out-dir", str(tmp_path / "demo")])

    assert exit_code == 0
    assert captured["config"].out_dir == str(tmp_path / "demo")
    output = capsys.readouterr().out
    assert "demo run: 6/6 passed" in output
    assert "open workbench" in output


def test_cli_batch_inspect(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    corpus_path.write_text("hello picochat", encoding="utf-8")
    from picochat.tokenizer import CharTokenizer

    CharTokenizer.train([corpus_path.read_text(encoding="utf-8")]).save(tokenizer_path)

    exit_code = main([
        "batch",
        "inspect",
        "--corpus",
        str(corpus_path),
        "--tokenizer",
        str(tokenizer_path),
        "--context-size",
        "4",
        "--examples",
        "1",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "num_sequences" in output
    assert "example 0" in output


def test_cli_generate(tmp_path, capsys):
    from picochat.checkpoint import save_checkpoint
    from picochat.model import GPTConfig, TinyGPT
    from picochat.tokenizer import CharTokenizer

    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    tokenizer = CharTokenizer.train(["hello"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    exit_code = main([
        "generate",
        "--checkpoint",
        str(checkpoint_path),
        "--tokenizer",
        str(tokenizer_path),
        "--prompt",
        "he",
        "--max-new-tokens",
        "2",
        "--temperature",
        "0",
    ])

    assert exit_code == 0
    assert capsys.readouterr().out.startswith("he")


def test_cli_train_sft(tmp_path, capsys):
    import json

    from picochat.checkpoint import save_checkpoint
    from picochat.model import GPTConfig, TinyGPT
    from picochat.tokenizer import CharTokenizer

    input_path = tmp_path / "chat.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "base"
    out_dir = tmp_path / "sft"
    rows = [
        {"user": "What is Picochat?", "assistant": "Picochat is small."},
        {"user": "What next?", "assistant": "Train chat format."},
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    tokenizer = CharTokenizer.train([
        "User: What is Picochat?\nAssistant: Picochat is small.\n"
        "User: What next?\nAssistant: Train chat format."
    ])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=64,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    exit_code = main([
        "train",
        "sft",
        "--input",
        str(input_path),
        "--tokenizer",
        str(tokenizer_path),
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(out_dir),
        "--max-steps",
        "1",
        "--log-every",
        "1",
        "--sample-tokens",
        "2",
        "--sampling",
        "category_balanced",
        "--sft-packing",
        "bos_bestfit",
    ])

    assert exit_code == 0
    assert (out_dir / "checkpoint" / "model.pt").exists()
    report = json.loads((out_dir / "sft_report.json").read_text(encoding="utf-8"))
    assert report["dataset"]["sampling"] == "category_balanced"
    assert report["dataset"]["packing"] == "bos_bestfit"
    assert "saved sft checkpoint" in capsys.readouterr().out


def test_cli_eval_chat(tmp_path, capsys):
    import json

    from picochat.checkpoint import save_checkpoint
    from picochat.model import GPTConfig, TinyGPT
    from picochat.tokenizer import CharTokenizer

    input_path = tmp_path / "eval.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "eval"
    input_path.write_text(json.dumps({"user": "hi"}), encoding="utf-8")
    tokenizer = CharTokenizer.train(["User: hi\nAssistant:"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=32,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    exit_code = main([
        "eval",
        "chat",
        "--input",
        str(input_path),
        "--checkpoint",
        str(checkpoint_path),
        "--tokenizer",
        str(tokenizer_path),
        "--out-dir",
        str(out_dir),
        "--max-new-tokens",
        "0",
    ])

    assert exit_code == 0
    assert (out_dir / "eval_report.json").exists()
    assert "chat eval: 1/1 passed" in capsys.readouterr().out


def test_cli_run_tiny(tmp_path, capsys):
    import json

    corpus_path = tmp_path / "corpus.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    out_dir = tmp_path / "tiny"
    corpus_path.write_text(
        "Picochat is small.\nUser: hi\nAssistant: hello\n" * 8,
        encoding="utf-8",
    )
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi"}), encoding="utf-8")

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(out_dir),
        "--corpus-input",
        str(corpus_path),
        "--chat-input",
        str(chat_path),
        "--eval-input",
        str(eval_path),
        "--context-size",
        "32",
        "--n-embd",
        "16",
        "--n-layer",
        "1",
        "--base-steps",
        "1",
        "--sft-steps",
        "1",
        "--base-batch-size",
        "2",
        "--base-dataset-mode",
        "sharded",
        "--base-shard-token-size",
        "64",
        "--base-shard-cache-size",
        "2",
        "--sft-batch-size",
        "1",
        "--sft-sampling",
        "category_balanced",
        "--sft-packing",
        "bos_bestfit",
        "--eval-max-new-tokens",
        "0",
        "--tokenizer-type",
        "byte",
        "--base-optimizer",
        "muon",
        "--sft-optimizer",
        "muon",
        "--base-muon-learning-rate",
        "0.01",
        "--sft-muon-learning-rate",
        "0.01",
        "--base-ema-decay",
        "0.5",
        "--sft-ema-decay",
        "0.5",
        "--attn-backend",
        "math",
        "--parallel-residual",
        "--allow-leaky-eval",
    ])

    assert exit_code == 0
    assert (out_dir / "summary.md").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["tokenizer_type"] == "byte"
    assert summary["config"]["base_dataset_mode"] == "sharded"
    assert summary["config"]["base_shard_token_size"] == 64
    assert summary["config"]["base_shard_cache_size"] == 2
    assert summary["config"]["sft_sampling"] == "category_balanced"
    assert summary["config"]["sft_packing"] == "bos_bestfit"
    assert summary["config"]["attn_backend"] == "math"
    assert summary["config"]["parallel_residual"] is True
    assert summary["config"]["base_optimizer"] == "muon"
    assert summary["config"]["sft_optimizer"] == "muon"
    assert summary["base"]["best_checkpoint"]["weights"] == "ema"
    assert summary["sft"]["best_checkpoint"]["weights"] == "ema"
    assert (out_dir / "base" / "ema_checkpoint" / "model.pt").exists()
    assert (out_dir / "sft" / "ema_checkpoint" / "model.pt").exists()
    assert summary["tokenizer"]["tokenizer_type"] == "byte"
    assert "tiny run: 1/1 passed" in capsys.readouterr().out


def test_cli_compare(tmp_path, capsys):
    import json

    run_a = tmp_path / "tiny-a"
    run_b = tmp_path / "tiny-b"
    out_path = tmp_path / "compare.md"
    for run_dir, passed, total in [(run_a, 3, 4), (run_b, 6, 6)]:
        run_dir.mkdir()
        summary = {
            "config": {"context_size": 128},
            "base": {
                "final_val_loss": 2.0,
                "num_parameters": 114609,
            },
            "sft": {
                "final_val_loss": 4.0,
                "truncated_examples": 0,
            },
            "eval": {
                "num_examples": total,
                "num_passed": passed,
                "num_failed": total - passed,
                "pass_rate": passed / total,
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main([
        "compare",
        str(run_a),
        str(run_b),
        "--out",
        str(out_path),
    ])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "tiny-a" in output
    assert "tiny-b" in output
    assert "Best eval run: tiny-b" in output
    assert "Champion gate:" in output
    assert out_path.exists()
