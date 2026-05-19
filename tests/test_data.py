import json

from picochat.data import (
    DocumentExtractionError,
    build_corpus,
    build_corpus_artifacts,
    find_text_files,
    inspect_documents,
    inspect_path,
    preview_corpus_sources,
    read_documents,
)


def test_find_text_files_ignores_unknown_extensions(tmp_path):
    keep = tmp_path / "notes.md"
    skip = tmp_path / "image.png"
    keep.write_text("hello", encoding="utf-8")
    skip.write_text("not really an image", encoding="utf-8")

    files = find_text_files(tmp_path)

    assert files == [keep]


def test_read_documents_and_inspect(tmp_path):
    (tmp_path / "a.txt").write_text("hello\nhello\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("picochat\n", encoding="utf-8")

    documents = read_documents(tmp_path)
    stats = inspect_path(tmp_path)

    assert len(documents) == 2
    assert stats.num_files == 2
    assert stats.num_documents == 2
    assert stats.num_characters > 0
    assert stats.duplicate_line_rate > 0
    assert stats.duplicate_document_rate == 0


def test_inspect_path_tracks_duplicate_documents(tmp_path):
    (tmp_path / "a.txt").write_text("same document", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Same   document\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("different document", encoding="utf-8")

    stats = inspect_path(tmp_path)

    assert stats.num_documents == 3
    assert round(stats.duplicate_document_rate, 2) == 0.33


def test_inspect_documents_tracks_near_duplicate_documents():
    template = " ".join(f"token{index}" for index in range(120))
    near_copy = template.replace("token55", "replacement55")
    unrelated = " ".join(f"other{index}" for index in range(120))

    stats = inspect_documents([template, near_copy, unrelated])

    assert stats.duplicate_document_rate == 0
    assert stats.near_duplicate_document_pairs >= 1
    assert stats.near_duplicate_document_rate > 0
    assert stats.near_duplicate_documents_checked == 3


def test_build_corpus_combines_documents(tmp_path):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "out" / "corpus.txt"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (input_dir / "b.md").write_text("beta", encoding="utf-8")

    stats = build_corpus(input_dir, output_path)

    assert output_path.read_text(encoding="utf-8") == "alpha\n\nbeta\n"
    assert stats.num_documents == 2


def test_build_corpus_artifacts_writes_manifest_and_report(tmp_path):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "out" / "corpus.txt"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (input_dir / "empty.md").write_text("   ", encoding="utf-8")
    (input_dir / "image.png").write_text("not text", encoding="utf-8")

    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
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

    report = build_corpus_artifacts(
        input_dir,
        output_path,
        chat_input=chat_path,
        eval_input=eval_path,
    )

    assert output_path.read_text(encoding="utf-8") == "alpha\n"
    assert (output_path.parent / "corpus_manifest.json").exists()
    assert (output_path.parent / "corpus_report.md").exists()
    assert report.stats.num_files == 3
    assert len(report.files) == 3
    assert [record.reason for record in report.files] == [
        "included",
        "empty_text",
        "unsupported_extension",
    ]
    assert report.readiness.status == "caution"
    assert report.budget.preset == "smoke"
    assert report.budget.suggested_context_size == 32
    assert "run tiny" in report.training_command.command
    assert str(chat_path) in report.training_command.command
    assert str(eval_path) in report.training_command.command
    assert "--base-steps 100" in report.training_command.command
    assert report.chat_data.status == "ready"
    assert report.eval_data.status == "ready"
    assert f"- Chat SFT input: `{chat_path}`" in (output_path.parent / "corpus_report.md").read_text(encoding="utf-8")
    assert "source file(s) were skipped" in report.warnings[-1]
    assert report.files[0].quality_score < 100
    assert "short_document" in report.files[0].quality_flags
    assert len(report.documents) == 1
    assert report.documents[0].char_start == 0
    assert report.documents[0].char_end == len("alpha")
    manifest = json.loads((output_path.parent / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert manifest["documents"][0]["path"] == str(input_dir / "a.txt")
    assert "## Documents" in (output_path.parent / "corpus_report.md").read_text(encoding="utf-8")


def test_corpus_report_caps_large_provenance_tables(tmp_path):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "out" / "corpus.txt"
    input_dir.mkdir()
    for index in range(505):
        (input_dir / f"doc-{index:03d}.txt").write_text(
            f"document {index} has enough text for a useful source record\n",
            encoding="utf-8",
        )

    report = build_corpus_artifacts(input_dir, output_path)
    markdown = (output_path.parent / "corpus_report.md").read_text(encoding="utf-8")
    manifest = json.loads((output_path.parent / "corpus_manifest.json").read_text(encoding="utf-8"))

    assert len(report.files) == 505
    assert len(manifest["files"]) == 505
    assert len(manifest["documents"]) == 505
    assert "305 more document(s) omitted" in markdown
    assert "5 more file(s) omitted" in markdown
    assert "doc-504.txt" not in markdown


def test_custom_corpus_preview_does_not_suggest_demo_tuning_data(tmp_path):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "out" / "corpus.txt"
    input_dir.mkdir()
    (input_dir / "domain.txt").write_text("domain text " * 200, encoding="utf-8")

    report = build_corpus_artifacts(input_dir, output_path)

    assert report.training_command.command == ""
    assert "demo tuning data" in report.training_command.note


def test_build_corpus_artifacts_extracts_document_sources(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "out" / "corpus.txt"
    input_dir.mkdir()
    (input_dir / "a.pdf").write_bytes(b"%PDF fake")
    (input_dir / "b.docx").write_bytes(b"PK fake")

    monkeypatch.setattr("picochat.data._extract_pdf_text", lambda path: "pdf text")
    monkeypatch.setattr("picochat.data._extract_docx_text", lambda path: "docx text")

    report = build_corpus_artifacts(input_dir, output_path)

    assert output_path.read_text(encoding="utf-8") == "pdf text\n\ndocx text\n"
    assert [record.reason for record in report.files] == ["included_pdf", "included_docx"]
    assert report.stats.num_documents == 2


def test_build_corpus_artifacts_records_missing_document_extractors(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "out" / "corpus.txt"
    input_dir.mkdir()
    (input_dir / "a.pdf").write_bytes(b"%PDF fake")
    (input_dir / "b.docx").write_bytes(b"PK fake")

    def missing_pdf(_path):
        raise DocumentExtractionError("missing_pdf_dependency")

    def missing_docx(_path):
        raise DocumentExtractionError("missing_docx_dependency")

    monkeypatch.setattr("picochat.data._extract_pdf_text", missing_pdf)
    monkeypatch.setattr("picochat.data._extract_docx_text", missing_docx)

    report = build_corpus_artifacts(input_dir, output_path)

    assert output_path.read_text(encoding="utf-8") == ""
    assert [record.reason for record in report.files] == [
        "missing_pdf_dependency",
        "missing_docx_dependency",
    ]
    assert report.stats.num_documents == 0
    assert report.readiness.status == "blocked"
    assert report.budget.preset == "blocked"
    assert report.budget.suggested_base_steps == 0
    assert report.training_command.command == ""


def test_build_corpus_artifacts_uses_recipe_labels_and_exclusions(tmp_path):
    source_dir = tmp_path / "sources"
    docs_dir = source_dir / "docs"
    output_path = tmp_path / "out" / "corpus.txt"
    docs_dir.mkdir(parents=True)
    (source_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (docs_dir / "manual.md").write_text("manual", encoding="utf-8")
    (docs_dir / "draft.md").write_text("draft", encoding="utf-8")
    (source_dir / "ignore.png").write_text("not text", encoding="utf-8")
    recipe_path = tmp_path / "corpus.recipe.json"
    recipe_path.write_text(json.dumps({
        "sources": [
            {"path": "sources/alpha.txt", "label": "seed"},
            {"path": "sources/docs", "label": "docs", "exclude": ["sources/docs/draft.md"]},
            {"path": "sources/ignore.png", "label": "asset", "include": False},
            {"path": "missing.txt", "label": "missing"},
        ],
    }), encoding="utf-8")

    report = build_corpus_artifacts(None, output_path, recipe_path=recipe_path)
    records = {record.path: record for record in report.files}

    assert output_path.read_text(encoding="utf-8") == "alpha\n\nmanual\n"
    assert report.recipe_path == str(recipe_path)
    assert records[str(source_dir / "alpha.txt")].label == "seed"
    assert records[str(docs_dir / "manual.md")].label == "docs"
    assert records[str(docs_dir / "draft.md")].reason == "recipe_excluded"
    assert records[str(source_dir / "ignore.png")].reason == "recipe_excluded"
    assert records[str(tmp_path / "missing.txt")].reason == "missing_source"


def test_preview_corpus_sources_is_read_only(tmp_path):
    source_path = tmp_path / "lesson.txt"
    recipe_path = tmp_path / "corpus.recipe.json"
    output_path = tmp_path / "corpus.txt"
    source_path.write_text("alpha beta gamma", encoding="utf-8")
    recipe_path.write_text(json.dumps({
        "sources": [
            {"path": "lesson.txt", "label": "lesson"},
        ],
    }), encoding="utf-8")

    report = preview_corpus_sources(
        recipe_path=recipe_path,
        preview_chars=10,
        chat_input="domain/chat.jsonl",
        eval_input="domain/eval.jsonl",
    )

    assert report.preview == "alpha beta"
    assert report.recipe_path == str(recipe_path)
    assert report.files[0].label == "lesson"
    assert report.stats.num_documents == 1
    assert report.readiness.status == "caution"
    assert report.budget.estimated_tokens == len("alpha beta gamma")
    assert report.budget.estimated_windows >= 0
    assert "--corpus-recipe" in report.training_command.command
    assert "--chat-input domain/chat.jsonl" in report.training_command.command
    assert "--eval-input domain/eval.jsonl" in report.training_command.command
    assert "--context-size 32" in report.training_command.command
    assert {check.name for check in report.readiness.checks} >= {
        "usable_documents",
        "corpus_size",
        "document_mix",
        "skipped_sources",
    }
    assert not output_path.exists()


def test_preview_corpus_sources_uses_dataset_pack(tmp_path):
    source_path = tmp_path / "lesson.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    source_path.write_text("alpha beta gamma", encoding="utf-8")
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

    report = preview_corpus_sources(dataset_pack=pack_path, preview_chars=5)

    assert report.dataset_pack == str(pack_path)
    assert report.input_path == str(source_path)
    assert report.training_command.dataset_pack == str(pack_path)
    assert "--dataset-pack" in report.training_command.command
    assert "--chat-input" not in report.training_command.command
    assert report.training_command.chat_input == str(chat_path)
    assert report.training_command.eval_input == str(eval_path)
    assert report.chat_data.status == "ready"
    assert report.eval_data.status == "ready"


def test_build_corpus_artifacts_reuses_imported_pack_corpus_without_collecting(tmp_path, monkeypatch):
    pack_dir = tmp_path / "imported"
    docs_dir = pack_dir / "documents"
    out_dir = tmp_path / "run"
    docs_dir.mkdir(parents=True)
    (docs_dir / "shard-000000.txt").write_text("alpha document\n\nbeta document\n", encoding="utf-8")
    (docs_dir / "shard-000001.txt").write_text("gamma document\n", encoding="utf-8")
    corpus_text = "alpha document\n\nbeta document\n\n\ngamma document\n"
    (pack_dir / "corpus.txt").write_text(corpus_text, encoding="utf-8")
    (pack_dir / "hf_import_report.json").write_text("{}", encoding="utf-8")
    recipe_path = pack_dir / "corpus_recipe.json"
    recipe_path.write_text(json.dumps({
        "sources": [{"path": "documents", "label": "climbmix"}],
    }), encoding="utf-8")
    chat_path = pack_dir / "chat.jsonl"
    eval_path = pack_dir / "eval.jsonl"
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
    pack_path = pack_dir / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": {"recipe": "corpus_recipe.json"},
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    def fail_collect(*_args, **_kwargs):
        raise AssertionError("imported dataset packs should not collect all documents into memory")

    monkeypatch.setattr("picochat.data._collect_corpus_sources", fail_collect)

    report = build_corpus_artifacts(
        None,
        out_dir / "corpus.txt",
        dataset_pack=pack_path,
    )

    assert (out_dir / "corpus.txt").read_text(encoding="utf-8") == corpus_text
    assert report.dataset_pack == str(pack_path)
    assert report.stats.num_documents == 2
    assert report.documents[0].path == str(docs_dir / "shard-000000.txt")
    assert report.documents[0].char_start == 0
    assert report.documents[0].char_end == len("alpha document\n\nbeta document")
    assert report.documents[1].char_start == report.documents[0].char_end + 2
    assert "fast imported dataset-pack path" in report.warnings[0]
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert manifest["documents"][1]["path"] == str(docs_dir / "shard-000001.txt")


def test_build_corpus_artifacts_uses_import_metadata_without_listing_documents(tmp_path, monkeypatch):
    pack_dir = tmp_path / "imported"
    out_dir = tmp_path / "run"
    pack_dir.mkdir()
    corpus_text = "alpha document\n\nbeta document\n"
    (pack_dir / "corpus.txt").write_text(corpus_text, encoding="utf-8")
    document_path = pack_dir / "documents" / "shard-000000.txt"
    (pack_dir / "hf_import_report.json").write_text(json.dumps({
        "rows_written": 2,
        "characters_written": len(corpus_text.rstrip("\n")),
        "document_files": [{
            "path": str(document_path),
            "num_documents": 2,
            "num_characters": len(corpus_text.rstrip("\n")),
            "num_lines": 3,
        }],
    }), encoding="utf-8")
    recipe_path = pack_dir / "corpus_recipe.json"
    recipe_path.write_text(json.dumps({
        "sources": [{"path": "documents", "label": "climbmix"}],
    }), encoding="utf-8")
    chat_path = pack_dir / "chat.jsonl"
    eval_path = pack_dir / "eval.jsonl"
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
    pack_path = pack_dir / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": {"recipe": "corpus_recipe.json"},
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    def fail_candidates(*_args, **_kwargs):
        raise AssertionError("import metadata should avoid listing document files")

    monkeypatch.setattr("picochat.data._recipe_source_candidates", fail_candidates)

    report = build_corpus_artifacts(
        None,
        out_dir / "corpus.txt",
        dataset_pack=pack_path,
    )

    assert (out_dir / "corpus.txt").read_text(encoding="utf-8") == corpus_text
    assert report.stats.num_documents == 2
    assert len(report.documents) == 1
    assert report.documents[0].path == str(document_path)
    assert "import-time document file metadata" in report.warnings[1]


def test_build_corpus_artifacts_uses_legacy_import_chunks_without_listing_documents(tmp_path, monkeypatch):
    pack_dir = tmp_path / "imported"
    out_dir = tmp_path / "run"
    pack_dir.mkdir()
    corpus_text = "alpha document\n\nbeta document\n"
    (pack_dir / "corpus.txt").write_text(corpus_text, encoding="utf-8")
    (pack_dir / "hf_import_report.json").write_text(json.dumps({
        "rows_written": 2,
        "characters_written": len(corpus_text.rstrip("\n")),
    }), encoding="utf-8")
    recipe_path = pack_dir / "corpus_recipe.json"
    recipe_path.write_text(json.dumps({
        "sources": [{"path": "documents", "label": "climbmix"}],
    }), encoding="utf-8")
    chat_path = pack_dir / "chat.jsonl"
    eval_path = pack_dir / "eval.jsonl"
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
    pack_path = pack_dir / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": {"recipe": "corpus_recipe.json"},
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    def fail_candidates(*_args, **_kwargs):
        raise AssertionError("legacy imported pack should chunk corpus instead of listing document files")

    monkeypatch.setattr("picochat.data._recipe_source_candidates", fail_candidates)

    report = build_corpus_artifacts(
        None,
        out_dir / "corpus.txt",
        dataset_pack=pack_path,
    )

    assert (out_dir / "corpus.txt").read_text(encoding="utf-8") == corpus_text
    assert report.stats.num_documents == 2
    assert len(report.documents) == 1
    assert report.documents[0].char_start == 0
    assert report.documents[0].char_end == len(corpus_text.rstrip("\n"))
    assert "legacy imported dataset pack" in report.warnings[1]


def test_preview_corpus_sources_reuses_imported_pack_corpus_without_collecting(tmp_path, monkeypatch):
    pack_dir = tmp_path / "imported"
    docs_dir = pack_dir / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "shard-000000.txt").write_text("alpha document\n\nbeta document\n", encoding="utf-8")
    (pack_dir / "corpus.txt").write_text("alpha document\n\nbeta document\n", encoding="utf-8")
    (pack_dir / "hf_import_report.json").write_text("{}", encoding="utf-8")
    recipe_path = pack_dir / "corpus_recipe.json"
    recipe_path.write_text(json.dumps({
        "sources": [{"path": "documents", "label": "climbmix"}],
    }), encoding="utf-8")
    chat_path = pack_dir / "chat.jsonl"
    eval_path = pack_dir / "eval.jsonl"
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
    pack_path = pack_dir / "dataset_pack.json"
    pack_path.write_text(json.dumps({
        "corpus": {"recipe": "corpus_recipe.json"},
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    def fail_collect(*_args, **_kwargs):
        raise AssertionError("imported dataset pack preview should not collect all documents into memory")

    monkeypatch.setattr("picochat.data._collect_corpus_sources", fail_collect)

    report = preview_corpus_sources(dataset_pack=pack_path, preview_chars=14)

    assert report.preview == "alpha document"
    assert report.dataset_pack == str(pack_path)
    assert report.stats.num_documents == 1
    assert "fast imported dataset-pack preview" in report.warnings[0]


def test_build_corpus_artifacts_can_filter_by_quality_score(tmp_path):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "out" / "corpus.txt"
    input_dir.mkdir()
    good_text = "\n".join(f"quality source line {index}" for index in range(80))
    (input_dir / "good.txt").write_text(good_text, encoding="utf-8")
    (input_dir / "short.txt").write_text("tiny", encoding="utf-8")

    report = build_corpus_artifacts(input_dir, output_path, min_quality_score=80)
    records = {record.path: record for record in report.files}

    assert output_path.read_text(encoding="utf-8") == f"{good_text}\n"
    assert report.min_quality_score == 80
    assert report.training_command.command == ""
    assert "demo tuning data" in report.training_command.note
    assert records[str(input_dir / "good.txt")].included is True
    assert records[str(input_dir / "good.txt")].quality_score >= 80
    assert records[str(input_dir / "short.txt")].included is False
    assert records[str(input_dir / "short.txt")].reason == "below_min_score"
    assert "short_document" in records[str(input_dir / "short.txt")].quality_flags
    assert "source file(s) were filtered by the minimum quality score" in " ".join(report.warnings)


def test_preview_corpus_sources_rejects_invalid_quality_score(tmp_path):
    source_path = tmp_path / "lesson.txt"
    source_path.write_text("lesson text", encoding="utf-8")

    try:
        preview_corpus_sources(source_path, min_quality_score=101)
    except ValueError as error:
        assert "min_quality_score" in str(error)
    else:
        raise AssertionError("preview should reject scores outside 0-100")
