import json

from picochat.data import (
    DocumentExtractionError,
    build_corpus,
    build_corpus_artifacts,
    find_text_files,
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

    report = build_corpus_artifacts(input_dir, output_path)

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
    assert "source file(s) were skipped" in report.warnings[-1]


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

    report = preview_corpus_sources(recipe_path=recipe_path, preview_chars=10)

    assert report.preview == "alpha beta"
    assert report.recipe_path == str(recipe_path)
    assert report.files[0].label == "lesson"
    assert report.stats.num_documents == 1
    assert not output_path.exists()
