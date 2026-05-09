import json

import pytest

from picochat.hf_import import HFImportConfig, HFDatasetsMissingError, import_hf_dataset


def test_import_hf_dataset_writes_local_corpus_and_reports(tmp_path):
    calls = []

    def fake_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {"text": " first useful row "},
            {"title": "missing text"},
            {"text": 42},
            {"text": "tiny"},
            {"text": "second useful row with enough text"},
        ]

    out_path = tmp_path / "hf" / "corpus.txt"
    report = import_hf_dataset(
        HFImportConfig(
            dataset="demo/dataset",
            config_name="plain",
            split="train[:1%]",
            text_column="text",
            out_path=str(out_path),
            max_rows=5,
            min_chars=10,
            streaming=True,
        ),
        loader=fake_loader,
    )

    assert calls == [(("demo/dataset", "plain"), {"split": "train[:1%]", "streaming": True})]
    assert out_path.read_text(encoding="utf-8") == "first useful row\n\nsecond useful row with enough text\n"
    assert report.rows_seen == 5
    assert report.rows_written == 2
    assert report.rows_skipped == 3
    assert report.rows[1].reason == "missing_text_column"
    assert report.rows[2].reason == "text_column_not_string"
    assert report.rows[3].reason == "below_min_chars"
    report_json = json.loads((tmp_path / "hf" / "hf_import_report.json").read_text(encoding="utf-8"))
    assert report_json["dataset"] == "demo/dataset"
    assert (tmp_path / "hf" / "hf_import_report.md").exists()


def test_import_hf_dataset_respects_max_rows(tmp_path):
    def fake_loader(*_args, **_kwargs):
        return [
            {"text": "row one is long enough"},
            {"text": "row two is long enough"},
            {"text": "row three is long enough"},
        ]

    report = import_hf_dataset(
        HFImportConfig(
            dataset="demo/dataset",
            out_path=str(tmp_path / "corpus.txt"),
            max_rows=2,
            min_chars=1,
        ),
        loader=fake_loader,
    )

    assert report.rows_seen == 2
    assert report.rows_written == 2


def test_import_hf_dataset_reports_missing_optional_dependency(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(HFDatasetsMissingError, match=r"\.\[hf\]"):
        import_hf_dataset(HFImportConfig(
            dataset="demo/dataset",
            out_path=str(tmp_path / "corpus.txt"),
        ))
