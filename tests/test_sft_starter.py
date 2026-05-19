import json

import pytest

from picochat.sft_starter import generate_sft_starter


def test_generate_sft_starter_writes_grouped_domain_rows(tmp_path):
    source = tmp_path / "docs"
    source.mkdir()
    (source / "lesson.txt").write_text(
        "Pico Cafe roasts careful beans for morning guests. "
        "Mira Chen founded Pico Cafe beside the train station. "
        "The cafe teaches new baristas to weigh water and beans. "
        "Evening classes explain bloom time and gentle pouring.",
        encoding="utf-8",
    )
    out_path = tmp_path / "chat.jsonl"

    report = generate_sft_starter(source, out_path, max_items=8, seed=1)

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert report.num_rows == len(rows)
    assert (tmp_path / "chat.md").exists()
    assert any(row["category"].startswith("domain_") for row in rows)
    assert any(row["category"].startswith("refusal_") for row in rows)
    assert all("group" in row for row in rows if row["category"].startswith("domain_"))
    assert not any("Using only the provided domain material" in row["user"] for row in rows)


def test_generate_sft_starter_refuses_overwrite_without_force(tmp_path):
    source = tmp_path / "lesson.txt"
    source.write_text("Pico Cafe roasts careful beans for morning guests.", encoding="utf-8")
    out_path = tmp_path / "chat.jsonl"
    out_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate_sft_starter(source, out_path)
