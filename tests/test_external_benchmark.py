import csv
import json

import pytest

from picochat.external_benchmark import ExternalBenchmarkConvertConfig, convert_external_benchmark


def test_convert_arc_jsonl_to_choice_eval(tmp_path):
    input_path = tmp_path / "arc.jsonl"
    output_path = tmp_path / "external_eval.jsonl"
    input_path.write_text(json.dumps({
        "id": "arc-1",
        "question": "What gas do plants need?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["Oxygen", "Carbon dioxide", "Helium", "Neon"],
        },
        "answerKey": "B",
    }) + "\n", encoding="utf-8")

    report = convert_external_benchmark(ExternalBenchmarkConvertConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        source_format="arc",
        benchmark_name="arc_easy_mini",
    ))
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert report["num_rows"] == 1
    assert rows[0]["choice_labels"] == ["A", "B", "C", "D"]
    assert rows[0]["correct_choice"] == "B"
    assert "B. Carbon dioxide" in rows[0]["user"]
    assert rows[0]["category"] == "external_arc_easy_mini"
    assert (tmp_path / "external_eval.jsonl.report.md").exists()


def test_convert_mmlu_csv_without_header(tmp_path):
    input_path = tmp_path / "mmlu.csv"
    output_path = tmp_path / "external_eval.jsonl"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["2+2?", "3", "4", "5", "6", "B"])

    convert_external_benchmark(ExternalBenchmarkConvertConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        source_format="mmlu_csv",
        benchmark_name="mmlu_mini",
    ))
    row = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert row["choice_labels"] == ["A", "B", "C", "D"]
    assert row["correct_choice"] == "B"
    assert "Answer with the single best option letter." in row["user"]


def test_convert_external_rejects_bad_answer(tmp_path):
    input_path = tmp_path / "bad.jsonl"
    output_path = tmp_path / "external_eval.jsonl"
    input_path.write_text(json.dumps({
        "question": "Bad row?",
        "A": "yes",
        "B": "no",
        "C": "maybe",
        "D": "unknown",
        "answer": "Z",
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not present in choice labels"):
        convert_external_benchmark(ExternalBenchmarkConvertConfig(
            input_path=str(input_path),
            output_path=str(output_path),
            source_format="mmlu",
        ))
