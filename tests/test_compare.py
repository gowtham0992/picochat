import json

from picochat.compare import compare_runs, comparison_markdown, comparison_table, load_compare_row


def write_summary(path, *, passed, total, base_val=2.0, sft_val=4.0, params=1234):
    path.mkdir()
    summary = {
        "config": {"context_size": 128},
        "base": {
            "final_val_loss": base_val,
            "num_parameters": params,
        },
        "sft": {
            "final_val_loss": sft_val,
            "truncated_examples": 0,
        },
        "eval": {
            "num_examples": total,
            "num_passed": passed,
            "num_failed": total - passed,
            "pass_rate": passed / total,
        },
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_load_compare_row_reads_summary(tmp_path):
    run_dir = tmp_path / "tiny-a"
    write_summary(run_dir, passed=3, total=4, params=114609)

    row = load_compare_row(run_dir)

    assert row.run == "tiny-a"
    assert row.eval_score == "3/4"
    assert row.pass_rate == 0.75
    assert row.num_parameters == 114609


def test_compare_runs_selects_best_eval_run(tmp_path):
    run_a = tmp_path / "tiny-a"
    run_b = tmp_path / "tiny-b"
    write_summary(run_a, passed=3, total=4)
    write_summary(run_b, passed=6, total=6)

    comparison = compare_runs([run_a, run_b])

    assert comparison["best_run"] == "tiny-b"
    assert len(comparison["rows"]) == 2


def test_comparison_table_and_markdown_include_metrics(tmp_path):
    run_dir = tmp_path / "tiny-a"
    write_summary(run_dir, passed=3, total=4, params=114609)
    comparison = compare_runs([run_dir])

    table = comparison_table(comparison)
    markdown = comparison_markdown(comparison)

    assert "tiny-a" in table
    assert "3/4" in table
    assert "114.6k" in table
    assert "# Picochat Run Comparison" in markdown
    assert "114,609" in markdown
