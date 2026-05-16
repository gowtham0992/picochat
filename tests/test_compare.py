import json

from picochat.compare import compare_runs, comparison_markdown, comparison_table, load_compare_row


def write_summary(
    path,
    *,
    passed,
    total,
    tokenizer_type="char",
    base_val=2.0,
    sft_val=4.0,
    base_bpb=1.5,
    sft_bpb=2.5,
    base_best_val=None,
    sft_best_val=None,
    base_best_bpb=None,
    sft_best_bpb=None,
    params=1234,
):
    path.mkdir()
    summary = {
        "config": {"context_size": 128, "tokenizer_type": tokenizer_type},
        "tokenizer": {"tokenizer_type": tokenizer_type},
        "base": {
            "final_val_loss": base_val,
            "final_val_bpb": base_bpb,
            "best_val_loss": base_best_val,
            "best_val_bpb": base_best_bpb,
            "best_checkpoint": {"step": 10, "val_loss": base_best_val, "val_bpb": base_best_bpb},
            "stop_reason": "max_steps",
            "loss_diagnostics": {"status": "stable"},
            "memorization": {"status": "low"},
            "num_parameters": params,
        },
        "sft": {
            "final_val_loss": sft_val,
            "final_val_bpb": sft_bpb,
            "best_val_loss": sft_best_val,
            "best_val_bpb": sft_best_bpb,
            "best_checkpoint": {"step": 20, "val_loss": sft_best_val, "val_bpb": sft_best_bpb},
            "stop_reason": "early_stop",
            "loss_diagnostics": {"status": "watch-gap"},
            "truncated_examples": 0,
        },
        "eval": {
            "num_examples": total,
            "num_passed": passed,
            "num_failed": total - passed,
            "pass_rate": passed / total,
            "non_choice_pass_rate": max(0.0, (passed - 1) / max(1, total - 1)),
            "support_match_rate": 0.5,
            "prompt_echo_rate": 0.25,
        },
        "sft_fit": {
            "num_examples": 10,
            "num_passed": 8,
            "num_failed": 2,
            "pass_rate": 0.8,
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
    assert row.non_choice_pass_rate == 2 / 3
    assert row.support_match_rate == 0.5
    assert row.prompt_echo_rate == 0.25
    assert row.sft_fit_rate == 0.8
    assert row.num_parameters == 114609
    assert row.tokenizer_type == "char"
    assert row.base_val_bpb == 1.5
    assert row.sft_val_bpb == 2.5
    assert row.base_best_step == 10
    assert row.sft_best_step == 20
    assert row.base_stop_reason == "max_steps"
    assert row.sft_stop_reason == "early_stop"
    assert row.memorization_status == "low"


def test_compare_runs_selects_best_eval_run(tmp_path):
    run_a = tmp_path / "tiny-a"
    run_b = tmp_path / "tiny-b"
    write_summary(run_a, passed=3, total=4, base_bpb=1.2, sft_bpb=2.0)
    write_summary(run_b, passed=6, total=6, base_bpb=1.4, sft_bpb=1.8)

    comparison = compare_runs([run_a, run_b])

    assert comparison["best_run"] == "tiny-b"
    assert comparison["best_eval_run"] == "tiny-b"
    assert comparison["best_base_bpb_run"] == "tiny-a"
    assert comparison["best_sft_bpb_run"] == "tiny-b"
    assert comparison["decision"]["baseline_run"] == "tiny-a"
    assert comparison["decision"]["champion_title"] == "Promote as reference"
    assert comparison["decision"]["next_title"] == "Separate compression from behavior"
    assert len(comparison["rows"]) == 2


def test_compare_uses_best_checkpoint_metrics_when_available(tmp_path):
    run_dir = tmp_path / "tiny-best"
    write_summary(
        run_dir,
        passed=3,
        total=4,
        base_val=3.0,
        sft_val=5.0,
        base_bpb=1.8,
        sft_bpb=2.8,
        base_best_val=2.5,
        sft_best_val=3.5,
        base_best_bpb=1.3,
        sft_best_bpb=2.1,
    )

    row = load_compare_row(run_dir)

    assert row.base_val_loss == 2.5
    assert row.sft_val_loss == 3.5
    assert row.base_val_bpb == 1.3
    assert row.sft_val_bpb == 2.1


def test_comparison_table_and_markdown_include_metrics(tmp_path):
    run_dir = tmp_path / "tiny-a"
    write_summary(run_dir, passed=3, total=4, params=114609)
    comparison = compare_runs([run_dir])

    table = comparison_table(comparison)
    markdown = comparison_markdown(comparison)

    assert "tiny-a" in table
    assert "3/4" in table
    assert "Base BPB" in table
    assert "NonChoice" in table
    assert "Support" in table
    assert "Echo" in table
    assert "SFT Fit" in table
    assert "50.00%" in table
    assert "25.00%" in table
    assert "80.00%" in table
    assert "1.5000" in table
    assert "10/20" in table
    assert "max/early" in table
    assert "low" in table
    assert "114.6k" in table
    assert "Champion gate" in table
    assert "# Picochat Run Comparison" in markdown
    assert "## Decision Gate" in markdown
    assert "Base Val BPB" in markdown
    assert "Non-Choice Pass" in markdown
    assert "Support Match" in markdown
    assert "Prompt Echo" in markdown
    assert "SFT Fit" in markdown
    assert "`stable`" in markdown
    assert "114,609" in markdown


def test_compare_handles_legacy_summary_without_bpb(tmp_path):
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    summary = {
        "config": {"context_size": 128},
        "base": {
            "final_val_loss": 2.0,
            "num_parameters": 1234,
        },
        "sft": {
            "final_val_loss": 4.0,
            "truncated_examples": 0,
        },
        "eval": {
            "num_examples": 4,
            "num_passed": 2,
            "num_failed": 2,
            "pass_rate": 0.5,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    comparison = compare_runs([run_dir])
    row = comparison["rows"][0]

    assert row["tokenizer_type"] == "unknown"
    assert row["base_val_bpb"] is None
    assert row["sft_val_bpb"] is None
    assert comparison["best_base_bpb_run"] is None
    assert "Base BPB" in comparison_table(comparison)
