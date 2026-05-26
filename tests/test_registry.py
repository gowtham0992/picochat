import json

from picochat.registry import (
    build_model_registry,
    discover_run_dirs,
    registry_markdown,
    registry_table,
    release_card_markdown,
    write_registry_json,
    write_registry_report,
    write_release_card,
)


def _write_run(root, name, *, status="approved", eval_rate=0.62):
    run = root / name
    (run / "sft" / "best_checkpoint").mkdir(parents=True)
    (run / "sft" / "best_checkpoint" / "model.pt").write_text("weights", encoding="utf-8")
    (run / "tokenizer.json").write_text("{}", encoding="utf-8")
    summary = {
        "preflight": {
            "status": "ready",
            "budget": {
                "estimated_parameters": 100,
                "base_planned_tokens": 2000,
                "target_param_data_ratio": 20.0,
            },
        },
        "eval": {
            "num_examples": 100,
            "num_passed": int(eval_rate * 100),
            "pass_rate": eval_rate,
        },
        "sft_fit": {
            "num_examples": 50,
            "num_passed": 40,
            "pass_rate": 0.8,
        },
        "honesty": {
            "status": "ready",
        },
        "external_evals": [{
            "name": "arc-mini",
            "summary": {"choice_accuracy": 0.55},
        }],
        "long_run_gate": {
            "status": status,
            "profile": "skill_release",
            "sft_fit_rate": 0.8,
            "sft_heldout_fit_rate": 0.7,
            "issues": [{
                "severity": "warn",
                "name": "external_eval_sample",
                "message": "External benchmark sample is small.",
            }],
        },
        "artifacts": {
            "preflight_report": str(run / "preflight.md"),
            "honesty_report": str(run / "honesty" / "report.md"),
            "eval_report": str(run / "eval" / "report.md"),
        },
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def test_build_model_registry_reports_gate_and_evidence(tmp_path):
    approved = _write_run(tmp_path, "approved-run", status="approved", eval_rate=0.7)
    blocked = _write_run(tmp_path, "blocked-run", status="blocked", eval_rate=0.9)

    registry = build_model_registry([blocked, approved])

    assert registry["best_run"] == "approved-run"
    assert registry["status_counts"] == {"blocked": 1, "approved": 1}
    row = registry["entries"][0]
    assert row["run"] == "approved-run"
    assert row["tokens_per_parameter"] == 20.0
    assert row["external_benchmark_count"] == 1
    assert row["best_checkpoint"].endswith("sft/best_checkpoint")
    table = registry_table(registry)
    assert "approved-run" in table
    markdown = registry_markdown(registry)
    assert "# Picochat Model Registry" in markdown
    assert "A registry row is not a release claim" in markdown


def test_discover_run_dirs_and_write_registry_outputs(tmp_path):
    run = _write_run(tmp_path, "run-a")
    (tmp_path / "scratch").mkdir()

    discovered = discover_run_dirs(tmp_path)
    assert discovered == [run]
    registry = build_model_registry(discovered)
    md_path = write_registry_report(registry, tmp_path / "registry.md")
    json_path = write_registry_json(registry, tmp_path / "registry.json")

    assert "# Picochat Model Registry" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["best_run"] == "run-a"


def test_release_card_markdown_lists_required_evidence(tmp_path):
    run = _write_run(tmp_path, "release-run", status="warn")

    card = release_card_markdown(run)
    assert "# Picochat Release Card: release-run" in card
    assert "## Required Evidence" in card
    assert "External benchmark sample is small." in card

    out = write_release_card(run, tmp_path / "release-card.md")
    assert "Do not present this model as production-ready" in out.read_text(encoding="utf-8")
