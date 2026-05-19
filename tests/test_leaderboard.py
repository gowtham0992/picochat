import json

from picochat.leaderboard import build_benchmark_leaderboard, leaderboard_table


def write_eval_report(run_dir, passed, total, category_passed):
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    eval_dir.joinpath("eval_report.json").write_text(json.dumps({
        "summary": {
            "num_examples": total,
            "num_passed": passed,
            "pass_rate": passed / total,
            "support_match_rate": 0.5,
            "prompt_echo_rate": 0.0,
            "category_breakdown": {
                "mmlu_science": {
                    "num_examples": 4,
                    "num_passed": category_passed,
                    "pass_rate": category_passed / 4,
                    "support_match_rate": 0.25,
                    "prompt_echo_rate": 0.0,
                },
                "domain_recall": {
                    "num_examples": 4,
                    "num_passed": 1,
                    "pass_rate": 0.25,
                },
            },
            "level_breakdown": {
                "choice": {
                    "num_examples": 4,
                    "num_passed": category_passed,
                    "pass_rate": category_passed / 4,
                },
            },
        },
    }), encoding="utf-8")


def test_build_benchmark_leaderboard_reports_overall_and_suites(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    write_eval_report(run_a, passed=2, total=8, category_passed=1)
    write_eval_report(run_b, passed=5, total=8, category_passed=3)

    leaderboard = build_benchmark_leaderboard([run_a, run_b])

    assert leaderboard["best_run"] == "run-b"
    suites = {(row["run"], row["suite"]) for row in leaderboard["rows"]}
    assert ("run-b", "overall") in suites
    assert ("run-b", "mmlu_science") in suites
    assert ("run-b", "level:choice") in suites

    table = leaderboard_table(leaderboard)
    assert "run-b" in table
    assert "mmlu_science" in table
