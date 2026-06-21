"""Leaderboard endpoint wrapper."""

from picochat.web import leaderboard_plan


def test_leaderboard_empty(tmp_path):
    assert leaderboard_plan(tmp_path) == {"rows": [], "best_run": None}


def test_leaderboard_degrades_when_runs_lack_eval(tmp_path):
    run = tmp_path / "r1"
    run.mkdir()
    (run / "summary.json").write_text("{}", encoding="utf-8")  # no eval report
    assert leaderboard_plan(tmp_path) == {"rows": [], "best_run": None}
