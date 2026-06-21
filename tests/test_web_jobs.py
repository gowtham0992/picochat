"""Restart-safe job tracking: orphan reconnect, reconcile, process-group cancel."""

import json
import os
import subprocess
import sys
import time

from picochat.web import cancel_run_plan, reconcile_orphan_jobs, run_status_plan


def _make_orphan(runs_dir, name, pid, *, started_at=None):
    """A run dir as left behind by a server that launched it then went away."""
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "web_run.log").write_text(
        "$ python -m picochat.cli run tiny\ntraining...\n", encoding="utf-8"
    )
    record = {
        "id": f"job-{name}",
        "run_name": name,
        "out_dir": str(run_dir),
        "command": "python -m picochat.cli run tiny",
        "pid": pid,
        "started_at": started_at or time.time(),
    }
    (run_dir / "web_job.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir, record


def test_orphan_with_live_pid_is_reconnectable(tmp_path):
    runs = tmp_path / "runs"
    # os.getpid() is guaranteed alive for the duration of the test.
    _run_dir, record = _make_orphan(runs, "live-run", os.getpid())

    job = run_status_plan(record["id"], runs)["job"]

    assert job["state"] == "running"
    assert job["can_cancel"] is True
    assert job["source"] == "orphan"
    assert job["pid"] == os.getpid()
    assert job["id"] == record["id"]


def test_dead_orphan_is_interrupted_then_reconciled(tmp_path):
    runs = tmp_path / "runs"
    probe = subprocess.Popen([sys.executable, "-c", "pass"])
    probe.wait()  # pid is now reaped and dead
    run_dir, record = _make_orphan(runs, "dead-run", probe.pid)

    # Discovery is read-only: it reports the dead pid as interrupted...
    job = run_status_plan(record["id"], runs)["job"]
    assert job["state"] == "interrupted"
    assert job["can_cancel"] is False
    assert not (run_dir / "web_returncode.txt").exists()

    # ...and a startup reconcile persists a terminal returncode.
    reconcile_orphan_jobs(runs)
    assert (run_dir / "web_returncode.txt").read_text(encoding="utf-8").strip() == "-1"
    assert run_status_plan(record["id"], runs)["job"]["state"] == "failed"


def test_reconcile_leaves_live_orphans_running(tmp_path):
    runs = tmp_path / "runs"
    run_dir, record = _make_orphan(runs, "still-live", os.getpid())

    reconcile_orphan_jobs(runs)

    assert not (run_dir / "web_returncode.txt").exists()
    assert run_status_plan(record["id"], runs)["job"]["state"] == "running"


def test_cancel_orphan_kills_process_group(tmp_path):
    runs = tmp_path / "runs"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        run_dir, record = _make_orphan(runs, "cancel-orphan", proc.pid)

        result = cancel_run_plan(runs, {"job_id": record["id"]})

        deadline = time.time() + 10
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None, "process group should have been killed"
        assert (run_dir / "web_returncode.txt").exists()
        assert result["job"]["state"] == "failed"
        assert result["job"]["can_cancel"] is False
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
