"""Remote (Modal) training launch from the dashboard."""

import pytest

import picochat.web as web
from picochat.web import remote_modal_start_plan, remote_status_plan


def test_remote_status_reports_modal(monkeypatch):
    monkeypatch.setattr("picochat.web.shutil.which", lambda name: None)
    status = remote_status_plan()
    assert status["modal_available"] is False
    assert "modal_script" in status


def test_remote_modal_requires_cli(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.shutil.which", lambda name: None)
    with pytest.raises(ValueError, match="modal"):
        remote_modal_start_plan(tmp_path, {"run_name": "r1"})


def test_remote_modal_builds_command(tmp_path, monkeypatch):
    captured = {}

    class CapturingProc:
        pid = 99

        def __init__(self, command, *args, **kwargs):
            captured["command"] = command

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr("picochat.web.shutil.which", lambda name: "/usr/bin/modal")
    monkeypatch.setattr("picochat.web.subprocess.Popen", CapturingProc)
    web._RUN_JOBS.clear()

    started = remote_modal_start_plan(tmp_path, {
        "run_name": "cloud1",
        "scale": "h100-100m",
        "gpu": "H100",
        "hf_dataset": "org/ds",
        "hf_max_rows": 1000,
    })

    command = captured["command"]
    assert command[0:2] == ["modal", "run"]
    assert "scripts/modal_picochat_train.py" in command[2]
    assert "--scale" in command and "h100-100m" in command
    assert "--gpu" in command and "H100" in command
    assert "--hf-dataset" in command and "org/ds" in command
    assert started["job"]["run_name"] == "cloud1"
    assert (tmp_path / "cloud1" / "web_run.log").exists()
