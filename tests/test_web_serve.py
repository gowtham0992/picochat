"""Managed `pico serve` lifecycle from the dashboard."""

import pytest

import picochat.web as web
from picochat.web import serve_start_plan, serve_status_plan, serve_stop_plan


class FakeProc:
    def __init__(self, *args, **kwargs):
        self._alive = True
        self.pid = 12345

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False


def _make_run(runs_dir, name):
    run_dir = runs_dir / name
    (run_dir / "sft" / "checkpoint").mkdir(parents=True)
    (run_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_serve_start_status_stop(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.subprocess.Popen", FakeProc)
    web._SERVE_JOBS.clear()
    _make_run(tmp_path, "demo")

    started = serve_start_plan(tmp_path, {"run": "demo"}, host="127.0.0.1")
    server = started["server"]
    assert server["run"] == "demo"
    assert server["port"] > 0
    assert server["api_key"] is None  # loopback bind needs no key
    assert server["state"] == "running"
    assert server["model_name"] == "demo"

    listed = serve_status_plan(tmp_path)
    assert [s["run"] for s in listed["servers"]] == ["demo"]

    # Starting again is idempotent (same port, no second process).
    again = serve_start_plan(tmp_path, {"run": "demo"})
    assert again["server"]["port"] == server["port"]

    stopped = serve_stop_plan({"run": "demo"})
    assert stopped["stopped"] is True
    assert serve_status_plan(tmp_path)["servers"] == []


def test_serve_non_loopback_mints_key(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.subprocess.Popen", FakeProc)
    web._SERVE_JOBS.clear()
    _make_run(tmp_path, "net")

    started = serve_start_plan(tmp_path, {"run": "net"}, host="0.0.0.0")
    assert started["server"]["api_key"]  # exposed bind requires a bearer key

    serve_stop_plan({"run": "net"})


def test_serve_requires_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.subprocess.Popen", FakeProc)
    web._SERVE_JOBS.clear()
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError):
        serve_start_plan(tmp_path, {"run": "empty"})


def test_serve_stop_unknown_run(tmp_path):
    web._SERVE_JOBS.clear()
    with pytest.raises(ValueError, match="no server running"):
        serve_stop_plan({"run": "ghost"})
