"""Auth, exposure, CSRF, health, and audit behavior for the dashboard server."""

from http.server import ThreadingHTTPServer
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from picochat.web import WebConfig, _is_loopback_host, _make_handler


def _serve(tmp_path, auth_token=None):
    config = WebConfig(runs_dir=str(tmp_path), host="127.0.0.1", port=0, auth_token=auth_token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _base(server):
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_loopback_host_detection():
    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("localhost")
    assert _is_loopback_host("::1")
    assert _is_loopback_host("")
    assert not _is_loopback_host("0.0.0.0")
    assert not _is_loopback_host("192.168.1.5")


def test_api_open_on_loopback_without_token(tmp_path):
    server = _serve(tmp_path, auth_token=None)
    try:
        with urllib.request.urlopen(f"{_base(server)}/api/runs") as resp:
            assert resp.status == 200
    finally:
        server.shutdown()


def test_api_requires_token_when_configured(tmp_path):
    server = _serve(tmp_path, auth_token="s3cret")
    try:
        base = _base(server)
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/api/runs")
        assert missing.value.code == 401

        with pytest.raises(urllib.error.HTTPError) as wrong:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/api/runs", headers={"X-Picochat-Token": "nope"})
            )
        assert wrong.value.code == 401

        good = urllib.request.Request(f"{base}/api/runs", headers={"X-Picochat-Token": "s3cret"})
        with urllib.request.urlopen(good) as resp:
            assert resp.status == 200

        bearer = urllib.request.Request(f"{base}/api/runs", headers={"Authorization": "Bearer s3cret"})
        with urllib.request.urlopen(bearer) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()


def test_static_shell_does_not_require_token(tmp_path):
    server = _serve(tmp_path, auth_token="s3cret")
    try:
        # The shell must load so the SPA can bootstrap the token from ?token=.
        with urllib.request.urlopen(f"{_base(server)}/assets/picochat-symbol.svg") as resp:
            assert resp.status == 200
    finally:
        server.shutdown()


def test_healthz_is_open_and_reports_version(tmp_path):
    server = _serve(tmp_path, auth_token="s3cret")
    try:
        # /healthz must be reachable without the token (container healthchecks).
        with urllib.request.urlopen(f"{_base(server)}/healthz") as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert data["version"]
        # Reflects live in-process job state; just assert it's a sane count.
        assert isinstance(data["active_jobs"], int)
        assert data["active_jobs"] >= 0
    finally:
        server.shutdown()


def test_audit_log_records_state_changing_posts(tmp_path):
    server = _serve(tmp_path, auth_token=None)
    try:
        request = urllib.request.Request(
            f"{_base(server)}/api/run/archive",
            data=json.dumps({"run_name": "ghost"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(request)  # archiving a missing run errors

        # The audit write happens on the server thread after the response is
        # sent, so wait briefly for it to land.
        audit = tmp_path / ".audit" / "audit.jsonl"
        deadline = time.time() + 5
        while not audit.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert audit.exists()
        record = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
        assert record["action"] == "/api/run/archive"
        assert record["outcome"] == "error"
        assert record["params"]["run_name"] == "ghost"
        assert record["actor"]
        assert record["ts"]
    finally:
        server.shutdown()


def test_post_rejects_cross_origin(tmp_path):
    server = _serve(tmp_path, auth_token=None)
    try:
        request = urllib.request.Request(
            f"{_base(server)}/api/run/start",
            data=b"{}",
            headers={"Origin": "http://evil.example", "Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 403
    finally:
        server.shutdown()
