"""Remote (Modal) training launch from the dashboard."""

import pytest

import picochat.web as web
from picochat.web import diagnose_remote_failure, remote_modal_pull_plan, remote_modal_start_plan, remote_status_plan


def test_remote_status_reports_modal(monkeypatch):
    monkeypatch.setattr("picochat.web.shutil.which", lambda name: None)
    status = remote_status_plan()
    assert status["modal_available"] is False
    assert "modal_script" in status


def test_remote_diagnostic_ignores_hf_rate_limit_warning():
    warning = (
        "Warning: You are sending unauthenticated requests to the HF Hub. "
        "Please set a HF_TOKEN to enable higher rate limits and faster downloads."
    )
    assert diagnose_remote_failure(warning) is None


def test_remote_diagnostic_detects_hf_auth_failure():
    diagnostic = diagnose_remote_failure("401 Unauthorized: access to this model is restricted")
    assert diagnostic is not None
    assert diagnostic["kind"] == "hf_auth"


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
    monkeypatch.setattr("picochat.web._probe_command", lambda *args, **kwargs: {
        "ok": True,
        "stdout": "test-profile",
        "stderr": "",
        "returncode": 0,
        "command": ["modal", "profile", "current"],
    })
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
    assert "--mode" in command and "native" in command
    assert "--hf-dataset" in command and "org/ds" in command
    assert started["job"]["launch_readiness"]["modal_authenticated"] is True
    assert started["job"]["run_name"] == "cloud1"
    assert (tmp_path / "cloud1" / "web_run.log").exists()


def test_remote_modal_builds_hf_sft_command(tmp_path, monkeypatch):
    captured = {}

    class CapturingProc:
        pid = 100

        def __init__(self, command, *args, **kwargs):
            captured["command"] = command

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr("picochat.web.shutil.which", lambda name: "/usr/bin/modal")
    monkeypatch.setattr("picochat.web._probe_command", lambda *args, **kwargs: {
        "ok": True,
        "stdout": "test-profile",
        "stderr": "",
        "returncode": 0,
        "command": ["modal", "profile", "current"],
    })
    monkeypatch.setattr("picochat.web.subprocess.Popen", CapturingProc)
    web._RUN_JOBS.clear()

    started = remote_modal_start_plan(tmp_path, {
        "run_name": "security-cloud",
        "mode": "hf-sft",
        "dataset_pack": "datasets/security-analyst/dataset_pack.json",
        "gpu": "A100",
        "hf_model": "HuggingFaceTB/SmolLM3-3B",
        "hf_sft_steps": 1200,
        "hf_batch_size": 2,
        "hf_grad_accum_steps": 8,
        "hf_eval_batches": 12,
        "hf_log_every": 20,
        "timeout_hours": 16,
        "security_source": "trendyol",
        "security_max_rows": 10000,
        "security_eval_rows": 500,
        "security_preference_rows": 128,
        "preference_input": "datasets/security-analyst/preferences.jsonl",
        "run_dpo": True,
    })

    command = captured["command"]
    assert "--mode" in command and "hf-sft" in command
    assert "--dataset-pack" in command and "datasets/security-analyst/dataset_pack.json" in command
    assert "--hf-dataset" not in command
    assert "--hf-max-rows" not in command
    assert "--hf-model" in command and "HuggingFaceTB/SmolLM3-3B" in command
    assert "--hf-sft-steps" in command and "1200" in command
    assert "--hf-batch-size" in command and "2" in command
    assert "--hf-grad-accum-steps" in command and "8" in command
    assert "--hf-eval-batches" in command and "12" in command
    assert "--hf-log-every" in command and "20" in command
    assert "--hf-learning-rate" in command and "2e-05" in command
    assert "--hf-max-length" in command and "1024" in command
    assert "--hf-lora-rank" in command and "16" in command
    assert "--hf-lora-alpha" in command and "32.0" in command
    assert "--hf-quantize" in command and "4bit" in command
    assert "--security-source" in command and "trendyol" in command
    assert "--security-max-rows" in command and "10000" in command
    assert "--security-eval-rows" in command and "500" in command
    assert "--security-preference-rows" in command and "128" in command
    assert "--timeout-hours" in command and "16" in command
    assert "--preference-input" in command and "datasets/security-analyst/preferences.jsonl" in command
    assert "--run-dpo" in command
    assert "--dpo-steps" in command and "100" in command
    assert "--dpo-beta" in command and "0.1" in command
    assert started["job"]["launch_config"]["mode"] == "hf-sft"
    assert started["job"]["launch_config"]["hf_dataset"] is None
    assert started["job"]["launch_config"]["hf_batch_size"] == 2
    assert started["job"]["launch_config"]["hf_grad_accum_steps"] == 8
    assert started["job"]["launch_config"]["security_source"] == "trendyol"
    assert started["job"]["launch_config"]["security_max_rows"] == 10000
    assert started["job"]["launch_config"]["security_eval_rows"] == 500
    assert started["job"]["launch_config"]["security_preference_rows"] == 128
    assert started["job"]["launch_config"]["preference_input"] == "datasets/security-analyst/preferences.jsonl"
    assert started["job"]["launch_config"]["run_dpo"] is True
    assert started["job"]["launch_config"]["dpo_steps"] == 100
    assert started["job"]["launch_config"]["dpo_beta"] == 0.1


def test_remote_pull_requires_cli(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.shutil.which", lambda name: None)
    with pytest.raises(ValueError, match="modal"):
        remote_modal_pull_plan(tmp_path, {"run": "r1"})


def test_remote_pull_builds_command(tmp_path, monkeypatch):
    captured = {}

    class CapturingProc:
        pid = 5

        def __init__(self, command, *args, **kwargs):
            captured["command"] = command

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr("picochat.web.shutil.which", lambda name: "/usr/bin/modal")
    monkeypatch.setattr("picochat.web.subprocess.Popen", CapturingProc)
    web._RUN_JOBS.clear()

    started = remote_modal_pull_plan(tmp_path, {"run": "cloud-run-1"})

    command = captured["command"]
    assert command[0:3] == ["modal", "volume", "get"]
    assert "picochat-runs" in command
    assert "cloud-run-1" in command
    assert str(tmp_path.resolve()) in command
    assert started["job"]["run_name"] == "pull-cloud-run-1"


def test_remote_pull_refuses_existing_local(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.shutil.which", lambda name: "/usr/bin/modal")
    (tmp_path / "exists").mkdir()
    (tmp_path / "exists" / "x.txt").write_text("y", encoding="utf-8")
    with pytest.raises(FileExistsError):
        remote_modal_pull_plan(tmp_path, {"run": "exists"})
