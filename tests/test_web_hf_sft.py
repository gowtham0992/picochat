"""Launching `train hf-sft` (fine-tune an existing HF model) from the dashboard."""

import json

import pytest

import picochat.web as web
from picochat.web import discover_runs, hf_sft_start_plan, serve_start_plan, serve_stop_plan


class FakeProc:
    pid = 7777

    def __init__(self, *args, **kwargs):
        pass

    def poll(self):
        return None


def _pack(tmp_path):
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({"corpus": "chat.jsonl", "chat": "chat.jsonl", "eval": "chat.jsonl"}), encoding="utf-8")
    return pack


def test_hf_sft_launches_job_from_pack(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.subprocess.Popen", FakeProc)
    web._RUN_JOBS.clear()
    pack = _pack(tmp_path)

    status = hf_sft_start_plan(tmp_path, {
        "model": "org/base",
        "dataset_pack": str(pack),
        "max_steps": 5,
        "peft": "lora",
        "device": "cpu",
        "run_name": "ft1",
    })

    job = status["job"]
    assert job["run_name"] == "ft1"
    command = job["command"]
    assert "train hf-sft" in command
    assert "--model org/base" in command
    assert "--peft lora" in command
    assert "--max-steps 5" in command
    assert job["launch_config"]["kind"] == "hf-sft"
    assert (tmp_path / "ft1" / "web_run.log").exists()
    assert (tmp_path / "ft1" / "web_job.json").exists()


def test_hf_sft_requires_model(tmp_path):
    web._RUN_JOBS.clear()
    with pytest.raises(ValueError, match="model is required"):
        hf_sft_start_plan(tmp_path, {"dataset_pack": str(_pack(tmp_path))})


def test_hf_sft_requires_data(tmp_path):
    web._RUN_JOBS.clear()
    with pytest.raises(ValueError, match="dataset_pack or input is required"):
        hf_sft_start_plan(tmp_path, {"model": "org/base"})


def _make_hf_run(tmp_path, name="hf-demo"):
    run = tmp_path / name
    (run / "final_model").mkdir(parents=True)
    (run / "hf_sft_report.json").write_text(
        json.dumps({"model": "org/base", "best_val_loss": 1.0}), encoding="utf-8"
    )
    return run


def test_discover_runs_surfaces_hf_run(tmp_path):
    _make_hf_run(tmp_path)
    rows = [r for r in discover_runs(tmp_path) if r["name"] == "hf-demo"]
    assert rows and rows[0]["kind"] == "hf-sft"
    assert rows[0]["base_model"] == "org/base"


def test_serve_hf_run_uses_hf_model_flag(tmp_path, monkeypatch):
    captured = {}

    class CapturingProc:
        pid = 4242

        def __init__(self, command, *args, **kwargs):
            captured["command"] = command

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr("picochat.web.subprocess.Popen", CapturingProc)
    web._SERVE_JOBS.clear()
    run = _make_hf_run(tmp_path)

    started = serve_start_plan(tmp_path, {"run": "hf-demo"}, host="127.0.0.1")
    assert started["server"]["run"] == "hf-demo"
    assert "--hf-model" in captured["command"]
    assert str(run / "final_model") in captured["command"]
    assert "--checkpoint" not in captured["command"]

    serve_stop_plan({"run": "hf-demo"})
