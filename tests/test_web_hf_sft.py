"""Launching `train hf-sft` (fine-tune an existing HF model) from the dashboard."""

import json

import pytest

import picochat.web as web
from picochat.web import hf_sft_start_plan


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
