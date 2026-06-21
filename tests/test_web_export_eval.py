"""Export-to-HF and re-run-eval from the dashboard."""

import json

import pytest

import picochat.web as web
from picochat.web import eval_run_plan, export_hf_run_plan


class FakeProc:
    pid = 321

    def __init__(self, command, *args, **kwargs):
        self.command = command

    def poll(self):
        return None


def _run_with_checkpoint(tmp_path, name="r1"):
    run = tmp_path / name
    (run / "sft" / "checkpoint").mkdir(parents=True)
    (run / "tokenizer.json").write_text("{}", encoding="utf-8")
    return run


def test_export_requires_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_hf_run_plan(tmp_path, {"run": "ghost"})


def test_export_calls_export_with_run_paths(tmp_path, monkeypatch):
    run = _run_with_checkpoint(tmp_path)
    captured = {}

    def fake_export(config):
        captured["config"] = config
        return {"out_dir": config.out_dir, "manifest": "m.json", "model_card": "README.md"}

    monkeypatch.setattr("picochat.hf_export.export_hf_checkpoint", fake_export)

    result = export_hf_run_plan(tmp_path, {"run": "r1"})

    assert result["run"] == "r1"
    assert result["out_dir"] == str(run / "export-hf")
    assert captured["config"].checkpoint_path == str(run / "sft" / "checkpoint")
    assert captured["config"].tokenizer_path == str(run / "tokenizer.json")


def test_eval_run_launches_job(tmp_path, monkeypatch):
    monkeypatch.setattr("picochat.web.subprocess.Popen", FakeProc)
    web._RUN_JOBS.clear()
    _run_with_checkpoint(tmp_path)
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")

    started = eval_run_plan(tmp_path, {"run": "r1", "input": str(eval_file)})

    job = started["job"]
    assert job["run_name"] == "eval-r1"
    assert "eval chat" in job["command"]
    assert str(eval_file) in job["command"]


def test_eval_run_requires_eval_set(tmp_path):
    _run_with_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="eval set"):
        eval_run_plan(tmp_path, {"run": "r1"})
