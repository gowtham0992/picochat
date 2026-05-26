import json
import sys

import pytest

from picochat.lm_harness import (
    LMEvalHarnessConfig,
    build_lm_eval_command,
    parse_lm_eval_tasks,
    run_lm_eval_harness,
)


def test_build_lm_eval_command_includes_trust_remote_code_and_tasks():
    command = build_lm_eval_command(LMEvalHarnessConfig(
        model_path="exports/pico",
        tasks=("arc_easy", "hellaswag"),
        out_dir="reports/lm_eval",
        device="cuda:0",
        batch_size="8",
        limit="10",
        num_fewshot=5,
        extra_model_args=("dtype=bfloat16",),
    ))

    assert command[:3] == [sys.executable, "-m", "lm_eval"]
    assert "--tasks" in command
    assert command[command.index("--tasks") + 1] == "arc_easy,hellaswag"
    model_args = command[command.index("--model_args") + 1]
    assert "pretrained=exports/pico" in model_args
    assert "trust_remote_code=True" in model_args
    assert "dtype=bfloat16" in model_args
    assert command[command.index("--limit") + 1] == "10"
    assert command[command.index("--num_fewshot") + 1] == "5"


def test_lm_eval_dry_run_writes_command_metadata(tmp_path):
    report = run_lm_eval_harness(LMEvalHarnessConfig(
        model_path="exports/pico",
        tasks=("arc_easy",),
        out_dir=str(tmp_path),
        dry_run=True,
    ))

    assert report["dry_run"] is True
    metadata = json.loads((tmp_path / "lm_eval_command.json").read_text(encoding="utf-8"))
    assert metadata["tasks"] == ["arc_easy"]
    assert "python" in metadata["command_text"] or sys.executable in metadata["command_text"]


def test_parse_lm_eval_tasks_rejects_empty():
    assert parse_lm_eval_tasks("arc_easy, hellaswag") == ("arc_easy", "hellaswag")
    with pytest.raises(ValueError, match="tasks cannot be empty"):
        parse_lm_eval_tasks(" , ")
