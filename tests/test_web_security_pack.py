import json
from pathlib import Path

from picochat.web import security_pack_plan


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_security_pack_plan_builds_seed_pack(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    _write_jsonl(seed / "chat.jsonl", [
        {"user": "How do I rotate a leaked API key?", "assistant": "Revoke it, create a new key, update consumers, and audit usage."},
    ])
    _write_jsonl(seed / "eval.jsonl", [
        {"user": "What is the first safe step after key exposure?"},
    ])
    _write_jsonl(seed / "preferences.jsonl", [
        {"user": "I found a public key.", "chosen": "Rotate it and audit access.", "rejected": "Keep using it."},
    ])

    report = security_pack_plan({
        "out_dir": str(tmp_path / "pack"),
        "seed_dir": str(seed),
        "include_trendyol": False,
        "preference_target_rows": 4,
        "force": True,
    }, runs_dir=tmp_path / "runs")

    assert Path(report["dataset_pack"]).is_file()
    assert Path(report["preferences"]).is_file()
    assert report["recommended_modal"]["mode"] == "hf-sft"
    assert report["recommended_modal"]["quantize"] == "4bit"
    assert "picochat data security-pack" in report["command"]
    assert "--source seed" in report["command"]
    assert "--preference-rows 4" in report["command"]
