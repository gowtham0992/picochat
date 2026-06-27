import json
from pathlib import Path

from picochat.dataset_pack import load_dataset_pack
from picochat.security_pack import SecurityPackConfig, build_security_analyst_pack


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_security_pack_builds_seed_only_pack(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    _write_jsonl(seed / "chat.jsonl", [
        {
            "category": "incident_triage",
            "user": "How should I triage a suspicious PowerShell alert?",
            "assistant": "Confirm host scope, preserve logs, isolate only if risk is confirmed, and document the timeline.",
        },
        {
            "category": "incident_triage",
            "user": "How should I triage a suspicious PowerShell alert?",
            "assistant": "Confirm host scope, preserve logs, isolate only if risk is confirmed, and document the timeline.",
        },
    ])
    _write_jsonl(seed / "eval.jsonl", [
        {"category": "incident_triage", "user": "What evidence should I collect first for a suspicious script?"},
    ])
    _write_jsonl(seed / "preferences.jsonl", [
        {
            "user": "Help with a phishing report.",
            "chosen": "Review headers safely and preserve evidence.",
            "rejected": "Click every link to see what happens.",
        },
    ])

    report = build_security_analyst_pack(SecurityPackConfig(
        out_dir=tmp_path / "out",
        seed_dir=seed,
        include_trendyol=False,
        preference_target_rows=8,
        force=True,
    ))

    assert report["chat_rows"] == 1
    assert report["duplicate_chat_rows_skipped"] == 1
    assert report["eval_rows"] == 1
    assert report["preference_rows"] == 8
    assert Path(report["preferences"]).is_file()

    pack = load_dataset_pack(report["dataset_pack"])
    assert Path(pack.chat_input).is_file()
    assert Path(pack.eval_input).is_file()
    assert Path(pack.corpus_input).is_file()
