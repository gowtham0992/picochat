import json
import subprocess
import sys


def test_pocket_tutor_practice_pack_builder_writes_qwen_ready_messages(tmp_path):
    out_dir = tmp_path / "pocket-tutor"
    subprocess.run(
        [
            sys.executable,
            "tools/build_pocket_tutor_practice.py",
            "--out-dir",
            str(out_dir),
            "--train-rows",
            "25",
            "--eval-rows",
            "10",
        ],
        check=True,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    train_rows = [
        json.loads(line)
        for line in (out_dir / "pocket_tutor_train_messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in (out_dir / "pocket_tutor_eval_prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["status"] == "practice-only"
    assert manifest["recommended_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert len(train_rows) == 25
    assert len(eval_rows) == 10
    assert set(manifest["categories"]) == {
        "hint_without_answer",
        "math_feedback",
        "parent_summary",
        "reading_feedback",
        "spelling_feedback",
    }

    sample = train_rows[0]
    assert sample["metadata"]["practice_only"] is True
    assert [message["role"] for message in sample["messages"]] == ["system", "user", "assistant"]
    assert "Pocket Tutor Lab" in sample["messages"][0]["content"]
    assistant_payload = json.loads(sample["messages"][-1]["content"])
    assert isinstance(assistant_payload, dict)
    assert sample["expected_answer"]

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "picochat train hf-sft" in readme
    assert "Rebuild official data during the hackathon window" in readme
