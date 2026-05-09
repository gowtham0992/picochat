import json

from picochat.run import TinyRunConfig, run_tiny


def test_run_tiny_writes_full_experiment_artifacts(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    out_dir = tmp_path / "run"
    corpus_path.write_text(
        "Picochat is small.\nUser: hi\nAssistant: hello\n" * 8,
        encoding="utf-8",
    )
    chat_path.write_text(
        json.dumps({"user": "hi", "assistant": "hello"}),
        encoding="utf-8",
    )
    eval_path.write_text(json.dumps({"user": "hi"}), encoding="utf-8")

    summary = run_tiny(TinyRunConfig(
        out_dir=str(out_dir),
        corpus_input=str(corpus_path),
        chat_input=str(chat_path),
        eval_input=str(eval_path),
        context_size=16,
        n_embd=16,
        n_head=4,
        n_layer=1,
        base_steps=1,
        sft_steps=1,
        base_batch_size=2,
        sft_batch_size=1,
        eval_max_new_tokens=0,
    ))

    assert (out_dir / "corpus.txt").exists()
    assert (out_dir / "corpus_manifest.json").exists()
    assert (out_dir / "corpus_report.md").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "base" / "checkpoint" / "model.pt").exists()
    assert (out_dir / "sft" / "checkpoint" / "model.pt").exists()
    assert (out_dir / "eval" / "eval_report.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    assert summary["eval"]["num_examples"] == 1
    assert "corpus_manifest" in summary["artifacts"]
