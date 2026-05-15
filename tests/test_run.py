import json

import pytest

from picochat.run import TinyRunConfig, _validation_log_every, run_tiny


def test_validation_log_every_keeps_long_runs_observable():
    assert _validation_log_every(1) == 1
    assert _validation_log_every(24) == 1
    assert _validation_log_every(240) == 10
    assert _validation_log_every(30000) == 1250


def test_run_tiny_writes_full_experiment_artifacts(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
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
    pack_path.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    summary = run_tiny(TinyRunConfig(
        out_dir=str(out_dir),
        dataset_pack=str(pack_path),
        context_size=32,
        n_embd=16,
        n_head=4,
        n_layer=1,
        base_steps=1,
        sft_steps=1,
        base_batch_size=2,
        sft_batch_size=1,
        eval_max_new_tokens=0,
        allow_leaky_eval=True,
    ))

    assert (out_dir / "corpus.txt").exists()
    assert (out_dir / "corpus_manifest.json").exists()
    assert (out_dir / "corpus_report.md").exists()
    assert (out_dir / "preflight.json").exists()
    assert (out_dir / "preflight.md").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "base" / "checkpoint" / "model.pt").exists()
    assert (out_dir / "base" / "best_checkpoint" / "model.pt").exists()
    assert (out_dir / "sft" / "checkpoint" / "model.pt").exists()
    assert (out_dir / "sft" / "best_checkpoint" / "model.pt").exists()
    assert (out_dir / "sft_fit" / "sft_fit_eval.jsonl").exists()
    assert (out_dir / "sft_fit" / "eval_report.json").exists()
    assert (out_dir / "eval" / "eval_report.json").exists()
    assert (out_dir / "honesty" / "honesty_report.json").exists()
    assert (out_dir / "honesty" / "report.md").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    assert summary["eval"]["num_examples"] == 1
    assert summary["config"]["tokenizer_type"] == "char"
    assert summary["tokenizer"]["tokenizer_type"] == "char"
    assert "corpus_manifest" in summary["artifacts"]
    assert "preflight_report" in summary["artifacts"]
    assert summary["preflight"]["status"] in {"ready", "warn"}
    assert summary["preflight"]["budget"]["target_param_data_ratio"] == 20.0
    assert summary["preflight"]["budget"]["recommended_base_steps"] >= 1
    assert summary["preflight"]["budget"]["planned_to_target_ratio"] is not None
    assert "honesty_report" in summary["artifacts"]
    assert summary["honesty"]["status"] == "blocked"
    assert summary["honesty"]["exact_prompt_leaks"] == 1
    assert summary["artifacts"]["sft_eval_checkpoint"] == str(out_dir / "sft" / "best_checkpoint")
    assert summary["artifacts"]["base_eval_checkpoint"] == str(out_dir / "base" / "best_checkpoint")
    assert summary["base"]["eval_checkpoint"] == str(out_dir / "base" / "best_checkpoint")
    assert summary["base"]["coverage"]["actual_steps"] == 1
    assert summary["base"]["stop_reason"] == "max_steps"
    assert summary["sft"]["eval_checkpoint"] == str(out_dir / "sft" / "best_checkpoint")
    assert summary["sft_fit"]["num_examples"] == 1
    assert summary["sft_fit_dataset"]["num_rows"] == 1
    assert summary["config"]["dataset_pack"] == str(pack_path)
    assert summary["config"]["chat_input"] == str(chat_path)
    assert summary["config"]["eval_input"] == str(eval_path)
    assert summary["long_run_gate"]["status"] in {"approved", "warn", "blocked"}


def test_run_tiny_blocks_leaky_eval_by_default(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    out_dir = tmp_path / "run"
    corpus_path.write_text("A clean training story.\n" * 8, encoding="utf-8")
    chat_path.write_text(
        json.dumps({"user": "hi", "assistant": "hello"}),
        encoding="utf-8",
    )
    eval_path.write_text(json.dumps({"user": "hi"}), encoding="utf-8")
    pack_path.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="data honesty blocked"):
        run_tiny(TinyRunConfig(
            out_dir=str(out_dir),
            dataset_pack=str(pack_path),
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

    assert (out_dir / "honesty" / "honesty_report.json").exists()
    assert not (out_dir / "tokenizer.json").exists()


def test_run_tiny_blocks_unsafe_long_run_before_training(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack_path = tmp_path / "dataset_pack.json"
    out_dir = tmp_path / "run"
    corpus_path.write_text("A clean training document about local models.\n" * 200, encoding="utf-8")
    chat_path.write_text(
        json.dumps({"user": "What is this?", "assistant": "This is a local model test."}) + "\n",
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps({"user": "Say local model.", "must_include": ["local model"]}) + "\n",
        encoding="utf-8",
    )
    pack_path.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="run preflight blocked"):
        run_tiny(TinyRunConfig(
            out_dir=str(out_dir),
            dataset_pack=str(pack_path),
            tokenizer_type="bpe",
            tokenizer_vocab_size=1024,
            context_size=64,
            n_embd=128,
            n_head=4,
            n_layer=4,
            base_steps=5000,
            sft_steps=1000,
            base_batch_size=8,
            sft_batch_size=8,
        ))

    assert (out_dir / "preflight.json").exists()
    assert not (out_dir / "honesty").exists()
    assert not (out_dir / "tokenizer.json").exists()


def test_run_tiny_blocks_custom_corpus_with_default_tuning_data(tmp_path):
    corpus_path = tmp_path / "domain.txt"
    out_dir = tmp_path / "run"
    corpus_path.write_text("This is a custom domain corpus about coffee.\n" * 40, encoding="utf-8")

    with pytest.raises(ValueError, match="demo tuning data"):
        run_tiny(TinyRunConfig(
            out_dir=str(out_dir),
            corpus_input=str(corpus_path),
            context_size=32,
            n_embd=16,
            n_head=4,
            n_layer=1,
            base_steps=1,
            sft_steps=1,
        ))

    assert (out_dir / "corpus.txt").exists()
    assert not (out_dir / "honesty").exists()
