import json

import pytest

from picochat.run import TinyRunConfig, _long_run_gate, _validation_log_every, run_tiny, run_tiny_multiseed


def test_validation_log_every_keeps_long_runs_observable():
    assert _validation_log_every(1) == 1
    assert _validation_log_every(24) == 1
    assert _validation_log_every(240) == 10
    assert _validation_log_every(30000) == 1250


def test_long_run_gate_blocks_choice_inflated_eval():
    gate = _long_run_gate(
        preflight_report={"status": "warn", "budget": {"long_run": True}},
        sft_fit_summary={"pass_rate": 0.76},
        sft_fit_heldout_summary={"pass_rate": 0.70},
        eval_summary={
            "pass_rate": 0.37,
            "non_choice_examples": 400,
            "non_choice_pass_rate": 0.24,
            "choice_pass_rate": 1.0,
            "refusal_pass_rate": 0.82,
            "prompt_echo_rate": 0.0,
            "unsupported_claim_rate": 0.0,
        },
        honesty={"status": "ready"},
    )

    assert gate["status"] == "blocked"
    assert gate["eval_non_choice_rate"] == 0.24
    assert any(issue["name"] == "eval_non_choice" for issue in gate["issues"])


def test_long_run_gate_blocks_weak_refusal_and_sft_heldout():
    gate = _long_run_gate(
        preflight_report={"status": "warn", "budget": {"long_run": True}},
        sft_fit_summary={"pass_rate": 0.80},
        sft_fit_heldout_summary={"pass_rate": 0.45},
        eval_summary={
            "pass_rate": 0.60,
            "non_choice_examples": 100,
            "non_choice_pass_rate": 0.55,
            "refusal_pass_rate": 0.70,
            "prompt_echo_rate": 0.0,
            "unsupported_claim_rate": 0.0,
        },
        honesty={"status": "ready"},
    )

    assert gate["status"] == "blocked"
    assert any(issue["name"] == "sft_heldout_fit" for issue in gate["issues"])
    assert any(issue["name"] == "refusal" for issue in gate["issues"])


def test_long_run_gate_first_release_profile_focuses_release_categories():
    gate = _long_run_gate(
        preflight_report={"status": "warn", "budget": {"long_run": True}},
        sft_fit_summary={
            "pass_rate": 0.45,
            "category_breakdown": {
                "bench_choice_language": {"num_passed": 90, "num_examples": 100},
                "identity": {"num_passed": 80, "num_examples": 100},
                "refusal": {"num_passed": 30, "num_examples": 40},
                "bench_math_addition": {"num_passed": 0, "num_examples": 100},
                "bench_spelling_reverse": {"num_passed": 0, "num_examples": 100},
            },
        },
        sft_fit_heldout_summary={
            "pass_rate": 0.35,
            "category_breakdown": {
                "bench_choice_language": {"num_passed": 40, "num_examples": 50},
                "identity": {"num_passed": 35, "num_examples": 50},
                "refusal": {"num_passed": 20, "num_examples": 25},
                "bench_math_addition": {"num_passed": 0, "num_examples": 100},
            },
        },
        eval_summary={
            "pass_rate": 0.30,
            "non_choice_examples": 240,
            "non_choice_pass_rate": 0.10,
            "refusal_pass_rate": 0.82,
            "prompt_echo_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "category_breakdown": {
                "bench_choice_language": {"num_passed": 45, "num_examples": 50},
                "identity": {"num_passed": 36, "num_examples": 50},
                "refusal": {"num_passed": 24, "num_examples": 30},
                "bench_math_addition": {"num_passed": 0, "num_examples": 120},
                "bench_spelling_reverse": {"num_passed": 0, "num_examples": 120},
            },
        },
        honesty={"status": "ready"},
        profile="first_release",
    )

    assert gate["status"] == "approved"
    assert gate["profile"] == "first_release"
    assert gate["sft_fit_rate"] == pytest.approx(200 / 240)
    assert gate["sft_heldout_fit_rate"] == pytest.approx(95 / 125)
    assert gate["first_release_eval_rate"] == pytest.approx(105 / 130)
    assert not any(issue["name"] == "eval_non_choice" for issue in gate["issues"])


def test_long_run_gate_first_release_profile_blocks_weak_release_categories():
    gate = _long_run_gate(
        preflight_report={"status": "warn", "budget": {"long_run": True}},
        sft_fit_summary={
            "pass_rate": 0.85,
            "category_breakdown": {
                "bench_choice_language": {"num_passed": 10, "num_examples": 50},
                "identity": {"num_passed": 12, "num_examples": 50},
                "refusal": {"num_passed": 8, "num_examples": 25},
                "bench_math_addition": {"num_passed": 100, "num_examples": 100},
            },
        },
        sft_fit_heldout_summary={
            "pass_rate": 0.80,
            "category_breakdown": {
                "bench_choice_language": {"num_passed": 8, "num_examples": 50},
                "identity": {"num_passed": 10, "num_examples": 50},
                "refusal": {"num_passed": 6, "num_examples": 25},
            },
        },
        eval_summary={
            "pass_rate": 0.80,
            "non_choice_examples": 100,
            "non_choice_pass_rate": 0.80,
            "refusal_pass_rate": 0.80,
            "prompt_echo_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "category_breakdown": {
                "bench_choice_language": {"num_passed": 10, "num_examples": 50},
                "identity": {"num_passed": 10, "num_examples": 50},
                "refusal": {"num_passed": 8, "num_examples": 25},
            },
        },
        honesty={"status": "ready"},
        profile="first_release",
    )

    assert gate["status"] == "blocked"
    assert any(issue["name"] == "sft_fit" for issue in gate["issues"])
    assert any(issue["name"] == "sft_heldout_fit" for issue in gate["issues"])
    assert any(issue["name"] == "first_release_eval" for issue in gate["issues"])


def test_long_run_gate_rejects_unknown_profile():
    with pytest.raises(ValueError, match="profile must be one of"):
        _long_run_gate(
            preflight_report={"status": "warn", "budget": {"long_run": True}},
            sft_fit_summary={"pass_rate": 1.0},
            eval_summary={"prompt_echo_rate": 0.0, "unsupported_claim_rate": 0.0},
            honesty={"status": "ready"},
            profile="demo",
        )


def test_run_tiny_multiseed_aggregates_seed_runs(tmp_path, monkeypatch):
    seen = []

    def fake_run_tiny(config):
        seen.append((config.seed, config.out_dir))
        pass_rate = 0.25 if config.seed == 42 else 0.75
        return {
            "eval": {
                "num_passed": int(pass_rate * 4),
                "num_examples": 4,
                "pass_rate": pass_rate,
                "non_choice_pass_rate": pass_rate / 2,
                "pass_rate_ci": {"low": pass_rate, "high": pass_rate, "confidence": 0.95},
            },
            "sft_fit": {"pass_rate": pass_rate / 2},
            "base": {
                "best_val_bpb": 1.5 + pass_rate,
                "best_val_loss": 2.5 + pass_rate,
                "final_val_bpb": 2.0 + pass_rate,
                "final_val_loss": 3.0 + pass_rate,
            },
            "sft": {
                "best_checkpoint": {
                    "val_bpb": 0.5 + pass_rate,
                    "val_loss": 1.5 + pass_rate,
                },
                "final_val_bpb": 1.0 + pass_rate,
                "final_val_loss": 2.0 + pass_rate,
            },
            "long_run_gate": {"status": "blocked"},
        }

    monkeypatch.setattr("picochat.run.run_tiny", fake_run_tiny)

    out_dir = tmp_path / "multi"
    summary = run_tiny_multiseed(TinyRunConfig(out_dir=str(out_dir)), n_seeds=2)

    assert seen == [
        (42, str(out_dir / "seed-42")),
        (43, str(out_dir / "seed-43")),
    ]
    assert summary["type"] == "multi_seed_tiny"
    assert summary["config"]["seeds"] == [42, 43]
    assert summary["aggregate"]["eval_pass_rate"]["mean"] == 0.5
    assert summary["aggregate"]["eval_non_choice_pass_rate"]["mean"] == 0.25
    assert summary["aggregate"]["base_val_bpb"]["mean"] == 2.0
    assert summary["aggregate"]["sft_val_bpb"]["mean"] == 1.0
    assert round(summary["aggregate"]["eval_pass_rate"]["std"], 4) == 0.3536
    assert (out_dir / "summary.json").exists()
    assert "Picochat Multi-Seed Tiny Run" in (out_dir / "summary.md").read_text(encoding="utf-8")


def test_run_tiny_rejects_sft_resume_without_base_resume(tmp_path):
    with pytest.raises(ValueError, match="sft_resume_from requires base_resume_from"):
        run_tiny(TinyRunConfig(
            out_dir=str(tmp_path / "run"),
            sft_resume_from=str(tmp_path / "run" / "sft" / "resume_checkpoint"),
        ))


def test_run_tiny_writes_full_experiment_artifacts(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    external_path = tmp_path / "arc_mini.jsonl"
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
    external_path.write_text(json.dumps({
        "question": "Which option says hello?",
        "choices": {"label": ["A", "B"], "text": ["hello", "bye"]},
        "answerKey": "A",
        "category": "arc_easy_mini",
    }), encoding="utf-8")
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
        external_eval_inputs=(f"arc-mini={external_path}",),
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
    assert (out_dir / "external_eval" / "arc-mini" / "external_eval.jsonl").exists()
    assert (out_dir / "external_eval" / "arc-mini" / "eval_report.json").exists()
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
    assert "contamination_matrix" in summary["honesty"]
    assert any(
        pair["name"] == "generated_vs_sft"
        for pair in summary["honesty"]["contamination_matrix"]["pairs"]
    )
    assert summary["artifacts"]["sft_eval_checkpoint"] == str(out_dir / "sft" / "best_checkpoint")
    assert summary["artifacts"]["base_eval_checkpoint"] == str(out_dir / "base" / "best_checkpoint")
    assert summary["base"]["eval_checkpoint"] == str(out_dir / "base" / "best_checkpoint")
    assert summary["base"]["best_val_loss"] is not None
    assert "best_val_bpb" in summary["base"]
    assert summary["base"]["coverage"]["actual_steps"] == 1
    assert summary["base"]["stop_reason"] == "max_steps"
    assert summary["sft"]["eval_checkpoint"] == str(out_dir / "sft" / "best_checkpoint")
    assert summary["sft"]["best_val_loss"] is not None
    assert "best_val_bpb" in summary["sft"]
    assert summary["sft_fit"]["num_examples"] == 1
    assert summary["external_evals"][0]["name"] == "arc-mini"
    assert summary["external_evals"][0]["summary"]["num_examples"] == 1
    assert "external_eval_reports" in summary["artifacts"]
    assert summary["sft_fit_dataset"]["num_rows"] == 1
    assert summary["config"]["dataset_pack"] == str(pack_path)
    assert summary["config"]["chat_input"] == str(chat_path)
    assert summary["config"]["eval_input"] == str(eval_path)
    assert summary["long_run_gate"]["status"] in {"approved", "warn", "blocked"}
    assert summary["timing"]["total_seconds"] >= 0
    assert [item["stage"] for item in summary["timing"]["stages"]] == [
        "corpus_build_preflight",
        "data_honesty",
        "tokenizer",
        "base_train",
        "sft_train",
        "sft_fit_eval",
        "chat_eval_gate",
    ]


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
