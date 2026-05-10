from picochat.report import (
    chat_eval_report_markdown,
    loss_diagnostics,
    sft_report_markdown,
    tiny_run_summary_markdown,
    training_report_markdown,
)


def test_loss_diagnostics_detects_validation_regression():
    diagnostics = loss_diagnostics([
        {"step": 1, "train_loss": 2.0, "val_loss": 2.1},
        {"step": 2, "train_loss": 0.4, "val_loss": 1.8},
        {"step": 3, "train_loss": 0.1, "val_loss": 3.2},
    ])

    assert diagnostics["status"] == "memorization-risk"
    assert diagnostics["best_val_step"] == 2
    assert round(diagnostics["final_gap"], 2) == 3.10
    assert round(diagnostics["val_regression"], 2) == 1.40


def test_training_report_markdown_contains_key_sections():
    report = {
        "config": {
            "corpus_path": "corpus.txt",
            "tokenizer_path": "tokenizer.json",
            "max_steps": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "val_fraction": 0.1,
        },
        "dataset": {
            "num_tokens": 100,
            "context_size": 8,
            "num_sequences": 92,
            "train_sequences": 83,
            "val_sequences": 9,
        },
        "model": {
            "num_parameters": 1234,
            "config": {
                "vocab_size": 20,
                "n_layer": 1,
                "n_embd": 16,
                "n_head": 4,
                "dropout": 0.0,
            },
        },
        "losses": [{"step": 1, "train_loss": 3.2, "val_loss": 3.4, "elapsed_sec": 0.1}],
        "sample": "hello",
        "checkpoint": "checkpoint",
    }

    markdown = training_report_markdown(report)

    assert "# Picochat Base Training Report" in markdown
    assert "## Dataset" in markdown
    assert "## Model" in markdown
    assert "## Training" in markdown
    assert "## Loss Diagnostics" in markdown
    assert "Best validation step" in markdown
    assert "hello" in markdown


def test_sft_report_markdown_contains_key_sections():
    report = {
        "config": {
            "input_path": "chat.jsonl",
            "tokenizer_path": "tokenizer.json",
            "max_steps": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
        },
        "base_checkpoint": {
            "path": "base/checkpoint",
            "step": 10,
            "train_loss": 2.5,
        },
        "dataset": {
            "num_examples": 5,
            "context_size": 32,
            "supervised_tokens": 100,
            "train_examples": 4,
            "val_examples": 1,
        },
        "model": {
            "num_parameters": 1234,
            "config": {
                "vocab_size": 20,
                "n_layer": 1,
                "n_embd": 16,
                "n_head": 4,
            },
        },
        "losses": [{"step": 1, "train_loss": 3.2, "val_loss": 3.4, "elapsed_sec": 0.1}],
        "sample": "User: hello\nAssistant:",
        "checkpoint": "checkpoint",
        "best_checkpoint": {
            "path": "best_checkpoint",
            "step": 1,
            "train_loss": 3.2,
            "val_loss": 3.4,
        },
    }

    markdown = sft_report_markdown(report)

    assert "# Picochat SFT Report" in markdown
    assert "## Dataset" in markdown
    assert "## Base Checkpoint" in markdown
    assert "## Training" in markdown
    assert "## Loss Diagnostics" in markdown
    assert "Best validation checkpoint" in markdown
    assert "User: hello" in markdown


def test_chat_eval_report_markdown_contains_key_sections():
    report = {
        "config": {
            "input_path": "eval.jsonl",
            "tokenizer_path": "tokenizer.json",
            "temperature": 0.0,
            "max_new_tokens": 20,
            "case_sensitive": False,
        },
        "checkpoint": {
            "path": "checkpoint",
            "step": 10,
            "train_loss": 2.5,
        },
        "summary": {
            "num_examples": 1,
            "num_passed": 0,
            "num_failed": 1,
            "pass_rate": 0.0,
            "num_answerable": 1,
            "num_unanswerable": 0,
            "unsupported_claims": 1,
            "unsupported_claim_rate": 1.0,
            "prompt_echoes": 1,
            "prompt_echo_rate": 1.0,
            "missing_support": 1,
            "missing_support_rate": 1.0,
            "category_breakdown": {
                "story_generation": {
                    "num_examples": 1,
                    "num_passed": 0,
                    "num_failed": 1,
                    "pass_rate": 0.0,
                    "num_answerable": 1,
                    "num_unanswerable": 0,
                    "unsupported_claims": 1,
                    "unsupported_claim_rate": 1.0,
                    "prompt_echoes": 1,
                    "prompt_echo_rate": 1.0,
                    "missing_support": 1,
                    "missing_support_rate": 1.0,
                },
            },
            "split_breakdown": {
                "transfer": {
                    "num_examples": 1,
                    "num_passed": 0,
                    "num_failed": 1,
                    "pass_rate": 0.0,
                    "num_answerable": 1,
                    "num_unanswerable": 0,
                    "unsupported_claims": 1,
                    "unsupported_claim_rate": 1.0,
                    "prompt_echoes": 1,
                    "prompt_echo_rate": 1.0,
                    "missing_support": 1,
                    "missing_support_rate": 1.0,
                },
            },
        },
        "examples": [{
            "user": "What is Picochat?",
            "split": "transfer",
            "reply": "I do not know.",
            "must_include": ["Picochat"],
            "must_include_any": [["educational", "learning"]],
            "must_not_include": ["I do not know"],
            "missing": ["Picochat"],
            "missing_any": [["educational", "learning"]],
            "found_forbidden": ["I do not know"],
            "prompt_echo": True,
            "prompt_echo_reasons": ["chat_role_label"],
            "passed": False,
        }],
    }

    markdown = chat_eval_report_markdown(report)

    assert "# Picochat Chat Eval Report" in markdown
    assert "## Summary" in markdown
    assert "FAIL" in markdown
    assert "## Honesty Metrics" in markdown
    assert "## Category Breakdown" in markdown
    assert "## Split Breakdown" in markdown
    assert "`story_generation`" in markdown
    assert "`transfer`" in markdown
    assert "Prompt echo rate" in markdown
    assert "Prompt echo: `chat_role_label`" in markdown
    assert "does not prove semantic truth" in markdown
    assert "Required any-phrase groups" in markdown
    assert "Missing any-group" in markdown
    assert "I do not know." in markdown


def test_tiny_run_summary_markdown_contains_key_sections():
    summary = {
        "config": {
            "context_size": 128,
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 2,
            "base_steps": 300,
            "sft_steps": 600,
            "device": "cpu",
        },
        "artifacts": {
            "corpus": "corpus.txt",
            "tokenizer": "tokenizer.json",
            "base_report": "base/report.md",
            "sft_report": "sft/report.md",
            "eval_report": "eval/report.md",
            "honesty_report": "honesty/report.md",
        },
        "base": {
            "final_train_loss": 2.0,
            "final_val_loss": 2.1,
            "loss_diagnostics": {
                "status": "stable",
                "final_gap": 0.1,
            },
        },
        "sft": {
            "final_train_loss": 0.1,
            "final_val_loss": 4.0,
            "truncated_examples": 0,
            "loss_diagnostics": {
                "status": "memorization-risk",
                "final_gap": 3.9,
            },
        },
        "eval": {
            "num_examples": 4,
            "num_passed": 3,
            "num_failed": 1,
            "pass_rate": 0.75,
            "unsupported_claim_rate": 0.25,
            "prompt_echo_rate": 0.25,
            "missing_support_rate": 0.5,
            "category_breakdown": {
                "story_generation": {
                    "num_examples": 4,
                    "num_passed": 3,
                    "num_failed": 1,
                    "pass_rate": 0.75,
                    "unsupported_claims": 1,
                    "prompt_echoes": 1,
                    "missing_support": 2,
                },
            },
            "split_breakdown": {
                "prompt_conditioned": {
                    "num_examples": 4,
                    "num_passed": 3,
                    "num_failed": 1,
                    "pass_rate": 0.75,
                    "unsupported_claims": 1,
                    "prompt_echoes": 1,
                    "missing_support": 2,
                },
            },
        },
        "honesty": {
            "status": "ready",
            "summary": "No obvious eval leakage was detected.",
            "exact_prompt_leaks": 0,
            "near_prompt_leaks": 0,
            "corpus_prompt_hits": 0,
            "duplicate_eval_prompts": 0,
            "max_sft_prompt_similarity": 0.25,
        },
    }

    markdown = tiny_run_summary_markdown(summary)

    assert "# Picochat Tiny Run Summary" in markdown
    assert "Eval passed: 3 / 4" in markdown
    assert "## Data Honesty" in markdown
    assert "Data honesty report" in markdown
    assert "## Eval Categories" in markdown
    assert "## Eval Splits" in markdown
    assert "`story_generation`" in markdown
    assert "`prompt_conditioned`" in markdown
    assert "Unsupported claim rate: 25.0000%" in markdown
    assert "Prompt echo rate: 25.0000%" in markdown
    assert "Base final train loss" in markdown
    assert "SFT loss status: `memorization-risk`" in markdown
    assert "eval/report.md" in markdown
