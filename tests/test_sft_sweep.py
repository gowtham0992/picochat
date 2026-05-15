import json

from picochat.checkpoint import save_checkpoint
from picochat.model import GPTConfig, TinyGPT
from picochat.sft_sweep import SFTSweepConfig, _candidate_name, run_sft_sweep, sft_sweep_markdown
from picochat.tokenizer import CharTokenizer


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_run_sft_sweep_writes_candidate_and_summary_artifacts(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "base"
    out_dir = tmp_path / "sweep"
    rows = [
        {"user": "What is Picochat?", "assistant": "Picochat is small.", "category": "identity"},
        {"user": "What comes next?", "assistant": "SFT fit comes next.", "category": "identity"},
    ]
    write_jsonl(chat_path, rows)
    write_jsonl(eval_path, [{"user": "Say hello.", "category": "smoke"}])
    tokenizer = CharTokenizer.train([
        "User: What is Picochat?\nAssistant: Picochat is small.\n"
        "User: What comes next?\nAssistant: SFT fit comes next.\n"
        "User: Say hello.\nAssistant: hello"
    ])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=64,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    report = run_sft_sweep(SFTSweepConfig(
        input_path=str(chat_path),
        eval_input_path=str(eval_path),
        tokenizer_path=str(tokenizer_path),
        checkpoint_path=str(checkpoint_path),
        out_dir=str(out_dir),
        learning_rates=(1e-4,),
        step_counts=(1,),
        samplings=("uniform",),
        batch_size=1,
        eval_batches=1,
        sample_tokens=4,
        eval_max_new_tokens=0,
        fit_max_rows=1,
        packing="bos_bestfit",
        precision="float32",
        torch_compile=False,
    ))

    row = report["rows"][0]
    candidate_dir = out_dir / row["candidate"]
    assert (out_dir / "sft_sweep.json").exists()
    assert (out_dir / "sft_sweep.md").exists()
    assert (candidate_dir / "sft" / "best_checkpoint" / "model.pt").exists()
    assert (candidate_dir / "sft_fit" / "eval_report.json").exists()
    assert (candidate_dir / "eval" / "eval_report.json").exists()
    assert (candidate_dir / "candidate_summary.json").exists()
    assert row["sft_fit_examples"] == 1
    assert row["sft_fit_split"] == "sft_train"
    assert row["sft_fit_selected_from_indices"] is True
    assert row["packing"] == "bos_bestfit"
    assert report["config"]["precision"] == "float32"
    assert report["config"]["torch_compile"] is False
    assert row["eval_score"] is not None
    assert "eval_non_choice_pass_rate" in row
    assert report["best_sft_fit"]["candidate"] == row["candidate"]
    assert report["best_non_choice_eval"]["candidate"] == row["candidate"]


def test_sft_sweep_candidate_name_preserves_fractional_learning_rate():
    one = _candidate_name(sampling="category_sqrt", learning_rate=1e-4, step_count=400)
    one_point_five = _candidate_name(sampling="category_sqrt", learning_rate=1.5e-4, step_count=400)
    seven_point_five = _candidate_name(sampling="category_sqrt", learning_rate=7.5e-5, step_count=400)

    assert one == "category-sqrt-lr1em04-steps400"
    assert one_point_five == "category-sqrt-lr1p5em04-steps400"
    assert seven_point_five == "category-sqrt-lr7p5em05-steps400"
    assert len({one, one_point_five, seven_point_five}) == 3


def test_sft_sweep_markdown_explains_sft_fit_first():
    markdown = sft_sweep_markdown({
        "config": {
            "input_path": "chat.jsonl",
            "tokenizer_path": "tokenizer.json",
            "checkpoint_path": "base",
            "eval_input_path": "eval.jsonl",
            "packing": "bos_bestfit",
            "precision": "bf16",
            "matmul_precision": "high",
            "torch_compile": True,
        },
        "rows": [{
            "candidate": "uniform-lr1em04-steps1",
            "candidate_dir": "sweep/uniform-lr1em04-steps1",
            "learning_rate": 1e-4,
            "step_count": 1,
            "sampling": "uniform",
            "packing": "bos_bestfit",
            "sft_fit_pass_rate": 0.5,
            "eval_pass_rate": 0.25,
            "eval_non_choice_pass_rate": 0.20,
            "eval_choice_pass_rate": 1.0,
            "eval_refusal_pass_rate": 0.75,
            "sft_final_val_bpb": 1.2,
            "stop_reason": "max_steps",
        }],
        "best_sft_fit": {"candidate": "uniform-lr1em04-steps1"},
        "best_eval": {"candidate": "uniform-lr1em04-steps1"},
        "best_non_choice_eval": {"candidate": "uniform-lr1em04-steps1"},
    })

    assert "# Picochat SFT Sweep" in markdown
    assert "Use SFT fit first" in markdown
    assert "SFT packing: `bos_bestfit`" in markdown
    assert "Precision: `bf16`" in markdown
    assert "Matmul precision: `high`" in markdown
    assert "torch.compile: `True`" in markdown
    assert "Best non-choice held-out eval" in markdown
    assert "uniform-lr1em04-steps1" in markdown
