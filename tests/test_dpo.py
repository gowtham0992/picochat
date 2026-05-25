import json
import math

import torch

from picochat.checkpoint import save_checkpoint
from picochat.dpo import (
    DPOConfig,
    PreferenceDataset,
    dpo_batch_metrics,
    load_preference_examples,
    train_dpo,
)
from picochat.model import GPTConfig, TinyGPT
from picochat.tokenizer import CharTokenizer


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _tokenizer_for_preferences(path):
    text = path.read_text(encoding="utf-8")
    return CharTokenizer.train([
        "User: Which answer is better?\nAssistant: ",
        text,
    ])


def test_load_preference_examples_accepts_alias_fields(tmp_path):
    path = tmp_path / "prefs.jsonl"
    _write_jsonl(path, [
        {
            "prompt": "Which answer is better?",
            "preferred": "The clear answer.",
            "dispreferred": "Bad.",
            "category": "helpfulness",
            "group": "g1",
        },
        {
            "user": "Pick one.",
            "winner": "A",
            "loser": "B",
        },
    ])

    examples = load_preference_examples(path)

    assert len(examples) == 2
    assert examples[0].user == "Which answer is better?"
    assert examples[0].chosen == "The clear answer."
    assert examples[0].rejected == "Bad."
    assert examples[0].category == "helpfulness"
    assert examples[0].group == "g1"


def test_load_preference_examples_rejects_identical_answers(tmp_path):
    path = tmp_path / "prefs.jsonl"
    _write_jsonl(path, [
        {"user": "Pick one.", "chosen": "same", "rejected": "same"},
    ])

    try:
        load_preference_examples(path)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("expected identical preference answers to fail")


def test_preference_dataset_masks_prompt_tokens(tmp_path):
    path = tmp_path / "prefs.jsonl"
    _write_jsonl(path, [
        {"user": "Pick one.", "chosen": "Good.", "rejected": "Bad."},
    ])
    tokenizer = _tokenizer_for_preferences(path)
    dataset = PreferenceDataset(load_preference_examples(path), tokenizer, context_size=64)

    chosen_x, chosen_labels, rejected_x, rejected_labels = dataset[0]

    assert chosen_x.shape == chosen_labels.shape == rejected_x.shape == rejected_labels.shape
    assert int((chosen_labels != -100).sum().item()) > 0
    assert int((rejected_labels != -100).sum().item()) > 0
    first_chosen_target = int((chosen_labels != -100).nonzero()[0].item())
    first_rejected_target = int((rejected_labels != -100).nonzero()[0].item())
    assert torch.all(chosen_labels[:first_chosen_target] == -100)
    assert torch.all(rejected_labels[:first_rejected_target] == -100)


def test_dpo_batch_metrics_same_policy_and_reference_is_log_two(tmp_path):
    path = tmp_path / "prefs.jsonl"
    _write_jsonl(path, [
        {"user": "Pick one.", "chosen": "Good.", "rejected": "Bad."},
    ])
    tokenizer = _tokenizer_for_preferences(path)
    dataset = PreferenceDataset(load_preference_examples(path), tokenizer, context_size=64)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=64,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    batch = tuple(tensor.unsqueeze(0) for tensor in dataset[0])

    metrics = dpo_batch_metrics(model, model, batch, beta=0.1)

    assert math.isclose(float(metrics["loss"].item()), math.log(2), rel_tol=1e-5)
    assert float(metrics["accuracy"].item()) == 0.0


def test_train_dpo_smoke(tmp_path):
    input_path = tmp_path / "prefs.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "sft"
    out_dir = tmp_path / "dpo"
    rows = [
        {"user": "Pick tone.", "chosen": "Use a clear tone.", "rejected": "Whatever.", "group": "a"},
        {"user": "Pick safety.", "chosen": "Refuse unsafe help.", "rejected": "Give unsafe details.", "group": "b"},
        {"user": "Pick math.", "chosen": "2 + 2 = 4.", "rejected": "2 + 2 = 5.", "group": "c"},
        {"user": "Pick spelling.", "chosen": "The word cat has three letters.", "rejected": "The word cat has five letters.", "group": "d"},
    ]
    _write_jsonl(input_path, rows)
    tokenizer = _tokenizer_for_preferences(input_path)
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=96,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    report = train_dpo(DPOConfig(
        input_path=str(input_path),
        tokenizer_path=str(tokenizer_path),
        checkpoint_path=str(checkpoint_path),
        out_dir=str(out_dir),
        batch_size=2,
        max_steps=1,
        log_every=1,
        eval_batches=2,
    ))

    assert (out_dir / "checkpoint" / "model.pt").exists()
    assert (out_dir / "best_checkpoint" / "model.pt").exists()
    assert (out_dir / "dpo_report.json").exists()
    assert report["checkpoint"] == str(out_dir / "checkpoint")
    assert report["dataset"]["train_examples"] >= 1
    assert report["dataset"]["val_examples"] >= 1
