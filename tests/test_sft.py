import json

from picochat.checkpoint import save_checkpoint
from picochat.model import GPTConfig, TinyGPT
from picochat.sft import (
    ChatExample,
    ChatSFTDataset,
    PackedChatSFTDataset,
    SFTConfig,
    category_balanced_weights,
    category_counts,
    category_sqrt_weights,
    load_chat_examples,
    split_chat_dataset,
    train_sft,
)
from picochat.tokenizer import CharTokenizer


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_load_chat_examples_from_jsonl(tmp_path):
    input_path = tmp_path / "chat.jsonl"
    write_jsonl(input_path, [{"user": "hi", "assistant": "hello", "category": "greet", "group": "greet-basic"}])

    examples = load_chat_examples(input_path)

    assert examples == [ChatExample(user="hi", assistant="hello", category="greet", group="greet-basic")]


def test_chat_sft_dataset_masks_prompt_tokens():
    tokenizer = CharTokenizer.train(["User: hi\nAssistant: hello"])
    dataset = ChatSFTDataset(
        [ChatExample(user="hi", assistant="hello")],
        tokenizer=tokenizer,
        context_size=32,
    )

    x, y = dataset[0]
    labels = y.tolist()

    assert len(x) == 32
    assert len(labels) == 32
    assert -100 in labels
    assert any(token_id != -100 for token_id in labels)
    assert dataset.stats().supervised_tokens > 0
    assert dataset.stats().masked_prompt_tokens > 0

    unmasked_labels = [token_id for token_id in labels if token_id != -100]
    assert unmasked_labels[0] == tokenizer.encode("hello", add_eos=True)[0]


def test_chat_sft_dataset_skips_rows_that_do_not_fit_context():
    tokenizer = CharTokenizer.train([
        "User: this prompt is far too long for a tiny context\nAssistant: ok"
        "User: hi\nAssistant: ok"
    ])

    dataset = ChatSFTDataset(
        [
            ChatExample(user="this prompt is far too long for a tiny context", assistant="ok"),
            ChatExample(user="hi", assistant="ok"),
        ],
        tokenizer=tokenizer,
        context_size=24,
    )

    assert len(dataset) == 1
    assert dataset.stats().truncated_examples == 0
    assert dataset.stats().skipped_long_examples == 1
    assert dataset.stats().supervised_tokens > 0


def test_packed_chat_sft_dataset_bestfit_packs_examples():
    tokenizer = CharTokenizer.train([
        "User: a\nAssistant: x\n"
        "User: b\nAssistant: y\n"
        "User: c\nAssistant: z\n"
        "User: d\nAssistant: w\n"
    ])
    source = ChatSFTDataset(
        [
            ChatExample(user="a", assistant="x", category="short"),
            ChatExample(user="b", assistant="y", category="short"),
            ChatExample(user="c", assistant="z", category="short"),
            ChatExample(user="d", assistant="w", category="short"),
        ],
        tokenizer=tokenizer,
        context_size=48,
    )
    packed = PackedChatSFTDataset(
        [source.tokenized_example(index) for index in range(len(source))],
        tokenizer=tokenizer,
        context_size=48,
    )

    assert len(packed) < len(source)
    assert packed.stats().packing == "bos_bestfit"
    assert packed.stats().source_examples == 4
    assert packed.stats().packed_sequences == len(packed)
    assert packed.stats().padded_tokens < source.stats().padded_tokens
    assert packed.stats().packing_efficiency > source.stats().packing_efficiency

    _, labels = packed[0]
    assert any(token_id != -100 for token_id in labels.tolist())
    assert packed.stats().supervised_tokens == source.stats().supervised_tokens


def test_split_chat_dataset_keeps_groups_out_of_both_sides():
    tokenizer = CharTokenizer.train([
        "User: a\nAssistant: one\n"
        "User: b\nAssistant: two\n"
        "User: c\nAssistant: three\n"
        "User: d\nAssistant: four\n"
    ])
    dataset = ChatSFTDataset(
        [
            ChatExample(user="a", assistant="one", group="alpha"),
            ChatExample(user="b", assistant="two", group="alpha"),
            ChatExample(user="c", assistant="three", group="beta"),
            ChatExample(user="d", assistant="four", group="gamma"),
        ],
        tokenizer=tokenizer,
        context_size=32,
    )

    split = split_chat_dataset(dataset, val_fraction=0.25, seed=1)
    train_groups = {dataset.group_key(index) for index in split.train.indices}
    val_groups = {dataset.group_key(index) for index in split.val.indices}

    assert split.method == "group"
    assert train_groups.isdisjoint(val_groups)
    assert split.num_groups == 3


def test_category_balanced_weights_boost_rare_categories():
    tokenizer = CharTokenizer.train([
        "User: a\nAssistant: one\n"
        "User: b\nAssistant: two\n"
        "User: c\nAssistant: three\n"
    ])
    dataset = ChatSFTDataset(
        [
            ChatExample(user="a", assistant="one", category="story"),
            ChatExample(user="b", assistant="two", category="story"),
            ChatExample(user="c", assistant="three", category="refusal"),
        ],
        tokenizer=tokenizer,
        context_size=32,
    )

    weights = category_balanced_weights(dataset).tolist()

    assert category_counts(dataset) == {"refusal": 1, "story": 2}
    assert weights[2] == weights[0] * 2
    assert dataset.stats().category_counts == {"refusal": 1, "story": 2}


def test_category_sqrt_weights_softly_boost_rare_categories():
    tokenizer = CharTokenizer.train([
        "User: a\nAssistant: one\n"
        "User: b\nAssistant: two\n"
        "User: c\nAssistant: three\n"
    ])
    dataset = ChatSFTDataset(
        [
            ChatExample(user="a", assistant="one", category="story"),
            ChatExample(user="b", assistant="two", category="story"),
            ChatExample(user="c", assistant="three", category="refusal"),
        ],
        tokenizer=tokenizer,
        context_size=32,
    )

    balanced = category_balanced_weights(dataset).tolist()
    sqrt_weights = category_sqrt_weights(dataset).tolist()

    assert sqrt_weights[2] > sqrt_weights[0]
    assert sqrt_weights[2] / sqrt_weights[0] < balanced[2] / balanced[0]


def test_train_sft_writes_artifacts(tmp_path):
    input_path = tmp_path / "chat.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "base"
    out_dir = tmp_path / "sft"
    rows = [
        {"user": "What is Picochat?", "assistant": "Picochat is small."},
        {"user": "What comes next?", "assistant": "Chat tuning comes next."},
    ]
    write_jsonl(input_path, rows)
    tokenizer = CharTokenizer.train([
        "User: What is Picochat?\nAssistant: Picochat is small.\n"
        "User: What comes next?\nAssistant: Chat tuning comes next."
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

    report = train_sft(SFTConfig(
        input_path=str(input_path),
        tokenizer_path=str(tokenizer_path),
        checkpoint_path=str(checkpoint_path),
        out_dir=str(out_dir),
        batch_size=2,
        max_steps=2,
        log_every=1,
        eval_batches=1,
        sample_tokens=8,
        lr_warmup_steps=1,
        lr_decay="linear",
        min_lr_ratio=0.5,
        grad_clip=1.0,
        sampling="category_balanced",
        packing="bos_bestfit",
    ))

    assert (out_dir / "checkpoint" / "model.pt").exists()
    assert (out_dir / "checkpoint" / "metadata.json").exists()
    assert (out_dir / "best_checkpoint" / "model.pt").exists()
    assert (out_dir / "sft_report.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "sample.txt").exists()
    assert report["dataset"]["num_examples"] == 2
    assert report["dataset"]["supervised_tokens"] > 0
    assert report["best_checkpoint"]["path"] == str(out_dir / "best_checkpoint")
    assert "val_bpb" in report["losses"][-1]
    assert "learning_rate" in report["losses"][-1]
    assert "grad_norm" in report["losses"][-1]
    assert report["coverage"]["actual_steps"] == 2
    assert report["stop_reason"] == "max_steps"
    assert report["loss_diagnostics"]["final_step"] == 2
    assert report["dataset"]["sampling"] == "category_balanced"
    assert report["dataset"]["packing"] == "bos_bestfit"
    assert report["dataset"]["source_examples"] == 2
    assert report["dataset"]["packed_sequences"] <= 2
    assert report["dataset"]["padded_tokens"] >= 0
    assert report["dataset"]["category_counts"] == {"chat": 2}
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Loss Diagnostics" in report_text
    assert "Best validation checkpoint" in report_text
    assert "SFT sampling" in report_text
    assert "Packing" in report_text


def test_train_sft_reports_gradient_accumulation(tmp_path):
    input_path = tmp_path / "chat.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "base"
    out_dir = tmp_path / "sft"
    rows = [
        {"user": "one", "assistant": "first"},
        {"user": "two", "assistant": "second"},
        {"user": "three", "assistant": "third"},
    ]
    write_jsonl(input_path, rows)
    tokenizer = CharTokenizer.train(["User: one\nAssistant: first\nUser: two\nAssistant: second\nUser: three\nAssistant: third"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=48,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    report = train_sft(SFTConfig(
        input_path=str(input_path),
        tokenizer_path=str(tokenizer_path),
        checkpoint_path=str(checkpoint_path),
        out_dir=str(out_dir),
        batch_size=1,
        grad_accum_steps=2,
        max_steps=2,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
    ))

    assert report["config"]["grad_accum_steps"] == 2
    assert report["config"]["effective_batch_size"] == 2
    assert report["coverage"]["examples_per_step_estimate"] == 2
    assert report["coverage"]["actual_example_updates"] == 4
    assert report["losses"][-1]["effective_batch_size"] == 2
