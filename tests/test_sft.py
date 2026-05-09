import json

from picochat.checkpoint import save_checkpoint
from picochat.model import GPTConfig, TinyGPT
from picochat.sft import ChatExample, ChatSFTDataset, SFTConfig, load_chat_examples, train_sft
from picochat.tokenizer import CharTokenizer


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_load_chat_examples_from_jsonl(tmp_path):
    input_path = tmp_path / "chat.jsonl"
    write_jsonl(input_path, [{"user": "hi", "assistant": "hello"}])

    examples = load_chat_examples(input_path)

    assert examples == [ChatExample(user="hi", assistant="hello")]


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


def test_chat_sft_dataset_keeps_answer_tokens_when_prompt_is_too_long():
    tokenizer = CharTokenizer.train([
        "User: this prompt is far too long for a tiny context\nAssistant: ok"
    ])

    dataset = ChatSFTDataset(
        [ChatExample(user="this prompt is far too long for a tiny context", assistant="ok")],
        tokenizer=tokenizer,
        context_size=8,
    )

    assert len(dataset) == 1
    assert dataset.stats().truncated_examples == 1
    assert dataset.stats().supervised_tokens > 0


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
    ))

    assert (out_dir / "checkpoint" / "model.pt").exists()
    assert (out_dir / "checkpoint" / "metadata.json").exists()
    assert (out_dir / "sft_report.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "sample.txt").exists()
    assert report["dataset"]["num_examples"] == 2
    assert report["dataset"]["supervised_tokens"] > 0
    assert report["loss_diagnostics"]["final_step"] == 2
    assert "Loss Diagnostics" in (out_dir / "report.md").read_text(encoding="utf-8")
