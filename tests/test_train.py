from picochat.tokenizer import CharTokenizer
from picochat.train import TrainConfig, train_base


def test_train_base_writes_artifacts(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "run"
    text = "hello picochat\n" * 20
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=8,
        batch_size=4,
        max_steps=2,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        val_fraction=0.2,
        eval_batches=1,
        sample_tokens=8,
        lr_warmup_steps=1,
        lr_decay="cosine",
        min_lr_ratio=0.5,
        grad_clip=1.0,
        weight_decay=0.02,
        weight_decay_decay="cosine_to_zero",
    ))

    assert (out_dir / "checkpoint" / "model.pt").exists()
    assert (out_dir / "best_checkpoint" / "model.pt").exists()
    assert (out_dir / "checkpoint" / "metadata.json").exists()
    assert (out_dir / "train_report.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "sample.txt").exists()
    assert report["model"]["num_parameters"] > 0
    assert "val_loss" in report["losses"][-1]
    assert "val_bpb" in report["losses"][-1]
    assert "learning_rate" in report["losses"][-1]
    assert "weight_decay" in report["losses"][-1]
    assert "grad_norm" in report["losses"][-1]
    assert report["config"]["weight_decay"] == 0.02
    assert report["config"]["weight_decay_decay"] == "cosine_to_zero"
    assert report["best_checkpoint"]["path"] == str(out_dir / "best_checkpoint")
    assert report["coverage"]["actual_steps"] == 2
    assert report["stop_reason"] == "max_steps"
    assert report["loss_diagnostics"]["final_step"] == 2
    assert "Loss Diagnostics" in (out_dir / "report.md").read_text(encoding="utf-8")
    assert "memorization" in report
    assert report["memorization"]["status"] in {"low", "medium", "high"}
    assert report["dataset"]["val_sequences"] > 0


def test_train_base_reports_gradient_accumulation(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "run"
    text = "accumulation teaches stable updates\n" * 20
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=8,
        batch_size=2,
        grad_accum_steps=3,
        max_steps=2,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
    ))

    assert report["config"]["grad_accum_steps"] == 3
    assert report["config"]["effective_batch_size"] == 6
    assert report["coverage"]["tokens_per_step_estimate"] == 48
    assert report["coverage"]["actual_training_tokens"] == 96
    assert report["losses"][-1]["effective_batch_size"] == 6


def test_train_base_records_precision_runtime(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "run"
    text = "precision runtime metadata\n" * 20
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=8,
        batch_size=4,
        max_steps=1,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
        precision="bf16",
    ))

    assert report["config"]["precision"] == "bf16"
    assert report["config"]["precision_runtime"]["requested"] == "bf16"
    assert report["config"]["precision_runtime"]["dtype_name"] == "bfloat16"
    assert report["config"]["torch_compile_metadata"]["enabled"] is False


def test_train_base_can_resume_from_training_state(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    first_dir = tmp_path / "first"
    resumed_dir = tmp_path / "resumed"
    text = "resume checkpoint keeps optimizer state\n" * 30
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    first = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(first_dir),
        context_size=8,
        batch_size=4,
        max_steps=1,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
        ema_decay=0.5,
    ))
    resumed = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(resumed_dir),
        context_size=8,
        batch_size=4,
        max_steps=3,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
        ema_decay=0.5,
        resume_from=first["resume_checkpoint"],
    ))

    assert (first_dir / "resume_checkpoint" / "training_state.pt").exists()
    assert (resumed_dir / "checkpoint" / "training_state.pt").exists()
    assert resumed["coverage"]["actual_steps"] == 3
    assert [row["step"] for row in resumed["losses"]] == [1, 2, 3]
    assert resumed["config"]["resume_from"] == first["resume_checkpoint"]


def test_train_base_can_use_sharded_dataset_mode(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "run"
    text = "sharded data path avoids one giant token tensor\n" * 40
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=8,
        batch_size=4,
        max_steps=1,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
        dataset_mode="sharded",
        shard_token_size=64,
    ))

    assert report["dataset"]["split_mode"] == "sharded"
    assert report["dataset"]["num_shards"] > 1
    assert (out_dir / "token_shards" / "token_shards_manifest.json").exists()
    assert report["coverage"]["actual_steps"] == 1


def test_train_base_records_gradient_checkpointing_and_ddp_metadata(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "run"
    text = "checkpoint activations during training\n" * 20
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=8,
        batch_size=4,
        max_steps=1,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
        gradient_checkpointing=True,
    ))

    assert report["model"]["config"]["gradient_checkpointing"] is True
    assert report["config"]["ddp_metadata"]["enabled"] is False


def test_train_base_uses_document_split_when_manifest_is_available(tmp_path):
    import json

    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "corpus_manifest.json"
    out_dir = tmp_path / "run"
    docs = [
        "alpha learning document " * 20,
        "beta validation document " * 20,
        "gamma training document " * 20,
    ]
    corpus = "\n\n".join(doc.strip() for doc in docs)
    corpus_path.write_text(f"{corpus}\n", encoding="utf-8")
    CharTokenizer.train([corpus]).save(tokenizer_path)
    offset = 0
    manifest_docs = []
    for index, doc in enumerate(doc.strip() for doc in docs):
        manifest_docs.append({
            "document_id": index,
            "path": f"doc-{index}.txt",
            "char_start": offset,
            "char_end": offset + len(doc),
        })
        offset += len(doc) + 2
    manifest_path.write_text(json.dumps({"documents": manifest_docs}), encoding="utf-8")

    report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=16,
        batch_size=4,
        max_steps=1,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        val_fraction=0.34,
        eval_batches=1,
        sample_tokens=8,
        split_mode="document",
        corpus_manifest_path=str(manifest_path),
        canary_count=1,
    ))

    assert report["dataset"]["split_mode"] == "document"
    assert report["dataset"]["val_documents"] == 1
    assert report["dataset"]["canaries_enabled"] is True
    assert report["dataset"]["canary_values"] == ["pico-canary-0042-00"]
    assert report["memorization"]["canary_values_in_train"] == ["pico-canary-0042-00"]
    assert "Memorization Diagnostics" in (out_dir / "report.md").read_text(encoding="utf-8")


def test_train_base_can_stop_early(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "run"
    text = "early stop picochat\n" * 30
    corpus_path.write_text(text, encoding="utf-8")
    CharTokenizer.train([text]).save(tokenizer_path)

    report = train_base(TrainConfig(
        corpus_path=str(corpus_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=8,
        batch_size=4,
        max_steps=5,
        n_embd=16,
        n_head=4,
        n_layer=1,
        log_every=1,
        eval_batches=1,
        sample_tokens=4,
        early_stop_patience=1,
        early_stop_min_delta=999.0,
    ))

    assert report["stop_reason"] == "early_stop"
    assert report["coverage"]["actual_steps"] < 5
