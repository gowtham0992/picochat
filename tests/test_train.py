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
    ))

    assert (out_dir / "checkpoint" / "model.pt").exists()
    assert (out_dir / "checkpoint" / "metadata.json").exists()
    assert (out_dir / "train_report.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "sample.txt").exists()
    assert report["model"]["num_parameters"] > 0
    assert "val_loss" in report["losses"][-1]
    assert report["loss_diagnostics"]["final_step"] == 2
    assert "Loss Diagnostics" in (out_dir / "report.md").read_text(encoding="utf-8")
    assert "memorization" in report
    assert report["memorization"]["status"] in {"low", "medium", "high"}
    assert report["dataset"]["val_sequences"] > 0


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
    ))

    assert report["dataset"]["split_mode"] == "document"
    assert report["dataset"]["val_documents"] == 1
    assert "Memorization Diagnostics" in (out_dir / "report.md").read_text(encoding="utf-8")
