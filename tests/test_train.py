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
    assert report["dataset"]["val_sequences"] > 0
