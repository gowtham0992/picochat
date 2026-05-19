import json

from picochat.cli import main
from picochat.lr_finder import LRRangeConfig, run_lr_range
from picochat.tokenizer import train_tokenizer


def _write_tiny_lr_assets(tmp_path):
    corpus = tmp_path / "corpus.txt"
    text = (
        "Picochat trains tiny language models honestly.\n"
        "Learning rate probes should be cheap before large runs.\n"
    ) * 80
    corpus.write_text(text, encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer("char", [text], vocab_size=128)
    tokenizer.save(tokenizer_path)
    return corpus, tokenizer_path


def test_lr_range_writes_report(tmp_path):
    corpus, tokenizer_path = _write_tiny_lr_assets(tmp_path)
    out_dir = tmp_path / "lr-range"

    report = run_lr_range(LRRangeConfig(
        corpus_path=str(corpus),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        context_size=16,
        batch_size=2,
        grad_accum_steps=1,
        steps=4,
        min_lr=1e-5,
        max_lr=5e-5,
        smoothing_beta=0.0,
        n_embd=24,
        n_head=2,
        n_layer=1,
        log_every=1,
    ))

    assert report["summary"]["steps_run"] == 4
    assert report["summary"]["recommended_lr"] is not None
    saved = json.loads((out_dir / "lr_range.json").read_text(encoding="utf-8"))
    assert saved["summary"]["steps_run"] == 4
    assert "Recommended LR" in (out_dir / "lr_range.md").read_text(encoding="utf-8")


def test_cli_train_lr_range(tmp_path, capsys):
    corpus, tokenizer_path = _write_tiny_lr_assets(tmp_path)
    out_dir = tmp_path / "cli-lr-range"

    code = main([
        "train",
        "lr-range",
        "--corpus",
        str(corpus),
        "--tokenizer",
        str(tokenizer_path),
        "--out-dir",
        str(out_dir),
        "--context-size",
        "16",
        "--batch-size",
        "2",
        "--steps",
        "3",
        "--min-lr",
        "0.00001",
        "--max-lr",
        "0.00005",
        "--smoothing-beta",
        "0",
        "--n-embd",
        "24",
        "--n-head",
        "2",
        "--n-layer",
        "1",
    ])

    assert code == 0
    output = capsys.readouterr().out
    assert "lr range recommended_lr" in output
    assert (out_dir / "lr_range.json").exists()
