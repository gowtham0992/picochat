import json

import torch

from picochat.checkpoint import save_checkpoint
from picochat.hf_export import HFExportConfig, export_hf_checkpoint
from picochat.model import GPTConfig, TinyGPT
from picochat.tokenizer import CharTokenizer


def test_export_hf_checkpoint_writes_release_folder(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "hf"
    tokenizer = CharTokenizer.train(["hello picochat"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=3, train_loss=1.23)

    report = export_hf_checkpoint(HFExportConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        model_name="picochat-test",
        dataset_summary="Synthetic unit-test data.",
        eval_summary="Unit-test export only.",
    ))
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "release_manifest.json").read_text(encoding="utf-8"))
    state = torch.load(out_dir / "pytorch_model.bin", map_location="cpu")

    assert report["out_dir"] == str(out_dir)
    assert config["model_type"] == "picochat"
    assert config["picochat_model_config"]["context_size"] == 8
    assert manifest["files"]["weights"] == "pytorch_model.bin"
    assert "token_embedding.weight" in state
    assert "Synthetic unit-test data." in (out_dir / "README.md").read_text(encoding="utf-8")
