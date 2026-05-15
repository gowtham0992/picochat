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
        n_kv_head=2,
        n_layer=1,
        qk_norm=True,
    ))
    save_checkpoint(checkpoint_path, model, step=3, train_loss=1.23)

    report = export_hf_checkpoint(HFExportConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        model_name="picochat-test",
        dataset_summary="Synthetic unit-test data.",
        eval_summary="Unit-test export only.",
        dynamic_int8=True,
    ))
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "release_manifest.json").read_text(encoding="utf-8"))
    serving_manifest = json.loads((out_dir / "serving_manifest.json").read_text(encoding="utf-8"))
    state = torch.load(out_dir / "pytorch_model.bin", map_location="cpu")

    assert report["out_dir"] == str(out_dir)
    assert report["dynamic_int8"] is True
    assert report["transformers_adapter"] is True
    assert config["model_type"] == "picochat"
    assert config["picochat_model_config"]["context_size"] == 8
    assert config["picochat_model_config"]["n_kv_head"] == 2
    assert config["picochat_model_config"]["qk_norm"] is True
    assert config["auto_map"]["AutoModelForCausalLM"] == "modeling_picochat.PicochatForCausalLM"
    assert manifest["files"]["weights"] == "pytorch_model.bin"
    assert manifest["files"]["dynamic_int8"] == "pytorch_model.dynamic_int8.bin"
    assert manifest["files"]["transformers_model_adapter"] == "modeling_picochat.py"
    assert manifest["transformers_adapter"] is True
    if report["safetensors"]:
        assert manifest["files"]["weights_safetensors"] == "model.safetensors"
        assert (out_dir / "model.safetensors").exists()
    else:
        assert report["safetensors_error"]
    assert serving_manifest["supports_kv_cache"] is True
    assert serving_manifest["transformers"]["requires_trust_remote_code"] is True
    assert serving_manifest["artifacts"]["dynamic_int8_weights"] == "pytorch_model.dynamic_int8.bin"
    assert "token_embedding.weight" in state
    assert (out_dir / "pytorch_model.dynamic_int8.bin").exists()
    assert (out_dir / "configuration_picochat.py").exists()
    assert (out_dir / "modeling_picochat.py").exists()
    assert (out_dir / "tokenization_picochat.py").exists()
    assert "transformers>=4.40" in (out_dir / "requirements.txt").read_text(encoding="utf-8")
    assert "Synthetic unit-test data." in (out_dir / "README.md").read_text(encoding="utf-8")
