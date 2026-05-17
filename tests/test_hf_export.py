import json
import importlib
import sys
import types

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
        initializer_range=0.015,
        qk_norm=True,
        attn_backend="math",
        parallel_residual=True,
        linear_bias=False,
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
    state = torch.load(out_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)

    assert report["out_dir"] == str(out_dir)
    assert report["dynamic_int8"] is True
    assert report["transformers_adapter"] is True
    assert config["model_type"] == "picochat"
    assert config["picochat_model_config"]["context_size"] == 8
    assert config["picochat_model_config"]["n_kv_head"] == 2
    assert config["picochat_model_config"]["qk_norm"] is True
    assert config["picochat_model_config"]["attn_backend"] == "math"
    assert config["picochat_model_config"]["parallel_residual"] is True
    assert config["picochat_model_config"]["linear_bias"] is False
    assert config["linear_bias"] is False
    assert config["initializer_range"] == 0.015
    assert config["picochat_model_config"]["initializer_range"] == 0.015
    assert "initializer_range=0.02" in (out_dir / "configuration_picochat.py").read_text(encoding="utf-8")
    assert "initializer_range=self.initializer_range" in (out_dir / "configuration_picochat.py").read_text(encoding="utf-8")
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
    assert serving_manifest["transformers"]["supports_padded_attention_mask"] is True
    assert serving_manifest["artifacts"]["dynamic_int8_weights"] == "pytorch_model.dynamic_int8.bin"
    assert "supports padded attention masks" in manifest["limitations"][2]
    assert "compacts padded rows" in serving_manifest["limitations"][2]
    assert "token_embedding.weight" in state
    assert (out_dir / "pytorch_model.dynamic_int8.bin").exists()
    assert (out_dir / "configuration_picochat.py").exists()
    assert (out_dir / "modeling_picochat.py").exists()
    assert (out_dir / "tokenization_picochat.py").exists()
    assert 'model_input_names = ["input_ids", "attention_mask"]' in (out_dir / "tokenization_picochat.py").read_text(encoding="utf-8")
    assert "transformers>=4.40" in (out_dir / "requirements.txt").read_text(encoding="utf-8")
    assert "Synthetic unit-test data." in (out_dir / "README.md").read_text(encoding="utf-8")
    assert "supports padded" in (out_dir / "README.md").read_text(encoding="utf-8")


def test_transformers_adapter_roundtrips_modern_config_without_transformers_package(tmp_path, monkeypatch):
    _install_minimal_transformers_stub(monkeypatch)

    checkpoint_path = tmp_path / "checkpoint"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "hf-modern"
    tokenizer = CharTokenizer.train(["hello picochat release adapter"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=8,
        n_embd=24,
        n_head=4,
        n_kv_head=2,
        n_layer=1,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="swiglu",
        tie_embeddings=True,
        qk_norm=True,
        attn_backend="math",
        parallel_residual=True,
    ))
    save_checkpoint(checkpoint_path, model, step=7, train_loss=0.5)
    export_hf_checkpoint(HFExportConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        model_name="picochat-modern",
    ))

    package_name = "_picochat_export_adapter_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(out_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    config_module = importlib.import_module(f"{package_name}.configuration_picochat")
    modeling_module = importlib.import_module(f"{package_name}.modeling_picochat")
    config_payload = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    adapter_config = config_module.PicochatConfig(**config_payload)
    adapter_model = modeling_module.PicochatForCausalLM(adapter_config)
    state = torch.load(out_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    adapter_model.load_state_dict(state)

    input_ids = torch.tensor([[tokenizer.bos_id, tokenizer.encode("hello")[0], tokenizer.eos_id]])
    model.eval()
    adapter_model.eval()
    with torch.no_grad():
        expected_logits, _ = model(input_ids)
        adapter_output = adapter_model(input_ids=input_ids, use_cache=False)

    torch.testing.assert_close(adapter_output.logits, expected_logits)


def test_transformers_adapter_uses_hf_shifted_labels_and_padded_masks(tmp_path, monkeypatch):
    _install_minimal_transformers_stub(monkeypatch)

    checkpoint_path = tmp_path / "checkpoint"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_dir = tmp_path / "hf-padding"
    tokenizer = CharTokenizer.train(["hello picochat adapter padding"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=8,
        n_embd=24,
        n_head=4,
        n_layer=1,
        attn_backend="math",
    ))
    save_checkpoint(checkpoint_path, model, step=11, train_loss=0.25)
    export_hf_checkpoint(HFExportConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        model_name="picochat-padding",
    ))

    package_name = "_picochat_export_padding_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(out_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    config_module = importlib.import_module(f"{package_name}.configuration_picochat")
    modeling_module = importlib.import_module(f"{package_name}.modeling_picochat")
    adapter_config = config_module.PicochatConfig(**json.loads((out_dir / "config.json").read_text(encoding="utf-8")))
    adapter_model = modeling_module.PicochatForCausalLM(adapter_config)
    adapter_model.load_state_dict(torch.load(out_dir / "pytorch_model.bin", map_location="cpu", weights_only=True))
    adapter_model.eval()
    model.eval()

    hello = tokenizer.encode("hello")[:3]
    pico = tokenizer.encode("pico")[:2]
    row0 = [tokenizer.bos_id, *hello, tokenizer.eos_id]
    row1_active = [tokenizer.bos_id, *pico, tokenizer.eos_id]
    max_len = max(len(row0), len(row1_active))
    row0 = row0 + [tokenizer.pad_id] * (max_len - len(row0))
    row1 = row1_active + [tokenizer.pad_id] * (max_len - len(row1_active))
    input_ids = torch.tensor([row0, row1])
    attention_mask = torch.tensor([
        [1] * len(hello + [tokenizer.bos_id, tokenizer.eos_id]),
        [1] * len(row1_active) + [0] * (max_len - len(row1_active)),
    ])

    with torch.no_grad():
        output = adapter_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=True,
        )
        expected0, loss0 = model(
            input_ids[:1, :len(row0)],
            targets=torch.tensor([[*row0[1:], -100]]),
        )
        row1_tensor = torch.tensor([row1_active])
        expected1, loss1 = model(
            row1_tensor,
            targets=torch.tensor([[*row1_active[1:], -100]]),
        )

    torch.testing.assert_close(output.logits[0], expected0[0])
    torch.testing.assert_close(output.logits[1, :len(row1_active)], expected1[0])
    assert torch.all(output.logits[1, len(row1_active):] == 0)
    expected_loss = ((loss0 * (len(row0) - 1)) + (loss1 * (len(row1_active) - 1))) / (
        (len(row0) - 1) + (len(row1_active) - 1)
    )
    torch.testing.assert_close(output.loss, expected_loss)
    assert output.past_key_values is None


def _install_minimal_transformers_stub(monkeypatch):
    transformers = types.ModuleType("transformers")
    modeling_outputs = types.ModuleType("transformers.modeling_outputs")

    class PretrainedConfig:
        model_type = "stub"

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class PreTrainedModel(torch.nn.Module):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__()
            self.config = config

    class CausalLMOutputWithPast:
        def __init__(self, loss=None, logits=None, past_key_values=None):
            self.loss = loss
            self.logits = logits
            self.past_key_values = past_key_values

    transformers.PretrainedConfig = PretrainedConfig
    transformers.PreTrainedModel = PreTrainedModel
    modeling_outputs.CausalLMOutputWithPast = CausalLMOutputWithPast
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_outputs", modeling_outputs)
