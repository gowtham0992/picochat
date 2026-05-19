import torch
import pytest

from picochat.lora import (
    LoRAConfig,
    apply_lora,
    load_lora_adapter,
    merged_lora_model,
    save_lora_adapter,
    subtract_lora_delta_from_base,
    trainable_parameter_report,
)
from picochat.model import GPTConfig, TinyGPT


def _tiny_model() -> TinyGPT:
    return TinyGPT(GPTConfig(
        vocab_size=32,
        context_size=16,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))


def _make_adapter_nonzero(model: TinyGPT) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_b"):
                parameter.fill_(0.01)


def test_apply_lora_freezes_base_and_reports_trainable_fraction():
    model = _tiny_model()

    report = apply_lora(model, LoRAConfig(rank=2, alpha=4.0, targets=("attn_qkv", "attn_proj")))
    parameter_report = trainable_parameter_report(model)

    assert report["mode"] == "lora"
    assert report["adapted_module_count"] == 2
    assert parameter_report["trainable_parameters"] == report["trainable_parameters"]
    assert 0 < parameter_report["trainable_fraction"] < 1
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".lora_" in name
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".base." in name
    )


def test_merged_lora_model_matches_adapter_forward():
    torch.manual_seed(7)
    model = _tiny_model()
    model.eval()
    apply_lora(model, LoRAConfig(rank=2, alpha=4.0, targets=("attn_qkv", "attn_proj")))
    _make_adapter_nonzero(model)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    with torch.no_grad():
        adapter_logits, _ = model(input_ids)
        with merged_lora_model(model):
            merged_logits, _ = model(input_ids)
        restored_logits, _ = model(input_ids)

    assert torch.allclose(adapter_logits, merged_logits, atol=1e-6)
    assert torch.allclose(adapter_logits, restored_logits, atol=1e-6)


def test_lora_adapter_roundtrip_from_merged_checkpoint(tmp_path):
    torch.manual_seed(11)
    config = LoRAConfig(rank=2, alpha=4.0, targets=("attn_qkv", "attn_proj"))
    model = _tiny_model()
    model.eval()
    apply_lora(model, config)
    _make_adapter_nonzero(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        expected_logits, _ = model(input_ids)
        with merged_lora_model(model):
            merged_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    adapter_dir = tmp_path / "adapter"
    save_lora_adapter(adapter_dir, model, config=config)

    resumed = _tiny_model()
    resumed.load_state_dict(merged_state)
    apply_lora(resumed, config)
    load_lora_adapter(adapter_dir, resumed)
    subtract_lora_delta_from_base(resumed)
    resumed.eval()

    with torch.no_grad():
        resumed_logits, _ = resumed(input_ids)

    assert torch.allclose(expected_logits, resumed_logits, atol=1e-6)


def test_load_lora_adapter_requires_lora_modules(tmp_path):
    model = _tiny_model()
    torch.save({}, tmp_path / "adapter_model.pt")

    with pytest.raises(ValueError, match="no LoRA adapters"):
        load_lora_adapter(tmp_path, model)
