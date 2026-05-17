from contextlib import contextmanager

import pytest
import torch

import picochat.model as model_module
from picochat.model import GPTConfig, RMSNorm, TinyGPT, sdpa_backend_context


def test_model_forward_shapes_and_loss():
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=2)
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (3, config.context_size))
    y = torch.randint(0, config.vocab_size, (3, config.context_size))

    logits, loss = model(x, y)

    assert logits.shape == (3, config.context_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_model_loss_ignores_masked_sft_targets():
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))
    y = torch.full((2, config.context_size), -100, dtype=torch.long)
    y[:, -1] = torch.randint(0, config.vocab_size, (2,))

    _, loss = model(x, y)

    assert loss is not None
    assert torch.isfinite(loss)


def test_model_rejects_too_long_sequence():
    config = GPTConfig(vocab_size=20, context_size=4, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (1, 5))

    with pytest.raises(ValueError):
        model(x)


def test_generate_adds_tokens():
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out = model.generate(prompt, max_new_tokens=5, temperature=0)

    assert out.shape == (1, 8)
    assert out[:, :3].tolist() == prompt.tolist()


def test_generate_with_cache_matches_uncached_greedy():
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=2)
    model = TinyGPT(config)
    model.eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    cached = model.generate(prompt, max_new_tokens=4, temperature=0, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=4, temperature=0, use_cache=False)

    assert cached.tolist() == uncached.tolist()


def test_forward_returns_kv_cache():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=2,
        position_encoding="rope",
    )
    model = TinyGPT(config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    logits, loss, past_kv = model(prompt, use_cache=True)
    next_logits, next_loss, next_kv = model(
        torch.tensor([[4]], dtype=torch.long),
        past_kv=past_kv,
        use_cache=True,
    )

    assert logits.shape == (1, 3, config.vocab_size)
    assert loss is None
    assert len(past_kv) == config.n_layer
    assert past_kv[0][0].shape == (1, config.n_head, 3, config.n_embd // config.n_head)
    assert next_logits.shape == (1, 1, config.vocab_size)
    assert next_loss is None
    assert next_kv[0][0].shape[-2] == 4


def test_generate_validates_sampling_controls():
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="top_p"):
        model.generate(prompt, max_new_tokens=1, top_p=0)

    with pytest.raises(ValueError, match="repetition_penalty"):
        model.generate(prompt, max_new_tokens=1, repetition_penalty=0)


def test_generate_stops_when_eos_is_generated():
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    eos_id = 2
    prompt = torch.tensor([[1, 3]], dtype=torch.long)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lm_head.bias[eos_id] = 10.0

    out = model.generate(prompt, max_new_tokens=5, temperature=0, eos_id=eos_id)

    assert out.shape == (1, 3)
    assert out[0, -1].item() == eos_id


def test_invalid_head_count_rejected():
    with pytest.raises(ValueError):
        TinyGPT(GPTConfig(vocab_size=20, context_size=8, n_embd=18, n_head=4, n_layer=1))


def test_invalid_grouped_query_attention_shape_raises():
    with pytest.raises(ValueError, match="n_head must be divisible by n_kv_head"):
        TinyGPT(GPTConfig(
            vocab_size=20,
            context_size=8,
            n_embd=16,
            n_head=4,
            n_kv_head=3,
            n_layer=1,
        ))


def test_model_supports_rmsnorm_and_rope():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="relu2",
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, loss = model(x, x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert loss is not None
    assert model.position_embedding is None


def test_model_supports_swiglu_activation():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        activation="swiglu",
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, loss = model(x, x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert loss is not None
    assert model.blocks[0].mlp.fc.out_features == 2 * int(8 * config.n_embd / 3)


def test_model_can_tie_input_and_output_embeddings():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        tie_embeddings=True,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, loss = model(x, x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert loss is not None
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.lm_head.bias is None


def test_model_can_disable_transformer_linear_biases():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        linear_bias=False,
    )
    model = TinyGPT(config)

    block = model.blocks[0]
    assert block.attn.qkv.bias is None
    assert block.attn.proj.bias is None
    assert block.mlp.fc.bias is None
    assert block.mlp.proj.bias is None
    assert model.lm_head.bias is None


def test_tied_embedding_initialization_keeps_starting_loss_sane():
    config = GPTConfig(
        vocab_size=2048,
        context_size=16,
        n_embd=128,
        n_head=8,
        n_kv_head=2,
        n_layer=2,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="swiglu",
        tie_embeddings=True,
        qk_norm=True,
        parallel_residual=True,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))
    y = torch.randint(0, config.vocab_size, (2, config.context_size))

    _, loss = model(x, y)

    assert loss is not None
    assert 6.0 < float(loss.item()) < 9.5
    assert model.token_embedding.weight.std().item() < 0.04
    assert model.lm_head.weight is model.token_embedding.weight


def test_model_supports_qk_norm():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        qk_norm=True,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, loss = model(x, x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert loss is not None
    assert isinstance(model.blocks[0].attn.q_norm, RMSNorm)
    assert isinstance(model.blocks[0].attn.k_norm, RMSNorm)


def test_qk_norm_preserves_attention_dtype_under_autocast(monkeypatch):
    observed = {}

    def fake_sdpa(q, k, v, **kwargs):
        observed["q_dtype"] = q.dtype
        observed["k_dtype"] = k.dtype
        observed["v_dtype"] = v.dtype
        return torch.zeros_like(q)

    monkeypatch.setattr(model_module.F, "scaled_dot_product_attention", fake_sdpa)
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        norm_type="rmsnorm",
        position_encoding="rope",
        qk_norm=True,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, loss = model(x, x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert loss is not None
    assert observed == {
        "q_dtype": torch.bfloat16,
        "k_dtype": torch.bfloat16,
        "v_dtype": torch.bfloat16,
    }


def test_model_supports_grouped_query_attention_cache():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_kv_head=1,
        n_layer=1,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 3))

    logits, loss, past_kv = model(x, use_cache=True)

    assert logits.shape == (2, 3, config.vocab_size)
    assert loss is None
    assert model.blocks[0].attn.qkv.out_features == config.n_embd + 2 * (
        config.n_embd // config.n_head
    )
    assert past_kv[0][0].shape == (2, 1, 3, config.n_embd // config.n_head)


def test_grouped_query_attention_uses_native_sdpa_gqa_when_available(monkeypatch):
    if not model_module._SDPA_SUPPORTS_ENABLE_GQA:
        pytest.skip("PyTorch build does not expose native SDPA GQA")
    calls = []

    def fake_sdpa(q, k, v, **kwargs):
        calls.append({
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "enable_gqa": kwargs.get("enable_gqa"),
        })
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_kv_head=1,
        n_layer=1,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, _ = model(x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert calls == [{
        "q_shape": (2, 4, 8, 4),
        "k_shape": (2, 1, 8, 4),
        "enable_gqa": True,
    }]


def test_grouped_query_attention_repeats_kv_when_native_gqa_is_unavailable(monkeypatch):
    calls = []

    def fake_sdpa(q, k, v, **kwargs):
        calls.append({
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "enable_gqa": kwargs.get("enable_gqa"),
        })
        return torch.zeros_like(q)

    monkeypatch.setattr(model_module, "_SDPA_SUPPORTS_ENABLE_GQA", False)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_kv_head=1,
        n_layer=1,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, _ = model(x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert calls == [{
        "q_shape": (2, 4, 8, 4),
        "k_shape": (2, 4, 8, 4),
        "enable_gqa": None,
    }]


def test_model_supports_parallel_residual_blocks():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        parallel_residual=True,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, loss = model(x, x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert loss is not None
    assert model.blocks[0].parallel_residual is True
    assert model.blocks[0].ln_2 is None


def test_model_can_softcap_logits():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        logit_softcap=2.0,
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))
    with torch.no_grad():
        model.lm_head.weight.fill_(10.0)
        model.lm_head.bias.fill_(10.0)

    logits, _ = model(x)

    assert float(logits.detach().abs().max()) <= 2.0001


def test_attention_uses_scaled_dot_product_attention(monkeypatch):
    calls = []

    def fake_sdpa(q, k, v, dropout_p=0.0, is_causal=False):
        calls.append({
            "shape": tuple(q.shape),
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        })
        return torch.zeros_like(v)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, _ = model(x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert calls == [{
        "shape": (2, 4, 8, 4),
        "dropout_p": 0.0,
        "is_causal": True,
    }]


def test_attention_backend_can_force_math_sdpa(monkeypatch):
    calls = []

    @contextmanager
    def fake_kernel(backend):
        calls.append(backend)
        yield

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", fake_kernel)
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        attn_backend="math",
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, _ = model(x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert calls == [torch.nn.attention.SDPBackend.MATH]


def test_attention_backend_can_use_optional_external_flash(monkeypatch):
    calls = []

    def fake_flash(q, k, v, dropout_p=0.0, causal=True):
        calls.append({
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "dropout_p": dropout_p,
            "causal": causal,
        })
        return torch.zeros_like(q)

    def fail_sdpa(*args, **kwargs):
        raise AssertionError("external_flash should bypass PyTorch SDPA")

    monkeypatch.setattr(model_module, "_external_flash_attn_func", lambda: fake_flash)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail_sdpa)
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        attn_backend="external_flash",
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    logits, _ = model(x)

    assert logits.shape == (2, config.context_size, config.vocab_size)
    assert calls == [{
        "q_shape": (2, 8, 4, 4),
        "k_shape": (2, 8, 4, 4),
        "dropout_p": 0.0,
        "causal": True,
    }]


def test_attention_backend_external_flash_requires_optional_package(monkeypatch):
    monkeypatch.setattr(model_module, "_external_flash_attn_func", lambda: None)
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        attn_backend="external_flash",
    )
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_size))

    with pytest.raises(RuntimeError, match="optional flash-attn package"):
        model(x)


def test_attention_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="attn_backend"):
        TinyGPT(GPTConfig(vocab_size=20, context_size=8, attn_backend="made_up"))


def test_sdpa_backend_context_auto_is_noop():
    with sdpa_backend_context("auto"):
        value = 1

    assert value == 1


def test_model_supports_gradient_checkpointing():
    config = GPTConfig(
        vocab_size=20,
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=2,
        gradient_checkpointing=True,
    )
    model = TinyGPT(config)
    model.train()
    x = torch.randint(0, config.vocab_size, (2, config.context_size))
    y = torch.randint(0, config.vocab_size, (2, config.context_size))

    _, loss = model(x, y)
    assert loss is not None
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())


def test_model_rejects_negative_logit_softcap():
    with pytest.raises(ValueError, match="logit_softcap"):
        TinyGPT(GPTConfig(vocab_size=20, context_size=8, logit_softcap=-1.0))


def test_rope_requires_even_head_dimension():
    with pytest.raises(ValueError, match="RoPE"):
        TinyGPT(GPTConfig(
            vocab_size=20,
            context_size=8,
            n_embd=15,
            n_head=3,
            n_layer=1,
            position_encoding="rope",
        ))
