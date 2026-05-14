import pytest
import torch

from picochat.model import GPTConfig, TinyGPT


def test_model_forward_shapes_and_loss():
    config = GPTConfig(vocab_size=20, context_size=8, n_embd=16, n_head=4, n_layer=2)
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (3, config.context_size))
    y = torch.randint(0, config.vocab_size, (3, config.context_size))

    logits, loss = model(x, y)

    assert logits.shape == (3, config.context_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


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
