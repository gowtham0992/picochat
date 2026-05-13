import torch
import pytest

from picochat.model import GPTConfig, TinyGPT
from picochat.optim import (
    ExponentialMovingAverage,
    create_optimizer,
    learning_rate_for_step,
    maybe_clip_grad_norm,
    set_optimizer_lr,
    using_ema_weights,
    validate_optim_controls,
)


def test_learning_rate_warmup_then_cosine_decay():
    first = learning_rate_for_step(
        base_learning_rate=1.0,
        step=1,
        max_steps=10,
        warmup_steps=2,
        decay="cosine",
        min_lr_ratio=0.1,
    )
    warm = learning_rate_for_step(
        base_learning_rate=1.0,
        step=2,
        max_steps=10,
        warmup_steps=2,
        decay="cosine",
        min_lr_ratio=0.1,
    )
    final = learning_rate_for_step(
        base_learning_rate=1.0,
        step=10,
        max_steps=10,
        warmup_steps=2,
        decay="cosine",
        min_lr_ratio=0.1,
    )

    assert first == 0.5
    assert warm == 1.0
    assert final == pytest.approx(0.1)


def test_validate_optim_controls_rejects_bad_decay():
    with pytest.raises(ValueError, match="lr_decay"):
        validate_optim_controls(
            max_steps=10,
            lr_warmup_steps=0,
            lr_decay="bad",
            min_lr_ratio=1.0,
            grad_clip=0.0,
        )


def test_maybe_clip_grad_norm_returns_norm():
    layer = torch.nn.Linear(2, 1)
    loss = layer(torch.ones(1, 2)).sum()
    loss.backward()

    norm = maybe_clip_grad_norm(layer, 1.0)

    assert norm is not None
    assert norm > 0.0


def test_muon_optimizer_updates_block_matrices_and_tracks_adamw_fallback():
    model = TinyGPT(GPTConfig(vocab_size=16, context_size=8, n_embd=16, n_head=4, n_layer=1))
    optimizer = create_optimizer(
        model,
        optimizer_type="muon",
        learning_rate=3e-4,
        muon_learning_rate=0.02,
    )

    assert optimizer.metadata["optimizer"] == "muon"
    assert optimizer.metadata["muon_parameters"] > 0
    assert optimizer.metadata["adamw_parameters"] > 0

    before = model.blocks[0].attn.qkv.weight.detach().clone()
    _, loss = model(
        torch.randint(0, 16, (2, 8)),
        torch.randint(0, 16, (2, 8)),
    )
    assert loss is not None
    loss.backward()
    set_optimizer_lr(optimizer, 1e-4)
    optimizer.step()

    after = model.blocks[0].attn.qkv.weight.detach()
    assert not torch.allclose(before, after)


def test_ema_weights_swap_and_restore_model_state():
    model = TinyGPT(GPTConfig(vocab_size=8, context_size=4, n_embd=8, n_head=2, n_layer=1))
    ema = ExponentialMovingAverage(model, decay=0.5)
    raw_before = model.token_embedding.weight.detach().clone()

    with torch.no_grad():
        model.token_embedding.weight.add_(2.0)
    raw_after = model.token_embedding.weight.detach().clone()
    ema.update(model)

    with using_ema_weights(model, ema):
        swapped = model.token_embedding.weight.detach().clone()
        assert torch.allclose(swapped, raw_before + 1.0)

    assert torch.allclose(model.token_embedding.weight.detach(), raw_after)
