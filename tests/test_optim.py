import pytest
import torch

from picochat.optim import learning_rate_for_step, maybe_clip_grad_norm, validate_optim_controls


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
