"""Small optimizer helpers for transparent local training runs."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch


LR_DECAYS = ("none", "linear", "cosine")
OPTIMIZER_TYPES = ("adamw", "muon")


def zeropower_via_newtonschulz5(
    gradient: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the zeroth power used by Muon for 2D matrix gradients."""
    if gradient.ndim != 2:
        raise ValueError("Muon Newton-Schulz update expects a 2D tensor")

    x = gradient.float()
    norm = x.norm()
    if float(norm.item()) == 0.0:
        return torch.zeros_like(gradient)
    x = x / (norm + eps)

    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.T
    return x.to(dtype=gradient.dtype)


class Muon(torch.optim.Optimizer):
    """Small single-process Muon optimizer for transformer matrix weights."""

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        if lr <= 0:
            raise ValueError("Muon learning rate must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Muon momentum must be in [0, 1)")
        if ns_steps < 1:
            raise ValueError("Muon Newton-Schulz steps must be at least 1")
        if weight_decay < 0:
            raise ValueError("Muon weight decay must be non-negative")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.ndim != 2:
                    raise ValueError("Muon only supports 2D matrix parameters")

                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient)
                update = gradient.add(buffer, alpha=momentum) if nesterov else buffer
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                update = update * max(1.0, update.size(0) / update.size(1)) ** 0.5

                if weight_decay > 0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr)
        return loss


class OptimizerBundle:
    """Adapter that lets training loops treat multiple optimizers as one."""

    def __init__(self, optimizers: list[torch.optim.Optimizer], metadata: dict):
        self.optimizers = optimizers
        self.metadata = metadata

    @property
    def param_groups(self) -> list[dict]:
        groups: list[dict] = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


def create_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_type: str = "adamw",
    learning_rate: float,
    muon_learning_rate: float = 0.02,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
) -> OptimizerBundle:
    """Create AdamW or a Muon+AdamW hybrid optimizer for Picochat models."""
    if optimizer_type not in OPTIMIZER_TYPES:
        raise ValueError(f"optimizer must be one of: {', '.join(OPTIMIZER_TYPES)}")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if muon_learning_rate <= 0:
        raise ValueError("muon_learning_rate must be positive")

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for group in optimizer.param_groups:
            group["lr_scale"] = 1.0
        return OptimizerBundle([optimizer], {
            "optimizer": "adamw",
            "adamw_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "muon_parameters": 0,
            "muon_learning_rate": None,
        })

    matrix_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    matrix_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2 and name.startswith("blocks."):
            matrix_params.append(parameter)
            matrix_names.append(name)
        else:
            adamw_params.append(parameter)

    optimizers: list[torch.optim.Optimizer] = []
    if matrix_params:
        muon = Muon(
            matrix_params,
            lr=muon_learning_rate,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
        )
        muon_scale = muon_learning_rate / learning_rate
        for group in muon.param_groups:
            group["lr_scale"] = muon_scale
        optimizers.append(muon)
    if adamw_params:
        adamw = torch.optim.AdamW(adamw_params, lr=learning_rate)
        for group in adamw.param_groups:
            group["lr_scale"] = 1.0
        optimizers.append(adamw)

    return OptimizerBundle(optimizers, {
        "optimizer": "muon",
        "adamw_parameters": sum(parameter.numel() for parameter in adamw_params),
        "muon_parameters": sum(parameter.numel() for parameter in matrix_params),
        "muon_learning_rate": muon_learning_rate,
        "muon_matrix_count": len(matrix_params),
        "muon_matrix_names": matrix_names[:12],
    })


class ExponentialMovingAverage:
    """Track float32 EMA weights without making them the live model state."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: tensor.detach().float().clone()
            if torch.is_floating_point(tensor)
            else tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor):
                self.shadow[name].mul_(self.decay).add_(
                    tensor.detach().float(),
                    alpha=1.0 - self.decay,
                )
            else:
                self.shadow[name].copy_(tensor.detach())

    def state_dict_for(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        current = model.state_dict()
        return {
            name: self.shadow[name].to(device=tensor.device, dtype=tensor.dtype)
            for name, tensor in current.items()
        }


@contextmanager
def using_ema_weights(
    model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
) -> Iterator[None]:
    """Temporarily swap EMA weights into the model."""
    if ema is None:
        yield
        return

    backup = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    model.load_state_dict(ema.state_dict_for(model), strict=True)
    try:
        yield
    finally:
        model.load_state_dict(backup, strict=True)


def validate_optim_controls(
    *,
    max_steps: int,
    lr_warmup_steps: int,
    lr_decay: str,
    min_lr_ratio: float,
    grad_clip: float,
    grad_accum_steps: int = 1,
    optimizer_type: str = "adamw",
    muon_learning_rate: float = 0.02,
    ema_decay: float = 0.0,
) -> None:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    if lr_decay not in LR_DECAYS:
        raise ValueError(f"lr_decay must be one of: {', '.join(LR_DECAYS)}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if optimizer_type not in OPTIMIZER_TYPES:
        raise ValueError(f"optimizer must be one of: {', '.join(OPTIMIZER_TYPES)}")
    if muon_learning_rate <= 0:
        raise ValueError("muon_learning_rate must be positive")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")


def learning_rate_for_step(
    *,
    base_learning_rate: float,
    step: int,
    max_steps: int,
    warmup_steps: int = 0,
    decay: str = "none",
    min_lr_ratio: float = 1.0,
) -> float:
    """Return the learning rate for a 1-indexed training step."""
    if step < 1:
        raise ValueError("step must be 1-indexed")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    if decay == "none":
        return base_learning_rate

    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    if decay == "linear":
        multiplier = 1.0 - progress
    elif decay == "cosine":
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported lr decay: {decay}")
    multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * multiplier
    return base_learning_rate * multiplier


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * group.get("lr_scale", 1.0)


def maybe_clip_grad_norm(model: torch.nn.Module, grad_clip: float) -> float | None:
    if grad_clip <= 0:
        return None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    return float(norm.item())
