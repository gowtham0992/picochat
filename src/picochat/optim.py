"""Small optimizer helpers for transparent local training runs."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch


LR_DECAYS = ("none", "linear", "cosine")
WEIGHT_DECAY_DECAYS = ("none", "cosine_to_zero")
MUON_MOMENTUM_SCHEDULES = ("none", "peaked")
OPTIMIZER_TYPES = ("adamw", "muon")


def _is_lora_parameter_name(name: str) -> bool:
    return name.endswith(".lora_a") or name.endswith(".lora_b") or ".lora_" in name


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

    def state_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state: dict) -> None:
        optimizer_states = state.get("optimizers", [])
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError("optimizer state does not match this optimizer bundle")
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)


def create_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_type: str = "adamw",
    learning_rate: float,
    weight_decay: float = 0.01,
    muon_learning_rate: float = 0.02,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
) -> OptimizerBundle:
    """Create AdamW or a Muon+AdamW hybrid optimizer for Picochat models."""
    if optimizer_type not in OPTIMIZER_TYPES:
        raise ValueError(f"optimizer must be one of: {', '.join(OPTIMIZER_TYPES)}")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if muon_learning_rate <= 0:
        raise ValueError("muon_learning_rate must be positive")

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("optimizer received no trainable parameters")

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        for group in optimizer.param_groups:
            group["lr_scale"] = 1.0
            group["weight_decay_scale"] = 1.0
        return OptimizerBundle([optimizer], {
            "optimizer": "adamw",
            "adamw_parameters": sum(parameter.numel() for parameter in trainable_params),
            "muon_parameters": 0,
            "muon_learning_rate": None,
            "weight_decay": weight_decay,
        })

    matrix_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    matrix_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2 and name.startswith("blocks.") and not _is_lora_parameter_name(name):
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
            weight_decay=weight_decay,
        )
        muon_scale = muon_learning_rate / learning_rate
        for group in muon.param_groups:
            group["lr_scale"] = muon_scale
            group["weight_decay_scale"] = 1.0
        optimizers.append(muon)
    if adamw_params:
        adamw = torch.optim.AdamW(
            adamw_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        for group in adamw.param_groups:
            group["lr_scale"] = 1.0
            group["weight_decay_scale"] = 1.0
        optimizers.append(adamw)

    return OptimizerBundle(optimizers, {
        "optimizer": "muon",
        "adamw_parameters": sum(parameter.numel() for parameter in adamw_params),
        "muon_parameters": sum(parameter.numel() for parameter in matrix_params),
        "muon_learning_rate": muon_learning_rate,
        "muon_momentum": muon_momentum,
        "muon_matrix_count": len(matrix_params),
        "muon_matrix_names": matrix_names[:12],
        "weight_decay": weight_decay,
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

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        if float(state.get("decay", self.decay)) != self.decay:
            raise ValueError("EMA decay in checkpoint does not match this run")
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow")
        if not isinstance(shadow, dict):
            raise ValueError("EMA state is missing shadow weights")
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in shadow.items()
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
    weight_decay: float = 0.01,
    weight_decay_decay: str = "none",
    muon_learning_rate: float = 0.02,
    muon_momentum_schedule: str = "none",
    ema_decay: float = 0.0,
    loss_spike_threshold: float = 2.5,
    loss_spike_lr_decay: float = 0.5,
    loss_spike_min_lr_scale: float = 0.1,
    loss_spike_snapshot_every: int = 10,
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
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if weight_decay_decay not in WEIGHT_DECAY_DECAYS:
        raise ValueError(f"weight_decay_decay must be one of: {', '.join(WEIGHT_DECAY_DECAYS)}")
    if muon_learning_rate <= 0:
        raise ValueError("muon_learning_rate must be positive")
    if muon_momentum_schedule not in MUON_MOMENTUM_SCHEDULES:
        raise ValueError(f"muon_momentum_schedule must be one of: {', '.join(MUON_MOMENTUM_SCHEDULES)}")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if loss_spike_threshold <= 1.0:
        raise ValueError("loss_spike_threshold must be greater than 1")
    if not 0.0 < loss_spike_lr_decay <= 1.0:
        raise ValueError("loss_spike_lr_decay must be in (0, 1]")
    if not 0.0 < loss_spike_min_lr_scale <= 1.0:
        raise ValueError("loss_spike_min_lr_scale must be in (0, 1]")
    if loss_spike_snapshot_every < 1:
        raise ValueError("loss_spike_snapshot_every must be at least 1")


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


def weight_decay_for_step(
    *,
    base_weight_decay: float,
    step: int,
    max_steps: int,
    decay: str = "none",
) -> float:
    """Return weight decay for a 1-indexed training step."""
    if step < 1:
        raise ValueError("step must be 1-indexed")
    if base_weight_decay < 0:
        raise ValueError("base_weight_decay must be non-negative")
    if decay == "none" or base_weight_decay == 0:
        return base_weight_decay
    if decay != "cosine_to_zero":
        raise ValueError(f"Unsupported weight decay schedule: {decay}")
    progress = min(1.0, max(0.0, (step - 1) / max(1, max_steps - 1)))
    return base_weight_decay * 0.5 * (1.0 + math.cos(math.pi * progress))


def muon_momentum_for_step(
    *,
    schedule: str,
    step: int,
    max_steps: int,
) -> float | None:
    """Return scheduled Muon momentum, or None when fixed momentum is requested."""
    if step < 1:
        raise ValueError("step must be 1-indexed")
    if schedule == "none":
        return None
    if schedule != "peaked":
        raise ValueError(f"Unsupported Muon momentum schedule: {schedule}")
    progress = min(1.0, max(0.0, (step - 1) / max(1, max_steps - 1)))
    if progress <= 0.5:
        return _linear_interpolate(0.85, 0.97, progress / 0.5)
    return _linear_interpolate(0.97, 0.90, (progress - 0.5) / 0.5)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * group.get("lr_scale", 1.0)


def set_optimizer_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    for group in optimizer.param_groups:
        if "weight_decay" in group:
            group["weight_decay"] = weight_decay * group.get("weight_decay_scale", 1.0)


def set_muon_momentum(optimizer: torch.optim.Optimizer, momentum: float | None) -> None:
    if momentum is None:
        return
    for group in optimizer.param_groups:
        if "momentum" in group and "ns_steps" in group:
            group["momentum"] = momentum


def maybe_clip_grad_norm(model: torch.nn.Module, grad_clip: float) -> float | None:
    if grad_clip <= 0:
        return None
    fsdp_clip = getattr(model, "clip_grad_norm_", None)
    norm = (
        fsdp_clip(grad_clip)
        if callable(fsdp_clip)
        else torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    )
    return float(norm.item())


def _linear_interpolate(start: float, end: float, t: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, t))
