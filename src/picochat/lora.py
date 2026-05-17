"""Native LoRA adapters for Picochat domain fine-tuning."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


PEFT_MODES = ("none", "lora")
LORA_TARGETS = ("attn_qkv", "attn_proj", "mlp_fc", "mlp_proj", "all_linear")
DEFAULT_LORA_TARGETS = ("attn_qkv", "attn_proj")


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    freeze_base: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["targets"] = list(self.targets)
        return payload


class LoRALinear(nn.Module):
    """Low-rank trainable adapter around a frozen Linear layer."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        self.merged = False
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.merged:
            return output
        adapter = F.linear(self.dropout(x), self.lora_a)
        adapter = F.linear(adapter, self.lora_b)
        return output + adapter * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.data.copy_(self.base.weight.data + self.delta_weight().to(dtype=self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


def parse_lora_targets(raw: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_LORA_TARGETS
    if isinstance(raw, str):
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
    else:
        values = tuple(str(item).strip() for item in raw if str(item).strip())
    if not values:
        raise ValueError("at least one LoRA target is required")
    bad = [value for value in values if value not in LORA_TARGETS]
    if bad:
        raise ValueError(f"unsupported LoRA target(s): {', '.join(bad)}")
    if "all_linear" in values and len(values) > 1:
        return ("all_linear",)
    return values


def apply_lora(model: nn.Module, config: LoRAConfig) -> dict[str, Any]:
    """Attach LoRA adapters to selected Picochat linear layers."""
    targets = parse_lora_targets(config.targets)
    if config.freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad = False

    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _matches_target(full_name, targets):
                replacements.append((module, child_name, full_name, child))

    if not replacements:
        raise ValueError(f"LoRA found no Linear modules for targets: {', '.join(targets)}")

    adapted: list[str] = []
    trainable_parameters = 0
    for parent, child_name, full_name, child in replacements:
        adapter = LoRALinear(child, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
        setattr(parent, child_name, adapter)
        adapted.append(full_name)
        trainable_parameters += adapter.lora_a.numel() + adapter.lora_b.numel()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "mode": "lora",
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "targets": list(targets),
        "adapted_modules": adapted,
        "adapted_module_count": len(adapted),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters if total_parameters else 0.0,
        "freeze_base": config.freeze_base,
    }


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_a"] = module.lora_a.detach().cpu()
            state[f"{name}.lora_b"] = module.lora_b.detach().cpu()
    return state


def save_lora_adapter(
    path: str | Path,
    model: nn.Module,
    *,
    config: LoRAConfig,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    if not state:
        raise ValueError("model has no LoRA adapters to save")
    torch.save(state, out_dir / "adapter_model.pt")
    payload = {
        "format": "picochat-lora",
        "peft_type": "lora",
        "config": config.to_dict(),
        "metadata": metadata or {},
        "adapter_tensors": sorted(state),
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "adapter_model": str(out_dir / "adapter_model.pt"),
        "adapter_config": str(out_dir / "adapter_config.json"),
    }


def load_lora_adapter(path: str | Path, model: nn.Module, *, map_location: str | torch.device = "cpu") -> None:
    state_path = Path(path) / "adapter_model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"LoRA adapter state not found: {state_path}")
    state = torch.load(state_path, map_location=map_location, weights_only=True)
    modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    if not modules:
        raise ValueError("model has no LoRA adapters to load")
    expected = {
        key
        for name in modules
        for key in (f"{name}.lora_a", f"{name}.lora_b")
    }
    unexpected = sorted(set(state) - expected)
    if unexpected:
        raise ValueError(f"LoRA adapter has unexpected tensor(s): {', '.join(unexpected[:8])}")
    missing: list[str] = []
    for name, module in modules.items():
        a_key = f"{name}.lora_a"
        b_key = f"{name}.lora_b"
        if a_key not in state or b_key not in state:
            missing.append(name)
            continue
        module.lora_a.data.copy_(state[a_key].to(device=module.lora_a.device, dtype=module.lora_a.dtype))
        module.lora_b.data.copy_(state[b_key].to(device=module.lora_b.device, dtype=module.lora_b.dtype))
    if missing:
        raise ValueError(f"LoRA adapter missing tensor(s) for: {', '.join(missing)}")


def subtract_lora_delta_from_base(model: nn.Module) -> None:
    """Turn merged base weights back into frozen-base + adapter weights."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.base.weight.sub_(module.delta_weight().to(device=module.base.weight.device, dtype=module.base.weight.dtype))


@contextmanager
def merged_lora_model(model: nn.Module) -> Iterator[None]:
    """Temporarily replace LoRA modules with merged Linear layers for checkpointing."""
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                replacements.append((module, child_name, child))
    if not replacements:
        yield
        return
    try:
        for parent, child_name, child in replacements:
            setattr(parent, child_name, child.merged_linear())
        yield
    finally:
        for parent, child_name, child in replacements:
            setattr(parent, child_name, child)


def trainable_parameter_report(model: nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def _matches_target(name: str, targets: tuple[str, ...]) -> bool:
    if "all_linear" in targets:
        return name.startswith("blocks.")
    return (
        ("attn_qkv" in targets and name.endswith(".attn.qkv"))
        or ("attn_proj" in targets and name.endswith(".attn.proj"))
        or ("mlp_fc" in targets and name.endswith(".mlp.fc"))
        or ("mlp_proj" in targets and name.endswith(".mlp.proj"))
    )
