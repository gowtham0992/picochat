"""Training precision and compile helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch


PRECISION_MODES = ("float32", "bf16", "fp16", "auto")
MATMUL_PRECISION_MODES = ("default", "highest", "high", "medium")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")


@dataclass(frozen=True)
class PrecisionRuntime:
    requested: str
    enabled: bool
    device_type: str
    dtype_name: str
    grad_scaler: bool
    note: str | None = None

    @property
    def dtype(self) -> torch.dtype | None:
        if self.dtype_name == "bfloat16":
            return torch.bfloat16
        if self.dtype_name == "float16":
            return torch.float16
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_precision(mode: str, device: torch.device) -> PrecisionRuntime:
    """Resolve a user precision request into a conservative runtime policy."""
    requested = mode.lower()
    if requested not in PRECISION_MODES:
        raise ValueError(f"precision must be one of: {', '.join(PRECISION_MODES)}")
    if requested == "float32":
        return PrecisionRuntime(
            requested=requested,
            enabled=False,
            device_type=device.type,
            dtype_name="float32",
            grad_scaler=False,
            note="full precision",
        )
    if requested == "auto":
        if device.type == "cuda":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return _runtime("auto", device, torch.bfloat16, note="auto selected bf16")
            return _runtime("auto", device, torch.float16, note="auto selected fp16")
        if device.type == "mps":
            return _runtime("auto", device, torch.float16, note="auto selected fp16")
        return PrecisionRuntime(
            requested=requested,
            enabled=False,
            device_type=device.type,
            dtype_name="float32",
            grad_scaler=False,
            note="auto kept float32 on this device",
        )
    dtype = torch.bfloat16 if requested == "bf16" else torch.float16
    return _runtime(requested, device, dtype)


def configure_float32_matmul_precision(mode: str) -> dict[str, Any]:
    """Configure PyTorch float32 matmul precision for CUDA tensor cores."""
    requested = mode.lower()
    if requested not in MATMUL_PRECISION_MODES:
        raise ValueError(f"matmul_precision must be one of: {', '.join(MATMUL_PRECISION_MODES)}")
    getter = getattr(torch, "get_float32_matmul_precision", None)
    setter = getattr(torch, "set_float32_matmul_precision", None)
    before = getter() if getter else None
    if requested != "default":
        if setter is None:
            raise RuntimeError("torch.set_float32_matmul_precision is not available in this PyTorch build")
        setter(requested)
    after = getter() if getter else before
    return {
        "requested": requested,
        "before": before,
        "after": after,
        "changed": requested != "default" and before != after,
    }


def autocast_context(runtime: PrecisionRuntime):
    """Return the autocast context for a resolved precision runtime."""
    if not runtime.enabled or runtime.dtype is None:
        return nullcontext()
    return torch.autocast(device_type=runtime.device_type, dtype=runtime.dtype)


def make_grad_scaler(runtime: PrecisionRuntime):
    """Create a GradScaler only for CUDA fp16."""
    enabled = runtime.grad_scaler
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def maybe_compile_model(
    model: torch.nn.Module,
    *,
    enabled: bool,
    mode: str = "default",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Compile a model when requested, keeping the original module saveable."""
    if mode not in COMPILE_MODES:
        raise ValueError(f"torch_compile_mode must be one of: {', '.join(COMPILE_MODES)}")
    if not enabled:
        return model, {"enabled": False, "mode": mode}
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    compile_kwargs = {} if mode == "default" else {"mode": mode}
    compiled = torch.compile(model, **compile_kwargs)
    return compiled, {"enabled": True, "mode": mode}


def _runtime(
    requested: str,
    device: torch.device,
    dtype: torch.dtype,
    note: str | None = None,
) -> PrecisionRuntime:
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 autocast is not supported for CPU training; use bf16 or float32")
    if device.type not in {"cuda", "cpu", "mps"}:
        raise ValueError(f"mixed precision is not supported for device type {device.type!r}")
    return PrecisionRuntime(
        requested=requested,
        enabled=True,
        device_type=device.type,
        dtype_name=str(dtype).replace("torch.", ""),
        grad_scaler=device.type == "cuda" and dtype == torch.float16,
        note=note,
    )
