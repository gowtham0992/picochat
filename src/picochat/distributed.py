"""Small DDP helpers for single-node Picochat training."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import os
from typing import Any

import torch


TORCHRUN_REQUIRED_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
DEFAULT_DDP_TIMEOUT_MINUTES = 120
DDP_TIMEOUT_ENV = "PICOCHAT_DDP_TIMEOUT_MINUTES"


def initialize_ddp(
    device: torch.device,
    enabled: bool = False,
) -> dict[str, Any]:
    """Initialize a single-node torchrun process group when DDP is enabled."""
    if not enabled:
        return {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    env = ddp_env_metadata(enabled=True)
    local_rank = int(env["local_rank"])
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
            "timeout": _ddp_timeout(),
        }
        if device.type == "cuda":
            try:
                torch.distributed.init_process_group(
                    **init_kwargs,
                    device_id=torch.device("cuda", local_rank),
                )
            except TypeError:
                # Older PyTorch releases do not expose the device_id kwarg.
                torch.distributed.init_process_group(**init_kwargs)
        else:
            torch.distributed.init_process_group(**init_kwargs)
    return {
        "enabled": True,
        "world_size": torch.distributed.get_world_size(),
        "rank": torch.distributed.get_rank(),
        "local_rank": local_rank,
        "backend": torch.distributed.get_backend(),
        "timeout_minutes": _ddp_timeout_minutes(),
    }


def prepare_ddp_model(
    model: torch.nn.Module,
    device: torch.device,
    enabled: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Wrap a model in DistributedDataParallel when explicitly requested."""
    if not enabled:
        return model, {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    metadata = initialize_ddp(device, enabled=True)
    local_rank = int(metadata["local_rank"])
    common_kwargs = {
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
        "static_graph": True,
    }
    if device.type == "cuda":
        ddp_kwargs = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            **common_kwargs,
        }
    else:
        ddp_kwargs = dict(common_kwargs)
    try:
        wrapped = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    except TypeError:
        # Older PyTorch builds may not support every performance hint. Keep
        # DDP functional instead of making portability depend on one kwarg.
        if device.type == "cuda":
            wrapped = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            wrapped = torch.nn.parallel.DistributedDataParallel(model)
    return wrapped, metadata


def ddp_env_metadata(enabled: bool = False) -> dict[str, Any]:
    """Return torchrun rank metadata before the process group is initialized."""
    if not enabled:
        return {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    missing = [name for name in TORCHRUN_REQUIRED_ENV if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP requires torchrun so every rank gets rendezvous metadata; "
            f"missing environment variable(s): {', '.join(missing)}. "
            "Launch with: torchrun --standalone --nproc_per_node=<gpus> -m picochat.cli ..."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be at least 1 for DDP")
    if rank < 0 or rank >= world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE) for DDP")
    return {
        "enabled": True,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }


def _ddp_timeout() -> timedelta:
    return timedelta(minutes=_ddp_timeout_minutes())


def _ddp_timeout_minutes() -> int:
    raw = os.environ.get(DDP_TIMEOUT_ENV, str(DEFAULT_DDP_TIMEOUT_MINUTES))
    try:
        minutes = int(raw)
    except ValueError:
        minutes = DEFAULT_DDP_TIMEOUT_MINUTES
    return max(1, minutes)


def is_main_process(metadata: dict[str, Any] | None = None) -> bool:
    if metadata is not None:
        return int(metadata.get("rank", 0)) == 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def barrier_if_distributed(metadata: dict[str, Any] | None = None) -> None:
    """Synchronize ranks after rank-sensitive side effects such as checkpoint writes."""
    if metadata is not None and not bool(metadata.get("enabled", False)):
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_rank = None if metadata is None else metadata.get("local_rank")
        if local_rank is not None and torch.distributed.get_backend() == "nccl":
            try:
                torch.distributed.barrier(device_ids=[int(local_rank)])
                return
            except TypeError:
                pass
        torch.distributed.barrier()


def mean_scalar_if_distributed(
    value: float,
    device: torch.device,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Average a scalar metric across ranks for artifact logging."""
    if metadata is not None and not bool(metadata.get("enabled", False)):
        return value
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)
    return float(tensor.item())


def broadcast_object_if_distributed(
    value: Any,
    *,
    src: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Broadcast a small Python object from one rank to every distributed worker."""
    if metadata is not None and not bool(metadata.get("enabled", False)):
        return value
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    rank = torch.distributed.get_rank()
    payload = [value if rank == src else None]
    torch.distributed.broadcast_object_list(payload, src=src)
    return payload[0]


def no_sync_if_distributed(model: torch.nn.Module, *, enabled: bool):
    """Skip DDP gradient all-reduce for intermediate gradient accumulation microsteps."""
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()
