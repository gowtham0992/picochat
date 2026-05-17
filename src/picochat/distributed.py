"""Small DDP helpers for single-node Picochat training."""

from __future__ import annotations

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
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=_ddp_timeout(),
        )
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
        torch.distributed.barrier()
