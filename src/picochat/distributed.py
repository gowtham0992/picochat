"""Small DDP helpers for single-node Picochat training."""

from __future__ import annotations

import os
from typing import Any

import torch


def prepare_ddp_model(
    model: torch.nn.Module,
    device: torch.device,
    enabled: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Wrap a model in DistributedDataParallel when explicitly requested."""
    if not enabled:
        return model, {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
    return wrapped, {
        "enabled": True,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "backend": torch.distributed.get_backend(),
    }


def is_main_process(metadata: dict[str, Any] | None = None) -> bool:
    if metadata is not None:
        return int(metadata.get("rank", 0)) == 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True
