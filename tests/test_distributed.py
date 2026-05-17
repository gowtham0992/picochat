from contextlib import contextmanager

import pytest
import torch

from picochat.distributed import (
    barrier_if_distributed,
    broadcast_object_if_distributed,
    ddp_env_metadata,
    initialize_ddp,
    is_main_process,
    mean_scalar_if_distributed,
    no_sync_if_distributed,
)


def test_is_main_process_uses_ddp_metadata():
    assert is_main_process({"enabled": True, "rank": 0}) is True
    assert is_main_process({"enabled": True, "rank": 1}) is False
    assert is_main_process({"enabled": False, "rank": 0}) is True


def test_barrier_if_distributed_noops_when_disabled():
    barrier_if_distributed({"enabled": False, "rank": 1})


def test_barrier_if_distributed_passes_nccl_device_ids(monkeypatch):
    captured = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    def fake_barrier(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)

    barrier_if_distributed({"enabled": True, "rank": 1, "local_rank": 3})

    assert captured["device_ids"] == [3]


def test_barrier_if_distributed_falls_back_for_old_torch(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    def fake_barrier(**kwargs):
        calls.append(kwargs)
        if "device_ids" in kwargs:
            raise TypeError("old torch")

    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)

    barrier_if_distributed({"enabled": True, "rank": 1, "local_rank": 3})

    assert calls == [{"device_ids": [3]}, {}]


def test_mean_scalar_if_distributed_noops_when_disabled(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail_all_reduce(*_args, **_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("all_reduce should not run for disabled DDP metadata")

    monkeypatch.setattr(torch.distributed, "all_reduce", fail_all_reduce)

    assert mean_scalar_if_distributed(1.25, torch.device("cpu"), {"enabled": False}) == 1.25


def test_mean_scalar_if_distributed_averages_initialized_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.AVG
        tensor.fill_(2.5)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert mean_scalar_if_distributed(1.25, torch.device("cpu"), {"enabled": True}) == 2.5


def test_broadcast_object_if_distributed_noops_when_disabled(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail_broadcast(*_args, **_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("broadcast should not run for disabled DDP metadata")

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fail_broadcast)

    payload = {"sha256": "abc"}
    assert broadcast_object_if_distributed(payload, metadata={"enabled": False}) == payload


def test_broadcast_object_if_distributed_receives_payload(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def fake_broadcast(payload, src):
        assert src == 0
        assert payload == [None]
        payload[0] = {"sha256": "from-rank-zero"}

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast)

    assert broadcast_object_if_distributed(None, metadata={"enabled": True}) == {
        "sha256": "from-rank-zero"
    }


def test_no_sync_if_distributed_uses_ddp_context_when_enabled():
    events = []

    class FakeDDP:
        @contextmanager
        def no_sync(self):
            events.append("enter")
            yield
            events.append("exit")

    with no_sync_if_distributed(FakeDDP(), enabled=True):
        events.append("body")

    assert events == ["enter", "body", "exit"]


def test_no_sync_if_distributed_noops_when_disabled():
    class FakeDDP:
        def no_sync(self):  # pragma: no cover - should not be called
            raise AssertionError("no_sync should not be used when disabled")

    with no_sync_if_distributed(FakeDDP(), enabled=False):
        pass


def test_ddp_env_metadata_requires_torchrun_env(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="DDP requires torchrun"):
        ddp_env_metadata(enabled=True)


def test_ddp_env_metadata_reads_torchrun_env(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    assert ddp_env_metadata(enabled=True) == {
        "enabled": True,
        "world_size": 8,
        "rank": 1,
        "local_rank": 1,
    }


def test_initialize_ddp_uses_long_preprocessing_timeout(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("PICOCHAT_DDP_TIMEOUT_MINUTES", "45")
    captured = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    def fake_init_process_group(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init_process_group)

    metadata = initialize_ddp(torch.device("cpu"), enabled=True)

    assert captured["timeout"].total_seconds() == 45 * 60
    assert metadata["timeout_minutes"] == 45


def test_initialize_ddp_passes_cuda_device_id(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    captured = {}

    monkeypatch.setattr(torch.cuda, "set_device", lambda local_rank: captured.setdefault("set_device", local_rank))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    def fake_init_process_group(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init_process_group)

    metadata = initialize_ddp(torch.device("cuda"), enabled=True)

    assert captured["set_device"] == 1
    assert captured["device_id"] == torch.device("cuda", 1)
    assert metadata["local_rank"] == 1


def test_initialize_ddp_falls_back_when_device_id_kwarg_is_missing(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    calls = []

    monkeypatch.setattr(torch.cuda, "set_device", lambda _local_rank: None)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    def fake_init_process_group(**kwargs):
        calls.append(kwargs)
        if "device_id" in kwargs:
            raise TypeError("old torch")

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init_process_group)

    initialize_ddp(torch.device("cuda"), enabled=True)

    assert calls[0]["device_id"] == torch.device("cuda", 1)
    assert "device_id" not in calls[1]
