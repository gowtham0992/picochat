from contextlib import contextmanager

import pytest
import torch

from picochat.distributed import (
    barrier_if_distributed,
    broadcast_object_if_distributed,
    ddp_env_metadata,
    fsdp_full_state_dict,
    initialize_ddp,
    is_main_process,
    mean_scalar_if_distributed,
    no_sync_if_distributed,
    prepare_distributed_model,
    prepare_ddp_model,
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

    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda local_rank: captured.setdefault("set_device", local_rank),
    )
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


def test_prepare_ddp_model_uses_static_graph_kwargs(monkeypatch):
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
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **_kwargs: None)

    class FakeDDP:
        def __init__(self, module, **kwargs):
            self.module = module
            captured["ddp_kwargs"] = kwargs

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    model = torch.nn.Linear(2, 2)

    wrapped, metadata = prepare_ddp_model(model, torch.device("cuda"), enabled=True)

    assert wrapped.module is model
    assert metadata["local_rank"] == 1
    assert captured["ddp_kwargs"]["device_ids"] == [1]
    assert captured["ddp_kwargs"]["output_device"] == 1
    assert captured["ddp_kwargs"]["broadcast_buffers"] is False
    assert captured["ddp_kwargs"]["gradient_as_bucket_view"] is True
    assert captured["ddp_kwargs"]["static_graph"] is True


def test_prepare_ddp_model_falls_back_for_old_ddp_kwargs(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    calls = []

    monkeypatch.setattr(torch.cuda, "set_device", lambda _local_rank: None)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **_kwargs: None)

    class FakeDDP:
        def __init__(self, module, **kwargs):
            calls.append(kwargs)
            if "static_graph" in kwargs:
                raise TypeError("old DDP")
            self.module = module

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    prepare_ddp_model(torch.nn.Linear(2, 2), torch.device("cuda"), enabled=True)

    assert calls[0]["static_graph"] is True
    assert calls[1] == {"device_ids": [0], "output_device": 0}


def test_prepare_distributed_model_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="distributed_strategy"):
        prepare_distributed_model(
            torch.nn.Linear(2, 2),
            torch.device("cpu"),
            enabled=True,
            strategy="zero",
        )


def test_prepare_distributed_model_disabled_records_strategy():
    model = torch.nn.Linear(2, 2)

    wrapped, metadata = prepare_distributed_model(
        model,
        torch.device("cpu"),
        enabled=False,
        strategy="fsdp",
    )

    assert wrapped is model
    assert metadata["enabled"] is False
    assert metadata["strategy"] == "fsdp"


def test_prepare_distributed_model_wraps_fsdp_with_full_shard(monkeypatch):
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
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **_kwargs: None)

    class FakeShardingStrategy:
        FULL_SHARD = object()

    class FakeFSDP:
        def __init__(self, module, **kwargs):
            self.module = module
            captured["fsdp_kwargs"] = kwargs

    import picochat.distributed as distributed_module

    monkeypatch.setattr(
        distributed_module,
        "_fsdp_components",
        lambda: (FakeFSDP, FakeShardingStrategy),
    )
    model = torch.nn.Linear(2, 2)

    wrapped, metadata = prepare_distributed_model(
        model,
        torch.device("cuda"),
        enabled=True,
        strategy="fsdp",
    )

    assert wrapped.module is model
    assert captured["set_device"] == 1
    assert captured["fsdp_kwargs"]["device_id"] == torch.device("cuda", 1)
    assert captured["fsdp_kwargs"]["sharding_strategy"] is FakeShardingStrategy.FULL_SHARD
    assert captured["fsdp_kwargs"]["use_orig_params"] is True
    assert metadata["strategy"] == "fsdp"
    assert metadata["fsdp_sharding_strategy"] == "full_shard"


def test_fsdp_full_state_dict_collects_on_all_ranks(monkeypatch):
    calls = []

    class FakeStateDictType:
        FULL_STATE_DICT = object()

    class FakeFullStateDictConfig:
        def __init__(self, **kwargs):
            calls.append(("config", kwargs))

    class FakeContext:
        def __enter__(self):
            calls.append(("enter", None))

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", None))

    class FakeFSDP:
        @staticmethod
        def state_dict_type(model, state_dict_type, config):
            calls.append(("state_dict_type", state_dict_type, config))
            return FakeContext()

    class FakeModel(torch.nn.Module):
        def state_dict(self, *args, **kwargs):
            calls.append(("state_dict", None))
            return {"weight": torch.tensor([1.0])}

    import picochat.distributed as distributed_module

    monkeypatch.setattr(
        distributed_module,
        "_fsdp_state_dict_components",
        lambda: (FakeFSDP, FakeStateDictType, FakeFullStateDictConfig),
    )

    main_state = fsdp_full_state_dict(
        FakeModel(),
        {"enabled": True, "strategy": "fsdp", "rank": 0},
    )
    worker_state = fsdp_full_state_dict(
        FakeModel(),
        {"enabled": True, "strategy": "fsdp", "rank": 1},
    )

    assert main_state is not None
    assert torch.equal(main_state["weight"], torch.tensor([1.0]))
    assert worker_state is None
    assert [call[0] for call in calls].count("state_dict") == 2
