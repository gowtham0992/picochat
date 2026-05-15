from picochat.distributed import barrier_if_distributed, is_main_process


def test_is_main_process_uses_ddp_metadata():
    assert is_main_process({"enabled": True, "rank": 0}) is True
    assert is_main_process({"enabled": True, "rank": 1}) is False
    assert is_main_process({"enabled": False, "rank": 0}) is True


def test_barrier_if_distributed_noops_when_disabled():
    barrier_if_distributed({"enabled": False, "rank": 1})
