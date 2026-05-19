import pytest

from picochat.device import DEVICE_CHOICES, resolve_device


def test_resolve_device_accepts_cpu_and_auto():
    assert "auto" in DEVICE_CHOICES
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}


def test_resolve_device_rejects_unknown_name():
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("quantum")
