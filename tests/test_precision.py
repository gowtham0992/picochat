import torch

from picochat.precision import maybe_compile_model


def test_maybe_compile_model_uses_static_shapes(monkeypatch):
    calls = []
    model = torch.nn.Linear(4, 4)

    def fake_compile(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    compiled, metadata = maybe_compile_model(model, enabled=True)

    assert compiled is model
    assert calls == [{"dynamic": False}]
    assert metadata == {"enabled": True, "mode": "default", "dynamic": False}


def test_maybe_compile_model_keeps_requested_compile_mode(monkeypatch):
    calls = []
    model = torch.nn.Linear(4, 4)

    def fake_compile(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    compiled, metadata = maybe_compile_model(
        model,
        enabled=True,
        mode="reduce-overhead",
    )

    assert compiled is model
    assert calls == [{"dynamic": False, "mode": "reduce-overhead"}]
    assert metadata == {"enabled": True, "mode": "reduce-overhead", "dynamic": False}
