import pytest
import torch

from picochat.checkpoint import load_checkpoint, load_training_state, save_checkpoint
from picochat.model import GPTConfig, TinyGPT
from picochat.resume import restore_rng_state


def test_save_and_load_checkpoint(tmp_path):
    model = TinyGPT(GPTConfig(vocab_size=10, context_size=4, n_embd=8, n_head=2, n_layer=1))

    save_checkpoint(tmp_path, model, step=3, train_loss=1.23)
    loaded, metadata = load_checkpoint(tmp_path)

    assert metadata["step"] == 3
    assert metadata["train_loss"] == 1.23
    assert loaded.config == model.config


def test_save_and_load_training_state(tmp_path):
    model = TinyGPT(GPTConfig(vocab_size=10, context_size=4, n_embd=8, n_head=2, n_layer=1))

    save_checkpoint(
        tmp_path,
        model,
        step=3,
        train_loss=1.23,
        training_state={"step": 3, "losses": [{"step": 3}]},
    )
    _, metadata = load_checkpoint(tmp_path)
    state = load_training_state(tmp_path)

    assert metadata["has_training_state"] is True
    assert state["step"] == 3
    assert state["losses"] == [{"step": 3}]


def test_save_checkpoint_accepts_materialized_state_dict(tmp_path):
    model = TinyGPT(GPTConfig(vocab_size=10, context_size=4, n_embd=8, n_head=2, n_layer=1))
    override = {
        name: torch.zeros_like(value)
        for name, value in model.state_dict().items()
    }

    save_checkpoint(
        tmp_path,
        model,
        step=5,
        train_loss=0.5,
        model_state_dict=override,
        model_config=model.config,
    )
    loaded, metadata = load_checkpoint(tmp_path)

    assert metadata["step"] == 5
    for value in loaded.state_dict().values():
        assert torch.count_nonzero(value).item() == 0


def test_save_checkpoint_keeps_previous_checkpoint_on_failed_write(tmp_path, monkeypatch):
    model = TinyGPT(GPTConfig(vocab_size=10, context_size=4, n_embd=8, n_head=2, n_layer=1))
    save_checkpoint(tmp_path, model, step=3, train_loss=1.23)

    real_save = __import__("torch").save

    def failing_save(payload, path):
        if str(path).endswith("training_state.pt"):
            raise OSError("simulated interrupted checkpoint write")
        return real_save(payload, path)

    monkeypatch.setattr("picochat.checkpoint.torch.save", failing_save)

    with pytest.raises(OSError, match="simulated interrupted"):
        save_checkpoint(
            tmp_path,
            model,
            step=4,
            train_loss=0.75,
            training_state={"step": 4},
        )

    _, metadata = load_checkpoint(tmp_path)
    assert metadata["step"] == 3
    assert not (tmp_path.parent / f".{tmp_path.name}.previous").exists()
    assert not list(tmp_path.parent.glob(f".{tmp_path.name}.tmp-*"))


def test_restore_rng_state_accepts_checkpoint_safe_payloads():
    expected = torch.get_rng_state()
    torch.manual_seed(999)

    restore_rng_state({"torch": expected.tolist()})

    assert torch.equal(torch.get_rng_state(), expected)
