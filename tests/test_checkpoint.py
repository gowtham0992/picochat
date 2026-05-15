from picochat.checkpoint import load_checkpoint, load_training_state, save_checkpoint
from picochat.model import GPTConfig, TinyGPT


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
