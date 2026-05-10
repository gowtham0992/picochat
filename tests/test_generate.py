from picochat.checkpoint import save_checkpoint
from picochat.generate import GenerateConfig, generate_text, generate_text_with_trace
from picochat.model import GPTConfig, TinyGPT
from picochat.tokenizer import CharTokenizer


def test_generate_text_from_checkpoint(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    tokenizer = CharTokenizer.train(["hello picochat"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    text = generate_text(GenerateConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        prompt="he",
        max_new_tokens=3,
        temperature=0,
    ))

    assert text.startswith("he")


def test_generate_text_with_trace_returns_token_metadata(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    tokenizer = CharTokenizer.train(["hello picochat"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    result = generate_text_with_trace(GenerateConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        prompt="he",
        max_new_tokens=2,
        temperature=0,
    ))

    assert result["text"].startswith("he")
    assert "completion" in result
    assert 1 <= len(result["generated_tokens"]) <= 2
    assert {"token", "id", "probability", "logprob"} <= set(result["generated_tokens"][0])
