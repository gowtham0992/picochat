import pytest

from picochat.serve import (
    ServeConfig,
    _chat_completion_response,
    _completion_response,
    _render_openai_messages,
)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["prompt"]
        return {
            "text": prompt + " Hello from Picochat.\nUser: stop",
            "completion": " Hello from Picochat.\nUser: stop",
            "generated_tokens": [{"id": 11}, {"id": 12}],
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "finish_reason": "length",
            "used_kv_cache": True,
        }


def test_completion_response_matches_openai_shape():
    engine = FakeEngine()
    config = ServeConfig(checkpoint_path="checkpoint", tokenizer_path="tokenizer.json", model_name="pico-test")

    response = _completion_response(engine, config, {
        "model": "pico-test",
        "prompt": "Complete this:",
        "max_tokens": 8,
        "temperature": 0,
        "top_k": 0,
    })

    assert response["object"] == "text_completion"
    assert response["model"] == "pico-test"
    assert response["choices"][0]["text"] == " Hello from Picochat.\nUser: stop"
    assert response["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }
    assert engine.calls[0]["prompt"] == "Complete this:"
    assert engine.calls[0]["max_new_tokens"] == 8
    assert engine.calls[0]["temperature"] == 0
    assert engine.calls[0]["top_k"] is None


def test_chat_completion_renders_messages_and_extracts_assistant_reply():
    engine = FakeEngine()
    config = ServeConfig(checkpoint_path="checkpoint", tokenizer_path="tokenizer.json")

    response = _chat_completion_response(engine, config, {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What are you?"},
        ],
        "max_tokens": 8,
        "stop": ["\nUser:"],
    })

    assert engine.calls[0]["prompt"] == "System: Be concise.\nUser: What are you?\nAssistant:"
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello from Picochat.",
    }
    assert response["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_rejects_streaming_until_native_streaming_exists():
    engine = FakeEngine()
    config = ServeConfig(checkpoint_path="checkpoint", tokenizer_path="tokenizer.json")

    with pytest.raises(ValueError, match="stream=true"):
        _chat_completion_response(engine, config, {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })


def test_render_openai_messages_rejects_unknown_roles():
    with pytest.raises(ValueError, match="unsupported message role"):
        _render_openai_messages([{"role": "tool", "content": "nope"}])
