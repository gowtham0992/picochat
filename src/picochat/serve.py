"""Local OpenAI-compatible serving for Picochat checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from time import time
import uuid

from picochat.chat import extract_assistant_reply
from picochat.generate import LoadedGenerator


@dataclass(frozen=True)
class ServeConfig:
    checkpoint_path: str
    tokenizer_path: str
    host: str = "127.0.0.1"
    port: int = 8000
    model_name: str = "picochat"
    device: str = "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 20
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    use_kv_cache: bool = True
    allow_origin: str = "*"


def serve_model(config: ServeConfig) -> None:
    """Start a blocking local OpenAI-compatible HTTP server."""
    engine = LoadedGenerator(
        checkpoint_path=config.checkpoint_path,
        tokenizer_path=config.tokenizer_path,
        device=config.device,
    )
    handler = _handler_factory(config, engine)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    print(f"pico serve: http://{config.host}:{config.port}")
    print(f"models:     http://{config.host}:{config.port}/v1/models")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def _handler_factory(config: ServeConfig, engine: LoadedGenerator):
    generation_lock = threading.Lock()

    class PicoServeHandler(BaseHTTPRequestHandler):
        server_version = "PicochatServe/0.1"

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._write_json({})

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._write_json({
                    "status": "ok",
                    "model": config.model_name,
                    "device": str(engine.device),
                })
                return
            if self.path == "/v1/models":
                self._write_json(_models_response(config.model_name))
                return
            self._write_json({"error": {"message": "not found"}}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                if self.path == "/v1/completions":
                    with generation_lock:
                        self._write_json(_completion_response(engine, config, payload))
                    return
                if self.path == "/v1/chat/completions":
                    with generation_lock:
                        self._write_json(_chat_completion_response(engine, config, payload))
                    return
                self._write_json({"error": {"message": "not found"}}, status=404)
            except ValueError as exc:
                self._write_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._write_json({"error": {"message": str(exc), "type": "server_error"}}, status=500)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _write_json(self, payload: dict, *, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", config.allow_origin)
            self.send_header("Access-Control-Allow-Headers", "content-type, authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

    return PicoServeHandler


def _models_response(model_name: str) -> dict:
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "picochat",
        }],
    }


def _completion_response(engine, config: ServeConfig, payload: dict) -> dict:
    _reject_unsupported_options(payload)
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        if len(prompt) != 1 or not isinstance(prompt[0], str):
            raise ValueError("prompt arrays must contain exactly one string")
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")

    result = _generate(engine, config, payload, prompt=prompt)
    text, stopped = _apply_stop(result["completion"], payload.get("stop"))
    return _openai_response(
        object_name="text_completion",
        model_name=_request_model_name(payload, config.model_name),
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=len(result["generated_tokens"]),
        choices=[{
            "text": text,
            "index": 0,
            "logprobs": None,
            "finish_reason": _finish_reason(result, stopped),
        }],
    )


def _chat_completion_response(engine, config: ServeConfig, payload: dict) -> dict:
    _reject_unsupported_options(payload)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    prompt = _render_openai_messages(messages)
    result = _generate(engine, config, payload, prompt=prompt)
    raw_reply = result["text"][len(prompt):] if result["text"].startswith(prompt) else result["completion"]
    role_stop = "\nUser:" in raw_reply
    content = extract_assistant_reply(prompt, result["text"])
    content, stopped = _apply_stop(content, payload.get("stop"))
    stopped = stopped or role_stop
    return _openai_response(
        object_name="chat.completion",
        model_name=_request_model_name(payload, config.model_name),
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=len(result["generated_tokens"]),
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": _finish_reason(result, stopped),
        }],
    )


def _generate(engine, config: ServeConfig, payload: dict, *, prompt: str) -> dict:
    max_new_tokens = _bounded_int(
        payload.get("max_tokens", payload.get("max_new_tokens", config.max_new_tokens)),
        0,
        8192,
        "max_tokens",
    )
    top_k_value = payload.get("top_k", config.top_k if config.top_k is not None else 0)
    top_k = _bounded_int(top_k_value, 0, 10000, "top_k")
    return engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=_bounded_float(payload.get("temperature", config.temperature), 0.0, 5.0, "temperature"),
        top_k=None if top_k <= 0 else top_k,
        top_p=_bounded_float(payload.get("top_p", config.top_p), 0.001, 1.0, "top_p"),
        repetition_penalty=_bounded_float(
            payload.get("repetition_penalty", config.repetition_penalty),
            0.01,
            10.0,
            "repetition_penalty",
        ),
        seed=_bounded_int(payload.get("seed", config.seed), 0, 2**31 - 1, "seed"),
        use_kv_cache=config.use_kv_cache,
    )


def _openai_response(
    *,
    object_name: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    choices: list[dict],
) -> dict:
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": object_name,
        "created": int(time()),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _render_openai_messages(messages: list[dict]) -> str:
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"message {index} must contain string role and content")
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            raise ValueError(f"unsupported message role: {role}")
    if not lines or not lines[-1].startswith("Assistant:"):
        lines.append("Assistant:")
    elif lines[-1] != "Assistant:":
        lines.append("Assistant:")
    return "\n".join(lines)


def _reject_unsupported_options(payload: dict) -> None:
    if payload.get("stream"):
        raise ValueError("stream=true is not supported by native pico serve yet")
    n = payload.get("n", 1)
    if n != 1:
        raise ValueError("native pico serve supports n=1")


def _request_model_name(payload: dict, default: str) -> str:
    model = payload.get("model", default)
    return model if isinstance(model, str) and model else default


def _finish_reason(result: dict, stopped: bool) -> str:
    if stopped:
        return "stop"
    reason = result.get("finish_reason")
    return reason if reason in {"stop", "length"} else "length"


def _apply_stop(text: str, stop) -> tuple[str, bool]:
    if stop is None:
        return text, False
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or not all(isinstance(item, str) for item in stops):
        raise ValueError("stop must be a string or array of strings")
    best = None
    for item in stops:
        if not item:
            continue
        index = text.find(item)
        if index >= 0 and (best is None or index < best):
            best = index
    if best is None:
        return text, False
    return text[:best], True


def _bounded_int(value, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return result


def _bounded_float(value, minimum: float, maximum: float, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return result
