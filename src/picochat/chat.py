"""Simple command-line chat helpers."""

from __future__ import annotations

from dataclasses import dataclass

from picochat.generate import GenerateConfig, generate_text


@dataclass(frozen=True)
class ChatConfig:
    checkpoint_path: str
    tokenizer_path: str
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_k: int | None = 20
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    device: str = "cpu"
    use_kv_cache: bool = True


def render_chat_prompt(history: list[tuple[str, str]], user_message: str) -> str:
    """Render conversation history into the plain chat format used by v1."""
    lines: list[str] = []
    for user, assistant in history:
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def extract_assistant_reply(prompt: str, generated_text: str) -> str:
    """Return only newly generated assistant text after the prompt."""
    if generated_text.startswith(prompt):
        reply = generated_text[len(prompt):]
    else:
        reply = generated_text

    stop = reply.find("\nUser:")
    if stop >= 0:
        reply = reply[:stop]
    return reply.strip()


def generate_reply(
    config: ChatConfig,
    history: list[tuple[str, str]],
    user_message: str,
) -> str:
    prompt = render_chat_prompt(history, user_message)
    generated_text = generate_text(GenerateConfig(
        checkpoint_path=config.checkpoint_path,
        tokenizer_path=config.tokenizer_path,
        prompt=prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        seed=config.seed,
        device=config.device,
        use_kv_cache=config.use_kv_cache,
    ))
    return extract_assistant_reply(prompt, generated_text)


def chat_loop(config: ChatConfig) -> int:
    """Run an interactive terminal chat session."""
    history: list[tuple[str, str]] = []
    print("Picochat CLI. Type 'quit' or 'exit' to leave. Type 'clear' to reset.")
    while True:
        try:
            user_message = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_message:
            continue
        if user_message.lower() in {"quit", "exit"}:
            return 0
        if user_message.lower() == "clear":
            history.clear()
            print("Conversation cleared.")
            continue

        reply = generate_reply(config, history, user_message)
        print(f"Assistant: {reply}")
        history.append((user_message, reply))
