from picochat.chat import extract_assistant_reply, render_chat_prompt


def test_render_chat_prompt_without_history():
    prompt = render_chat_prompt([], "hello")

    assert prompt == "User: hello\nAssistant:"


def test_render_chat_prompt_with_history():
    prompt = render_chat_prompt([("hi", "hello")], "what is picochat")

    assert "User: hi" in prompt
    assert "Assistant: hello" in prompt
    assert prompt.endswith("User: what is picochat\nAssistant:")


def test_extract_assistant_reply_removes_prompt_and_next_user_turn():
    prompt = "User: hello\nAssistant:"
    generated = prompt + " hi there\nUser: next"

    reply = extract_assistant_reply(prompt, generated)

    assert reply == "hi there"

