import json

from picochat.checkpoint import save_checkpoint
from picochat.eval import (
    ChatEvalConfig,
    ChatEvalItem,
    analyze_eval_failures,
    detect_prompt_echo,
    load_chat_eval_items,
    run_chat_eval,
    score_reply,
)
from picochat.model import GPTConfig, TinyGPT
from picochat.tokenizer import CharTokenizer


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_load_chat_eval_items_supports_expected_alias(tmp_path):
    input_path = tmp_path / "eval.jsonl"
    write_jsonl(input_path, [{
        "user": "hi",
        "expected": "hello",
        "must_include_any": [["hey", "hello"]],
        "answerable": False,
        "category": "refusal",
        "split": "safety",
    }])

    items = load_chat_eval_items(input_path)

    assert items == [
        ChatEvalItem(
            user="hi",
            must_include=("hello",),
            must_include_any=(("hey", "hello"),),
            answerable=False,
            category="refusal",
            split="safety",
            level="adversarial",
            reference_answer="hello",
        )
    ]


def test_score_reply_checks_required_and_forbidden_phrases():
    item = ChatEvalItem(
        user="What is Picochat?",
        must_include=("educational", "LLM"),
        must_include_any=(("factory", "lab"),),
        must_not_include=("unknown",),
    )

    passing = score_reply("Picochat is an educational tiny LLM.", item)
    passing_with_any = score_reply("Picochat is an educational tiny LLM lab.", item)
    failing = score_reply("Picochat is unknown.", item)

    assert passing["passed"] is False
    assert passing["missing_any"] == [["factory", "lab"]]
    assert passing["support_matched"] == 2
    assert passing["support_total"] == 3
    assert passing_with_any["passed"] is True
    assert passing_with_any["support_match_rate"] == 1.0
    assert failing["passed"] is False
    assert failing["missing"] == ["educational", "LLM"]
    assert failing["missing_any"] == [["factory", "lab"]]
    assert failing["found_forbidden"] == ["unknown"]


def test_score_reply_blocks_prompt_echo():
    item = ChatEvalItem(
        user="Subject = turtle; lesson = sharing.",
        must_include=("turtle",),
    )

    score = score_reply("User: Subject = turtle; lesson = sharing.\nAssistant: turtle", item)

    assert score["passed"] is False
    assert score["prompt_echo"] is True
    assert score["prompt_echo_reasons"] == ["chat_role_label", "starts_with_user_prompt"]
    assert detect_prompt_echo("Subject: turtle. Story: One day.", item.user) == []


def test_score_reply_tracks_richer_diagnostics():
    item = ChatEvalItem(
        user="Who founded Pico Cafe?",
        must_include=("Pico Cafe",),
        reference_answer="Pico Cafe was founded by Mira Chen.",
        required_entities=("Mira Chen",),
        min_words=4,
        max_words=12,
        require_corpus_support=True,
    )

    score = score_reply(
        "Pico Cafe was founded by Mira Chen.",
        item,
        support_corpus_text="Pico Cafe was founded by Mira Chen in a small downtown shop.",
    )

    assert score["passed"] is True
    assert score["reference_token_f1"] == 1.0
    assert score["reference_rouge_l"] == 1.0
    assert score["entity_match_rate"] == 1.0
    assert score["length_violations"] == []
    assert score["corpus_support_failed"] is False
    assert score["corpus_support_rate"] > 0


def test_score_reply_can_fail_entity_length_and_corpus_support():
    item = ChatEvalItem(
        user="Who founded Pico Cafe?",
        reference_answer="Pico Cafe was founded by Mira Chen.",
        required_entities=("Mira Chen",),
        max_words=3,
        require_corpus_support=True,
    )

    score = score_reply(
        "A totally unrelated answer with many extra words.",
        item,
        support_corpus_text="Pico Cafe was founded by Mira Chen.",
    )

    assert score["passed"] is False
    assert score["missing_entities"] == ["Mira Chen"]
    assert score["length_violations"] == ["max_words:3"]
    assert score["corpus_support_failed"] is True


def test_analyze_eval_failures_recommends_next_actions():
    rows = [{
        "index": 1,
        "category": "story_generation",
        "split": "heldout",
        "answerable": True,
        "reply": "User: write a story",
        "missing": ["kind fox"],
        "missing_any": [["sharing", "helping"]],
        "found_forbidden": ["User:"],
        "prompt_echo": True,
        "prompt_echo_reasons": ["chat_role_label"],
        "passed": False,
        "support_total": 2,
        "support_matched": 0,
    }]

    analysis = analyze_eval_failures(rows)

    assert analysis["failure_counts"]["missing_required"] == 1
    assert analysis["failure_counts"]["missing_any_group"] == 1
    assert analysis["failure_counts"]["forbidden_phrase"] == 1
    assert analysis["failure_counts"]["prompt_echo"] == 1
    assert analysis["cluster_counts"]["content_mismatch"] == 1
    assert analysis["weak_categories"][0]["category"] == "story_generation"
    assert any(item["area"] == "story_generation" for item in analysis["recommendations"])


def test_run_chat_eval_writes_artifacts(tmp_path):
    input_path = tmp_path / "eval.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "eval"
    write_jsonl(input_path, [{"user": "hi"}])
    tokenizer = CharTokenizer.train(["User: hi\nAssistant:"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=32,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    report = run_chat_eval(ChatEvalConfig(
        input_path=str(input_path),
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        max_new_tokens=0,
    ))

    assert (out_dir / "eval_report.json").exists()
    assert (out_dir / "report.md").exists()
    assert report["summary"]["num_examples"] == 1
    assert report["summary"]["num_passed"] == 1
    assert report["summary"]["unsupported_claim_rate"] == 0.0
    assert report["summary"]["prompt_echo_rate"] == 0.0
    assert report["summary"]["support_match_rate"] == 1.0
    assert report["summary"]["category_breakdown"]["answerable"]["num_examples"] == 1
    assert report["summary"]["category_breakdown"]["answerable"]["num_passed"] == 1
    assert report["summary"]["split_breakdown"]["default"]["num_examples"] == 1
    assert report["examples"][0]["answerable"] is True
    assert report["examples"][0]["category"] == "answerable"
    assert report["examples"][0]["split"] == "default"
    assert "analysis" in report
    assert report["analysis"]["recommendations"]
