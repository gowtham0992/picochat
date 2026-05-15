import json

import picochat.eval as eval_module
from picochat.checkpoint import save_checkpoint
from picochat.eval import (
    ChatEvalConfig,
    ChatEvalItem,
    _choice_continuation_candidates,
    analyze_eval_failures,
    detect_prompt_echo,
    load_chat_eval_items,
    run_chat_eval,
    score_reply,
    write_sft_fit_eval,
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


def test_load_chat_eval_items_supports_choice_eval_fields(tmp_path):
    input_path = tmp_path / "eval.jsonl"
    write_jsonl(input_path, [{
        "user": "Pick one.\nA. red\nB. blue\nRespond only with the letter.",
        "must_include": ["B"],
        "choice_labels": ["A", "B"],
        "correct_choice": "B",
    }])

    items = load_chat_eval_items(input_path)

    assert items[0].choice_labels == ("A", "B")
    assert items[0].correct_choice == "B"


def test_write_sft_fit_eval_converts_chat_rows(tmp_path):
    input_path = tmp_path / "chat.jsonl"
    output_path = tmp_path / "fit.jsonl"
    write_jsonl(input_path, [
        {
            "user": "What is Picochat?",
            "assistant": "Picochat is a tiny local model.",
            "category": "identity",
            "curriculum_stage": "identity_l1_name",
        },
        {
            "user": "What is the secret key?",
            "assistant": "I do not know from the provided material.",
            "category": "refusal",
            "answerable": False,
        },
    ])

    report = write_sft_fit_eval(input_path, output_path)
    items = load_chat_eval_items(output_path)

    assert report["num_rows"] == 2
    assert report["category_counts"] == {"identity": 1, "refusal": 1}
    assert items[0].split == "sft_train"
    assert items[0].level == "identity"
    assert items[0].curriculum_stage == "identity_l1_name"
    assert items[0].must_include == ("Picochat is a tiny local model.",)
    assert items[0].max_words == 14
    assert items[1].answerable is False


def test_write_sft_fit_eval_can_select_split_indices(tmp_path):
    input_path = tmp_path / "chat.jsonl"
    output_path = tmp_path / "fit.jsonl"
    write_jsonl(input_path, [
        {"user": "u0", "assistant": "a0", "category": "alpha"},
        {"user": "u1", "assistant": "a1", "category": "beta"},
        {"user": "u2", "assistant": "a2", "category": "alpha"},
    ])

    report = write_sft_fit_eval(
        input_path,
        output_path,
        include_indices=[0, 2],
        split_label="sft_heldout",
    )
    items = load_chat_eval_items(output_path)

    assert report["num_rows"] == 2
    assert report["selected_from_indices"] is True
    assert report["split_label"] == "sft_heldout"
    assert report["category_counts"] == {"alpha": 2}
    assert [item.user for item in items] == ["u0", "u2"]
    assert {item.split for item in items} == {"sft_heldout"}


def test_write_sft_fit_eval_uses_declared_fit_phrases(tmp_path):
    input_path = tmp_path / "chat.jsonl"
    output_path = tmp_path / "fit.jsonl"
    write_jsonl(input_path, [
        {
            "user": "Solve 2 + 3 with scratchpad.",
            "assistant": "Scratchpad:\n- Compute: 2 + 3.\n- Result: 5.\nFinal answer: 5",
            "category": "bench_math_addition",
            "fit_must_include": ["Scratchpad:", "Final answer: 5"],
            "fit_reference_answer": "5",
            "fit_max_words": 80,
        },
    ])

    report = write_sft_fit_eval(input_path, output_path)
    items = load_chat_eval_items(output_path)

    assert report["num_rows"] == 1
    assert items[0].must_include == ("Scratchpad:", "Final answer: 5")
    assert items[0].reference_answer == "5"
    assert items[0].max_words == 80


def test_choice_continuation_candidates_cover_sft_and_bare_styles():
    tokenizer = CharTokenizer.train(["Assistant: A B\n"])

    candidates = _choice_continuation_candidates(tokenizer, "A")

    assert candidates
    assert candidates[0].variant == "space+eos"
    assert {"space", "bare"}.issubset({candidate.variant for candidate in candidates})
    assert len({candidate.token_ids for candidate in candidates}) == len(candidates)


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


def test_score_reply_uses_word_aware_phrase_matching():
    item = ChatEvalItem(
        user="Should refuse weather?",
        must_include=("do not know",),
        must_include_any=(("story model", "tiny story"),),
        must_not_include=("no", "raining"),
        answerable=False,
    )

    score = score_reply(
        "I do not know. I am a tiny story model, so I should not give training facts.",
        item,
    )

    assert score["passed"] is True
    assert score["found_forbidden"] == []


def test_score_reply_keeps_format_markers_literal():
    item = ChatEvalItem(
        user="Tell a story.",
        must_include=("Story:",),
        must_not_include=("User:",),
    )

    score = score_reply("A story can be short, but this has no label.", item)
    tagged = score_reply("Story: A short tale.", item)

    assert score["passed"] is False
    assert score["missing"] == ["Story:"]
    assert tagged["passed"] is True


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


def test_analyze_eval_failures_tracks_wrong_choice():
    rows = [{
        "index": 1,
        "category": "arc_easy",
        "split": "benchmark",
        "level": "choice",
        "answerable": True,
        "reply": "A",
        "choice_predicted": "A",
        "correct_choice": "B",
        "missing": ["B"],
        "missing_any": [],
        "found_forbidden": [],
        "prompt_echo": False,
        "passed": False,
        "support_total": 1,
        "support_matched": 0,
    }]

    analysis = analyze_eval_failures(rows)

    assert analysis["failure_counts"]["wrong_choice"] == 1
    assert analysis["cluster_counts"]["choice_mismatch"] == 1
    assert any(item["area"] == "choice_eval" for item in analysis["recommendations"])


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
    assert report["summary"]["pass_rate_ci"]["method"] == "bootstrap"
    assert report["summary"]["pass_rate_ci"]["n"] == 1
    assert report["summary"]["prompt_echo_rate"] == 0.0
    assert report["summary"]["support_match_rate"] == 1.0
    assert report["summary"]["non_choice_examples"] == 1
    assert report["summary"]["non_choice_pass_rate"] == 1.0
    assert report["summary"]["category_breakdown"]["answerable"]["num_examples"] == 1
    assert report["summary"]["category_breakdown"]["answerable"]["num_passed"] == 1
    assert report["summary"]["category_breakdown"]["answerable"]["pass_rate_ci"]["method"] == "bootstrap"


def test_run_chat_eval_prints_progress_when_requested(tmp_path, capsys):
    input_path = tmp_path / "eval.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "eval"
    write_jsonl(input_path, [
        {"user": "one"},
        {"user": "two"},
    ])
    tokenizer = CharTokenizer.train(["User: one\nAssistant:"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=32,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    run_chat_eval(ChatEvalConfig(
        input_path=str(input_path),
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        max_new_tokens=0,
        log_every=1,
    ))

    output = capsys.readouterr().out
    assert "eval 0000/0002" in output
    assert "eval 0001/0002" in output
    assert "eval 0002/0002" in output
    assert "eta" in output


def test_generation_token_budget_uses_public_length_constraints():
    tokenizer = CharTokenizer.train(["short answer"])
    config = ChatEvalConfig(
        input_path="eval.jsonl",
        checkpoint_path="checkpoint",
        tokenizer_path="tokenizer.json",
        out_dir="out",
        max_new_tokens=80,
    )

    char_budget = eval_module._generation_max_new_tokens(
        config,
        tokenizer,
        ChatEvalItem(user="say it briefly", max_chars=12, max_words=2),
    )
    unrestricted_budget = eval_module._generation_max_new_tokens(
        config,
        tokenizer,
        ChatEvalItem(user="say anything"),
    )

    assert char_budget == 20
    assert unrestricted_budget == 80


def test_run_chat_eval_indexes_support_corpus_once(tmp_path, monkeypatch):
    input_path = tmp_path / "eval.jsonl"
    support_path = tmp_path / "support.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "eval"
    write_jsonl(input_path, [
        {"user": f"question {index}", "require_corpus_support": True}
        for index in range(3)
    ])
    support_path.write_text("Pico Cafe was founded by Mira Chen.", encoding="utf-8")
    tokenizer = CharTokenizer.train(["User: question\nAssistant:"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=32,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    calls = 0
    original = eval_module._corpus_support_token_set

    def counted(corpus_text):
        nonlocal calls
        calls += 1
        return original(corpus_text)

    monkeypatch.setattr(eval_module, "_corpus_support_token_set", counted)

    report = run_chat_eval(ChatEvalConfig(
        input_path=str(input_path),
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        max_new_tokens=0,
        support_corpus_path=str(support_path),
    ))

    assert calls == 1
    assert report["summary"]["split_breakdown"]["default"]["num_examples"] == 3
    assert report["examples"][0]["answerable"] is True
    assert report["examples"][0]["category"] == "answerable"
    assert report["examples"][0]["split"] == "default"
    assert "analysis" in report
    assert report["analysis"]["recommendations"]
