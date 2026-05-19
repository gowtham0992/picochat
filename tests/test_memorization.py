from picochat.memorization import memorization_diagnostics


def test_memorization_diagnostics_flags_long_exact_copy():
    train_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    generated = "prefix alpha beta gamma delta epsilon zeta eta theta iota kappa suffix"

    report = memorization_diagnostics(generated, train_text, ngram_size=4)

    assert report["status"] in {"medium", "high"}
    assert report["train_overlap_rate"] > 0
    assert report["longest_train_overlap_tokens"] >= 10


def test_memorization_diagnostics_flags_canary_hit():
    report = memorization_diagnostics(
        "The secret is pico-canary-7F2Q9.",
        "Training includes pico-canary-7F2Q9.",
        ngram_size=3,
    )

    assert report["status"] == "high"
    assert report["canary_hits"] == ["pico-canary-7F2Q9"]


def test_memorization_diagnostics_low_for_unrelated_text():
    report = memorization_diagnostics(
        "fresh words are unrelated",
        "alpha beta gamma delta epsilon",
        ngram_size=3,
    )

    assert report["status"] == "low"
    assert report["train_overlap_rate"] == 0.0
