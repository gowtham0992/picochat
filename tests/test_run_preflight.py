from picochat.data import (
    CorpusPreviewReport,
    CorpusReadiness,
    CorpusReadinessCheck,
    CorpusStats,
    CorpusTrainingBudget,
    CorpusTrainingCommand,
)
from picochat.model import GPTConfig, TinyGPT
from picochat.run import TinyRunConfig
from picochat.run_preflight import assess_run_preflight
from picochat.tuning_data import ChatEvalDataReport, ChatSFTDataReport


def test_preflight_warns_that_sharded_base_mode_is_not_document_holdout():
    report = assess_run_preflight(
        _h100_like_config(base_dataset_mode="sharded"),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert report.status == "warn"
    assert checks["document_split"].status == "warn"
    assert checks["document_split"].metric == "sharded"
    assert "token shard" in checks["document_split"].message
    assert checks["document_boundaries"].status == "pass"
    assert checks["document_boundaries"].metric == "bos/eos in token shards"
    assert "token-shard holdout" in checks["document_boundaries"].message


def test_preflight_reports_document_boundaries_for_memory_document_split():
    report = assess_run_preflight(
        _h100_like_config(base_dataset_mode="memory"),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert checks["document_split"].status == "pass"
    assert checks["document_split"].metric == "document"
    assert checks["document_boundaries"].status == "pass"
    assert checks["document_boundaries"].metric == "bos/eos per document"


def test_preflight_reports_packed_base_mode_as_document_holdout():
    report = assess_run_preflight(
        _h100_like_config(base_dataset_mode="packed"),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert checks["document_split"].status == "pass"
    assert checks["document_split"].metric == "packed"
    assert "BOS-bestfit" in checks["document_split"].message
    assert checks["document_boundaries"].status == "pass"
    assert checks["document_boundaries"].metric == "bos-bestfit packed rows"
    assert "held-out documents" in checks["document_boundaries"].message


def test_preflight_does_not_claim_boundaries_for_window_split():
    config = _h100_like_config(base_dataset_mode="memory", split_mode="window")
    report = assess_run_preflight(config, _ready_large_corpus())
    checks = _checks_by_name(report)

    assert checks["document_split"].status == "block"
    assert checks["document_boundaries"].status == "warn"
    assert checks["document_boundaries"].metric == "disabled"


def test_preflight_warns_for_terse_long_run_sft_answers():
    report = assess_run_preflight(
        _h100_like_config(),
        _ready_large_corpus(assistant_avg_words=2.5),
    )
    checks = _checks_by_name(report)

    assert checks["sft_answer_length"].status == "warn"
    assert checks["sft_answer_length"].metric == "2.50"
    assert "Very terse SFT answers" in checks["sft_answer_length"].message


def test_preflight_accepts_narrow_first_release_sft_focus():
    report = assess_run_preflight(
        _h100_like_config(long_run_gate_profile="first_release"),
        _ready_large_corpus(categories={"identity": 1360, "refusal": 240}),
    )
    checks = _checks_by_name(report)

    assert checks["sft_category_balance"].status == "pass"
    assert checks["sft_category_balance"].metric == "2"
    assert checks["sft_category_balance"].threshold == ">= 2 first-release behavior categories"
    assert "separate diagnostics" in checks["sft_category_balance"].message


def test_preflight_still_warns_for_narrow_research_sft_focus():
    report = assess_run_preflight(
        _h100_like_config(),
        _ready_large_corpus(categories={"identity": 1360, "refusal": 240}),
    )
    checks = _checks_by_name(report)

    assert checks["sft_category_balance"].status == "warn"
    assert checks["sft_category_balance"].threshold == ">= 4 categories preferred"


def test_preflight_counts_ddp_world_size_in_effective_batch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    report = assess_run_preflight(
        _h100_like_config(ddp=True, base_steps=100, sft_steps=10),
        _ready_large_corpus(),
    )

    assert report.budget.ddp_world_size == 8
    assert report.budget.base_effective_batch_size == 8 * 16 * 8
    assert report.budget.sft_effective_batch_size == 8 * 4 * 8
    assert report.budget.base_effective_tokens_per_step == 8 * 16 * 8 * 512


def test_preflight_surfaces_ddp_auto_lr_scaling_risk(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    report = assess_run_preflight(
        _h100_like_config(
            ddp=True,
            auto_lr_scaling=True,
            base_steps=4100,
            base_learning_rate=0.00005,
            sft_learning_rate=0.00001,
        ),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert report.budget.auto_lr_scaling is True
    assert report.budget.base_effective_learning_rate > 0.0005
    assert checks["base_effective_lr"].status == "pass"
    assert checks["ddp_auto_lr_scaling"].status == "warn"
    assert "DDP world size" in checks["ddp_auto_lr_scaling"].message


def test_preflight_blocks_ddp_loss_spike_rollback(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    report = assess_run_preflight(
        _h100_like_config(ddp=True, loss_spike_rollback=True),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert report.status == "blocked"
    assert checks["ddp_loss_spike_rollback"].status == "block"
    assert "rank-local" in checks["ddp_loss_spike_rollback"].message


def test_preflight_blocks_ddp8_scale_without_eight_rank_budget(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    report = assess_run_preflight(
        _h100_like_config(
            scale="h100-100m-ddp8",
            ddp=False,
            base_steps=4100,
        ),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert checks["ddp_scale_launch"].status == "block"
    assert report.status == "blocked"


def test_preflight_allows_ddp8_scale_with_simulated_world_size(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    report = assess_run_preflight(
        _h100_like_config(
            scale="h100-100m-ddp8",
            ddp=True,
            ddp_world_size=8,
            base_steps=4100,
        ),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert report.budget.ddp_world_size == 8
    assert checks["ddp_scale_launch"].status == "pass"


def test_preflight_blocks_flash_attention_without_cuda_mixed_precision():
    report = assess_run_preflight(
        _h100_like_config(attn_backend="flash", device="mps", precision="auto"),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert report.status == "blocked"
    assert checks["attention_backend_runtime"].status == "block"
    assert "--device cuda" in checks["attention_backend_runtime"].threshold

    report = assess_run_preflight(
        _h100_like_config(attn_backend="flash", device="cuda", precision="float32"),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert report.status == "blocked"
    assert checks["attention_backend_runtime"].status == "block"
    assert "bf16" in checks["attention_backend_runtime"].threshold


def test_preflight_allows_flash_attention_on_cuda_bf16():
    report = assess_run_preflight(
        _h100_like_config(attn_backend="flash", device="cuda", precision="bf16"),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert checks["attention_backend_runtime"].status == "pass"
    assert report.status == "warn"


def test_preflight_allows_external_flash_attention_on_cuda_bf16():
    report = assess_run_preflight(
        _h100_like_config(attn_backend="external_flash", device="cuda", precision="bf16"),
        _ready_large_corpus(),
    )
    checks = _checks_by_name(report)

    assert checks["attention_backend_runtime"].status == "pass"
    assert "flash-attn package" in checks["attention_backend_runtime"].message


def test_preflight_parameter_estimate_matches_modern_model():
    config = _h100_like_config(
        tokenizer_vocab_size=1024,
        context_size=128,
        n_embd=64,
        n_head=8,
        n_kv_head=2,
        n_layer=3,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="swiglu",
        tie_embeddings=True,
        qk_norm=True,
        parallel_residual=True,
        base_steps=100,
    )
    report = assess_run_preflight(config, _ready_large_corpus())

    assert report.budget.estimated_parameters == _actual_model_parameters(config)


def test_preflight_parameter_estimate_matches_legacy_model():
    config = _h100_like_config(
        tokenizer_vocab_size=512,
        context_size=96,
        n_embd=48,
        n_head=4,
        n_kv_head=None,
        n_layer=2,
        norm_type="layernorm",
        position_encoding="learned",
        activation="gelu",
        tie_embeddings=False,
        qk_norm=False,
        parallel_residual=False,
        base_steps=100,
    )
    report = assess_run_preflight(config, _ready_large_corpus())

    assert report.budget.estimated_parameters == _actual_model_parameters(config)


def _h100_like_config(**overrides) -> TinyRunConfig:
    values = {
        "out_dir": "runs/test-preflight",
        "tokenizer_type": "hf_bpe",
        "tokenizer_vocab_size": 8192,
        "context_size": 512,
        "n_embd": 384,
        "n_head": 8,
        "n_kv_head": 2,
        "n_layer": 8,
        "norm_type": "rmsnorm",
        "position_encoding": "rope",
        "activation": "swiglu",
        "tie_embeddings": True,
        "qk_norm": True,
        "base_steps": 5000,
        "sft_steps": 700,
        "base_batch_size": 8,
        "base_grad_accum_steps": 16,
        "sft_batch_size": 8,
        "sft_grad_accum_steps": 4,
        "base_learning_rate": 0.0001,
        "sft_learning_rate": 0.00001,
        "split_mode": "document",
        "base_dataset_mode": "memory",
    }
    values.update(overrides)
    return TinyRunConfig(**values)


def _ready_large_corpus(
    *,
    assistant_avg_words: float = 8.0,
    categories: dict[str, int] | None = None,
) -> CorpusPreviewReport:
    categories = categories or {"choice": 400, "math": 400, "spelling": 400, "refusal": 400}
    return CorpusPreviewReport(
        input_path="corpus_recipe.json",
        recipe_path="corpus_recipe.json",
        dataset_pack="dataset_pack.json",
        stats=CorpusStats(
            num_files=80_000,
            num_documents=79_651,
            num_characters=235_000_000,
            num_lines=5_000_000,
            average_document_chars=2950.0,
            duplicate_document_rate=0.0,
            duplicate_line_rate=0.06,
            non_ascii_rate=0.001,
            empty_line_rate=0.2,
        ),
        files=(),
        readiness=CorpusReadiness(
            status="ready",
            summary="Corpus looks ready for a long run.",
            checks=(
                CorpusReadinessCheck(
                    name="usable_documents",
                    status="pass",
                    metric="79651",
                    threshold=">= 1",
                    message="At least one source has usable text.",
                ),
            ),
        ),
        budget=CorpusTrainingBudget(
            preset="test",
            estimated_tokens=65_800_000,
            suggested_context_size=512,
            estimated_windows=65_799_488,
            suggested_batch_size=8,
            suggested_base_steps=5000,
            estimated_tokens_per_step=4096,
            estimated_passes=4.9,
            note="test",
        ),
        training_command=CorpusTrainingCommand(
            out_dir="runs/test",
            chat_input="chat.jsonl",
            eval_input="eval.jsonl",
            dataset_pack="dataset_pack.json",
            command="picochat",
            note="test",
        ),
        chat_data=ChatSFTDataReport(
            path="chat.jsonl",
            status="ready",
            summary="Chat SFT file looks usable.",
            num_rows=1600,
            num_examples=1600,
            empty_rows=0,
            invalid_rows=0,
            average_user_chars=60.0,
            average_assistant_chars=40.0,
            duplicate_user_rate=0.0,
            duplicate_user_prompts=0,
            duplicate_user_samples=(),
            near_duplicate_user_pairs=0,
            near_duplicate_user_samples=(),
            categories=categories,
            category_entropy=2.0,
            category_entropy_normalized=1.0,
            assistant_length_distribution={
                "count": 1600,
                "min_chars": 2,
                "max_chars": 120,
                "avg_chars": 40.0,
                "min_words": 1,
                "max_words": 24,
                "avg_words": assistant_avg_words,
            },
            template_families={},
            answer_styles={},
            curriculum_label="mixed_sft",
            curriculum_breakdown={},
            quality_warnings=(),
            issues=(),
            preview=(),
        ),
        eval_data=ChatEvalDataReport(
            path="eval.jsonl",
            status="ready",
            summary="Eval file looks usable.",
            num_rows=320,
            num_items=320,
            empty_rows=0,
            invalid_rows=0,
            answerable_items=288,
            unanswerable_items=32,
            must_include_rules=320,
            must_include_any_groups=160,
            must_not_include_rules=32,
            duplicate_user_rate=0.0,
            duplicate_user_prompts=0,
            duplicate_user_samples=(),
            near_duplicate_user_pairs=0,
            near_duplicate_user_samples=(),
            categories={"choice": 80, "math": 80, "spelling": 80, "refusal": 80},
            category_entropy=2.0,
            category_entropy_normalized=1.0,
            splits={"benchmark": 240, "behavior": 48, "adversarial": 32},
            levels={"choice": 80, "math": 80, "spelling": 80, "refusal": 80},
            heldout_categories={"choice": 80, "math": 80, "spelling": 80, "refusal": 80},
            answer_length_distribution={},
            template_families={},
            curriculum_label="mixed_eval",
            curriculum_breakdown={},
            quality_warnings=(),
            issues=(),
            preview=(),
        ),
        warnings=(),
        preview="sample",
    )


def _checks_by_name(report):
    return {check.name: check for check in report.checks}


def _actual_model_parameters(config: TinyRunConfig) -> int:
    model = TinyGPT(
        GPTConfig(
            vocab_size=config.tokenizer_vocab_size,
            context_size=config.context_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_layer=config.n_layer,
            norm_type=config.norm_type,
            position_encoding=config.position_encoding,
            activation=config.activation,
            tie_embeddings=config.tie_embeddings,
            qk_norm=config.qk_norm,
            parallel_residual=config.parallel_residual,
        )
    )
    return model.num_parameters()
