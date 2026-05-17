from picochat.cli import main
from picochat.model import GPTConfig, TinyGPT
from picochat.scale_planner import estimate_parameters, parse_count, plan_scale, render_scale_plan_markdown


def test_parse_count_supports_compact_suffixes():
    assert parse_count("100m") == 100_000_000
    assert parse_count("2.5b") == 2_500_000_000
    assert parse_count("524,288") == 524_288


def test_scale_plan_matches_100m_shape_and_exact_parameter_count():
    plan = plan_scale(target_parameters=100_000_000, depth=16, dataset_tokens=667_278_610)

    assert plan.n_layer == 16
    assert plan.n_embd == 768
    assert plan.n_head == 12
    assert plan.n_kv_head == 4
    assert plan.global_batch_tokens == 65_536
    assert plan.recommended_base_steps == 32_645
    assert 3.1 < plan.estimated_epochs < 3.3
    assert plan.linear_bias is False
    assert plan.scaled_residual_init is True

    config = GPTConfig(
        vocab_size=8192,
        context_size=512,
        n_embd=768,
        n_head=12,
        n_kv_head=4,
        n_layer=16,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="swiglu",
        tie_embeddings=True,
        qk_norm=True,
        parallel_residual=True,
        linear_bias=False,
    )
    assert plan.estimated_parameters == estimate_parameters(config)


def test_parameter_estimate_matches_model_on_small_shape():
    config = GPTConfig(
        vocab_size=128,
        context_size=32,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=3,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="swiglu",
        tie_embeddings=True,
        qk_norm=True,
        parallel_residual=True,
        linear_bias=False,
    )
    assert estimate_parameters(config) == TinyGPT(config).num_parameters()


def test_scale_plan_marks_ddp_overrides():
    plan = plan_scale(target_parameters=100_000_000, depth=16, world_size=8)

    assert plan.global_batch_tokens == 524_288
    assert plan.recommended_base_steps < 5_000
    assert "--ddp" in plan.run_tiny_overrides()
    assert "--scaled-residual-init" in plan.run_tiny_overrides()
    assert plan.base_learning_rate == 0.0002
    assert plan.batch_scaled_learning_rate > plan.base_learning_rate
    assert "--long-run-gate-profile" in plan.run_tiny_overrides()
    assert "skill_release" in plan.run_tiny_overrides()


def test_scale_plan_markdown_contains_copyable_command():
    plan = plan_scale(target_parameters=100_000_000, depth=16)
    markdown = render_scale_plan_markdown(plan)

    assert "# Picochat Scale Plan" in markdown
    assert "--n-embd 768" in markdown
    assert "--no-linear-bias" in markdown
    assert "--scaled-residual-init" in markdown
    assert "--base-dataset-mode" in markdown
    assert "--long-run-gate-profile skill_release" in markdown
    assert "--profile release_skills --skill-answer-style scratchpad" in markdown


def test_cli_scale_plan_prints_report(capsys):
    code = main(["scale", "plan", "--target-params", "100m", "--depth", "16", "--dataset-tokens", "667m"])

    assert code == 0
    output = capsys.readouterr().out
    assert "Picochat Scale Plan" in output
    assert "Estimated corpus epochs" in output


def test_cli_scale_plan_accepts_hopper_attention_and_release_gate(capsys):
    code = main([
        "scale",
        "plan",
        "--target-params",
        "1b",
        "--world-size",
        "8",
        "--attn-backend",
        "fa3",
        "--long-run-gate-profile",
        "skill_release",
    ])

    assert code == 0
    output = capsys.readouterr().out
    assert "--attn-backend fa3" in output
    assert "--long-run-gate-profile skill_release" in output
