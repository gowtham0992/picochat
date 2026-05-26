import json
import tarfile
from types import SimpleNamespace

from picochat.cli import main


def test_cli_version(capsys):
    exit_code = main(["--version"])

    assert exit_code == 0
    assert "picochat" in capsys.readouterr().out


def test_cli_train_base_accepts_experimental_fsdp_strategy(tmp_path, monkeypatch):
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {
            "checkpoint": str(tmp_path / "run" / "checkpoint"),
            "sample": "ok",
            "config": {"artifacts_written": True},
        }

    monkeypatch.setattr("picochat.cli.train_base", fake_train)

    exit_code = main([
        "train",
        "base",
        "--corpus",
        str(tmp_path / "corpus.txt"),
        "--tokenizer",
        str(tmp_path / "tokenizer.json"),
        "--out-dir",
        str(tmp_path / "run"),
        "--ddp",
        "--distributed-strategy",
        "fsdp",
    ])

    assert exit_code == 0
    assert captured["config"].ddp is True
    assert captured["config"].distributed_strategy == "fsdp"


def test_cli_sanity_preh100(tmp_path, capsys, monkeypatch):
    def fake_run(config):
        assert config.out_dir == str(tmp_path / "sanity")
        assert config.precision == "float32"
        assert config.matmul_precision == "high"
        assert config.attn_backend == "math"
        assert config.include_compile is True
        assert config.capacity_scale == "smoke"
        assert config.capacity_batch_size == 2
        assert config.capacity_min_free_fraction == 0.2
        return {
            "status": "passed",
            "report_path": str(tmp_path / "sanity" / "preh100_sanity.json"),
            "markdown_path": str(tmp_path / "sanity" / "preh100_sanity.md"),
            "checks": [
                {"name": "precision_backward", "status": "pass", "detail": "ok"},
            ],
        }

    monkeypatch.setattr("picochat.cli.run_preh100_sanity", fake_run)

    exit_code = main([
        "sanity",
        "preh100",
        "--out-dir",
        str(tmp_path / "sanity"),
        "--precision",
        "float32",
        "--matmul-precision",
        "high",
        "--attn-backend",
        "math",
        "--include-compile",
        "--capacity-scale",
        "smoke",
        "--capacity-batch-size",
        "2",
        "--capacity-min-free-fraction",
        "0.2",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "sanity: passed" in output
    assert "precision_backward: pass ok" in output


def test_cli_eval_chat_accepts_runtime_precision(tmp_path, capsys, monkeypatch):
    def fake_run(config):
        assert config.precision == "bf16"
        assert config.matmul_precision == "high"
        assert config.device == "cuda"
        return {
            "summary": {
                "num_passed": 1,
                "num_examples": 2,
                "pass_rate": 0.5,
            },
        }

    monkeypatch.setattr("picochat.cli.run_chat_eval", fake_run)

    exit_code = main([
        "eval",
        "chat",
        "--input",
        "eval.jsonl",
        "--checkpoint",
        "checkpoint",
        "--tokenizer",
        "tokenizer.json",
        "--out-dir",
        str(tmp_path / "eval"),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--matmul-precision",
        "high",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "chat eval: 1/2 passed (50.00%)" in output


def test_cli_serve_builds_openai_compatible_server_config(monkeypatch):
    captured = {}

    def fake_serve(config):
        captured["config"] = config

    monkeypatch.setattr("picochat.cli.serve_model", fake_serve)

    exit_code = main([
        "serve",
        "--checkpoint",
        "runs/demo/sft/checkpoint",
        "--tokenizer",
        "runs/demo/tokenizer.json",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--model-name",
        "pico-demo",
        "--device",
        "cpu",
        "--top-k",
        "0",
        "--no-kv-cache",
    ])

    assert exit_code == 0
    config = captured["config"]
    assert config.checkpoint_path == "runs/demo/sft/checkpoint"
    assert config.tokenizer_path == "runs/demo/tokenizer.json"
    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.model_name == "pico-demo"
    assert config.top_k is None
    assert config.use_kv_cache is False


def test_cli_run_tiny_multiseed(tmp_path, capsys, monkeypatch):
    def fake_run(config, n_seeds):
        assert config.out_dir == str(tmp_path / "multi")
        assert config.seed == 7
        assert n_seeds == 3
        return {
            "aggregate": {
                "eval_pass_rate": {
                    "n": 3,
                    "mean": 0.25,
                    "std": 0.05,
                },
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny_multiseed", fake_run)

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "multi"),
        "--n-seeds",
        "3",
        "--seed",
        "7",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "multi-seed tiny run: n=3 eval pass mean 25.00% std 5.00%" in output


def test_cli_run_tiny_h100_scale_applies_modern_defaults(tmp_path, capsys, monkeypatch):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "eval": {
                "num_passed": 1,
                "num_examples": 1,
                "pass_rate": 1.0,
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run)

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "h100"),
        "--scale",
        "h100-pilot",
        "--long-run-gate-profile",
        "first_release",
        "--sft-peft",
        "lora",
        "--sft-lora-rank",
        "4",
        "--sft-lora-alpha",
        "8",
        "--sft-lora-targets",
        "attn_qkv,attn_proj",
        "--xsa-last-n",
        "4",
    ])

    assert exit_code == 0
    config = captured["config"]
    assert config.tokenizer_type == "hf_bpe"
    assert config.activation == "swiglu"
    assert config.n_kv_head == 2
    assert config.tie_embeddings is True
    assert config.qk_norm is True
    assert config.parallel_residual is True
    assert config.xsa_last_n == 4
    assert config.linear_bias is False
    assert config.attn_backend == "flash"
    assert config.precision == "bf16"
    assert config.matmul_precision == "high"
    assert config.torch_compile is True
    assert config.base_dataset_mode == "sharded"
    assert config.sft_learning_rate == 0.00001
    assert config.auto_lr_scaling is True
    assert config.loss_spike_rollback is True
    assert config.long_run_gate_profile == "first_release"
    assert config.sft_peft == "lora"
    assert config.sft_lora_rank == 4
    assert config.sft_lora_alpha == 8.0
    assert config.sft_lora_targets == ("attn_qkv", "attn_proj")
    assert "tiny run: 1/1 passed" in capsys.readouterr().out


def test_cli_run_tiny_ddp_worker_completion_does_not_expect_eval(tmp_path, capsys, monkeypatch):
    def fake_run(config):
        assert config.out_dir == str(tmp_path / "worker")
        assert config.ddp is True
        return {
            "status": "ddp_worker_complete",
            "rank": 1,
            "world_size": 8,
            "out_dir": str(tmp_path / "worker"),
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run)

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "worker"),
        "--ddp",
        "--ddp-world-size",
        "8",
    ])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_cli_run_tiny_can_override_h100_scale_linear_bias(tmp_path, monkeypatch):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "eval": {
                "num_passed": 1,
                "num_examples": 1,
                "pass_rate": 1.0,
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run)

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "h100-bias"),
        "--scale",
        "h100-pilot",
        "--linear-bias",
    ])

    assert exit_code == 0
    assert captured["config"].linear_bias is True


def test_cli_run_tiny_accepts_optional_dpo_stage(tmp_path, monkeypatch):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "eval": {
                "num_passed": 1,
                "num_examples": 1,
                "pass_rate": 1.0,
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run)

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "with-dpo"),
        "--dpo-input",
        str(tmp_path / "preferences.jsonl"),
        "--dpo-steps",
        "12",
        "--dpo-batch-size",
        "2",
        "--dpo-learning-rate",
        "0.000003",
        "--dpo-beta",
        "0.2",
        "--dpo-length-normalize",
    ])

    assert exit_code == 0
    assert captured["config"].dpo_input == str(tmp_path / "preferences.jsonl")
    assert captured["config"].dpo_steps == 12
    assert captured["config"].dpo_batch_size == 2
    assert captured["config"].dpo_learning_rate == 0.000003
    assert captured["config"].dpo_beta == 0.2
    assert captured["config"].dpo_length_normalize is True


def test_cli_run_tiny_h200_1b_scale_defaults_to_skill_release(tmp_path, monkeypatch):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "eval": {
                "num_passed": 1,
                "num_examples": 1,
                "pass_rate": 1.0,
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run)

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "h200"),
        "--scale",
        "h200-1b-ddp8",
    ])

    assert exit_code == 0
    config = captured["config"]
    assert config.n_embd == 2048
    assert config.n_layer == 24
    assert config.context_size == 2048
    assert config.tokenizer_vocab_size == 32768
    assert config.attn_backend == "fa3"
    assert config.base_grad_accum_steps == 8
    assert config.target_param_data_ratio == 20.0
    assert config.scaled_residual_init is True
    assert config.gradient_checkpointing is True
    assert config.long_run_gate_profile == "skill_release"


def test_cli_run_tiny_accepts_phase_resume_paths(tmp_path, monkeypatch):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "eval": {
                "num_passed": 1,
                "num_examples": 1,
                "pass_rate": 1.0,
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run)

    base_resume = tmp_path / "run" / "base" / "resume_checkpoint"
    sft_resume = tmp_path / "run" / "sft" / "resume_checkpoint"
    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "run"),
        "--base-resume-from",
        str(base_resume),
        "--sft-resume-from",
        str(sft_resume),
    ])

    assert exit_code == 0
    config = captured["config"]
    assert config.base_resume_from == str(base_resume)
    assert config.sft_resume_from == str(sft_resume)


def test_cli_run_tiny_accepts_ddp_world_size_for_preflight(tmp_path, monkeypatch):
    captured = {}

    def fake_preview(*args, **kwargs):
        captured["preview_kwargs"] = kwargs
        return {"stats": {}, "readiness": {"status": "ready"}}

    def fake_preflight(config, preview):
        captured["config"] = config
        return type("Report", (), {
            "status": "ready",
            "to_dict": lambda self: {},
            "summary": "ready",
        })()

    monkeypatch.setattr("picochat.cli.preview_corpus_sources", fake_preview)
    monkeypatch.setattr("picochat.cli.assess_run_preflight", fake_preflight)
    monkeypatch.setattr("picochat.cli.preflight_markdown", lambda report: "# preflight")

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "ddp8"),
        "--dataset-pack",
        "pack.json",
        "--scale",
        "h100-100m-ddp8",
        "--ddp",
        "--ddp-world-size",
        "8",
        "--preflight-only",
    ])

    assert exit_code == 0
    assert captured["config"].ddp is True
    assert captured["config"].ddp_world_size == 8


def test_cli_run_bundle_packages_partial_checkpoint_without_corpus(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "partial-run"
    resume = run_dir / "base" / "resume_checkpoint"
    best = run_dir / "base" / "best_checkpoint"
    resume.mkdir(parents=True)
    best.mkdir(parents=True)
    (resume / "model.pt").write_text("weights", encoding="utf-8")
    (resume / "training_state.pt").write_text("state", encoding="utf-8")
    (resume / "progress.md").write_text("# progress", encoding="utf-8")
    (best / "model.pt").write_text("best", encoding="utf-8")
    (run_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (run_dir / "preflight.md").write_text("# preflight", encoding="utf-8")
    (run_dir / "corpus.txt").write_text("large corpus", encoding="utf-8")
    (run_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "train.log").write_text("step 1", encoding="utf-8")
    bundle = tmp_path / "partial.tgz"

    exit_code = main([
        "run",
        "bundle",
        "--run-dir",
        str(run_dir),
        "--out",
        str(bundle),
        "--logs-dir",
        str(logs_dir),
        "--strict",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "bundle:" in output
    assert "excluded_large: corpus.txt, corpus_manifest.json" in output
    manifest_path = tmp_path / "partial.tgz.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["included_file_count"] >= 5
    assert manifest["excluded_large"] == ["corpus.txt", "corpus_manifest.json"]
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
    assert "partial-run/base/resume_checkpoint/model.pt" in names
    assert "partial-run/base/resume_checkpoint/training_state.pt" in names
    assert "partial-run/base/best_checkpoint/model.pt" in names
    assert "partial-run/corpus.txt" not in names
    assert "partial.tgz.manifest.json" in names


def test_cli_run_inspect_bundle_reports_resume_checkpoint(tmp_path, capsys):
    source = tmp_path / "source"
    checkpoint = source / "partial-run" / "base" / "resume_checkpoint"
    checkpoint.mkdir(parents=True)
    metadata = {
        "step": 4125,
        "train_loss": 3.25,
        "checkpoint_kind": "resume",
        "has_training_state": True,
        "model_config": {
            "vocab_size": 8192,
            "context_size": 512,
            "n_layer": 16,
            "n_embd": 768,
            "n_head": 12,
            "n_kv_head": 4,
        },
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint / "model.pt").write_text("weights", encoding="utf-8")
    (checkpoint / "training_state.pt").write_text("state", encoding="utf-8")
    (source / "partial-run" / "tokenizer.json").write_text("{}", encoding="utf-8")
    bundle = tmp_path / "partial.tgz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(source / "partial-run", arcname="runs/partial-run")

    exit_code = main([
        "run",
        "inspect-bundle",
        "--bundle",
        str(bundle),
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "# Picochat Bundle Inspection" in output
    assert "`runs/partial-run`" in output
    assert "`runs/partial-run/base/resume_checkpoint`" in output
    assert "| `runs/partial-run/base/resume_checkpoint` | `base` | `resume` | 4125 | `True` | `True` |" in output
    assert "--base-resume-from runs/partial-run/base/resume_checkpoint" in output
    assert "excludes `corpus.txt`" in output


def test_cli_run_inspect_bundle_json(tmp_path, capsys):
    source = tmp_path / "source"
    checkpoint = source / "run" / "sft" / "resume_checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({
        "step": 80,
        "train_loss": 0.1,
        "checkpoint_kind": "resume",
        "has_training_state": True,
        "model_config": {"vocab_size": 128, "context_size": 16, "n_layer": 1, "n_embd": 8, "n_head": 1},
    }), encoding="utf-8")
    (checkpoint / "model.pt").write_text("weights", encoding="utf-8")
    (checkpoint / "training_state.pt").write_text("state", encoding="utf-8")
    bundle = tmp_path / "run.tgz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(source / "run", arcname="run")

    exit_code = main(["run", "inspect-bundle", "--bundle", str(bundle), "--json"])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["resume_capable_checkpoints"][0]["path"] == "run/sft/resume_checkpoint"
    assert report["resume_capable_checkpoints"][0]["step"] == 80
    assert report["manifest_found"] is False


def test_cli_run_inspect_bundle_requires_training_state_file_for_resume(tmp_path, capsys):
    source = tmp_path / "source"
    checkpoint = source / "run" / "base" / "resume_checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({
        "step": 12,
        "train_loss": 1.0,
        "checkpoint_kind": "resume",
        "has_training_state": True,
        "model_config": {"vocab_size": 128, "context_size": 16, "n_layer": 1, "n_embd": 8, "n_head": 1},
    }), encoding="utf-8")
    (checkpoint / "model.pt").write_text("weights", encoding="utf-8")
    bundle = tmp_path / "broken.tgz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(source / "run", arcname="run")

    exit_code = main(["run", "inspect-bundle", "--bundle", str(bundle), "--json"])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoints"][0]["metadata_has_training_state"] is True
    assert report["checkpoints"][0]["has_training_state"] is False
    assert report["resume_capable_checkpoints"] == []


def test_cli_registry_builds_markdown_json_and_release_card(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "registered"
    (run_dir / "base" / "best_checkpoint").mkdir(parents=True)
    (run_dir / "base" / "best_checkpoint" / "model.pt").write_text("weights", encoding="utf-8")
    (run_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({
        "preflight": {
            "status": "ready",
            "budget": {
                "estimated_parameters": 100,
                "base_planned_tokens": 2000,
            },
        },
        "eval": {"num_examples": 10, "num_passed": 6, "pass_rate": 0.6},
        "sft_fit": {"num_examples": 10, "num_passed": 8, "pass_rate": 0.8},
        "honesty": {"status": "ready"},
        "long_run_gate": {
            "status": "approved",
            "profile": "skill_release",
            "sft_fit_rate": 0.8,
        },
    }), encoding="utf-8")
    registry_md = tmp_path / "registry.md"
    registry_json = tmp_path / "registry.json"
    release_card = tmp_path / "release-card.md"

    exit_code = main([
        "registry",
        "--runs-dir",
        str(tmp_path / "runs"),
        "--out",
        str(registry_md),
        "--json-out",
        str(registry_json),
        "--release-card",
        str(release_card),
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Best registered run: registered" in output
    assert "# Picochat Model Registry" in registry_md.read_text(encoding="utf-8")
    assert json.loads(registry_json.read_text(encoding="utf-8"))["best_run"] == "registered"
    assert "# Picochat Release Card: registered" in release_card.read_text(encoding="utf-8")


def test_cli_eval_lm_harness_dry_run_writes_command(tmp_path, capsys):
    exit_code = main([
        "eval",
        "lm-harness",
        "--model-path",
        "exports/pico",
        "--tasks",
        "arc_easy,hellaswag",
        "--out-dir",
        str(tmp_path / "lm-eval"),
        "--device",
        "cuda:0",
        "--batch-size",
        "2",
        "--model-arg",
        "dtype=bfloat16",
        "--dry-run",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "lm_eval" in output
    assert "arc_easy,hellaswag" in output
    metadata = json.loads((tmp_path / "lm-eval" / "lm_eval_command.json").read_text(encoding="utf-8"))
    assert metadata["tasks"] == ["arc_easy", "hellaswag"]


def test_cli_data_preference_starter(tmp_path, capsys):
    chat = tmp_path / "chat.jsonl"
    out = tmp_path / "preferences.jsonl"
    chat.write_text(json.dumps({"user": "What are you?", "assistant": "Picochat.", "category": "identity"}) + "\n", encoding="utf-8")

    exit_code = main([
        "data",
        "preference-starter",
        "--input",
        str(chat),
        "--out",
        str(out),
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "preference starter:" in output
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["chosen"] == "Picochat."
    assert row["category"] == "identity_preference"


def test_cli_train_sft_sweep_uses_dataset_pack(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    chat = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack = tmp_path / "dataset_pack.json"
    corpus.write_text("base corpus\n", encoding="utf-8")
    chat.write_text(json.dumps({"user": "who", "assistant": "Picochat"}) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "who", "must_include": ["Picochat"]}) + "\n", encoding="utf-8")
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    def fake_run(config):
        assert config.input_path == str(chat)
        assert config.eval_input_path == str(eval_path)
        assert config.support_corpus_path == str(corpus)
        assert config.tokenizer_path == "tok.json"
        assert config.checkpoint_path == "base"
        assert config.eval_log_every == 50
        assert config.matmul_precision == "default"
        assert config.peft == "lora"
        assert config.lora_rank == 2
        assert config.lora_alpha == 4.0
        assert config.lora_targets == ("attn_qkv",)
        return {
            "best_sft_fit": {
                "candidate": "uniform-lr1em04-steps1",
                "sft_fit_pass_rate": 0.9,
            },
            "best_eval": {
                "candidate": "uniform-lr1em04-steps1",
                "eval_pass_rate": 0.4,
            },
        }

    monkeypatch.setattr("picochat.cli.run_sft_sweep", fake_run)

    exit_code = main([
        "train",
        "sft-sweep",
        "--dataset-pack",
        str(pack),
        "--tokenizer",
        "tok.json",
        "--checkpoint",
        "base",
        "--out-dir",
        str(tmp_path / "sweep"),
        "--peft",
        "lora",
        "--lora-rank",
        "2",
        "--lora-alpha",
        "4",
        "--lora-targets",
        "attn_qkv",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "sft sweep report:" in output
    assert "best sft fit: uniform-lr1em04-steps1 (90.00%)" in output
    assert "best eval: uniform-lr1em04-steps1 (40.00%)" in output


def test_cli_tokenizer_train(tmp_path, capsys):
    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("hello picochat", encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
    ])

    assert exit_code == 0
    assert tokenizer_path.exists()
    output = capsys.readouterr().out
    assert "trained tokenizer" in output
    assert "type: char" in output
    assert "vocab_size" in output


def test_cli_tokenizer_train_byte(tmp_path, capsys):
    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("hello café", encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
        "--type",
        "byte",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "type: byte" in output
    assert "vocab_size: 260" in output


def test_cli_tokenizer_train_bpe(tmp_path, capsys):
    import json

    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("picochat picochat learns small stories", encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
        "--type",
        "bpe",
        "--vocab-size",
        "32",
        "--min-freq",
        "2",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert "type: bpe" in output
    assert data["type"] == "bpe"
    assert data["merges"]


def test_cli_tokenizer_train_hf_bpe(tmp_path, capsys):
    import json
    import pytest

    pytest.importorskip("tokenizers")
    data_path = tmp_path / "data.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    data_path.write_text("picochat trains fast tokenizers\n" * 20, encoding="utf-8")

    exit_code = main([
        "tok",
        "train",
        "--input",
        str(data_path),
        "--out",
        str(tokenizer_path),
        "--type",
        "hf_bpe",
        "--vocab-size",
        "300",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert "type: hf_bpe" in output
    assert data["type"] == "hf_bpe"
    assert data["backend"] == "huggingface_tokenizers"


def test_cli_data_inspect(tmp_path, capsys):
    data_path = tmp_path / "data.txt"
    data_path.write_text("hello\npicochat\n", encoding="utf-8")

    exit_code = main(["data", "inspect", "--input", str(data_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "num_documents" in output
    assert "num_characters" in output


def test_cli_data_build(tmp_path, capsys):
    input_dir = tmp_path / "input"
    output_path = tmp_path / "corpus.txt"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha", encoding="utf-8")

    exit_code = main(["data", "build", "--input", str(input_dir), "--out", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert (output_path.parent / "corpus_manifest.json").exists()
    output = capsys.readouterr().out
    assert "built corpus" in output
    assert "manifest:" in output


def test_cli_data_build_from_recipe(tmp_path, capsys):
    import json

    source_path = tmp_path / "lesson.txt"
    recipe_path = tmp_path / "corpus.recipe.json"
    output_path = tmp_path / "corpus.txt"
    source_path.write_text("lesson text", encoding="utf-8")
    recipe_path.write_text(json.dumps({
        "sources": [
            {"path": "lesson.txt", "label": "lesson"},
        ],
    }), encoding="utf-8")

    exit_code = main(["data", "build", "--recipe", str(recipe_path), "--out", str(output_path)])

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "lesson text\n"
    manifest = json.loads((output_path.parent / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recipe_path"] == str(recipe_path)
    assert manifest["files"][0]["label"] == "lesson"
    output = capsys.readouterr().out
    assert "built corpus" in output


def test_cli_run_tiny_preflight_only(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    chat = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pack = tmp_path / "dataset_pack.json"
    corpus.write_text("Picochat trains tiny local models.\n" * 20, encoding="utf-8")
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(tmp_path / "run"),
        "--dataset-pack",
        str(pack),
        "--preflight-only",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Picochat Run Preflight" in output
    assert "Recommended base steps" in output
    assert "assistant_only_masking" in output


def test_cli_data_skills_corpus(tmp_path, capsys):
    base_path = tmp_path / "base.txt"
    out_path = tmp_path / "skills.txt"
    docs_path = tmp_path / "skills_docs"
    recipe_path = tmp_path / "skills_recipe.json"
    base_path.write_text("base text", encoding="utf-8")

    exit_code = main([
        "data",
        "skills-corpus",
        "--out",
        str(out_path),
        "--math-rows",
        "3",
        "--spelling-rows",
        "3",
        "--choice-rows",
        "2",
        "--base-corpus",
        str(base_path),
        "--recipe-out",
        str(recipe_path),
        "--documents-dir",
        str(docs_path),
        "--rows-per-shard",
        "4",
    ])

    assert exit_code == 0
    assert out_path.exists()
    assert len(list(docs_path.glob("shard-*.txt"))) == 2
    assert recipe_path.exists()
    output = capsys.readouterr().out
    assert "skills corpus:" in output
    assert "documents_dir:" in output
    assert "shards: 2" in output
    assert "skills_math" in output
    assert "recipe:" in output


def test_cli_data_benchmark_pack(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains tiny local models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "corpus": str(corpus),
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "benchmark-pack",
        "--dataset-pack",
        str(pack),
        "--sft-rows",
        "32",
        "--eval-rows",
        "16",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "benchmark chat SFT:" in output
    assert "benchmark eval:" in output
    assert "source_status: offline" in output
    assert "profile: full" in output
    assert "skill_answer_style: direct" in output
    assert "contamination: ready" in output
    assert "promoted_to_pack: True" in output
    assert (tmp_path / "chat_benchmark.jsonl").exists()
    assert (tmp_path / "eval_benchmark.jsonl").exists()


def test_cli_data_task_pack(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains tiny local models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"user": "hi", "assistant": "hello"}) + "\n", encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"user": "hi", "must_include": ["hello"]}) + "\n", encoding="utf-8")
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "corpus": str(corpus),
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "task-pack",
        "--dataset-pack",
        str(pack),
        "--sft-rows",
        "64",
        "--eval-rows",
        "24",
        "--profile",
        "capability",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "task-mixture chat SFT:" in output
    assert "task-mixture eval:" in output
    assert "profile: capability" in output
    assert "contamination:" in output
    assert "promoted_to_pack: True" in output
    assert "weak_skills" in output
    assert "behavior_anchor" in output
    assert (tmp_path / "chat_task_mixture_capability.jsonl").exists()
    assert (tmp_path / "eval_task_mixture_capability.jsonl").exists()


def test_cli_data_slice_pack(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains tiny local models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        json.dumps({"user": "who", "assistant": "Picochat", "category": "identity"})
        + "\n"
        + json.dumps({"user": "add", "assistant": "4", "category": "bench_math_addition"})
        + "\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"user": "who", "must_include": ["Picochat"], "category": "identity"})
        + "\n"
        + json.dumps({"user": "add", "must_include": ["4"], "category": "bench_math_addition"})
        + "\n",
        encoding="utf-8",
    )
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "slice-pack",
        "--dataset-pack",
        str(pack),
        "--out-dir",
        str(tmp_path / "identity-pack"),
        "--include-categories",
        "identity",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "sliced dataset pack:" in output
    assert "chat rows: 1/2" in output
    assert "eval rows: 1/2" in output
    assert "- identity: 1" in output
    assert (tmp_path / "identity-pack" / "dataset_pack.json").exists()
    assert (tmp_path / "identity-pack" / "tuning_slice.md").exists()


def test_cli_data_stage_pack(tmp_path, capsys):
    import json

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Picochat trains tiny local models.\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        json.dumps({"user": "who", "assistant": "Picochat", "category": "identity"})
        + "\n"
        + json.dumps({"user": "unknown", "assistant": "I do not know.", "category": "refusal"})
        + "\n"
        + json.dumps({"user": "choice", "assistant": "A", "category": "bench_choice_language"})
        + "\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"user": "who", "must_include": ["Picochat"], "category": "identity"})
        + "\n"
        + json.dumps({"user": "unknown", "must_include": ["I do not know"], "category": "refusal"})
        + "\n"
        + json.dumps({"user": "choice", "must_include": ["A"], "category": "bench_choice_language"})
        + "\n",
        encoding="utf-8",
    )
    pack = tmp_path / "dataset_pack.json"
    pack.write_text(json.dumps({
        "corpus": "corpus.txt",
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "stage-pack",
        "--dataset-pack",
        str(pack),
        "--out-dir",
        str(tmp_path / "staged"),
        "--stages",
        "behavior,choice",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "staged tuning pack:" in output
    assert "- behavior: chat 2/3, eval 2/3" in output
    assert "- choice: chat 1/3, eval 1/3" in output
    assert (tmp_path / "staged" / "staged_tuning_pack.md").exists()
    assert (tmp_path / "staged" / "behavior" / "dataset_pack.json").exists()
    assert (tmp_path / "staged" / "choice" / "dataset_pack.json").exists()


def test_cli_data_preview_from_recipe(tmp_path, capsys):
    import json

    source_path = tmp_path / "lesson.txt"
    recipe_path = tmp_path / "corpus.recipe.json"
    source_path.write_text("lesson text", encoding="utf-8")
    recipe_path.write_text(json.dumps({
        "sources": [
            {"path": "lesson.txt", "label": "lesson"},
        ],
    }), encoding="utf-8")

    exit_code = main([
        "data",
        "preview",
        "--recipe",
        str(recipe_path),
        "--preview-chars",
        "6",
        "--chat-input",
        "domain/chat.jsonl",
        "--eval-input",
        "domain/eval.jsonl",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "source plan:" in output
    assert "readiness:" in output
    assert "readiness checks:" in output
    assert "training budget:" in output
    assert "suggested_context_size" in output
    assert "suggested run:" in output
    assert "run tiny" in output
    assert "chat_input: domain/chat.jsonl" in output
    assert "eval_input: domain/eval.jsonl" in output
    assert "--chat-input domain/chat.jsonl" in output
    assert "chat/eval data:" in output
    assert "chat_sft: blocked" in output
    assert "eval: blocked" in output
    assert "include" in output
    assert "score=" in output
    assert "label=lesson" in output
    assert "lesson" in output
    assert not (tmp_path / "corpus_manifest.json").exists()


def test_cli_data_preview_limits_large_source_plan(tmp_path, capsys):
    for index in range(30):
        (tmp_path / f"doc-{index:02d}.txt").write_text(f"document {index} text", encoding="utf-8")

    exit_code = main([
        "data",
        "preview",
        "--input",
        str(tmp_path),
        "--preview-chars",
        "0",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "25 more" not in output
    assert "5 more source file(s) omitted" in output
    assert "include" in output
    assert "score=" in output
    assert "doc-00.txt" in output
    assert "doc-29.txt" not in output
    assert not (tmp_path / "corpus_manifest.json").exists()


def test_cli_data_preview_from_dataset_pack(capsys):
    exit_code = main([
        "data",
        "preview",
        "--dataset-pack",
        "examples/tiny_dataset_pack.json",
        "--preview-chars",
        "6",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "dataset_pack: examples/tiny_dataset_pack.json" in output
    assert "--dataset-pack examples/tiny_dataset_pack.json" in output
    assert "chat_sft: ready" in output
    assert "eval: ready" in output


def test_cli_data_init_pack_creates_starter_pack(tmp_path, capsys):
    corpus_dir = tmp_path / "docs"
    out_dir = tmp_path / "pack"
    corpus_dir.mkdir()

    exit_code = main([
        "data",
        "init-pack",
        "--name",
        "lesson-pack",
        "--corpus",
        str(corpus_dir),
        "--out",
        str(out_dir),
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "initialized dataset pack:" in output
    assert "data preview --dataset-pack" in output
    assert (out_dir / "dataset_pack.json").exists()
    assert (out_dir / "corpus_recipe.json").exists()
    assert (out_dir / "chat.jsonl").exists()
    assert (out_dir / "eval.jsonl").exists()


def test_cli_data_eval_starter(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.txt"
    out_path = tmp_path / "eval.jsonl"
    corpus_path.write_text(
        "Pico Cafe roasts careful beans for morning guests. "
        "Mira Chen founded Pico Cafe beside the train station.",
        encoding="utf-8",
    )

    exit_code = main([
        "data",
        "eval-starter",
        "--input",
        str(corpus_path),
        "--out",
        str(out_path),
        "--max-items",
        "6",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "eval starter:" in output
    assert out_path.exists()
    assert (tmp_path / "eval.md").exists()


def test_cli_data_hf_import_uses_importer(tmp_path, capsys, monkeypatch):
    from picochat.hf_import import HFImportReport

    captured = {}

    def fake_import(config):
        captured["config"] = config
        return HFImportReport(
            dataset=config.dataset,
            config_name=config.config_name,
            split=config.split,
            text_column=config.text_column,
            streaming=config.streaming,
            max_rows=config.max_rows,
            min_chars=config.min_chars,
            out_path=config.out_path,
            report_path=config.report_path or str(tmp_path / "hf_import_report.json"),
            documents_dir=config.documents_dir or str(tmp_path / "documents"),
            document_shard_rows=config.document_shard_rows,
            document_files_written=2,
            rows_seen=3,
            rows_written=2,
            rows_skipped=1,
            characters_written=42,
            rows=(),
        )

    monkeypatch.setattr("picochat.cli.import_hf_dataset", fake_import)

    exit_code = main([
        "data",
        "hf-import",
        "--dataset",
        "demo/dataset",
        "--config",
        "plain",
        "--split",
        "train",
        "--text-column",
        "body",
        "--out",
        str(tmp_path / "corpus.txt"),
        "--documents-dir",
        str(tmp_path / "docs"),
        "--max-rows",
        "3",
        "--min-chars",
        "5",
        "--document-shard-rows",
        "2",
        "--no-streaming",
    ])

    assert exit_code == 0
    assert captured["config"].dataset == "demo/dataset"
    assert captured["config"].config_name == "plain"
    assert captured["config"].text_column == "body"
    assert captured["config"].streaming is False
    assert captured["config"].documents_dir == str(tmp_path / "docs")
    assert captured["config"].document_shard_rows == 2
    output = capsys.readouterr().out
    assert "imported dataset: demo/dataset" in output
    assert "rows_written: 2" in output
    assert f"documents_dir: {tmp_path / 'docs'}" in output
    assert "document_files_written: 2" in output


def test_cli_climbmix_import_auto_shards_large_imports(tmp_path, capsys, monkeypatch):
    from picochat.hf_import import HFImportReport

    captured = {}

    def fake_import(config):
        captured["config"] = config
        return HFImportReport(
            dataset=config.dataset,
            config_name=config.config_name,
            split=config.split,
            text_column=config.text_column,
            streaming=config.streaming,
            max_rows=config.max_rows,
            min_chars=config.min_chars,
            out_path=config.out_path,
            report_path=config.report_path,
            documents_dir=config.documents_dir,
            document_shard_rows=config.document_shard_rows,
            document_files_written=800,
            rows_seen=800000,
            rows_written=796000,
            rows_skipped=4000,
            characters_written=123,
            rows=(),
            data_files=tuple(config.data_files),
        )

    def fake_pack(**kwargs):
        return SimpleNamespace(
            dataset_pack=str(tmp_path / "climbmix" / "dataset_pack.json"),
            corpus_recipe=str(tmp_path / "climbmix" / "corpus_recipe.json"),
        )

    monkeypatch.setattr("picochat.cli.import_hf_dataset", fake_import)
    monkeypatch.setattr("picochat.cli.init_dataset_pack", fake_pack)

    exit_code = main([
        "data",
        "climbmix-import",
        "--out-dir",
        str(tmp_path / "climbmix"),
        "--shards",
        "170",
        "--max-rows",
        "800000",
        "--min-chars",
        "100",
        "--force",
    ])

    assert exit_code == 0
    assert captured["config"].document_shard_rows == 1000
    assert len(captured["config"].data_files) == 170
    output = capsys.readouterr().out
    assert "document_shard_rows: 1000" in output
    assert "document_files_written: 800" in output


def test_cli_climbmix_import_preserves_explicit_row_documents(tmp_path, monkeypatch):
    from picochat.hf_import import HFImportReport

    captured = {}

    def fake_import(config):
        captured["config"] = config
        return HFImportReport(
            dataset=config.dataset,
            config_name=config.config_name,
            split=config.split,
            text_column=config.text_column,
            streaming=config.streaming,
            max_rows=config.max_rows,
            min_chars=config.min_chars,
            out_path=config.out_path,
            report_path=config.report_path,
            documents_dir=config.documents_dir,
            document_shard_rows=config.document_shard_rows,
            document_files_written=2,
            rows_seen=2,
            rows_written=2,
            rows_skipped=0,
            characters_written=42,
            rows=(),
            data_files=tuple(config.data_files),
        )

    monkeypatch.setattr("picochat.cli.import_hf_dataset", fake_import)
    monkeypatch.setattr(
        "picochat.cli.init_dataset_pack",
        lambda **_kwargs: SimpleNamespace(
            dataset_pack=str(tmp_path / "climbmix" / "dataset_pack.json"),
            corpus_recipe=str(tmp_path / "climbmix" / "corpus_recipe.json"),
        ),
    )

    exit_code = main([
        "data",
        "climbmix-import",
        "--out-dir",
        str(tmp_path / "climbmix"),
        "--max-rows",
        "800000",
        "--document-shard-rows",
        "1",
        "--force",
    ])

    assert exit_code == 0
    assert captured["config"].document_shard_rows == 1


def test_cli_demo_uses_default_pipeline(tmp_path, capsys, monkeypatch):
    captured = {}

    def fake_run_tiny(config):
        captured["config"] = config
        return {
            "eval": {
                "num_passed": 6,
                "num_examples": 6,
                "pass_rate": 1.0,
            },
        }

    monkeypatch.setattr("picochat.cli.run_tiny", fake_run_tiny)

    exit_code = main(["demo", "--out-dir", str(tmp_path / "demo")])

    assert exit_code == 0
    assert captured["config"].out_dir == str(tmp_path / "demo")
    output = capsys.readouterr().out
    assert "demo run: 6/6 passed" in output
    assert "open workbench" in output


def test_cli_batch_inspect(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    corpus_path.write_text("hello picochat", encoding="utf-8")
    from picochat.tokenizer import CharTokenizer

    CharTokenizer.train([corpus_path.read_text(encoding="utf-8")]).save(tokenizer_path)

    exit_code = main([
        "batch",
        "inspect",
        "--corpus",
        str(corpus_path),
        "--tokenizer",
        str(tokenizer_path),
        "--context-size",
        "4",
        "--examples",
        "1",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "num_sequences" in output
    assert "example 0" in output


def test_cli_generate(tmp_path, capsys):
    from picochat.checkpoint import save_checkpoint
    from picochat.model import GPTConfig, TinyGPT
    from picochat.tokenizer import CharTokenizer

    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    tokenizer = CharTokenizer.train(["hello"])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    exit_code = main([
        "generate",
        "--checkpoint",
        str(checkpoint_path),
        "--tokenizer",
        str(tokenizer_path),
        "--prompt",
        "he",
        "--max-new-tokens",
        "2",
        "--temperature",
        "0",
    ])

    assert exit_code == 0
    assert capsys.readouterr().out.startswith("he")


def test_cli_train_sft(tmp_path, capsys):
    import json

    from picochat.checkpoint import save_checkpoint
    from picochat.model import GPTConfig, TinyGPT
    from picochat.tokenizer import CharTokenizer

    input_path = tmp_path / "chat.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "base"
    out_dir = tmp_path / "sft"
    rows = [
        {"user": "What is Picochat?", "assistant": "Picochat is small."},
        {"user": "What next?", "assistant": "Train chat format."},
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    tokenizer = CharTokenizer.train([
        "User: What is Picochat?\nAssistant: Picochat is small.\n"
        "User: What next?\nAssistant: Train chat format."
    ])
    tokenizer.save(tokenizer_path)
    model = TinyGPT(GPTConfig(
        vocab_size=len(tokenizer),
        context_size=64,
        n_embd=16,
        n_head=4,
        n_layer=1,
    ))
    save_checkpoint(checkpoint_path, model, step=0, train_loss=0.0)

    exit_code = main([
        "train",
        "sft",
        "--input",
        str(input_path),
        "--tokenizer",
        str(tokenizer_path),
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(out_dir),
        "--max-steps",
        "1",
        "--log-every",
        "1",
        "--sample-tokens",
        "2",
        "--sampling",
        "category_balanced",
        "--sft-packing",
        "bos_bestfit",
        "--peft",
        "lora",
        "--lora-rank",
        "2",
        "--lora-alpha",
        "4",
        "--lora-targets",
        "attn_qkv",
    ])

    assert exit_code == 0
    assert (out_dir / "checkpoint" / "model.pt").exists()
    assert (out_dir / "checkpoint" / "adapter_model.pt").exists()
    report = json.loads((out_dir / "sft_report.json").read_text(encoding="utf-8"))
    assert report["dataset"]["sampling"] == "category_balanced"
    assert report["dataset"]["packing"] == "bos_bestfit"
    assert report["config"]["peft"]["mode"] == "lora"
    assert "saved sft checkpoint" in capsys.readouterr().out


def test_cli_train_dpo_builds_config(tmp_path, capsys, monkeypatch):
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {
            "checkpoint": str(tmp_path / "dpo" / "checkpoint"),
            "best_checkpoint": {"path": str(tmp_path / "dpo" / "best_checkpoint")},
        }

    monkeypatch.setattr("picochat.cli.train_dpo", fake_train)

    exit_code = main([
        "train",
        "dpo",
        "--input",
        "prefs.jsonl",
        "--tokenizer",
        "tokenizer.json",
        "--checkpoint",
        "sft/checkpoint",
        "--reference-checkpoint",
        "sft/reference",
        "--out-dir",
        str(tmp_path / "dpo"),
        "--batch-size",
        "2",
        "--max-steps",
        "3",
        "--learning-rate",
        "0.000005",
        "--beta",
        "0.2",
        "--precision",
        "bf16",
        "--length-normalize",
    ])

    assert exit_code == 0
    config = captured["config"]
    assert config.input_path == "prefs.jsonl"
    assert config.reference_checkpoint_path == "sft/reference"
    assert config.batch_size == 2
    assert config.max_steps == 3
    assert config.learning_rate == 0.000005
    assert config.beta == 0.2
    assert config.precision == "bf16"
    assert config.length_normalize is True
    output = capsys.readouterr().out
    assert "saved dpo checkpoint" in output
    assert "best dpo checkpoint" in output


def test_cli_eval_chat(tmp_path, capsys):
    import json

    from picochat.checkpoint import save_checkpoint
    from picochat.model import GPTConfig, TinyGPT
    from picochat.tokenizer import CharTokenizer

    input_path = tmp_path / "eval.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "eval"
    input_path.write_text(json.dumps({"user": "hi"}), encoding="utf-8")
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

    exit_code = main([
        "eval",
        "chat",
        "--input",
        str(input_path),
        "--checkpoint",
        str(checkpoint_path),
        "--tokenizer",
        str(tokenizer_path),
        "--out-dir",
        str(out_dir),
        "--max-new-tokens",
        "0",
    ])

    assert exit_code == 0
    assert (out_dir / "eval_report.json").exists()
    assert "chat eval: 1/1 passed" in capsys.readouterr().out


def test_cli_run_tiny(tmp_path, capsys):
    import json

    corpus_path = tmp_path / "corpus.txt"
    chat_path = tmp_path / "chat.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    out_dir = tmp_path / "tiny"
    corpus_path.write_text(
        "Picochat is small.\nUser: hi\nAssistant: hello\n" * 8,
        encoding="utf-8",
    )
    chat_path.write_text(json.dumps({"user": "hi", "assistant": "hello"}), encoding="utf-8")
    eval_path.write_text(json.dumps({"user": "hi"}), encoding="utf-8")

    exit_code = main([
        "run",
        "tiny",
        "--out-dir",
        str(out_dir),
        "--corpus-input",
        str(corpus_path),
        "--chat-input",
        str(chat_path),
        "--eval-input",
        str(eval_path),
        "--context-size",
        "32",
        "--n-embd",
        "16",
        "--n-layer",
        "1",
        "--base-steps",
        "1",
        "--sft-steps",
        "1",
        "--base-batch-size",
        "2",
        "--base-dataset-mode",
        "sharded",
        "--base-shard-token-size",
        "64",
        "--base-shard-cache-size",
        "2",
        "--sft-batch-size",
        "1",
        "--sft-sampling",
        "category_balanced",
        "--sft-packing",
        "bos_bestfit",
        "--eval-max-new-tokens",
        "0",
        "--tokenizer-type",
        "byte",
        "--base-optimizer",
        "muon",
        "--sft-optimizer",
        "muon",
        "--base-muon-learning-rate",
        "0.01",
        "--sft-muon-learning-rate",
        "0.01",
        "--base-ema-decay",
        "0.5",
        "--sft-ema-decay",
        "0.5",
        "--attn-backend",
        "math",
        "--parallel-residual",
        "--allow-leaky-eval",
    ])

    assert exit_code == 0
    assert (out_dir / "summary.md").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["tokenizer_type"] == "byte"
    assert summary["config"]["base_dataset_mode"] == "sharded"
    assert summary["config"]["base_shard_token_size"] == 64
    assert summary["config"]["base_shard_cache_size"] == 2
    assert summary["config"]["sft_sampling"] == "category_balanced"
    assert summary["config"]["sft_packing"] == "bos_bestfit"
    assert summary["config"]["attn_backend"] == "math"
    assert summary["config"]["parallel_residual"] is True
    assert summary["config"]["base_optimizer"] == "muon"
    assert summary["config"]["sft_optimizer"] == "muon"
    assert summary["base"]["best_checkpoint"]["weights"] == "ema"
    assert summary["sft"]["best_checkpoint"]["weights"] == "ema"
    assert (out_dir / "base" / "ema_checkpoint" / "model.pt").exists()
    assert (out_dir / "sft" / "ema_checkpoint" / "model.pt").exists()
    assert summary["tokenizer"]["tokenizer_type"] == "byte"
    assert "tiny run: 1/1 passed" in capsys.readouterr().out


def test_cli_compare(tmp_path, capsys):
    import json

    run_a = tmp_path / "tiny-a"
    run_b = tmp_path / "tiny-b"
    out_path = tmp_path / "compare.md"
    for run_dir, passed, total in [(run_a, 3, 4), (run_b, 6, 6)]:
        run_dir.mkdir()
        summary = {
            "config": {"context_size": 128},
            "base": {
                "final_val_loss": 2.0,
                "num_parameters": 114609,
            },
            "sft": {
                "final_val_loss": 4.0,
                "truncated_examples": 0,
            },
            "eval": {
                "num_examples": total,
                "num_passed": passed,
                "num_failed": total - passed,
                "pass_rate": passed / total,
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main([
        "compare",
        str(run_a),
        str(run_b),
        "--out",
        str(out_path),
    ])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "tiny-a" in output
    assert "tiny-b" in output
    assert "Best eval run: tiny-b" in output
    assert "Champion gate:" in output
    assert out_path.exists()
