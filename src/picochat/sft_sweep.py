"""Controlled SFT sweeps for diagnosing chat fine-tuning quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from picochat.eval import ChatEvalConfig, run_chat_eval, write_sft_fit_eval
from picochat.sft import SFTConfig, SFT_PACKING_MODES, SFT_SAMPLING_MODES, train_sft


@dataclass(frozen=True)
class SFTSweepConfig:
    input_path: str
    tokenizer_path: str
    checkpoint_path: str
    out_dir: str
    eval_input_path: str | None = None
    support_corpus_path: str | None = None
    learning_rates: tuple[float, ...] = (3e-5, 5e-5, 1e-4)
    step_counts: tuple[int, ...] = (160, 400, 800)
    samplings: tuple[str, ...] = ("category_sqrt",)
    batch_size: int = 8
    seed: int = 42
    device: str = "cpu"
    eval_max_new_tokens: int = 120
    fit_max_rows: int = 500
    val_fraction: float = 0.2
    eval_batches: int = 10
    sample_prompt: str = "What is Picochat?"
    sample_tokens: int = 120
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    max_minutes: float | None = None
    lr_warmup_steps: int = 20
    lr_decay: str = "cosine"
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    weight_decay_decay: str = "none"
    muon_learning_rate: float = 0.02
    muon_momentum_schedule: str = "none"
    ema_decay: float = 0.0
    packing: str = "separate"
    precision: str = "float32"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    eval_log_every: int = 50


def run_sft_sweep(config: SFTSweepConfig) -> dict:
    """Train and score a small grid of SFT candidates from one base checkpoint."""
    _validate_sweep_config(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_candidates: set[str] = set()
    for sampling in config.samplings:
        for learning_rate in config.learning_rates:
            for step_count in config.step_counts:
                candidate = _candidate_name(
                    sampling=sampling,
                    learning_rate=learning_rate,
                    step_count=step_count,
                )
                if candidate in seen_candidates:
                    raise ValueError(
                        "sweep candidate name collision; adjust candidate naming "
                        f"for sampling={sampling}, learning_rate={learning_rate:g}, "
                        f"steps={step_count}"
                    )
                seen_candidates.add(candidate)
                candidate_dir = out_dir / candidate
                print(
                    f"sft sweep candidate: {candidate} "
                    f"| lr {learning_rate:g} | steps {step_count} | sampling {sampling}"
                )
                row = _run_candidate(
                    config,
                    candidate=candidate,
                    candidate_dir=candidate_dir,
                    sampling=sampling,
                    learning_rate=learning_rate,
                    step_count=step_count,
                )
                rows.append(row)

    report = {
        "config": asdict(config),
        "rows": rows,
        "best_sft_fit": _best_row(rows, "sft_fit_pass_rate"),
        "best_eval": _best_row(rows, "eval_pass_rate"),
        "best_non_choice_eval": _best_row(rows, "eval_non_choice_pass_rate"),
    }
    (out_dir / "sft_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "sft_sweep.md").write_text(sft_sweep_markdown(report), encoding="utf-8")
    return report


def sft_sweep_markdown(report: dict) -> str:
    """Render a compact SFT sweep report."""
    config = report["config"]
    rows = report["rows"]
    lines = [
        "# Picochat SFT Sweep",
        "",
        "This report compares SFT schedules from the same base checkpoint. "
        "Use SFT fit first: if exact-fit is low, held-out eval is not yet the main blocker.",
        "",
        "## Inputs",
        "",
        f"- Chat SFT: `{config['input_path']}`",
        f"- Tokenizer: `{config['tokenizer_path']}`",
        f"- Base checkpoint: `{config['checkpoint_path']}`",
        f"- Held-out eval: `{config.get('eval_input_path') or 'not run'}`",
        f"- SFT packing: `{config.get('packing', 'separate')}`",
        f"- Precision: `{config.get('precision', 'float32')}`",
        f"- torch.compile: `{config.get('torch_compile', False)}`",
        "",
        "## Results",
        "",
        "| Candidate | LR | Steps | Sampling | Packing | SFT Fit | Eval | NonChoice | Choice | Refusal | SFT Val BPB | Stop | Path |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | {row['learning_rate']:g} | {row['step_count']} | "
            f"`{row['sampling']}` | `{row.get('packing', 'separate')}` | {_percent(row.get('sft_fit_pass_rate'))} | "
            f"{_percent(row.get('eval_pass_rate'))} | {_percent(row.get('eval_non_choice_pass_rate'))} | "
            f"{_percent(row.get('eval_choice_pass_rate'))} | {_percent(row.get('eval_refusal_pass_rate'))} | "
            f"{_float_or_dash(row.get('sft_final_val_bpb'))} | "
            f"`{row.get('stop_reason', 'unknown')}` | `{row['candidate_dir']}` |"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
        f"- Best SFT fit: `{_best_name(report.get('best_sft_fit'))}`",
        f"- Best held-out eval: `{_best_name(report.get('best_eval'))}`",
        f"- Best non-choice held-out eval: `{_best_name(report.get('best_non_choice_eval'))}`",
        "",
        "When choice accuracy is saturated, prefer the non-choice winner for transfer diagnosis. "
        "Aggregate eval can otherwise hide weak math, spelling, identity, or refusal behavior.",
        "",
        "If best SFT fit is below 70%, increase SFT fit with more steps, a simpler "
        "curriculum, or a narrower starter set before blaming the base model.",
        "",
    ])
    return "\n".join(lines)


def _run_candidate(
    config: SFTSweepConfig,
    *,
    candidate: str,
    candidate_dir: Path,
    sampling: str,
    learning_rate: float,
    step_count: int,
) -> dict:
    sft_dir = candidate_dir / "sft"
    sft_report = train_sft(SFTConfig(
        input_path=config.input_path,
        tokenizer_path=config.tokenizer_path,
        checkpoint_path=config.checkpoint_path,
        out_dir=str(sft_dir),
        batch_size=config.batch_size,
        max_steps=step_count,
        learning_rate=learning_rate,
        seed=config.seed,
        device=config.device,
        log_every=max(1, step_count // 12),
        val_fraction=config.val_fraction,
        eval_batches=config.eval_batches,
        sample_prompt=config.sample_prompt,
        sample_tokens=config.sample_tokens,
        early_stop_patience=config.early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        max_minutes=config.max_minutes,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay=config.lr_decay,
        min_lr_ratio=config.min_lr_ratio,
        grad_clip=config.grad_clip,
        sampling=sampling,
        grad_accum_steps=config.grad_accum_steps,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        weight_decay_decay=config.weight_decay_decay,
        muon_learning_rate=config.muon_learning_rate,
        muon_momentum_schedule=config.muon_momentum_schedule,
        ema_decay=config.ema_decay,
        packing=config.packing,
        precision=config.precision,
        torch_compile=config.torch_compile,
        torch_compile_mode=config.torch_compile_mode,
    ))
    eval_checkpoint = sft_report.get("best_checkpoint", {}).get(
        "path",
        str(sft_dir / "checkpoint"),
    )
    fit_dir = candidate_dir / "sft_fit"
    fit_input = fit_dir / "sft_fit_eval.jsonl"
    train_indices = sft_report.get("dataset", {}).get("train_indices")
    fit_dataset = write_sft_fit_eval(
        config.input_path,
        fit_input,
        max_rows=None if config.fit_max_rows <= 0 else config.fit_max_rows,
        include_indices=train_indices if isinstance(train_indices, list) else None,
        split_label="sft_train",
    )
    fit_report = run_chat_eval(ChatEvalConfig(
        input_path=str(fit_input),
        checkpoint_path=eval_checkpoint,
        tokenizer_path=config.tokenizer_path,
        out_dir=str(fit_dir),
        max_new_tokens=config.eval_max_new_tokens,
        seed=config.seed,
        device=config.device,
        support_corpus_path=config.support_corpus_path,
        log_every=config.eval_log_every,
    ))

    eval_report = None
    if config.eval_input_path:
        eval_report = run_chat_eval(ChatEvalConfig(
            input_path=config.eval_input_path,
            checkpoint_path=eval_checkpoint,
            tokenizer_path=config.tokenizer_path,
            out_dir=str(candidate_dir / "eval"),
            max_new_tokens=config.eval_max_new_tokens,
            seed=config.seed,
            device=config.device,
            support_corpus_path=config.support_corpus_path,
            log_every=config.eval_log_every,
        ))

    row = {
        "candidate": candidate,
        "candidate_dir": str(candidate_dir),
        "sft_dir": str(sft_dir),
        "checkpoint": eval_checkpoint,
        "learning_rate": learning_rate,
        "step_count": step_count,
        "sampling": sampling,
        "packing": config.packing,
        "sft_final_train_loss": sft_report["losses"][-1]["train_loss"],
        "sft_final_val_loss": sft_report["losses"][-1]["val_loss"],
        "sft_final_val_bpb": sft_report["losses"][-1].get("val_bpb"),
        "stop_reason": sft_report.get("stop_reason"),
        "actual_steps": sft_report.get("coverage", {}).get("actual_steps"),
        "sft_fit_examples": fit_dataset["num_rows"],
        "sft_fit_split": fit_dataset.get("split_label"),
        "sft_fit_selected_from_indices": fit_dataset.get("selected_from_indices"),
        "sft_fit_pass_rate": fit_report["summary"]["pass_rate"],
        "sft_fit_score": (
            f"{fit_report['summary']['num_passed']}/{fit_report['summary']['num_examples']}"
        ),
        "eval_pass_rate": eval_report["summary"]["pass_rate"] if eval_report else None,
        "eval_score": (
            f"{eval_report['summary']['num_passed']}/{eval_report['summary']['num_examples']}"
            if eval_report else None
        ),
        "eval_non_choice_examples": (
            eval_report["summary"].get("non_choice_examples") if eval_report else None
        ),
        "eval_non_choice_pass_rate": (
            eval_report["summary"].get("non_choice_pass_rate") if eval_report else None
        ),
        "eval_non_choice_score": (
            f"{eval_report['summary'].get('non_choice_passed', 0)}/"
            f"{eval_report['summary'].get('non_choice_examples', 0)}"
            if eval_report else None
        ),
        "eval_choice_pass_rate": (
            eval_report["summary"].get("choice_pass_rate") if eval_report else None
        ),
        "eval_refusal_pass_rate": (
            eval_report["summary"].get("refusal_pass_rate") if eval_report else None
        ),
        "eval_answerable_pass_rate": (
            eval_report["summary"].get("answerable_pass_rate") if eval_report else None
        ),
        "eval_domain_pass_rate": (
            eval_report["summary"].get("domain_pass_rate") if eval_report else None
        ),
    }
    (candidate_dir / "candidate_summary.json").write_text(
        json.dumps(row, indent=2),
        encoding="utf-8",
    )
    return row


def _validate_sweep_config(config: SFTSweepConfig) -> None:
    if not config.learning_rates:
        raise ValueError("at least one learning rate is required")
    if not config.step_counts:
        raise ValueError("at least one step count is required")
    if not config.samplings:
        raise ValueError("at least one sampling mode is required")
    if any(value <= 0 for value in config.learning_rates):
        raise ValueError("learning rates must be positive")
    if any(value <= 0 for value in config.step_counts):
        raise ValueError("step counts must be positive")
    bad_samplings = [value for value in config.samplings if value not in SFT_SAMPLING_MODES]
    if bad_samplings:
        raise ValueError(f"unsupported SFT sampling mode(s): {', '.join(bad_samplings)}")
    if config.packing not in SFT_PACKING_MODES:
        raise ValueError(f"unsupported SFT packing mode: {config.packing}")
    if config.fit_max_rows < 0:
        raise ValueError("fit_max_rows must be non-negative")


def _candidate_name(*, sampling: str, learning_rate: float, step_count: int) -> str:
    lr_text = _learning_rate_slug(learning_rate)
    return f"{sampling.replace('_', '-')}-lr{lr_text}-steps{step_count}"


def _learning_rate_slug(learning_rate: float) -> str:
    mantissa_text, exponent_text = f"{learning_rate:.6e}".split("e")
    mantissa_text = mantissa_text.rstrip("0").rstrip(".").replace(".", "p")
    exponent = int(exponent_text)
    sign = "m" if exponent < 0 else ""
    return f"{mantissa_text}e{sign}{abs(exponent):02d}"


def _best_row(rows: list[dict], metric: str) -> dict | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row[metric], -(row.get("sft_final_val_loss") or 0.0)))


def _best_name(row: dict | None) -> str:
    return str(row.get("candidate")) if row else "n/a"


def _percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}%"


def _float_or_dash(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"
