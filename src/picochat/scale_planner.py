"""Compute auditable scale recipes for larger Picochat runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from picochat.model import GPTConfig


_COUNT_SUFFIXES = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}


@dataclass(frozen=True)
class ScalePlan:
    name: str
    target_parameters: int
    estimated_parameters: int
    parameter_error_rate: float
    vocab_size: int
    context_size: int
    n_embd: int
    n_head: int
    n_kv_head: int
    n_layer: int
    head_dim: int
    activation: str
    norm_type: str
    position_encoding: str
    tie_embeddings: bool
    qk_norm: bool
    parallel_residual: bool
    linear_bias: bool
    global_batch_tokens: int
    per_device_batch_size: int
    grad_accum_steps: int
    world_size: int
    target_param_data_ratio: float
    target_training_tokens: int
    recommended_base_steps: int
    planned_training_tokens: int
    planned_token_param_ratio: float
    dataset_tokens: int | None
    estimated_epochs: float | None
    base_learning_rate: float
    batch_scaled_learning_rate: float
    sft_learning_rate: float
    base_warmup_steps: int
    sft_warmup_steps: int
    sft_steps: int
    tokenizer_type: str
    tokenizer_vocab_size: int
    bpe_pretokenizer: str
    base_dataset_mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def run_tiny_overrides(self) -> list[str]:
        parts = [
            "--device", "cuda",
            "--precision", "bf16",
            "--matmul-precision", "high",
            "--attn-backend", "flash",
            "--torch-compile",
            "--tokenizer-type", self.tokenizer_type,
            "--tokenizer-vocab-size", str(self.tokenizer_vocab_size),
            "--bpe-pretokenizer", self.bpe_pretokenizer,
            "--context-size", str(self.context_size),
            "--n-embd", str(self.n_embd),
            "--n-head", str(self.n_head),
            "--n-kv-head", str(self.n_kv_head),
            "--n-layer", str(self.n_layer),
            "--norm-type", self.norm_type,
            "--position-encoding", self.position_encoding,
            "--activation", self.activation,
            "--tie-embeddings",
            "--qk-norm",
            "--parallel-residual",
        ]
        if not self.linear_bias:
            parts.append("--no-linear-bias")
        parts.extend([
            "--base-dataset-mode", self.base_dataset_mode,
            "--base-steps", str(self.recommended_base_steps),
            "--base-batch-size", str(self.per_device_batch_size),
            "--base-grad-accum-steps", str(self.grad_accum_steps),
            "--base-learning-rate", _format_float(self.base_learning_rate),
            "--base-lr-warmup-steps", str(self.base_warmup_steps),
            "--base-lr-decay", "cosine",
            "--base-min-lr-ratio", "0.1",
            "--base-grad-clip", "1.0",
            "--sft-steps", str(self.sft_steps),
            "--sft-batch-size", str(self.per_device_batch_size),
            "--sft-grad-accum-steps", "4",
            "--sft-learning-rate", _format_float(self.sft_learning_rate),
            "--sft-lr-warmup-steps", str(self.sft_warmup_steps),
            "--sft-lr-decay", "cosine",
            "--sft-min-lr-ratio", "0.1",
            "--sft-grad-clip", "1.0",
            "--sft-packing", "bos_bestfit",
            "--sft-sampling", "category_sqrt",
            "--eval-max-new-tokens", "120",
            "--long-run-gate-profile", "first_release",
        ])
        if self.world_size > 1:
            parts.append("--ddp")
        return parts


def parse_count(value: str | int | float) -> int:
    """Parse compact counts such as 100m, 2.1b, or 524288."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().lower().replace("_", "").replace(",", "")
    if not text:
        raise ValueError("count cannot be empty")
    suffix = text[-1]
    if suffix in _COUNT_SUFFIXES:
        number = float(text[:-1])
        return int(round(number * _COUNT_SUFFIXES[suffix]))
    return int(text)


def plan_scale(
    *,
    target_parameters: int,
    dataset_tokens: int | None = None,
    depth: int | None = None,
    aspect_ratio: int = 48,
    head_dim: int = 64,
    vocab_size: int = 8192,
    context_size: int = 512,
    world_size: int = 1,
    per_device_batch_size: int = 8,
    grad_accum_steps: int = 16,
    target_param_data_ratio: float = 20.0,
    tokenizer_type: str = "hf_bpe",
    bpe_pretokenizer: str = "regex",
    activation: str = "swiglu",
    norm_type: str = "rmsnorm",
    position_encoding: str = "rope",
    tie_embeddings: bool = True,
    qk_norm: bool = True,
    parallel_residual: bool = True,
    linear_bias: bool = False,
    base_dataset_mode: str = "sharded",
) -> ScalePlan:
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if context_size <= 0 or per_device_batch_size <= 0 or grad_accum_steps <= 0 or world_size <= 0:
        raise ValueError("context, batch, grad accumulation, and world size must be positive")
    if target_param_data_ratio <= 0:
        raise ValueError("target_param_data_ratio must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even number")

    if depth is None:
        candidates = []
        for candidate_depth in range(4, 65):
            candidate_embd = _round_to_multiple(candidate_depth * aspect_ratio, head_dim)
            if candidate_embd < head_dim:
                continue
            candidate = _shape_for_depth(
                candidate_depth,
                candidate_embd,
                head_dim=head_dim,
                vocab_size=vocab_size,
                context_size=context_size,
                activation=activation,
                norm_type=norm_type,
                position_encoding=position_encoding,
                tie_embeddings=tie_embeddings,
                qk_norm=qk_norm,
                parallel_residual=parallel_residual,
                linear_bias=linear_bias,
            )
            candidates.append(candidate)
        shape = min(candidates, key=lambda item: abs(item["estimated_parameters"] - target_parameters))
    else:
        n_embd = _round_to_multiple(depth * aspect_ratio, head_dim)
        shape = _shape_for_depth(
            depth,
            n_embd,
            head_dim=head_dim,
            vocab_size=vocab_size,
            context_size=context_size,
            activation=activation,
            norm_type=norm_type,
            position_encoding=position_encoding,
            tie_embeddings=tie_embeddings,
            qk_norm=qk_norm,
            parallel_residual=parallel_residual,
            linear_bias=linear_bias,
        )

    estimated_parameters = int(shape["estimated_parameters"])
    global_batch_tokens = context_size * per_device_batch_size * grad_accum_steps * world_size
    target_training_tokens = int(round(estimated_parameters * target_param_data_ratio))
    recommended_base_steps = max(1, int(round(target_training_tokens / global_batch_tokens)))
    planned_training_tokens = recommended_base_steps * global_batch_tokens
    estimated_epochs = None
    if dataset_tokens:
        estimated_epochs = planned_training_tokens / dataset_tokens

    base_lr, batch_scaled_lr = _base_learning_rates(
        n_embd=int(shape["n_embd"]),
        global_batch_tokens=global_batch_tokens,
    )
    sft_steps = _recommended_sft_steps(world_size=world_size)
    sft_warmup = max(1, min(20, sft_steps // 4))
    name = f"picochat-{_compact_count(estimated_parameters)}-w{world_size}"
    return ScalePlan(
        name=name,
        target_parameters=target_parameters,
        estimated_parameters=estimated_parameters,
        parameter_error_rate=(estimated_parameters - target_parameters) / target_parameters,
        vocab_size=vocab_size,
        context_size=context_size,
        n_embd=int(shape["n_embd"]),
        n_head=int(shape["n_head"]),
        n_kv_head=int(shape["n_kv_head"]),
        n_layer=int(shape["n_layer"]),
        head_dim=head_dim,
        activation=activation,
        norm_type=norm_type,
        position_encoding=position_encoding,
        tie_embeddings=tie_embeddings,
        qk_norm=qk_norm,
        parallel_residual=parallel_residual,
        linear_bias=linear_bias,
        global_batch_tokens=global_batch_tokens,
        per_device_batch_size=per_device_batch_size,
        grad_accum_steps=grad_accum_steps,
        world_size=world_size,
        target_param_data_ratio=target_param_data_ratio,
        target_training_tokens=target_training_tokens,
        recommended_base_steps=recommended_base_steps,
        planned_training_tokens=planned_training_tokens,
        planned_token_param_ratio=planned_training_tokens / estimated_parameters,
        dataset_tokens=dataset_tokens,
        estimated_epochs=estimated_epochs,
        base_learning_rate=base_lr,
        batch_scaled_learning_rate=batch_scaled_lr,
        sft_learning_rate=min(2e-5, base_lr / 10),
        base_warmup_steps=max(1, int(round(recommended_base_steps * 0.045))),
        sft_warmup_steps=sft_warmup,
        sft_steps=sft_steps,
        tokenizer_type=tokenizer_type,
        tokenizer_vocab_size=vocab_size,
        bpe_pretokenizer=bpe_pretokenizer,
        base_dataset_mode=base_dataset_mode,
    )


def render_scale_plan_markdown(plan: ScalePlan) -> str:
    lines = [
        "# Picochat Scale Plan",
        "",
        "This is a recipe estimate, not a claim of model quality. Run preflight and a small sweep before spending on a full release run.",
        "",
        "## Model",
        "",
        f"- Name: `{plan.name}`",
        f"- Target parameters: {plan.target_parameters:,}",
        f"- Estimated parameters: {plan.estimated_parameters:,} ({plan.parameter_error_rate:+.2%})",
        f"- Shape: {plan.n_layer} layers, {plan.n_embd} embedding, {plan.n_head} query heads, {plan.n_kv_head} KV heads, {plan.head_dim} head dim",
        f"- Architecture: {plan.norm_type}, {plan.position_encoding}, {plan.activation}, tied embeddings={plan.tie_embeddings}, qk_norm={plan.qk_norm}, parallel_residual={plan.parallel_residual}, linear_bias={plan.linear_bias}",
        "",
        "## Training Budget",
        "",
        f"- Context size: {plan.context_size}",
        f"- World size: {plan.world_size}",
        f"- Per-device batch: {plan.per_device_batch_size}",
        f"- Gradient accumulation: {plan.grad_accum_steps}",
        f"- Global batch tokens: {plan.global_batch_tokens:,}",
        f"- Target token/parameter ratio: {plan.target_param_data_ratio:.2f}",
        f"- Target training tokens: {plan.target_training_tokens:,}",
        f"- Recommended base steps: {plan.recommended_base_steps:,}",
        f"- Planned training tokens: {plan.planned_training_tokens:,}",
        f"- Planned token/parameter ratio: {plan.planned_token_param_ratio:.2f}",
    ]
    if plan.dataset_tokens is not None:
        lines.append(f"- Estimated corpus tokens: {plan.dataset_tokens:,}")
        lines.append(f"- Estimated corpus epochs: {plan.estimated_epochs:.2f}" if plan.estimated_epochs is not None else "- Estimated corpus epochs: unknown")
    lines.extend([
        "",
        "## Optimizer Hints",
        "",
        f"- Conservative base LR: {_format_float(plan.base_learning_rate)}",
        f"- Batch-scaled LR candidate: {_format_float(plan.batch_scaled_learning_rate)}",
        f"- SFT LR: {_format_float(plan.sft_learning_rate)}",
        f"- Base warmup steps: {plan.base_warmup_steps:,}",
        f"- SFT steps: {plan.sft_steps:,}",
        f"- SFT warmup steps: {plan.sft_warmup_steps:,}",
        "",
        "## Run Tiny Overrides",
        "",
        "```bash",
        _wrap_command(["PYTHONPATH=src", "python", "-m", "picochat.cli", "run", "tiny", *plan.run_tiny_overrides()]),
        "```",
    ])
    if plan.world_size > 1:
        lines.extend([
            "",
            "For a real multi-GPU launch, use `torchrun --standalone --nproc_per_node "
            f"{plan.world_size}` and keep `--ddp` in the run command.",
        ])
    return "\n".join(lines) + "\n"


def _shape_for_depth(
    n_layer: int,
    n_embd: int,
    *,
    head_dim: int,
    vocab_size: int,
    context_size: int,
    activation: str,
    norm_type: str,
    position_encoding: str,
    tie_embeddings: bool,
    qk_norm: bool,
    parallel_residual: bool,
    linear_bias: bool,
) -> dict[str, int]:
    if n_embd % head_dim != 0:
        raise ValueError("n_embd must be divisible by head_dim")
    n_head = n_embd // head_dim
    n_kv_head = _kv_heads_for(n_head)
    config = GPTConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_layer=n_layer,
        norm_type=norm_type,
        position_encoding=position_encoding,
        activation=activation,
        tie_embeddings=tie_embeddings,
        qk_norm=qk_norm,
        parallel_residual=parallel_residual,
        linear_bias=linear_bias,
    )
    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "estimated_parameters": estimate_parameters(config),
    }


def estimate_parameters(config: GPTConfig) -> int:
    n = config.n_embd
    head_dim = config.n_embd // config.n_head
    kv_heads = config.n_kv_head or config.n_head
    kv_dim = kv_heads * head_dim
    total = config.vocab_size * n
    if config.position_encoding == "learned":
        total += config.context_size * n
    norm_params = n if config.norm_type == "rmsnorm" else 2 * n
    qk_norm_params = 2 * head_dim if config.qk_norm else 0
    bias = config.linear_bias
    if config.activation == "swiglu":
        hidden = max(1, int(8 * n / 3))
        mlp_params = n * (2 * hidden) + ((2 * hidden) if bias else 0)
        mlp_params += hidden * n + (n if bias else 0)
    else:
        hidden = 4 * n
        mlp_params = n * hidden + (hidden if bias else 0)
        mlp_params += hidden * n + (n if bias else 0)
    attn_params = n * (n + 2 * kv_dim) + ((n + 2 * kv_dim) if bias else 0)
    attn_params += n * n + (n if bias else 0)
    per_block_norms = norm_params if config.parallel_residual else 2 * norm_params
    block_params = per_block_norms + qk_norm_params + attn_params + mlp_params
    total += config.n_layer * block_params
    total += norm_params
    if not config.tie_embeddings:
        total += config.vocab_size * n + (config.vocab_size if bias else 0)
    return int(total)


def _kv_heads_for(n_head: int) -> int:
    divisors = [value for value in range(1, n_head + 1) if n_head % value == 0]
    target = max(1, n_head / 3)
    return min(divisors, key=lambda value: (abs(value - target), value))


def _base_learning_rates(*, n_embd: int, global_batch_tokens: int) -> tuple[float, float]:
    reference_lr = 2e-4
    reference_embd = 768
    reference_batch_tokens = 512 * 8 * 16
    dim_scale = math.sqrt(reference_embd / n_embd)
    batch_scale = math.sqrt(global_batch_tokens / reference_batch_tokens)
    candidate = reference_lr * dim_scale * batch_scale
    conservative = min(2e-4, candidate)
    return conservative, candidate


def _recommended_sft_steps(*, world_size: int) -> int:
    return max(24, int(round(180 / max(1, world_size))))


def _round_to_multiple(value: int, multiple: int) -> int:
    rounded = int(round(value / multiple) * multiple)
    return max(multiple, rounded)


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _compact_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}b".replace(".0b", "b")
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}m"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def _wrap_command(parts: list[str]) -> str:
    lines = []
    current = ""
    for part in parts:
        addition = part if not current else f" {part}"
        if current and len(current) + len(addition) > 88:
            lines.append(current + " \\")
            current = f"  {part}"
        else:
            current += addition
    if current:
        lines.append(current)
    return "\n".join(lines)
