"""Named local training scales for Picochat experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunScale:
    name: str
    label: str
    description: str
    tokenizer_type: str
    tokenizer_vocab_size: int | None
    tokenizer_min_freq: int
    context_size: int
    n_embd: int
    n_head: int
    n_layer: int
    norm_type: str
    position_encoding: str
    activation: str
    base_steps: int
    sft_steps: int
    base_batch_size: int
    sft_batch_size: int
    base_learning_rate: float
    sft_learning_rate: float
    base_lr_warmup_steps: int
    sft_lr_warmup_steps: int
    base_lr_decay: str
    sft_lr_decay: str
    base_min_lr_ratio: float
    sft_min_lr_ratio: float
    base_grad_clip: float
    sft_grad_clip: float
    base_grad_accum_steps: int
    sft_grad_accum_steps: int
    sft_sampling: str
    base_early_stop_patience: int
    sft_early_stop_patience: int
    canary_count: int
    eval_max_new_tokens: int
    sft_peft: str = "none"
    sft_lora_rank: int = 8
    sft_lora_alpha: float = 16.0
    sft_lora_dropout: float = 0.0
    sft_lora_targets: tuple[str, ...] = ("attn_qkv", "attn_proj")
    base_optimizer: str = "adamw"
    sft_optimizer: str = "adamw"
    base_muon_learning_rate: float = 0.02
    sft_muon_learning_rate: float = 0.02
    base_ema_decay: float = 0.0
    sft_ema_decay: float = 0.0
    n_kv_head: int | None = None
    tie_embeddings: bool = False
    qk_norm: bool = False
    attn_backend: str = "auto"
    parallel_residual: bool = False
    linear_bias: bool = True
    scaled_residual_init: bool = False
    bpe_pretokenizer: str = "regex"
    sft_packing: str = "separate"
    base_dataset_mode: str = "memory"
    base_shard_token_size: int = 1_000_000
    base_shard_cache_size: int = 2
    precision: str = "float32"
    matmul_precision: str = "default"
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    gradient_checkpointing: bool = False
    auto_lr_scaling: bool = False
    loss_spike_rollback: bool = False
    target_param_data_ratio: float = 20.0
    long_run_gate_profile: str = "research"

    def to_dict(self) -> dict:
        return asdict(self)

    def tiny_run_values(self) -> dict:
        values = self.to_dict()
        values.pop("name")
        values.pop("label")
        values.pop("description")
        return values


RUN_SCALES: dict[str, RunScale] = {
    "smoke": RunScale(
        name="smoke",
        label="Smoke",
        description="Fast CPU sanity check for data, reports, and UI wiring.",
        tokenizer_type="char",
        tokenizer_vocab_size=None,
        tokenizer_min_freq=1,
        context_size=128,
        n_embd=32,
        n_head=4,
        n_layer=1,
        norm_type="layernorm",
        position_encoding="learned",
        activation="gelu",
        base_steps=40,
        sft_steps=60,
        base_batch_size=4,
        sft_batch_size=4,
        base_learning_rate=3e-4,
        sft_learning_rate=1e-3,
        base_lr_warmup_steps=0,
        sft_lr_warmup_steps=0,
        base_lr_decay="none",
        sft_lr_decay="none",
        base_min_lr_ratio=1.0,
        sft_min_lr_ratio=1.0,
        base_grad_clip=0.0,
        sft_grad_clip=0.0,
        base_grad_accum_steps=1,
        sft_grad_accum_steps=1,
        sft_sampling="uniform",
        base_early_stop_patience=4,
        sft_early_stop_patience=4,
        canary_count=1,
        eval_max_new_tokens=80,
    ),
    "pico": RunScale(
        name="pico",
        label="Pico",
        description="First serious local run: BPE tokenizer, stronger tiny model, and scheduled training.",
        tokenizer_type="bpe",
        tokenizer_vocab_size=512,
        tokenizer_min_freq=2,
        context_size=256,
        n_embd=96,
        n_head=4,
        n_layer=3,
        norm_type="layernorm",
        position_encoding="learned",
        activation="gelu",
        base_steps=10000,
        sft_steps=1000,
        base_batch_size=4,
        sft_batch_size=4,
        base_learning_rate=3e-4,
        sft_learning_rate=3e-4,
        base_lr_warmup_steps=200,
        sft_lr_warmup_steps=50,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=1,
        sft_grad_accum_steps=1,
        sft_sampling="category_sqrt",
        base_early_stop_patience=3,
        sft_early_stop_patience=4,
        canary_count=3,
        eval_max_new_tokens=120,
    ),
    "small": RunScale(
        name="small",
        label="Small",
        description="Slower local SLM experiment for larger corpora after a pico run is healthy.",
        tokenizer_type="bpe",
        tokenizer_vocab_size=1024,
        tokenizer_min_freq=2,
        context_size=512,
        n_embd=128,
        n_head=4,
        n_layer=4,
        norm_type="layernorm",
        position_encoding="learned",
        activation="gelu",
        base_steps=25000,
        sft_steps=1500,
        base_batch_size=4,
        sft_batch_size=4,
        base_learning_rate=3e-4,
        sft_learning_rate=2e-4,
        base_lr_warmup_steps=500,
        sft_lr_warmup_steps=100,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=4,
        sft_grad_accum_steps=2,
        sft_sampling="category_sqrt",
        base_early_stop_patience=3,
        sft_early_stop_patience=4,
        canary_count=5,
        eval_max_new_tokens=160,
    ),
    "medium": RunScale(
        name="medium",
        label="Medium",
        description="Overnight-class Mac experiment; use only after tokenizer/data diagnostics look good.",
        tokenizer_type="bpe",
        tokenizer_vocab_size=2048,
        tokenizer_min_freq=2,
        context_size=512,
        n_embd=192,
        n_head=6,
        n_layer=6,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="relu2",
        base_steps=60000,
        sft_steps=2500,
        base_batch_size=4,
        sft_batch_size=4,
        base_learning_rate=2e-4,
        sft_learning_rate=1.5e-4,
        base_lr_warmup_steps=1000,
        sft_lr_warmup_steps=150,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=8,
        sft_grad_accum_steps=4,
        sft_sampling="category_sqrt",
        base_early_stop_patience=4,
        sft_early_stop_patience=5,
        canary_count=5,
        eval_max_new_tokens=160,
    ),
    "mps-local": RunScale(
        name="mps-local",
        label="MPS Local",
        description="Mac GPU-oriented local run with accumulation for a larger effective batch.",
        tokenizer_type="bpe",
        tokenizer_vocab_size=1024,
        tokenizer_min_freq=2,
        context_size=512,
        n_embd=128,
        n_head=4,
        n_layer=4,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="relu2",
        base_steps=30000,
        sft_steps=1500,
        base_batch_size=4,
        sft_batch_size=4,
        base_learning_rate=3e-4,
        sft_learning_rate=2e-4,
        base_lr_warmup_steps=500,
        sft_lr_warmup_steps=100,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=8,
        sft_grad_accum_steps=4,
        sft_sampling="category_sqrt",
        base_early_stop_patience=4,
        sft_early_stop_patience=5,
        canary_count=5,
        eval_max_new_tokens=160,
    ),
    "climbmix-pilot": RunScale(
        name="climbmix-pilot",
        label="ClimbMix Pilot",
        description="First closed-book public-data pilot for ClimbMix samples.",
        tokenizer_type="bpe",
        tokenizer_vocab_size=8192,
        tokenizer_min_freq=2,
        context_size=512,
        n_embd=192,
        n_head=6,
        n_layer=6,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="relu2",
        base_steps=60000,
        sft_steps=1000,
        base_batch_size=4,
        sft_batch_size=4,
        base_learning_rate=2e-4,
        sft_learning_rate=1.5e-4,
        base_lr_warmup_steps=1000,
        sft_lr_warmup_steps=100,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=16,
        sft_grad_accum_steps=4,
        sft_sampling="category_sqrt",
        base_early_stop_patience=4,
        sft_early_stop_patience=4,
        canary_count=5,
        eval_max_new_tokens=160,
    ),
    "h100-pilot": RunScale(
        name="h100-pilot",
        label="H100 Pilot",
        description="Single-H100 modern pilot with HF BPE, GQA, SwiGLU, FlashAttention, and sharded base data.",
        tokenizer_type="hf_bpe",
        tokenizer_vocab_size=8192,
        tokenizer_min_freq=2,
        context_size=512,
        n_embd=384,
        n_head=8,
        n_kv_head=2,
        n_layer=8,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="swiglu",
        tie_embeddings=True,
        qk_norm=True,
        attn_backend="flash",
        parallel_residual=True,
        linear_bias=False,
        base_steps=5000,
        sft_steps=180,
        base_batch_size=8,
        sft_batch_size=8,
        base_learning_rate=1e-4,
        sft_learning_rate=1e-5,
        base_lr_warmup_steps=500,
        sft_lr_warmup_steps=20,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=16,
        sft_grad_accum_steps=4,
        sft_sampling="category_sqrt",
        sft_packing="bos_bestfit",
        base_dataset_mode="sharded",
        base_shard_token_size=1_000_000,
        base_shard_cache_size=2,
        base_early_stop_patience=4,
        sft_early_stop_patience=4,
        canary_count=1,
        eval_max_new_tokens=120,
        precision="bf16",
        matmul_precision="high",
        torch_compile=True,
        torch_compile_mode="default",
        auto_lr_scaling=True,
        loss_spike_rollback=True,
        long_run_gate_profile="first_release",
    ),
    "h100-100m": RunScale(
        name="h100-100m",
        label="H100 100M",
        description=(
            "Single-H100/H200 100M-parameter public-proof run. Target a curated "
            "SmolLM/FineWeb-Edu-scale pack near 2B tokens, then train every local "
            "token shard under the skill_release gate."
        ),
        tokenizer_type="hf_bpe",
        tokenizer_vocab_size=8192,
        tokenizer_min_freq=2,
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
        attn_backend="flash",
        parallel_residual=True,
        linear_bias=False,
        base_steps=33000,
        sft_steps=180,
        base_batch_size=8,
        sft_batch_size=8,
        base_learning_rate=5e-5,
        sft_learning_rate=1e-5,
        base_lr_warmup_steps=1500,
        sft_lr_warmup_steps=20,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=16,
        sft_grad_accum_steps=4,
        sft_sampling="category_sqrt",
        sft_packing="bos_bestfit",
        base_dataset_mode="sharded",
        base_shard_token_size=1_000_000,
        base_shard_cache_size=2,
        base_early_stop_patience=4,
        sft_early_stop_patience=4,
        canary_count=1,
        eval_max_new_tokens=120,
        precision="bf16",
        matmul_precision="high",
        torch_compile=True,
        torch_compile_mode="default",
        auto_lr_scaling=True,
        loss_spike_rollback=True,
        long_run_gate_profile="skill_release",
    ),
    "h100-100m-ddp8": RunScale(
        name="h100-100m-ddp8",
        label="H100 100M DDP8",
        description=(
            "DDP-aware 100M recipe for 8 GPUs. Keeps base tokens and SFT example "
            "exposure near the single-GPU 100M target by reducing steps, uses "
            "explicit tuned LRs, and defaults to the skill_release gate."
        ),
        tokenizer_type="hf_bpe",
        tokenizer_vocab_size=8192,
        tokenizer_min_freq=2,
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
        attn_backend="flash",
        parallel_residual=True,
        linear_bias=False,
        base_steps=4100,
        sft_steps=24,
        base_batch_size=8,
        sft_batch_size=8,
        base_learning_rate=2e-4,
        sft_learning_rate=2e-5,
        base_lr_warmup_steps=200,
        sft_lr_warmup_steps=5,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=16,
        sft_grad_accum_steps=4,
        sft_sampling="category_sqrt",
        sft_packing="bos_bestfit",
        base_dataset_mode="sharded",
        base_shard_token_size=1_000_000,
        base_shard_cache_size=2,
        base_early_stop_patience=4,
        sft_early_stop_patience=4,
        canary_count=1,
        eval_max_new_tokens=120,
        precision="bf16",
        matmul_precision="high",
        torch_compile=True,
        torch_compile_mode="default",
        auto_lr_scaling=False,
        loss_spike_rollback=False,
        long_run_gate_profile="skill_release",
    ),
    "h200-1b-ddp8": RunScale(
        name="h200-1b-ddp8",
        label="H200 1B DDP8",
        description=(
            "Eight-H200 1B-class skill-release recipe. Uses 2048-token context, "
            "1M-token global base batches, FA3, 32k HF BPE, a Chinchilla-class "
            "20 tokens/parameter base budget, and the skill_release gate; import "
            "substantially more ClimbMix/data-mix tokens before treating this as a release candidate."
        ),
        tokenizer_type="hf_bpe",
        tokenizer_vocab_size=32768,
        tokenizer_min_freq=2,
        context_size=2048,
        n_embd=2048,
        n_head=16,
        n_kv_head=4,
        n_layer=24,
        norm_type="rmsnorm",
        position_encoding="rope",
        activation="swiglu",
        tie_embeddings=True,
        qk_norm=True,
        attn_backend="fa3",
        parallel_residual=True,
        linear_bias=False,
        scaled_residual_init=True,
        base_steps=21400,
        sft_steps=64,
        base_batch_size=8,
        sft_batch_size=8,
        base_learning_rate=1e-4,
        sft_learning_rate=1e-5,
        base_lr_warmup_steps=500,
        sft_lr_warmup_steps=10,
        base_lr_decay="cosine",
        sft_lr_decay="cosine",
        base_min_lr_ratio=0.1,
        sft_min_lr_ratio=0.1,
        base_grad_clip=1.0,
        sft_grad_clip=1.0,
        base_grad_accum_steps=8,
        sft_grad_accum_steps=4,
        sft_sampling="category_sqrt",
        sft_packing="bos_bestfit",
        base_dataset_mode="sharded",
        base_shard_token_size=10_000_000,
        base_shard_cache_size=2,
        base_early_stop_patience=4,
        sft_early_stop_patience=4,
        canary_count=1,
        eval_max_new_tokens=160,
        precision="bf16",
        matmul_precision="high",
        torch_compile=True,
        torch_compile_mode="default",
        gradient_checkpointing=True,
        auto_lr_scaling=False,
        loss_spike_rollback=False,
        target_param_data_ratio=20.0,
        long_run_gate_profile="skill_release",
    ),
}

RUN_SCALE_NAMES = tuple(RUN_SCALES)
