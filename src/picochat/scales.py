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
    sft_sampling: str
    base_early_stop_patience: int
    sft_early_stop_patience: int
    canary_count: int
    eval_max_new_tokens: int

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
        context_size=64,
        n_embd=32,
        n_head=4,
        n_layer=1,
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
        sft_sampling="category_balanced",
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
        sft_sampling="category_balanced",
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
        sft_sampling="category_balanced",
        base_early_stop_patience=4,
        sft_early_stop_patience=5,
        canary_count=5,
        eval_max_new_tokens=160,
    ),
}

RUN_SCALE_NAMES = tuple(RUN_SCALES)
