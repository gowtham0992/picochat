from picochat.scales import RUN_SCALES


def test_run_scales_order_capacity():
    assert RUN_SCALES["smoke"].base_steps < RUN_SCALES["pico"].base_steps
    assert RUN_SCALES["small"].n_layer >= RUN_SCALES["pico"].n_layer
    assert RUN_SCALES["medium"].tokenizer_vocab_size >= RUN_SCALES["small"].tokenizer_vocab_size


def test_pico_scale_uses_bpe_and_training_controls():
    scale = RUN_SCALES["pico"]

    assert scale.tokenizer_type == "bpe"
    assert scale.tokenizer_vocab_size == 512
    assert scale.base_lr_decay == "cosine"
    assert scale.base_grad_clip == 1.0
    assert scale.sft_sampling == "category_sqrt"


def test_h100_scale_uses_modern_runtime_defaults():
    scale = RUN_SCALES["h100-pilot"]

    assert scale.tokenizer_type == "hf_bpe"
    assert scale.activation == "swiglu"
    assert scale.n_kv_head == 2
    assert scale.tie_embeddings is True
    assert scale.qk_norm is True
    assert scale.attn_backend == "flash"
    assert scale.linear_bias is False
    assert scale.base_dataset_mode == "sharded"
    assert scale.precision == "bf16"
    assert scale.matmul_precision == "high"
    assert scale.torch_compile is True
    assert scale.sft_learning_rate == 0.00001
    assert scale.sft_steps == 180
    assert scale.sft_lr_warmup_steps == 20


def test_h100_100m_scale_matches_release_pilot_recipe():
    scale = RUN_SCALES["h100-100m"]

    assert scale.tokenizer_type == "hf_bpe"
    assert scale.tokenizer_vocab_size == 8192
    assert scale.n_embd == 768
    assert scale.n_layer == 16
    assert scale.n_head == 12
    assert scale.n_kv_head == 4
    assert scale.base_steps == 33000
    assert scale.base_learning_rate == 0.00005
    assert scale.sft_learning_rate == 0.00001
    assert scale.sft_steps == 180
    assert scale.base_dataset_mode == "sharded"
    assert scale.attn_backend == "flash"
    assert scale.linear_bias is False
    assert scale.precision == "bf16"
    assert scale.long_run_gate_profile == "skill_release"


def test_h100_100m_ddp8_scale_uses_global_budget_recipe():
    scale = RUN_SCALES["h100-100m-ddp8"]

    assert scale.n_embd == RUN_SCALES["h100-100m"].n_embd
    assert scale.n_layer == RUN_SCALES["h100-100m"].n_layer
    assert scale.base_steps == 4100
    assert scale.base_batch_size == 8
    assert scale.base_grad_accum_steps == 16
    assert scale.base_learning_rate == 0.0002
    assert scale.sft_learning_rate == 0.00002
    assert scale.sft_steps == 24
    assert scale.sft_lr_warmup_steps == 5
    assert scale.auto_lr_scaling is False
    assert scale.loss_spike_rollback is False
    assert scale.base_dataset_mode == "sharded"
    assert scale.target_param_data_ratio == 20.0
    assert scale.long_run_gate_profile == "skill_release"


def test_h200_1b_ddp8_scale_uses_hopper_release_recipe():
    scale = RUN_SCALES["h200-1b-ddp8"]

    assert scale.tokenizer_type == "hf_bpe"
    assert scale.tokenizer_vocab_size == 32768
    assert scale.context_size == 2048
    assert scale.n_embd == 2048
    assert scale.n_layer == 24
    assert scale.n_head == 16
    assert scale.n_kv_head == 4
    assert scale.attn_backend == "fa3"
    assert scale.base_batch_size == 8
    assert scale.base_grad_accum_steps == 8
    assert scale.base_steps == 9200
    assert scale.base_shard_token_size == 10_000_000
    assert scale.scaled_residual_init is True
    assert scale.target_param_data_ratio == 8.5
    assert scale.long_run_gate_profile == "skill_release"
