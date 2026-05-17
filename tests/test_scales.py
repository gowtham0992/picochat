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
    assert scale.precision == "bf16"
