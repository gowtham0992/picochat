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
    assert scale.sft_sampling == "category_balanced"
