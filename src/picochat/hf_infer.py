"""Inference for fine-tuned Hugging Face models (the output of `train hf-sft`).

Exposes the same `.generate(...)` shape as the native `LoadedGenerator` so the
serving handler and the web generate endpoint can treat both interchangeably.
Requires the `hf` extra (transformers); import this module lazily.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from picochat.device import resolve_device


class HFGenerator:
    """Checkpoint-backed generator for a saved Hugging Face causal LM."""

    def __init__(self, *, model_path: str, device: str = "cpu", base_only: bool = False) -> None:
        self.model_path = model_path
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # A LoRA fine-tune saves only the adapter; loading model_path auto-applies
        # base+adapter. base_only loads the untouched base model instead, so the
        # Playground's base/sft toggle becomes a real before/after-fine-tuning
        # comparison rather than serving the same weights for both.
        load_path = model_path
        if base_only:
            adapter_cfg = Path(model_path) / "adapter_config.json"
            if adapter_cfg.exists():
                base_name = json.loads(adapter_cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
                if base_name:
                    load_path = base_name
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str = "",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 20,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int = 42,
        use_kv_cache: bool = True,
    ) -> dict:
        set_seed(seed)
        # Match training: hf_sft formats rows with the model's chat template, so
        # inference must apply it too — otherwise an instruct model is fed a raw
        # string it was never tuned on and answers worse (or ignores its system
        # role). Fall back to the raw prompt for base models with no template.
        text_input = prompt
        if getattr(self.tokenizer, "chat_template", None):
            try:
                text_input = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text_input = prompt
        encoded = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        prompt_len = int(encoded["input_ids"].shape[1])
        do_sample = temperature is not None and temperature > 0

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "use_cache": use_kv_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k is not None and top_k > 0:
                gen_kwargs["top_k"] = top_k

        output = self.model.generate(**encoded, **gen_kwargs)
        full_ids = output[0].tolist()
        gen_ids = full_ids[prompt_len:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        stopped = bool(gen_ids) and gen_ids[-1] == self.tokenizer.eos_token_id
        generated_tokens = [
            {"token": self.tokenizer.decode([tid]), "id": int(tid)} for tid in gen_ids
        ]
        return {
            "text": text,
            "completion": completion,
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_len,
            "completion_tokens": len(gen_ids),
            "total_tokens": len(full_ids),
            "finish_reason": "stop" if stopped else "length",
            "used_kv_cache": bool(use_kv_cache),
        }
