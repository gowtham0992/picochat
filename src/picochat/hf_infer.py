"""Inference for fine-tuned Hugging Face models (the output of `train hf-sft`).

Exposes the same `.generate(...)` shape as the native `LoadedGenerator` so the
serving handler and the web generate endpoint can treat both interchangeably.
Requires the `hf` extra (transformers); import this module lazily.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from picochat.device import resolve_device


class HFGenerator:
    """Checkpoint-backed generator for a saved Hugging Face causal LM."""

    def __init__(self, *, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
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
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
