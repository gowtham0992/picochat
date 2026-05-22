"""Text generation from a saved Picochat checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from picochat.checkpoint import load_checkpoint
from picochat.device import resolve_device
from picochat.tokenizer import load_tokenizer


@dataclass(frozen=True)
class GenerateConfig:
    checkpoint_path: str
    tokenizer_path: str
    prompt: str = ""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int | None = 20
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    device: str = "cpu"
    use_kv_cache: bool = True


class LoadedGenerator:
    """Reusable checkpoint-backed generator for CLI, web, and serving paths."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.device = resolve_device(device)
        self.model, _ = load_checkpoint(checkpoint_path, map_location=self.device)
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
        return _generate_with_loaded(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            use_kv_cache=use_kv_cache,
        )


def generate_text(config: GenerateConfig) -> str:
    """Load a checkpoint and generate text from a prompt."""
    return generate_text_with_trace(config)["text"]


@torch.no_grad()
def generate_text_with_trace(config: GenerateConfig) -> dict:
    """Generate text and return token-level sampling details."""
    engine = LoadedGenerator(
        checkpoint_path=config.checkpoint_path,
        tokenizer_path=config.tokenizer_path,
        device=config.device,
    )
    return engine.generate(
        prompt=config.prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        seed=config.seed,
        use_kv_cache=config.use_kv_cache,
    )


@torch.no_grad()
def _generate_with_loaded(
    *,
    model,
    tokenizer,
    device: torch.device,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 20,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seed: int = 42,
    use_kv_cache: bool = True,
) -> dict:
    """Generate text with already-loaded model/tokenizer objects."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    generated_ids: list[int] = []
    generated_tokens: list[dict] = []
    stopped_eos = False

    use_cache = (
        use_kv_cache
        and ids.size(1) > 0
        and ids.size(1) + max_new_tokens <= model.config.context_size
    )
    past_kv = None
    logits = None
    if use_cache and max_new_tokens > 0:
        logits, _, past_kv = model(ids, use_cache=True)

    for step in range(max_new_tokens):
        if not use_cache:
            context = ids[:, -model.config.context_size:]
            logits, _ = model(context)
        next_logits = logits[:, -1, :]
        next_logits = _apply_repetition_penalty(
            next_logits,
            ids,
            repetition_penalty,
        )

        if temperature == 0:
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            sample_logits = next_logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(sample_logits, min(top_k, sample_logits.size(-1)))
                sample_logits = sample_logits.masked_fill(
                    sample_logits < values[:, [-1]],
                    float("-inf"),
                )
            sample_logits = _apply_top_p(sample_logits, top_p)
            probs = F.softmax(sample_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)

        token_id = int(next_id.item())
        probability = float(probs[0, token_id].clamp_min(1e-12).item())
        generated_ids.append(token_id)
        generated_tokens.append({
            "token": tokenizer.id_to_token.get(token_id, "<unk>"),
            "id": token_id,
            "probability": probability,
            "logprob": float(torch.log(torch.tensor(probability)).item()),
        })

        ids = torch.cat([ids, next_id], dim=1)
        if token_id == tokenizer.eos_id:
            stopped_eos = True
            break
        if use_cache and step != max_new_tokens - 1:
            logits, _, past_kv = model(
                next_id,
                past_kv=past_kv,
                use_cache=True,
            )

    output_ids = ids[0].tolist()
    return {
        "text": tokenizer.decode(output_ids),
        "completion": tokenizer.decode(generated_ids),
        "generated_tokens": generated_tokens,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(generated_ids),
        "total_tokens": len(output_ids),
        "finish_reason": "stop" if stopped_eos else "length",
        "used_kv_cache": use_cache,
    }


def _apply_repetition_penalty(
    logits: torch.Tensor,
    ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    for batch_index in range(ids.size(0)):
        token_ids = torch.unique(ids[batch_index])
        token_logits = adjusted[batch_index, token_ids]
        adjusted[batch_index, token_ids] = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )
    return adjusted


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_sorted = cumulative_probs > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))
