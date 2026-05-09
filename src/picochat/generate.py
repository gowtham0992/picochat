"""Text generation from a saved Picochat checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from picochat.checkpoint import load_checkpoint
from picochat.tokenizer import CharTokenizer


@dataclass(frozen=True)
class GenerateConfig:
    checkpoint_path: str
    tokenizer_path: str
    prompt: str = ""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int | None = 20
    seed: int = 42
    device: str = "cpu"


def generate_text(config: GenerateConfig) -> str:
    """Load a checkpoint and generate text from a prompt."""
    return generate_text_with_trace(config)["text"]


@torch.no_grad()
def generate_text_with_trace(config: GenerateConfig) -> dict:
    """Generate text and return token-level sampling details."""
    tokenizer = CharTokenizer.load(config.tokenizer_path)
    device = torch.device(config.device)
    model, _ = load_checkpoint(config.checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(config.prompt, add_bos=True)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    generated_ids: list[int] = []
    generated_tokens: list[dict] = []

    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if config.temperature < 0:
        raise ValueError("temperature must be non-negative")

    for _ in range(config.max_new_tokens):
        context = ids[:, -model.config.context_size:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]

        if config.temperature == 0:
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            sample_logits = next_logits / config.temperature
            if config.top_k is not None and config.top_k > 0:
                values, _ = torch.topk(sample_logits, min(config.top_k, sample_logits.size(-1)))
                sample_logits = sample_logits.masked_fill(
                    sample_logits < values[:, [-1]],
                    float("-inf"),
                )
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
            break

    output_ids = ids[0].tolist()
    return {
        "text": tokenizer.decode(output_ids),
        "completion": tokenizer.decode(generated_ids),
        "generated_tokens": generated_tokens,
        "prompt_tokens": len(prompt_ids),
        "total_tokens": len(output_ids),
    }
