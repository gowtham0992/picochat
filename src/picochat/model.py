"""A tiny decoder-only Transformer language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    activation: str = "gelu"
    rope_base: float = 10000.0
    logit_softcap: float = 0.0
    gradient_checkpointing: bool = False

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by many modern small LMs."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def make_norm(norm_type: str, size: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(size)
    if norm_type == "rmsnorm":
        return RMSNorm(size)
    raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.position_encoding = config.position_encoding
        self.rope_base = config.rope_base
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be 'learned' or 'rope'")
        if self.position_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, embd_size = x.shape
        q, k, v = self.qkv(x).split(embd_size, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if self.position_encoding == "rope":
            q, k = apply_rope(q, k, base=self.rope_base, start_pos=start_pos)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(-2)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        attn_mask = None
        is_causal = past_kv is None
        if past_kv is not None and seq_len > 1:
            key_len = past_len + seq_len
            query_positions = torch.arange(
                past_len,
                past_len + seq_len,
                device=x.device,
            )[:, None]
            key_positions = torch.arange(key_len, device=x.device)[None, :]
            allowed = key_positions <= query_positions
            attn_mask = torch.zeros(seq_len, key_len, dtype=q.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))

        sdpa_kwargs = {
            "dropout_p": self.dropout.p if self.training else 0.0,
            "is_causal": is_causal,
        }
        if attn_mask is not None:
            sdpa_kwargs["attn_mask"] = attn_mask

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            **sdpa_kwargs,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_size)
        y = self.dropout(self.proj(y))
        if use_cache:
            return y, (k, v)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.activation not in {"gelu", "relu2"}:
            raise ValueError("activation must be 'gelu' or 'relu2'")
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = config.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.activation == "relu2":
            x = F.relu(x).square()
        else:
            x = F.gelu(x)
        return self.dropout(self.proj(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = make_norm(config.norm_type, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config.norm_type, config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out = self.attn(
            self.ln_1(x),
            past_kv=past_kv,
            start_pos=start_pos,
            use_cache=use_cache,
        )
        present_kv = None
        if use_cache:
            attn_out, present_kv = attn_out
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if use_cache:
            return x, present_kv
        return x


class TinyGPT(nn.Module):
    """Small GPT-style model for next-token prediction."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be 'learned' or 'rope'")
        if config.logit_softcap < 0:
            raise ValueError("logit_softcap must be non-negative")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            nn.Embedding(config.context_size, config.n_embd)
            if config.position_encoding == "learned"
            else None
        )
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = make_norm(config.norm_type, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
        start_pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, KVCache]:
        batch_size, seq_len = input_ids.shape
        cache_start = _infer_start_pos(past_kv) if start_pos is None else start_pos
        if cache_start < 0:
            raise ValueError("start_pos must be non-negative")
        if cache_start + seq_len > self.config.context_size:
            raise ValueError(
                f"Sequence length {cache_start + seq_len} exceeds context size {self.config.context_size}"
            )
        if past_kv is not None and len(past_kv) != len(self.blocks):
            raise ValueError("past_kv must contain one key/value pair per transformer block")
        if targets is not None and use_cache:
            raise ValueError("targets are not supported when use_cache=True")

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(
                cache_start,
                cache_start + seq_len,
                device=input_ids.device,
            )
            x = x + self.position_embedding(positions)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            layer_past = None if past_kv is None else past_kv[index]
            if self.config.gradient_checkpointing and self.training and not use_cache:
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
                block_out = block(
                    x,
                    past_kv=layer_past,
                    start_pos=cache_start,
                    use_cache=use_cache,
                )
                if use_cache:
                    x, present_kv = block_out
                    presents.append(present_kv)
                else:
                    x = block_out
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.config.logit_softcap > 0:
            logits = torch.tanh(logits / self.config.logit_softcap) * self.config.logit_softcap

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, -1),
                targets.view(batch_size * seq_len),
                ignore_index=-100,
            )
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int = 42,
        eos_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed)
        ids = input_ids

        if (
            use_cache
            and ids.size(1) > 0
            and ids.size(1) + max_new_tokens <= self.config.context_size
        ):
            return self._generate_with_cache(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generator=generator,
                eos_id=eos_id,
            )

        for _ in range(max_new_tokens):
            context = ids[:, -self.config.context_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            logits = _apply_repetition_penalty(logits, ids, repetition_penalty)

            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                logits = _apply_top_p(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)

            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break

        return ids

    def _generate_with_cache(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
        repetition_penalty: float,
        generator: torch.Generator,
        eos_id: int | None,
    ) -> torch.Tensor:
        if max_new_tokens == 0:
            return ids

        logits, _, past_kv = self(ids, use_cache=True)
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            next_logits = _apply_repetition_penalty(next_logits, ids, repetition_penalty)
            next_id = _sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )

            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
            if step == max_new_tokens - 1:
                break
            logits, _, past_kv = self(
                next_id,
                past_kv=past_kv,
                use_cache=True,
            )
        return ids

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


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


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
    logits = _apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _infer_start_pos(past_kv: KVCache | None) -> int:
    if past_kv is None:
        return 0
    if not past_kv:
        return 0
    return int(past_kv[0][0].size(-2))


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    base: float = 10000.0,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    seq_len = q.size(-2)
    head_dim = q.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    half_dim = head_dim // 2
    positions = torch.arange(
        start_pos,
        start_pos + seq_len,
        device=q.device,
        dtype=torch.float32,
    )
    inv_freq = base ** (
        -torch.arange(0, half_dim, device=q.device, dtype=torch.float32) / half_dim
    )
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]
    return _rotate_rope(q, cos, sin), _rotate_rope(k, cos, sin)


def _rotate_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)


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
