"""Small trainable character tokenizer.

The first Picochat tokenizer is intentionally character-level. It is slower and
less capable than BPE, but it makes the first training pipeline easy to inspect:
each input character becomes one token, plus a few reserved control tokens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


@dataclass(frozen=True)
class TokenizerStats:
    vocab_size: int
    num_special_tokens: int
    num_text_tokens: int


class CharTokenizer:
    """A tiny tokenizer that learns a vocabulary of characters from text."""

    def __init__(self, token_to_id: dict[str, int]):
        missing = [token for token in SPECIAL_TOKENS if token not in token_to_id]
        if missing:
            raise ValueError(f"Missing required special tokens: {missing}")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("Token ids must be unique")

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int | None = None,
        min_freq: int = 1,
    ) -> "CharTokenizer":
        """Build a character vocabulary from a list of training strings."""
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1")
        if vocab_size is not None and vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("vocab_size must leave room for special tokens")

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text)

        chars = [
            char for char, count in counts.items()
            if count >= min_freq and char not in SPECIAL_TOKENS
        ]
        chars.sort(key=lambda char: (-counts[char], char))

        if vocab_size is not None:
            chars = chars[: vocab_size - len(SPECIAL_TOKENS)]

        tokens = [*SPECIAL_TOKENS, *chars]
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def stats(self) -> TokenizerStats:
        return TokenizerStats(
            vocab_size=len(self),
            num_special_tokens=len(SPECIAL_TOKENS),
            num_text_tokens=len(self) - len(SPECIAL_TOKENS),
        )

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(char, self.unk_id) for char in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "special_tokens": list(SPECIAL_TOKENS),
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "char":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        if tuple(data.get("special_tokens", [])) != SPECIAL_TOKENS:
            raise ValueError("Tokenizer special tokens do not match this version")
        return cls({token: int(idx) for token, idx in data["token_to_id"].items()})

