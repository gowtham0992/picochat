"""Small tokenizers for Picochat experiments.

Picochat starts with character and byte tokenizers because they are easy to
inspect. The BPE tokenizer adds the first compression step: it learns frequent
adjacent token pairs and stores the merge table so the process stays
dependency-free and explainable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
DEFAULT_BPE_VOCAB_SIZE = 512
BPE_PRETOKENIZERS = ("char", "regex")
DEFAULT_BPE_PRETOKENIZER = "regex"
BPE_REGEX_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+(?:[.,]\d+)*%?| ?[^\s\w]+|\s+(?!\S)|\s+",
    flags=re.IGNORECASE | re.UNICODE,
)
HF_BPE_REGEX_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"
    r"\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)


@dataclass(frozen=True)
class TokenizerStats:
    tokenizer_type: str
    vocab_size: int
    num_special_tokens: int
    num_text_tokens: int


class CharTokenizer:
    """A tiny tokenizer that learns a vocabulary of characters from text."""

    tokenizer_type = "char"

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
        texts: Iterable[str],
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
            tokenizer_type=self.tokenizer_type,
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
            "type": self.tokenizer_type,
            "special_tokens": list(SPECIAL_TOKENS),
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = _read_tokenizer_json(path)
        if data.get("type") != "char":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: dict) -> "CharTokenizer":
        if tuple(data.get("special_tokens", [])) != SPECIAL_TOKENS:
            raise ValueError("Tokenizer special tokens do not match this version")
        return cls({token: int(idx) for token, idx in data["token_to_id"].items()})


class ByteTokenizer:
    """A fixed UTF-8 byte tokenizer with the same interface as CharTokenizer."""

    tokenizer_type = "byte"

    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = dict(token_to_id) if token_to_id is not None else _byte_token_to_id()
        missing = [token for token in SPECIAL_TOKENS if token not in self.token_to_id]
        if missing:
            raise ValueError(f"Missing required special tokens: {missing}")
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("Token ids must be unique")
        self.byte_to_id = {
            byte: self.token_to_id[_byte_token(byte)]
            for byte in range(256)
        }
        self.id_to_byte = {token_id: byte for byte, token_id in self.byte_to_id.items()}

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int | None = None,
        min_freq: int = 1,
    ) -> "ByteTokenizer":
        """Return the fixed byte vocabulary.

        The arguments match CharTokenizer.train so callers can switch tokenizer
        type without changing the training flow.
        """
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1")
        if vocab_size is not None and vocab_size != len(SPECIAL_TOKENS) + 256:
            raise ValueError("byte tokenizer has a fixed vocab size of 260")
        return cls()

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
            tokenizer_type=self.tokenizer_type,
            vocab_size=len(self),
            num_special_tokens=len(SPECIAL_TOKENS),
            num_text_tokens=256,
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
        ids.extend(self.byte_to_id[byte] for byte in text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        buffer = bytearray()
        for idx in ids:
            token_id = int(idx)
            byte = self.id_to_byte.get(token_id)
            if byte is not None:
                buffer.append(byte)
                continue
            if buffer:
                pieces.append(bytes(buffer).decode("utf-8", errors="replace"))
                buffer.clear()
            token = self.id_to_token.get(token_id, "<unk>")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        if buffer:
            pieces.append(bytes(buffer).decode("utf-8", errors="replace"))
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": self.tokenizer_type,
            "special_tokens": list(SPECIAL_TOKENS),
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = _read_tokenizer_json(path)
        if data.get("type") != "byte":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: dict) -> "ByteTokenizer":
        if tuple(data.get("special_tokens", [])) != SPECIAL_TOKENS:
            raise ValueError("Tokenizer special tokens do not match this version")
        return cls({token: int(idx) for token, idx in data["token_to_id"].items()})


class BPETokenizer:
    """A tiny deterministic character-BPE tokenizer.

    This is intentionally not a production tiktoken/SentencePiece replacement.
    It exists so Picochat can compare character, byte, and subword-like
    tokenization without adding another dependency or hiding the algorithm.
    """

    tokenizer_type = "bpe"

    def __init__(
        self,
        token_to_id: dict[str, int],
        merges: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        pretokenizer: str = DEFAULT_BPE_PRETOKENIZER,
    ):
        missing = [token for token in SPECIAL_TOKENS if token not in token_to_id]
        if missing:
            raise ValueError(f"Missing required special tokens: {missing}")
        if pretokenizer not in BPE_PRETOKENIZERS:
            raise ValueError(f"pretokenizer must be one of {BPE_PRETOKENIZERS}")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("Token ids must be unique")
        self.merges = tuple((str(left), str(right)) for left, right in merges)
        self.pretokenizer = pretokenizer

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int | None = None,
        min_freq: int = 1,
        pretokenizer: str = DEFAULT_BPE_PRETOKENIZER,
    ) -> "BPETokenizer":
        """Learn a small character-BPE vocabulary from training strings."""
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1")
        if pretokenizer not in BPE_PRETOKENIZERS:
            raise ValueError(f"pretokenizer must be one of {BPE_PRETOKENIZERS}")
        target_vocab_size = vocab_size or DEFAULT_BPE_VOCAB_SIZE
        if target_vocab_size < len(SPECIAL_TOKENS) + 1:
            raise ValueError("vocab_size must leave room for special tokens and text tokens")

        counts: Counter[str] = Counter()
        raw_sequences: list[list[str]] = []
        for text in texts:
            if not text:
                continue
            counts.update(text)
            raw_sequences.extend(
                list(piece)
                for piece in _pretokenize_for_bpe(text, pretokenizer)
                if piece
            )

        chars = [char for char in counts if char not in SPECIAL_TOKENS]
        chars.sort(key=lambda char: (-counts[char], char))
        chars = chars[: target_vocab_size - len(SPECIAL_TOKENS)]
        token_to_id = {token: idx for idx, token in enumerate([*SPECIAL_TOKENS, *chars])}

        known_chars = set(chars)
        sequences = [
            [char for char in sequence if char in known_chars]
            for sequence in raw_sequences
        ]
        sequences = [sequence for sequence in sequences if len(sequence) >= 2]

        merges: list[tuple[str, str]] = []
        # One-off merges compress the training corpus but often behave like tiny
        # memorization shortcuts, so BPE merges must appear at least twice.
        merge_min_freq = max(2, min_freq)
        while len(token_to_id) < target_vocab_size:
            pair_counts = _pair_counts(sequences)
            candidates = [
                (pair, count) for pair, count in pair_counts.items()
                if count >= merge_min_freq and pair[0] + pair[1] not in token_to_id
            ]
            if not candidates:
                break
            best_pair, _ = min(candidates, key=lambda item: (-item[1], item[0][0], item[0][1]))
            merged_token = best_pair[0] + best_pair[1]
            token_to_id[merged_token] = len(token_to_id)
            merges.append(best_pair)
            sequences = [
                _apply_merge_to_sequence(sequence, best_pair)
                for sequence in sequences
            ]
            sequences = [sequence for sequence in sequences if len(sequence) >= 2]

        return cls(token_to_id, merges, pretokenizer=pretokenizer)

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
            tokenizer_type=self.tokenizer_type,
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
        units: list[str] = []
        for piece in _pretokenize_for_bpe(text, self.pretokenizer):
            units.extend(_apply_merges(list(piece), self.merges))
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(unit, self.unk_id) for unit in units)
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
            "type": self.tokenizer_type,
            "special_tokens": list(SPECIAL_TOKENS),
            "token_to_id": self.token_to_id,
            "merges": [[left, right] for left, right in self.merges],
            "pretokenizer": self.pretokenizer,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = _read_tokenizer_json(path)
        if data.get("type") != "bpe":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: dict) -> "BPETokenizer":
        if tuple(data.get("special_tokens", [])) != SPECIAL_TOKENS:
            raise ValueError("Tokenizer special tokens do not match this version")
        merges = []
        for pair in data.get("merges", []):
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                raise ValueError("BPE merges must be pairs")
            merges.append((str(pair[0]), str(pair[1])))
        return cls(
            {token: int(idx) for token, idx in data["token_to_id"].items()},
            merges,
            pretokenizer=str(data.get("pretokenizer", "char")),
        )


class HuggingFaceBPETokenizer:
    """Production BPE tokenizer backed by Hugging Face's compiled tokenizers.

    Picochat's ``bpe`` tokenizer is deliberately small and readable. This
    backend is the long-run path: it keeps the same Picochat special-token
    contract, but delegates BPE training and encoding to compiled Rust code so
    H100 runs do not waste paid time in Python tokenizer loops.
    """

    tokenizer_type = "hf_bpe"

    def __init__(self, backend, pretokenizer: str = DEFAULT_BPE_PRETOKENIZER):
        if pretokenizer not in BPE_PRETOKENIZERS:
            raise ValueError(f"pretokenizer must be one of {BPE_PRETOKENIZERS}")
        self.backend = backend
        self.pretokenizer = pretokenizer
        self.token_to_id = {token: int(idx) for token, idx in backend.get_vocab().items()}
        missing = [token for token in SPECIAL_TOKENS if token not in self.token_to_id]
        if missing:
            raise ValueError(f"Missing required special tokens: {missing}")
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("Token ids must be unique")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int | None = None,
        min_freq: int = 1,
        pretokenizer: str = DEFAULT_BPE_PRETOKENIZER,
    ) -> "HuggingFaceBPETokenizer":
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1")
        if pretokenizer not in BPE_PRETOKENIZERS:
            raise ValueError(f"pretokenizer must be one of {BPE_PRETOKENIZERS}")
        target_vocab_size = vocab_size or DEFAULT_BPE_VOCAB_SIZE
        if target_vocab_size < len(SPECIAL_TOKENS) + 256:
            raise ValueError("hf_bpe vocab_size must leave room for special tokens and byte fallback")

        tokenizers = _require_tokenizers()
        backend = tokenizers["Tokenizer"](tokenizers["BPE"](
            byte_fallback=True,
            unk_token="<unk>",
            fuse_unk=False,
        ))
        backend.normalizer = None
        if pretokenizer == "regex":
            backend.pre_tokenizer = tokenizers["pre_tokenizers"].Sequence([
                tokenizers["pre_tokenizers"].Split(
                    pattern=tokenizers["Regex"](HF_BPE_REGEX_PATTERN),
                    behavior="isolated",
                    invert=False,
                ),
                tokenizers["pre_tokenizers"].ByteLevel(add_prefix_space=False, use_regex=False),
            ])
        else:
            backend.pre_tokenizer = tokenizers["pre_tokenizers"].ByteLevel(
                add_prefix_space=False,
                use_regex=False,
            )
        backend.decoder = tokenizers["decoders"].ByteLevel()
        backend.post_processor = None
        trainer = tokenizers["BpeTrainer"](
            vocab_size=target_vocab_size,
            show_progress=True,
            min_frequency=min_freq,
            initial_alphabet=tokenizers["pre_tokenizers"].ByteLevel.alphabet(),
            special_tokens=list(SPECIAL_TOKENS),
        )
        backend.train_from_iterator((text for text in texts if text), trainer)
        return cls(backend, pretokenizer=pretokenizer)

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
        return self.backend.get_vocab_size()

    def stats(self) -> TokenizerStats:
        return TokenizerStats(
            tokenizer_type=self.tokenizer_type,
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
        ids.extend(self.backend.encode(text, add_special_tokens=False).ids)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.backend.decode([int(idx) for idx in ids], skip_special_tokens=skip_special)

    def token_byte_lengths(self) -> list[int]:
        lengths = [0] * len(self)
        for token_id in range(len(self)):
            token = self.id_to_token.get(token_id)
            if token in SPECIAL_TOKENS:
                lengths[token_id] = 0
                continue
            piece = self.backend.decode([token_id], skip_special_tokens=False)
            lengths[token_id] = len(piece.encode("utf-8"))
        return lengths

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": self.tokenizer_type,
            "special_tokens": list(SPECIAL_TOKENS),
            "token_to_id": self.token_to_id,
            "pretokenizer": self.pretokenizer,
            "backend": "huggingface_tokenizers",
            "hf_tokenizer": json.loads(self.backend.to_str()),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HuggingFaceBPETokenizer":
        data = _read_tokenizer_json(path)
        if data.get("type") != "hf_bpe":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: dict) -> "HuggingFaceBPETokenizer":
        if tuple(data.get("special_tokens", [])) != SPECIAL_TOKENS:
            raise ValueError("Tokenizer special tokens do not match this version")
        if "hf_tokenizer" not in data:
            raise ValueError("hf_bpe tokenizer JSON is missing hf_tokenizer payload")
        tokenizers = _require_tokenizers()
        backend = tokenizers["Tokenizer"].from_str(json.dumps(data["hf_tokenizer"]))
        return cls(backend, pretokenizer=str(data.get("pretokenizer", DEFAULT_BPE_PRETOKENIZER)))


Tokenizer = CharTokenizer | ByteTokenizer | BPETokenizer | HuggingFaceBPETokenizer
TOKENIZER_TYPES = ("char", "byte", "bpe", "hf_bpe")


def token_byte_lengths(tokenizer: Tokenizer) -> list[int]:
    """Return byte length for each token id, with special tokens counted as zero."""
    if isinstance(tokenizer, HuggingFaceBPETokenizer):
        return tokenizer.token_byte_lengths()
    lengths = [0] * len(tokenizer)
    for token_id, token in tokenizer.id_to_token.items():
        token_id = int(token_id)
        if token in SPECIAL_TOKENS:
            lengths[token_id] = 0
        elif isinstance(tokenizer, ByteTokenizer) and token_id in tokenizer.id_to_byte:
            lengths[token_id] = 1
        else:
            lengths[token_id] = len(token.encode("utf-8"))
    return lengths


def train_tokenizer(
    tokenizer_type: str,
    texts: Iterable[str],
    vocab_size: int | None = None,
    min_freq: int = 1,
    bpe_pretokenizer: str = DEFAULT_BPE_PRETOKENIZER,
) -> Tokenizer:
    """Train or construct a tokenizer by type."""
    if tokenizer_type == "char":
        return CharTokenizer.train(texts, vocab_size=vocab_size, min_freq=min_freq)
    if tokenizer_type == "byte":
        return ByteTokenizer.train(texts, vocab_size=vocab_size, min_freq=min_freq)
    if tokenizer_type == "bpe":
        return BPETokenizer.train(
            texts,
            vocab_size=vocab_size,
            min_freq=min_freq,
            pretokenizer=bpe_pretokenizer,
        )
    if tokenizer_type == "hf_bpe":
        return HuggingFaceBPETokenizer.train(
            texts,
            vocab_size=vocab_size,
            min_freq=min_freq,
            pretokenizer=bpe_pretokenizer,
        )
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load any supported Picochat tokenizer."""
    data = _read_tokenizer_json(path)
    tokenizer_type = data.get("type")
    if tokenizer_type == "char":
        return CharTokenizer.from_data(data)
    if tokenizer_type == "byte":
        return ByteTokenizer.from_data(data)
    if tokenizer_type == "bpe":
        return BPETokenizer.from_data(data)
    if tokenizer_type == "hf_bpe":
        return HuggingFaceBPETokenizer.from_data(data)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def _read_tokenizer_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_tokenizers():
    try:
        from tokenizers import Regex
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import decoders, pre_tokenizers
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "hf_bpe requires the 'tokenizers' package. Install Picochat with "
            "`pip install -e '.[hf]'` or install `tokenizers>=0.20`."
        ) from exc
    return {
        "Tokenizer": HFTokenizer,
        "Regex": Regex,
        "BPE": BPE,
        "BpeTrainer": BpeTrainer,
        "pre_tokenizers": pre_tokenizers,
        "decoders": decoders,
    }


def _pair_counts(sequences: list[list[str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for sequence in sequences:
        counts.update(zip(sequence, sequence[1:]))
    return counts


def _apply_merges(sequence: list[str], merges: tuple[tuple[str, str], ...]) -> list[str]:
    for merge in merges:
        sequence = _apply_merge_to_sequence(sequence, merge)
    return sequence


def _pretokenize_for_bpe(text: str, pretokenizer: str) -> list[str]:
    if pretokenizer == "char":
        return [text] if text else []
    if pretokenizer != "regex":
        raise ValueError(f"pretokenizer must be one of {BPE_PRETOKENIZERS}")
    pieces: list[str] = []
    cursor = 0
    for match in BPE_REGEX_PATTERN.finditer(text):
        if match.start() > cursor:
            pieces.extend(text[cursor:match.start()])
        pieces.append(match.group(0))
        cursor = match.end()
    if cursor < len(text):
        pieces.extend(text[cursor:])
    return [piece for piece in pieces if piece]


def _apply_merge_to_sequence(sequence: list[str], pair: tuple[str, str]) -> list[str]:
    if len(sequence) < 2:
        return sequence
    merged = pair[0] + pair[1]
    output: list[str] = []
    index = 0
    while index < len(sequence):
        if (
            index < len(sequence) - 1
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            output.append(merged)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


def _byte_token(byte: int) -> str:
    return f"<byte:{byte:02x}>"


def _byte_token_to_id() -> dict[str, int]:
    token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for byte in range(256):
        token_to_id[_byte_token(byte)] = len(SPECIAL_TOKENS) + byte
    return token_to_id
