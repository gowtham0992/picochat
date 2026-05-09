"""Dataset pack parsing for corpus, chat SFT, and eval inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetPack:
    path: str
    name: str
    description: str
    corpus_input: str | None
    corpus_recipe: str | None
    chat_input: str
    eval_input: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def load_dataset_pack(path: str | Path) -> DatasetPack:
    """Load a dataset pack JSON file and resolve paths relative to the pack."""
    pack_path = Path(path)
    try:
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid dataset pack JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError("Dataset pack must be a JSON object.")

    name = _optional_string(payload.get("name"), pack_path.stem)
    description = _optional_string(payload.get("description"), "")
    corpus_input, corpus_recipe = _parse_corpus(payload.get("corpus"), pack_path)
    chat_input = _resolve_required_path(_first_present(payload, ("chat", "chat_input")), pack_path, "chat")
    eval_input = _resolve_required_path(_first_present(payload, ("eval", "eval_input")), pack_path, "eval")

    return DatasetPack(
        path=str(pack_path),
        name=name,
        description=description,
        corpus_input=corpus_input,
        corpus_recipe=corpus_recipe,
        chat_input=chat_input,
        eval_input=eval_input,
    )


def _parse_corpus(value: Any, pack_path: Path) -> tuple[str | None, str | None]:
    if isinstance(value, str):
        return _resolve_path(value, pack_path), None
    if not isinstance(value, dict):
        raise ValueError("Dataset pack must define corpus as a path string or object.")

    corpus_input = _optional_path(value.get("input"), pack_path, "corpus.input")
    corpus_recipe = _optional_path(value.get("recipe"), pack_path, "corpus.recipe")
    if bool(corpus_input) == bool(corpus_recipe):
        raise ValueError("Dataset pack corpus must define exactly one of input or recipe.")
    return corpus_input, corpus_recipe


def _first_present(payload: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _optional_string(value: Any, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("Dataset pack name and description must be strings.")
    return value.strip() or default


def _optional_path(value: Any, pack_path: Path, field: str) -> str | None:
    if value is None:
        return None
    return _resolve_required_path(value, pack_path, field)


def _resolve_required_path(value: Any, pack_path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset pack field '{field}' must be a non-empty path string.")
    return _resolve_path(value, pack_path)


def _resolve_path(value: str, pack_path: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(pack_path.parent / path)
