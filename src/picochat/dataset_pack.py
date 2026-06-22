"""Dataset pack parsing for corpus, chat SFT, and eval inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
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


@dataclass(frozen=True)
class DatasetPackInitReport:
    out_dir: str
    dataset_pack: str
    corpus_recipe: str
    chat_input: str
    eval_input: str
    created: tuple[str, ...]
    overwritten: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "created": list(self.created),
            "overwritten": list(self.overwritten),
        }


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


def init_dataset_pack(
    out_dir: str | Path,
    corpus_path: str | Path,
    name: str = "picochat-pack",
    description: str = "Starter Picochat dataset pack.",
    force: bool = False,
) -> DatasetPackInitReport:
    """Create editable starter files for a corpus + chat + eval dataset pack."""
    out_dir = Path(out_dir)
    if not str(corpus_path).strip():
        raise ValueError("corpus_path must be a non-empty path")
    name = name.strip()
    if not name:
        raise ValueError("name must be non-empty")

    dataset_pack_path = out_dir / "dataset_pack.json"
    corpus_recipe_path = out_dir / "corpus_recipe.json"
    chat_path = out_dir / "chat.jsonl"
    eval_path = out_dir / "eval.jsonl"
    targets = (dataset_pack_path, corpus_recipe_path, chat_path, eval_path)
    existing = tuple(str(path) for path in targets if path.exists())
    if existing and not force:
        names = ", ".join(existing)
        raise FileExistsError(f"Refusing to overwrite existing pack file(s): {names}")

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_ref = _relative_path(corpus_path, out_dir)
    files = {
        dataset_pack_path: json.dumps({
            "name": name,
            "description": description,
            "corpus": {"recipe": "corpus_recipe.json"},
            "chat": "chat.jsonl",
            "eval": "eval.jsonl",
        }, indent=2) + "\n",
        corpus_recipe_path: json.dumps({
            "name": name,
            "description": f"Corpus recipe for {name}.",
            "sources": [
                {
                    "path": corpus_ref,
                    "label": "corpus",
                },
            ],
            "exclude": [
                "**/.DS_Store",
                "**/.git/**",
                "**/__pycache__/**",
            ],
        }, indent=2) + "\n",
        chat_path: _jsonl([
            {
                "category": "domain_qa",
                "group": "starter-answerable-1",
                "answerable": True,
                "user": "Replace this with a real user question from your domain.",
                "assistant": "Replace this with the concise answer you want the model to learn.",
            },
            {
                "category": "refusal",
                "group": "starter-refusal-1",
                "answerable": False,
                "user": "Replace this with a question that is outside your corpus or should not be answered.",
                "assistant": "I do not know from the provided domain material.",
            },
            {
                "category": "style",
                "group": "starter-style-1",
                "answerable": True,
                "user": "Replace this with a request that demonstrates the tone and format you want.",
                "assistant": "Replace this with an answer in the desired tone and format.",
            },
        ]),
        eval_path: _jsonl([
            {
                "user": "Replace this with a held-out domain question your model should answer.",
                "category": "domain_qa",
                "split": "heldout",
                "answerable": True,
                "must_include": [
                    "Replace this with a required phrase from the correct answer",
                ],
                "must_not_include": [
                    "Replace this with a common wrong claim",
                ],
            },
            {
                "user": "Replace this with a held-out out-of-domain or unsafe question.",
                "category": "refusal",
                "split": "heldout",
                "answerable": False,
                "must_include_any": [
                    [
                        "I do not know",
                        "provided domain material",
                    ],
                ],
            },
        ]),
    }

    overwritten: list[str] = []
    created: list[str] = []
    for path, content in files.items():
        existed = path.exists()
        path.write_text(content, encoding="utf-8")
        if existed:
            overwritten.append(str(path))
        else:
            created.append(str(path))

    return DatasetPackInitReport(
        out_dir=str(out_dir),
        dataset_pack=str(dataset_pack_path),
        corpus_recipe=str(corpus_recipe_path),
        chat_input=str(chat_path),
        eval_input=str(eval_path),
        created=tuple(created),
        overwritten=tuple(overwritten),
    )


def update_dataset_pack_tuning_paths(
    path: str | Path,
    chat_input: str | Path | None = None,
    eval_input: str | Path | None = None,
) -> DatasetPack:
    """Point a dataset pack at new chat/eval JSONL files without rewriting them."""
    pack_path = Path(path)
    try:
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid dataset pack JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError("Dataset pack must be a JSON object.")
    if chat_input is None and eval_input is None:
        raise ValueError("chat_input or eval_input is required")

    if chat_input is not None:
        chat_path = _existing_file(chat_input, "chat_input")
        payload["chat"] = _relative_path(chat_path, pack_path.parent)
    if eval_input is not None:
        eval_path = _existing_file(eval_input, "eval_input")
        payload["eval"] = _relative_path(eval_path, pack_path.parent)

    pack_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return load_dataset_pack(pack_path)


def clone_dataset_pack(path: str | Path, target_dir: str | Path) -> str:
    """Copy a dataset pack's chat/eval files into a writable directory.

    Used to make a working copy of a read-only bundled example so generation
    never mutates the shipped files. The corpus (which is never rewritten by
    data generation) is referenced in place via an absolute path, so only the
    chat/eval JSONL files are duplicated.

    Returns the path to the new ``dataset_pack.json``.
    """
    pack = load_dataset_pack(path)
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chat_src = _existing_file(pack.chat_input, "chat")
    eval_src = _existing_file(pack.eval_input, "eval")
    chat_dst = out_dir / chat_src.name
    eval_dst = out_dir / eval_src.name
    shutil.copyfile(chat_src, chat_dst)
    shutil.copyfile(eval_src, eval_dst)

    payload: dict[str, Any] = {
        "name": pack.name,
        "description": pack.description,
        "chat": chat_dst.name,
        "eval": eval_dst.name,
    }
    if pack.corpus_recipe:
        payload["corpus"] = {"recipe": str(Path(pack.corpus_recipe).resolve())}
    elif pack.corpus_input:
        payload["corpus"] = {"input": str(Path(pack.corpus_input).resolve())}

    new_pack = out_dir / "dataset_pack.json"
    new_pack.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(new_pack)


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


def _relative_path(path: str | Path, base_dir: Path) -> str:
    source = Path(path)
    target = source if source.is_absolute() else Path.cwd() / source
    try:
        relative = os.path.relpath(target.resolve(strict=False), start=base_dir.resolve(strict=False))
    except ValueError:
        return str(target.resolve(strict=False))
    return Path(relative).as_posix()


def _existing_file(path: str | Path, field: str) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset pack {field} does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Dataset pack {field} must be a file: {candidate}")
    return candidate


def _jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(row) for row in rows) + "\n"
