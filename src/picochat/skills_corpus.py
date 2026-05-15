"""Micro-skills corpus generation for tiny closed-book models."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import string
from typing import Any


TRAIN_WORDS = (
    "planet", "garden", "silver", "bridge", "rocket", "window", "little", "forest",
    "button", "orange", "paper", "pencil", "castle", "yellow", "circle", "flower",
    "rabbit", "mirror", "pocket", "stream", "cloudy", "friend", "gentle", "simple",
    "bright", "branch", "kitten", "lesson", "mother", "smooth", "ticket", "velvet",
)

HELDOUT_WORDS = (
    "market", "river", "winter", "summer", "candle", "basket", "needle", "school",
    "travel", "purple", "camera", "animal", "doctor", "engine", "island", "ladder",
    "magnet", "napkin", "pillow", "square", "temple", "wonder", "zipper", "artist",
)

@dataclass(frozen=True)
class SkillsCorpusReport:
    output_path: str
    report_path: str
    recipe_path: str | None
    base_corpus: str | None
    total_rows: int
    categories: dict[str, int]
    characters_written: int
    seed: int
    heldout_words_excluded: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "heldout_words_excluded": list(self.heldout_words_excluded),
        }


def generate_skills_corpus(
    out_path: str | Path,
    *,
    math_rows: int = 50_000,
    spelling_rows: int = 50_000,
    choice_rows: int = 10_000,
    seed: int = 42,
    force: bool = False,
    base_corpus: str | Path | None = None,
    recipe_out: str | Path | None = None,
) -> SkillsCorpusReport:
    """Write a deterministic micro-skills corpus for base pretraining."""
    if math_rows < 0 or spelling_rows < 0 or choice_rows < 0:
        raise ValueError("row counts must be non-negative")
    if math_rows + spelling_rows + choice_rows <= 0:
        raise ValueError("at least one skills row is required")

    out = Path(out_path)
    if out.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing skills corpus: {out}")

    rng = random.Random(seed)
    rows = [
        *(_math_example(index, rng) for index in range(math_rows)),
        *(_spelling_example(index, rng) for index in range(spelling_rows)),
        *(_choice_example(index, rng) for index in range(choice_rows)),
    ]
    rng.shuffle(rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(row["text"] for row in rows) + "\n"
    out.write_text(text, encoding="utf-8")

    recipe_path = None
    if recipe_out:
        recipe_path = _write_recipe(
            recipe_out,
            skills_path=out,
            base_corpus=Path(base_corpus) if base_corpus else None,
            force=force,
        )

    report_path = out.with_suffix(".report.md")
    report = SkillsCorpusReport(
        output_path=str(out),
        report_path=str(report_path),
        recipe_path=str(recipe_path) if recipe_path else None,
        base_corpus=str(base_corpus) if base_corpus else None,
        total_rows=len(rows),
        categories=dict(sorted(Counter(row["category"] for row in rows).items())),
        characters_written=len(text),
        seed=seed,
        heldout_words_excluded=HELDOUT_WORDS,
    )
    report_path.write_text(_report_markdown(report), encoding="utf-8")
    return report


def _math_example(index: int, rng: random.Random) -> dict[str, str]:
    kind = index % 5
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    c = rng.randint(1, 12)
    if kind == 0:
        question = f"{a} + {b}"
        answer = str(a + b)
    elif kind == 1:
        left = max(a, b)
        right = min(a, b)
        question = f"{left} - {right}"
        answer = str(left - right)
    elif kind == 2:
        x = rng.randint(2, 25)
        y = rng.randint(2, 12)
        question = f"{x} * {y}"
        answer = str(x * y)
    elif kind == 3:
        question = f"Start with {a + c}. Remove {c}. What number remains?"
        answer = str(a)
    else:
        question = f"Combine {a} objects and {b} objects. What is the total count?"
        answer = str(a + b)
    templates = (
        "Arithmetic drill: question {question}; answer {answer}.",
        "Compute exactly: {question} = {answer}.",
        "Math fact: the answer to {question} is {answer}.",
        "Return only the number: problem {question}; number {answer}.",
    )
    return {
        "category": "skills_math",
        "text": templates[index % len(templates)].format(question=question, answer=answer),
    }


def _spelling_example(index: int, rng: random.Random) -> dict[str, str]:
    word = _skill_word(index, rng)
    mode = index % 6
    if mode == 0:
        task = f"spaced letters of {word}"
        answer = " ".join(word)
    elif mode == 1:
        task = f"reverse of {word}"
        answer = word[::-1]
    elif mode == 2:
        task = f"letter count of {word}"
        answer = str(len(word))
    elif mode == 3:
        task = f"first letter of {word}"
        answer = word[0]
    elif mode == 4:
        task = f"last letter of {word}"
        answer = word[-1]
    else:
        task = f"characters in {word}"
        answer = " | ".join(word)
    templates = (
        "Character drill: task {task}; answer {answer}.",
        "Word skill: {task} -> {answer}.",
        "Spelling fact: {task}; correct answer {answer}.",
    )
    return {
        "category": "skills_spelling",
        "text": templates[(index // 6) % len(templates)].format(task=task, answer=answer),
    }


def _choice_example(index: int, rng: random.Random) -> dict[str, str]:
    kind = index % 4
    if kind == 0:
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        correct = str(a + b)
        distractors = _numeric_distractors(a + b, rng)
        question = f"Which option equals {a} + {b}?"
    elif kind == 1:
        word = _skill_word(index + 100_000, rng)
        correct = word[0]
        distractors = _letter_distractors(correct, rng)
        question = f"Which option is the first letter of {word}?"
    elif kind == 2:
        word = _skill_word(index + 200_000, rng)
        correct = str(len(word))
        distractors = _numeric_distractors(len(word), rng)
        question = f"Which option is the letter count of {word}?"
    else:
        word = _skill_word(index + 300_000, rng)
        correct = word[::-1]
        distractors = _word_distractors(word, rng)
        question = f"Which option is {word} reversed?"
    options = [correct, *distractors[:3]]
    rng.shuffle(options)
    labels = ("A", "B", "C", "D")
    correct_label = labels[options.index(correct)]
    text = (
        f"Choice drill: question {question}; "
        f"A. {options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}; "
        f"answer letter {correct_label}; answer text {correct}."
    )
    return {"category": "skills_choice", "text": text}


def _numeric_distractors(answer: int, rng: random.Random) -> list[str]:
    distractors: set[int] = set()
    while len(distractors) < 3:
        offset = rng.choice((-9, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 9))
        distractor = max(0, answer + offset)
        if distractor != answer:
            distractors.add(distractor)
    return [str(value) for value in sorted(distractors)]


def _letter_distractors(answer: str, rng: random.Random) -> list[str]:
    letters = [letter for letter in string.ascii_lowercase if letter != answer]
    rng.shuffle(letters)
    return letters[:3]


def _word_distractors(word: str, rng: random.Random) -> list[str]:
    answer = word[::-1]
    distractors = {
        word,
        "".join(sorted(word)),
        word[1:] + word[:1],
        word[-1:] + word[:-1],
    }
    distractors.discard(answer)
    while len(distractors) < 4:
        candidate = "".join(rng.choice(string.ascii_lowercase) for _ in range(len(word)))
        if candidate != answer:
            distractors.add(candidate)
    return sorted(distractors)[:3]


def _skill_word(index: int, rng: random.Random) -> str:
    if index % 10 == 0:
        return TRAIN_WORDS[(index // 10) % len(TRAIN_WORDS)]
    length = 4 + (index % 5)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _write_recipe(
    recipe_out: str | Path,
    *,
    skills_path: Path,
    base_corpus: Path | None,
    force: bool,
) -> Path:
    recipe_path = Path(recipe_out)
    if recipe_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing corpus recipe: {recipe_path}")
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, str]] = []
    if base_corpus:
        sources.append({"path": _relative_path(base_corpus, recipe_path.parent), "label": "base"})
    sources.append({"path": _relative_path(skills_path, recipe_path.parent), "label": "micro_skills"})
    recipe_path.write_text(json.dumps({
        "name": "Picochat base plus micro-skills",
        "description": "Mixes the base corpus with arithmetic, spelling, and choice drills for pretraining.",
        "sources": sources,
        "exclude": [
            "**/.DS_Store",
            "**/.git/**",
            "**/__pycache__/**",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    return recipe_path


def _relative_path(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return str(path)


def _report_markdown(report: SkillsCorpusReport) -> str:
    categories = "\n".join(f"- {name}: {count}" for name, count in report.categories.items())
    return f"""# Picochat Micro-Skills Corpus

Output: `{report.output_path}`

Rows: `{report.total_rows}`

Characters: `{report.characters_written}`

Seed: `{report.seed}`

Base corpus: `{report.base_corpus or 'none'}`

Recipe: `{report.recipe_path or 'none'}`

## Categories

{categories}

## Honesty Notes

- This is pretraining data, not eval data.
- Drills are compact one-example-per-line records so duplicate-line checks catch true replay, not repeated formatting labels.
- Held-out eval spelling words are intentionally excluded from the fixed word pool.
- Random synthetic strings are included so character operations are not only memorized words.
- Use a separate held-out eval after training to verify transfer.
"""
