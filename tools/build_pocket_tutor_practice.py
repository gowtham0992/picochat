#!/usr/bin/env python3
"""Build practice-only Pocket Tutor Lab SFT and eval rows.

This generator is for pre-hackathon recipe practice. Rebuild the official
dataset during the hackathon window before submitting anything.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable


SYSTEM_PROMPT = (
    "You are Pocket Tutor Lab, a private small-model tutor for one learner. "
    "Give short, kind, age-appropriate feedback. Prefer hints before answers. "
    "When the app provides the correct answer, use it for feedback but do not "
    "lecture. Return valid compact JSON with the requested keys."
)

SPELLING_WORDS = [
    "planet", "window", "garden", "silver", "basket", "pencil", "rabbit",
    "summer", "yellow", "forest", "button", "market", "dragon", "school",
    "animal", "castle", "bridge", "little", "number", "orange", "people",
    "purple", "rocket", "turtle", "winter", "friend", "mother", "simple",
]

READING_PASSAGES = [
    (
        "Mina watered the basil before school. By dinner, the leaves stood taller "
        "and smelled fresh.",
        "What did Mina water?",
        "the basil",
        "reading_detail",
    ),
    (
        "Aiden packed a raincoat because gray clouds covered the sky. He stayed "
        "dry on the walk home.",
        "Why did Aiden pack a raincoat?",
        "because gray clouds covered the sky",
        "reading_reason",
    ),
    (
        "The class turtle hid under a rock when the room became loud. It came out "
        "again after everyone whispered.",
        "What helped the turtle come out again?",
        "everyone whispered",
        "reading_cause",
    ),
]


@dataclass(frozen=True)
class TutorRow:
    category: str
    skill: str
    difficulty: str
    messages: list[dict[str, str]]
    expected_answer: str | None = None
    metadata: dict[str, str | int | bool] | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/pocket-tutor-practice-pack-v1")
    parser.add_argument("--train-rows", type=int, default=360)
    parser.add_argument("--eval-rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_rows = build_rows(args.train_rows, rng, split="train")
    eval_rows = build_rows(args.eval_rows, rng, split="eval")

    train_path = out_dir / "pocket_tutor_train_messages.jsonl"
    eval_path = out_dir / "pocket_tutor_eval_prompts.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    manifest = {
        "name": "pocket-tutor-practice-pack",
        "status": "practice-only",
        "warning": "Rebuild official data during the hackathon window.",
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "categories": count_categories(train_rows),
        "eval_categories": count_categories(eval_rows),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "recommended_model": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(readme_text(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def build_rows(count: int, rng: random.Random, *, split: str) -> list[TutorRow]:
    builders = [
        build_math_feedback,
        build_spelling_feedback,
        build_hint_without_answer,
        build_reading_feedback,
        build_parent_summary,
    ]
    rows = [builders[index % len(builders)](rng, split=split, index=index) for index in range(count)]
    rng.shuffle(rows)
    return rows


def build_math_feedback(rng: random.Random, *, split: str, index: int) -> TutorRow:
    op = rng.choice(["+", "-", "x"])
    if op == "+":
        a, b = rng.randint(8, 99), rng.randint(3, 77)
        answer = a + b
        mistake = answer + rng.choice([-10, -1, 1, 10])
        stage = "addition_carry" if (a % 10) + (b % 10) >= 10 else "addition_basic"
    elif op == "-":
        a, b = rng.randint(30, 140), rng.randint(2, 29)
        if b > a:
            a, b = b, a
        answer = a - b
        mistake = answer + rng.choice([-10, -1, 1, 10])
        stage = "subtraction_borrow" if (a % 10) < (b % 10) else "subtraction_basic"
    else:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        answer = a * b
        mistake = answer + rng.choice([-b, -1, 1, b])
        stage = "multiplication_basic"
    correct = rng.random() < 0.35
    learner_answer = answer if correct else mistake
    user = (
        f"Learner: Maya, grade 4\n"
        f"Skill: arithmetic/{stage}\n"
        f"Problem: {a} {op} {b}\n"
        f"Learner answer: {learner_answer}\n"
        f"Correct answer: {answer}\n"
        "Return JSON keys: verdict, feedback, hint, next_step, review_tag."
    )
    assistant = tutor_json(
        verdict="correct" if correct else "not_yet",
        feedback=(
            "Correct. Nice work keeping the numbers lined up."
            if correct else
            f"Not yet. The correct answer is {answer}; check the place value before trying again."
        ),
        hint=(
            "Try one more similar problem without rushing."
            if correct else
            "Write the ones column first, then the tens column."
        ),
        next_step="Move this skill up one notch." if correct else "Try a nearby problem with smaller numbers.",
        review_tag=stage,
    )
    return row("math_feedback", "arithmetic", stage, user, assistant, str(answer), index, split)


def build_spelling_feedback(rng: random.Random, *, split: str, index: int) -> TutorRow:
    word = rng.choice(SPELLING_WORDS)
    mode = rng.choice(["spell", "first_letter", "last_letter", "count_letters", "reverse"])
    if mode == "spell":
        prompt = f"Spell the word: {word}"
        answer = word
        wrong = word[:-1] + rng.choice("aeiou")
    elif mode == "first_letter":
        prompt = f"What is the first letter in {word}?"
        answer = word[0]
        wrong = word[-1]
    elif mode == "last_letter":
        prompt = f"What is the last letter in {word}?"
        answer = word[-1]
        wrong = word[0]
    elif mode == "count_letters":
        prompt = f"How many letters are in {word}?"
        answer = str(len(word))
        wrong = str(len(word) + rng.choice([-1, 1]))
    else:
        prompt = f"Write {word} backward."
        answer = word[::-1]
        wrong = word[::-1][:-1]
    correct = rng.random() < 0.35
    learner_answer = answer if correct else wrong
    user = (
        f"Learner: Maya, grade 4\n"
        f"Skill: spelling/{mode}\n"
        f"Prompt: {prompt}\n"
        f"Learner answer: {learner_answer}\n"
        f"Correct answer: {answer}\n"
        "Return JSON keys: verdict, feedback, hint, next_step, review_tag."
    )
    assistant = tutor_json(
        verdict="correct" if correct else "not_yet",
        feedback=(
            "Correct. You looked carefully at the word."
            if correct else
            f"Not yet. Look at each letter in {word} slowly."
        ),
        hint="Say the letters out loud and point to each one.",
        next_step="Review one similar word." if correct else "Try the same word with a smaller hint.",
        review_tag=mode,
    )
    return row("spelling_feedback", "spelling", mode, user, assistant, answer, index, split)


def build_hint_without_answer(rng: random.Random, *, split: str, index: int) -> TutorRow:
    a, b = rng.randint(4, 18), rng.randint(3, 17)
    answer = a + b
    user = (
        f"Learner: Maya, grade 4\n"
        f"Problem: {a} + {b}\n"
        "The learner clicked Hint before answering. Give one hint only. "
        "Do not reveal the answer. Return JSON keys: hint, no_answer_given, next_step."
    )
    assistant = json.dumps({
        "hint": "Break the bigger number into tens and ones, then add the smaller number.",
        "no_answer_given": True,
        "next_step": "Ask the learner to write the ones-column result first.",
    }, separators=(",", ":"))
    return row("hint_without_answer", "arithmetic", "hint_ladder", user, assistant, str(answer), index, split)


def build_reading_feedback(rng: random.Random, *, split: str, index: int) -> TutorRow:
    passage, question, answer, stage = rng.choice(READING_PASSAGES)
    correct = rng.random() < 0.45
    learner_answer = answer if correct else rng.choice(["the sky", "a rock", "before school", "the class"])
    user = (
        f"Learner: Maya, grade 4\n"
        f"Skill: reading/{stage}\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        f"Learner answer: {learner_answer}\n"
        f"Expected answer: {answer}\n"
        "Return JSON keys: verdict, feedback, hint, next_step, review_tag."
    )
    assistant = tutor_json(
        verdict="correct" if correct else "not_yet",
        feedback=(
            "Correct. You found the detail in the passage."
            if correct else
            "Not yet. The answer is stated in the passage, so reread the sentence with the clue."
        ),
        hint="Underline the sentence that mentions the question words.",
        next_step="Try one more detail question." if correct else "Reread the short passage and answer again.",
        review_tag=stage,
    )
    return row("reading_feedback", "reading", stage, user, assistant, answer, index, split)


def build_parent_summary(rng: random.Random, *, split: str, index: int) -> TutorRow:
    weak = rng.choice(["subtraction with borrowing", "reversing longer words", "reading detail questions"])
    strong = rng.choice(["single-digit addition", "first-letter spelling", "staying focused for five cards"])
    user = (
        "Learner: Maya, grade 4\n"
        "Round summary: 5 cards completed. Correct: 3. Hints used: 2.\n"
        f"Strong area: {strong}.\n"
        f"Needs review: {weak}.\n"
        "Write a parent/tutor summary in JSON keys: praise, needs_review, next_round, caution."
    )
    assistant = json.dumps({
        "praise": f"Maya did well with {strong}.",
        "needs_review": weak,
        "next_round": f"Give two easier {weak} cards, then one mixed review card.",
        "caution": "Keep the session short and stop if the learner is frustrated.",
    }, separators=(",", ":"))
    return row("parent_summary", "progress", "summary", user, assistant, weak, index, split)


def tutor_json(*, verdict: str, feedback: str, hint: str, next_step: str, review_tag: str) -> str:
    return json.dumps({
        "verdict": verdict,
        "feedback": feedback,
        "hint": hint,
        "next_step": next_step,
        "review_tag": review_tag,
    }, separators=(",", ":"))


def row(
    category: str,
    skill: str,
    difficulty: str,
    user: str,
    assistant: str,
    expected_answer: str,
    index: int,
    split: str,
) -> TutorRow:
    return TutorRow(
        category=category,
        skill=skill,
        difficulty=difficulty,
        expected_answer=expected_answer,
        metadata={"index": index, "split": split, "practice_only": True},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    )


def write_jsonl(path: Path, rows: Iterable[TutorRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def count_categories(rows: list[TutorRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rows:
        counts[item.category] = counts.get(item.category, 0) + 1
    return dict(sorted(counts.items()))


def readme_text(manifest: dict) -> str:
    command = (
        "picochat train hf-sft \\\n"
        "  --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
        f"  --input {manifest['train_path']} \\\n"
        "  --out-dir runs/qwen-pocket-tutor-practice-lora-v1 \\\n"
        "  --max-steps 200 \\\n"
        "  --batch-size 1 \\\n"
        "  --grad-accum-steps 8 \\\n"
        "  --learning-rate 0.00002 \\\n"
        "  --lr-warmup-steps 20 \\\n"
        "  --max-length 1024 \\\n"
        "  --device cuda \\\n"
        "  --precision bf16 \\\n"
        "  --gradient-checkpointing \\\n"
        "  --peft lora \\\n"
        "  --lora-rank 16 \\\n"
        "  --lora-alpha 32"
    )
    return (
        "# Pocket Tutor Lab Practice Pack\n\n"
        "Status: practice-only. Rebuild official data during the hackathon window.\n\n"
        f"- Train rows: {manifest['train_rows']}\n"
        f"- Eval rows: {manifest['eval_rows']}\n"
        f"- Recommended model: `{manifest['recommended_model']}`\n\n"
        "## Practice fine-tune command\n\n"
        "```bash\n"
        f"{command}\n"
        "```\n\n"
        "The rows train short JSON tutor behavior for arithmetic, spelling, reading, "
        "hint ladders, and parent summaries. Deterministic app code should still "
        "grade arithmetic and spelling answers; the model should explain and adapt.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
