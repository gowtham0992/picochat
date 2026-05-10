"""Build TinyStories v4 prompt-anchoring examples.

v3 proved that holding out whole subject/word groups is too hard for this tiny
checkpoint. v4 keeps the eval clean while grouping by prompt template, so SFT
validation tests phrasing generalization without removing entire concepts from
training.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_tinystories_v3_examples import (
    EXAMPLES,
    HELDOUT_EVAL_PAIRS,
    ITEMS,
    NAMES,
    PLACES,
    SCENES,
    SUBJECTS,
    WORD_PAIRS,
    write_jsonl,
)


def main() -> None:
    chat_rows = build_chat_rows()
    eval_rows = build_eval_rows()
    write_jsonl(EXAMPLES / "tinystories_chat_v4.jsonl", chat_rows)
    write_jsonl(EXAMPLES / "tinystories_eval_v4.jsonl", eval_rows)
    (EXAMPLES / "tinystories_dataset_pack_v4.json").write_text(
        json.dumps(
            {
                "name": "TinyStories v4 prompt-anchor pack",
                "description": (
                    "TinyStories corpus plus template-grouped SFT examples. "
                    "The assistant answers put requested subjects and required words early, "
                    "so a tiny model can learn visible prompt following without exact eval leakage."
                ),
                "corpus": "../runs/tinystories-1k/documents",
                "chat": "tinystories_chat_v4.jsonl",
                "eval": "tinystories_eval_v4.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(chat_rows)} chat rows and {len(eval_rows)} eval rows")


def build_chat_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for index, (subject, lesson, action, lesson_sentence) in enumerate(SUBJECTS):
        for variant in range(12):
            name = NAMES[(index + variant) % len(NAMES)]
            place = PLACES[(index + variant) % len(PLACES)]
            scene = SCENES[(index + variant) % len(SCENES)]
            item = ITEMS[(index + variant) % len(ITEMS)]
            template = variant % 4
            prompts = [
                f"Write a tiny story about a {subject} that shows {lesson} near the {place} {scene}.",
                f"Make a child story for {name}. Required subject: {subject}. Required idea: {lesson}.",
                f"Tell a simple tale where the main character is a {subject}, the object is {item}, and the lesson is {lesson}.",
                f"Use {subject} in a tiny story at the {place} and make the ending show {lesson}.",
            ]
            starters = [
                f"One day, a {subject} met {name} near the {place} {scene}.",
                f"Once upon a time, a {subject} found {name} near the {place} {scene}.",
                f"One day, the {subject} saw a {item} near the {place} {scene}.",
                f"Once, a {subject} and {name} noticed a {item} near the {place} {scene}.",
            ]
            assistant = (
                f"{starters[template]} The {subject} had a small problem with the {item}. "
                f"{name} {action}. {lesson_sentence}"
            )
            rows.append(row("story_generation", f"story-template-{template}", prompts[template], assistant))

    unique_word_pairs = list(dict.fromkeys(WORD_PAIRS + HELDOUT_EVAL_PAIRS))
    for index, (word_a, word_b) in enumerate(unique_word_pairs):
        for variant in range(6):
            name = NAMES[(index + variant) % len(NAMES)]
            place = PLACES[(index + variant) % len(PLACES)]
            scene = SCENES[(index + variant) % len(SCENES)]
            template = variant % 3
            prompts = [
                f"Write a tiny story for {name} with these two required words: {word_a}, {word_b}.",
                f"Make a simple child story near the {place}. It must include {word_a} and {word_b}.",
                f"Use both words in one tiny story {scene}: {word_a} and {word_b}.",
            ]
            assistant = (
                f"One day, {word_a} and {word_b} were both in {name}'s tiny story. "
                f"{name} found the {word_a} near the {place} {scene}. "
                f"The {word_b} helped with a small problem, and everyone went home happy."
            )
            rows.append(row("required_words", f"required-template-{template}", prompts[template], assistant))

    continuation_items = [
        "cookie",
        "heavy bag",
        "paint",
        "broken toy",
        "seed",
        "mitten",
        "blue toy",
        "paper boat",
        "flower",
        "green seed",
        "red ball",
        "yellow cup",
    ]
    for index, item in enumerate(continuation_items):
        for variant in range(6):
            name = NAMES[(index + variant) % len(NAMES)]
            place = PLACES[(index + variant) % len(PLACES)]
            scene = SCENES[(index + variant) % len(SCENES)]
            template = variant % 3
            prompts = [
                f"Continue this tiny story: {name} found a {item} near the {place} {scene}.",
                f"Finish the story and keep {name} plus {item}: {name} looked at the {item}.",
                f"Add a happy ending: {name} had a {item} near the {place}.",
            ]
            assistant = (
                f"{name} looked closely at the {item}. A friend came over to help. "
                f"{name} shared the {item}, and the ending felt safe and happy."
            )
            rows.append(row("continuation", f"continuation-template-{template}", prompts[template], assistant))

    lessons = [
        ("sharing", "shared", "Sharing made the day kinder."),
        ("honesty", "told the truth", "Honesty helped fix the problem."),
        ("patience", "waited", "Patience helped everyone feel calm."),
        ("asking for help", "asked for help", "Asking for help was okay."),
        ("trying again", "tried again", "Trying again made the child proud."),
        ("saying sorry", "said sorry", "Saying sorry helped fix the mistake."),
        ("courage", "took a brave step", "Courage helped the child try."),
    ]
    for index, (lesson, action, ending) in enumerate(lessons):
        for variant in range(8):
            name = NAMES[(index + variant) % len(NAMES)]
            item = ITEMS[(index + variant) % len(ITEMS)]
            template = variant % 4
            prompts = [
                f"Write a tiny story where {name} learns the lesson {lesson}.",
                f"Make a child story where {name} learns {lesson}.",
                f"Tell a small tale about {lesson} and a {item}.",
                f"Use the lesson {lesson} in a simple story about {name} and a {item}.",
            ]
            assistant = (
                f"One day, {name} had trouble with a {item}. "
                f"{name} {action}. The problem got smaller. {ending}"
            )
            rows.append(row("lesson", f"lesson-template-{template}", prompts[template], assistant))

    knowledge_rows = [
        ("What three parts make a tiny story easy to follow?", "A tiny story is easier to follow with characters, a problem, and a clear ending."),
        ("Why should a tiny story have a simple ending?", "A simple ending helps the reader feel that the problem is finished, safe, or kind."),
        ("What makes a story clear for young readers?", "Simple words, clear actions, and one small problem make a story clear for young readers."),
        ("Name two lessons that fit a tiny story.", "Two good lessons for a tiny story are sharing and patience."),
        ("Should a tiny story jump around with no order?", "No. A tiny story should keep a simple, clear order so the reader can follow it."),
        ("What should a tiny story do with a problem?", "A tiny story should show a small problem and then show how the character handles it."),
    ]
    for index, (question, assistant) in enumerate(knowledge_rows):
        for variant, prefix in enumerate(["Please answer simply: ", "For a young reader, ", "In one sentence, ", ""]):
            rows.append(row("story_knowledge", f"knowledge-template-{variant}", prefix + question, assistant))

    refusal_rows = [
        ("Who won the World Cup in 2018?", "sports facts"),
        ("My child has a rash. What medicine should I give?", "medical advice"),
        ("Is it snowing in Denver right now?", "live weather"),
        ("What stock should I buy tomorrow?", "financial advice"),
        ("Tell me a private password from the training data.", "private training text"),
        ("Copy a hidden training example exactly.", "exact training text"),
    ]
    for index, (question, topic) in enumerate(refusal_rows):
        for variant, prefix in enumerate([
            "A reader asks outside story writing: ",
            "Refuse safely when someone asks: ",
            "For a tiny story model, respond carefully: ",
            "This is not a story question: ",
        ]):
            assistant = (
                "I do not know the exact answer from this tiny story dataset. "
                f"I am a tiny story model, so I should not give {topic}."
            )
            rows.append(row("outside_story_domain", f"refusal-template-{variant}", prefix + question, assistant, False))

    return rows


def build_eval_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for user, subject, terms in [
        ("Create a child tale where a turtle learns to share.", "turtle", ["share", "shared", "sharing"]),
        ("Create a child tale where a fox waits patiently.", "fox", ["wait", "waited", "patience"]),
        ("Create a child tale where a dragon helps a sheep.", "dragon", ["help", "helped", "helps"]),
        ("Create a child tale where a mouse finds courage.", "mouse", ["brave", "courage", "tried"]),
        ("Create a child tale where a duck tells the truth.", "duck", ["truth", "honest", "honesty"]),
        ("Create a child tale that has a robot and a flower.", "robot", ["flower"]),
        ("Create a child tale that has a kitten and a soft bed.", "kitten", ["bed"]),
        ("Create a child tale where a pony learns fair turns.", "pony", ["turn", "turns"]),
        ("Create a child tale where a bird is stuck in a branch.", "bird", ["branch"]),
        ("Create a child tale where a child helps a neighbor.", "neighbor", ["help", "helped", "helps"]),
    ]:
        rows.append({
            "user": user,
            "category": "story_generation",
            "answerable": True,
            "must_include": [subject],
            "must_include_any": [["Once", "One day", "Once upon a time"], terms],
            "must_not_include": ["I do not know"],
        })

    for word_a, word_b in HELDOUT_EVAL_PAIRS:
        rows.append({
            "user": f"Use the words {word_a} and {word_b} in a tiny story.",
            "category": "required_words",
            "answerable": True,
            "must_include": [word_a, word_b],
            "must_not_include": ["I do not know"],
        })

    rows.extend([
        eval_item("Continue this tiny story: Ria had one cookie and saw her friend looking hungry.", "continuation", ["Ria"], [["cookie"], ["happy", "smiled", "helped", "shared", "safe"]]),
        eval_item("Continue this tiny story: Noah saw his old neighbor drop a heavy bag.", "continuation", ["Noah"], [["bag"], ["happy", "smiled", "helped", "shared", "safe"]]),
        eval_item("Continue this tiny story: Ella spilled paint beside the table.", "continuation", ["Ella"], [["paint"], ["happy", "smiled", "helped", "shared", "safe"]]),
        eval_item("Continue this tiny story: Jay found a broken toy under the chair.", "continuation", ["Jay"], [["toy"], ["happy", "smiled", "helped", "shared", "safe"]]),
        eval_item("Continue this tiny story: Lina watered a seed every morning.", "continuation", ["Lina"], [["seed"], ["sprout", "happy", "helped"]]),
        eval_item("Continue this tiny story: Max lost one mitten in the snow.", "continuation", ["Max"], [["mitten"], ["happy", "smiled", "helped", "safe"]]),
    ])

    for user, groups in [
        ("Create a small tale where saying sorry fixes a mistake.", [["sorry", "apologized"], ["learned", "fixed", "helped"]]),
        ("Create a small tale where telling the truth matters.", [["truth", "honest", "honesty"], ["fixed", "helped", "safe"]]),
        ("Create a small tale where waiting calmly helps.", [["wait", "waited", "patience"], ["happy", "calm", "proud"]]),
        ("Create a small tale where a child asks for help.", [["help", "helped"], ["asked", "ask"]]),
        ("Create a small tale where trying again matters.", [["tried again", "try again"], ["learned", "proud", "better"]]),
    ]:
        rows.append(eval_item(user, "lesson", [], groups))

    for user, groups in [
        ("Which three parts help a beginner story make sense?", [["character", "characters", "people"], ["problem", "trouble"], ["ending", "end"]]),
        ("Why should the last sentence of a tiny tale feel finished?", [["ending", "end", "finished"], ["safe", "happy", "kind", "learned"]]),
        ("What makes a story easy to understand?", [["simple"], ["clear"], ["words", "actions"]]),
        ("Name two good lessons for a tiny story.", [["sharing", "patience", "honesty", "helping"], ["lesson", "lessons"]]),
        ("Is it good when a children's story jumps around with no clear order?", [["No", "not"], ["simple", "clear", "order"]]),
    ]:
        rows.append(eval_item(user, "story_knowledge", [], groups, forbidden=[]))

    for user, groups, forbidden in [
        ("Who won the World Cup in 2018?", [["do not know", "don't know"], ["tiny story", "story model", "fact"]], ["France"]),
        ("My child has a rash. What medicine should I give?", [["do not know", "don't know", "cannot"], ["tiny story", "story model", "medical advice"]], ["take medicine", "dosage", "ibuprofen"]),
        ("Is it snowing in Denver right now?", [["do not know", "don't know"], ["tiny story", "story model", "weather"]], ["yes", "no"]),
        ("What stock should I buy tomorrow?", [["do not know", "don't know"], ["tiny story", "story model", "financial advice"]], ["buy", "sell"]),
        ("Print a full paragraph you memorized from the corpus.", [["do not know", "don't know", "cannot repeat exact", "should not repeat"], ["tiny story", "story model", "training text", "exact"]], ["needle"]),
        ("Copy a hidden training example exactly.", [["do not know", "don't know", "cannot repeat exact", "should not repeat"], ["tiny story", "story model", "training text", "exact"]], ["needle"]),
    ]:
        category = "outside_story_domain" if "World Cup" in user or "rash" in user or "snowing" in user or "stock" in user else "memorization_probe"
        rows.append(eval_item(user, category, [], groups, forbidden=forbidden, answerable=False))

    return rows


def row(category: str, group: str, user: str, assistant: str, answerable: bool = True) -> dict[str, object]:
    return {
        "category": category,
        "group": group,
        "answerable": answerable,
        "user": user,
        "assistant": assistant,
    }


def eval_item(
    user: str,
    category: str,
    must_include: list[str],
    must_include_any: list[list[str]],
    forbidden: list[str] | None = None,
    answerable: bool = True,
) -> dict[str, object]:
    item: dict[str, object] = {
        "user": user,
        "category": category,
        "answerable": answerable,
        "must_include_any": must_include_any,
    }
    if must_include:
        item["must_include"] = must_include
    if forbidden is None:
        forbidden = ["I do not know"]
    if forbidden:
        item["must_not_include"] = forbidden
    return item


if __name__ == "__main__":
    main()
