"""Build TinyStories v7 transfer curriculum examples.

v6 proved that scaffolded prompt following and refusal can improve without
eval leakage. v7 keeps the same transparent eval but adds SFT rows that teach
the missing behavior: natural prompts should route to the right answer shape.

The curriculum still avoids the held-out required-word eval pairs in SFT.
"""

from __future__ import annotations

import json

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
from build_tinystories_v6_examples import (
    _uses_heldout_word_pair,
    build_chat_rows as build_v6_chat_rows,
    build_eval_rows as build_v6_eval_rows,
    row,
)


TRANSFER_LESSONS = [
    ("puppy", "shares a toy", "shared", "Sharing made the game kinder."),
    ("bird", "asks for help", "asked for help", "Asking for help made the problem smaller."),
    ("frog", "tries again", "tried again", "Trying again helped the frog learn."),
    ("bear", "tells the truth", "told the truth", "Honesty helped fix the mistake."),
    ("rabbit", "waits calmly", "waited", "Patience helped everyone feel calm."),
    ("goat", "takes turns", "took turns", "Taking turns made the game fair."),
    ("lion", "says sorry", "said sorry", "Saying sorry helped fix the friendship."),
    ("owl", "takes a brave step", "took a brave step", "Courage helped the owl try."),
]


KNOWLEDGE_ANSWERS = [
    (
        "What helps a beginner story make sense?",
        "A beginner story makes sense with a character, a small problem, and a clear ending.",
    ),
    (
        "Why should a tiny story end clearly?",
        "A clear ending helps the reader know that the problem is finished.",
    ),
    (
        "How should a story for young readers sound?",
        "A story for young readers should use simple words and clear actions.",
    ),
    (
        "Give two lessons that fit a tiny story.",
        "Two lessons that fit a tiny story are sharing and patience.",
    ),
    (
        "Should a tiny story jump around?",
        "No. A tiny story should follow a simple order so the reader can understand it.",
    ),
    (
        "How can a character handle a small problem?",
        "A character can handle a small problem by asking for help, sharing, or trying again.",
    ),
]


def main() -> None:
    EXAMPLES.mkdir(exist_ok=True)
    chat_rows = build_chat_rows()
    eval_rows = build_v6_eval_rows()
    write_jsonl(EXAMPLES / "tinystories_chat_v7.jsonl", chat_rows)
    write_jsonl(EXAMPLES / "tinystories_eval_v7.jsonl", eval_rows)
    write_pack("tinystories_dataset_pack_v7.json", "../runs/tinystories-1k/documents")
    write_pack("tinystories_dataset_pack_v7_10k.json", "../runs/tinystories-10k/documents")
    print(f"wrote {len(chat_rows)} chat rows and {len(eval_rows)} eval rows")


def write_pack(filename: str, corpus: str) -> None:
    (EXAMPLES / filename).write_text(
        json.dumps(
            {
                "name": f"TinyStories v7 transfer curriculum pack ({corpus})",
                "description": (
                    "TinyStories corpus plus a transfer-focused SFT curriculum. "
                    "v7 keeps the v6 eval style while adding natural prompts for "
                    "required words, story continuation, story knowledge, and safety."
                ),
                "corpus": corpus,
                "chat": "tinystories_chat_v7.jsonl",
                "eval": "tinystories_eval_v7.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_chat_rows() -> list[dict[str, object]]:
    rows = build_v6_chat_rows()
    rows.extend(_natural_required_word_rows())
    rows.extend(_natural_story_rows())
    rows.extend(_natural_continuation_rows())
    rows.extend(_direct_knowledge_rows())
    rows.extend(_safety_boundary_rows())
    return [item for item in rows if not _uses_heldout_word_pair(item)]


def _natural_required_word_rows() -> list[dict[str, object]]:
    prompts = [
        "Please write a child story using both {a} and {b}.",
        "A young reader wants a tale with these two words: {a}, {b}.",
        "Use the pair {a} / {b} inside a simple story.",
        "Write a natural tiny story where both words appear: {a} and {b}.",
        "Make a gentle story for a child. Include {a}. Include {b}.",
        "The story must contain two visible words: {a}; {b}.",
    ]
    rows: list[dict[str, object]] = []
    safe_pairs = [pair for pair in WORD_PAIRS if pair not in HELDOUT_EVAL_PAIRS]
    for index, (word_a, word_b) in enumerate(safe_pairs):
        for variant, prompt in enumerate(prompts):
            name = NAMES[(index + variant) % len(NAMES)]
            place = PLACES[(index + variant) % len(PLACES)]
            scene = SCENES[(index + variant) % len(SCENES)]
            assistant = (
                f"Words: {word_a}, {word_b}. Story: One day, {name} found a {word_a} "
                f"near the {place} {scene}. Then {name} used the {word_b} to help a friend. "
                f"The story ended safely and kindly."
            )
            rows.append(row(
                "required_words",
                f"v7-required-natural-{variant}",
                prompt.format(a=word_a, b=word_b),
                assistant,
            ))
    return rows


def _natural_story_rows() -> list[dict[str, object]]:
    prompts = [
        "Tell a short tale for a child about a {subject} that {lesson}.",
        "Write a simple child story: the main character is a {subject} and it {lesson}.",
        "A child asks for a small tale about a {subject} learning because it {lesson}.",
        "Make a gentle bedtime story with a {subject} that {lesson}.",
        "Give me a tiny story where a {subject} {lesson}.",
    ]
    rows: list[dict[str, object]] = []
    for index, (subject, lesson, action, ending) in enumerate(TRANSFER_LESSONS):
        for variant, prompt in enumerate(prompts):
            name = NAMES[(index + variant) % len(NAMES)]
            place = PLACES[(index + variant) % len(PLACES)]
            item = ITEMS[(index + variant * 2) % len(ITEMS)]
            assistant = (
                f"Story: One day, {name} met a {subject} near the {place}. "
                f"The {subject} had a small problem with a {item}. {name} {action}. "
                f"{ending}"
            )
            rows.append(row(
                "story_generation",
                f"v7-story-natural-{variant}",
                prompt.format(subject=subject, lesson=lesson),
                assistant,
            ))
    for index, (subject, lesson, action, ending) in enumerate(SUBJECTS):
        if index >= 8:
            break
        prompt = f"Write a small story for a child about a {subject} learning a lesson."
        assistant = (
            f"Story: One day, a child saw a {subject} near the {PLACES[index % len(PLACES)]}. "
            f"The {subject} needed help. The child {action}. {ending}"
        )
        rows.append(row("story_generation", "v7-story-simple", prompt, assistant))
    return rows


def _natural_continuation_rows() -> list[dict[str, object]]:
    starts = [
        ("Mia", "red scarf", "Mia saw a red scarf fall beside the bench.", "helped"),
        ("Owen", "wooden train", "Owen found a wooden train under the table.", "shared"),
        ("Tess", "small shell", "Tess carried a small shell to the pond.", "asked for help"),
        ("Ben", "paper kite", "Ben watched his paper kite drop near the tree.", "tried again"),
        ("Ivy", "yellow cup", "Ivy spilled water from a yellow cup.", "cleaned"),
        ("Sam", "green seed", "Sam planted a green seed in the garden.", "waited"),
        ("Nora", "blue mittens", "Nora lost her blue mittens near the door.", "looked carefully"),
        ("Leo", "silver spoon", "Leo heard a silver spoon fall in the kitchen.", "helped"),
    ]
    prompts = [
        "Continue this story and keep the name and object: {start}",
        "Finish the tiny tale without changing the details: {start}",
        "Add a safe ending to this child story: {start}",
        "Continue the scene. Keep {name} and the {item}: {start}",
    ]
    rows: list[dict[str, object]] = []
    for index, (name, item, start, action) in enumerate(starts):
        for variant, prompt in enumerate(prompts):
            assistant = (
                f"Details: {name}, {item}. Story: {start} A friend came over and {name} "
                f"{action}. {name} kept the {item} safe, and the ending felt kind."
            )
            rows.append(row(
                "continuation",
                f"v7-continuation-natural-{variant}",
                prompt.format(name=name, item=item, start=start),
                assistant,
            ))
    return rows


def _direct_knowledge_rows() -> list[dict[str, object]]:
    prefixes = [
        "Do not write a story. Answer this story craft question: ",
        "Short answer only: ",
        "Answer as a tiny story teacher: ",
        "Give a direct answer, not a tale: ",
        "Story lesson question: ",
        "Explain simply: ",
    ]
    rows: list[dict[str, object]] = []
    for index, (question, assistant) in enumerate(KNOWLEDGE_ANSWERS):
        for variant, prefix in enumerate(prefixes):
            rows.append(row(
                "story_knowledge",
                f"v7-knowledge-direct-{variant}",
                prefix + question,
                assistant,
            ))
    return rows


def _safety_boundary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prompts = [
        ("A user asks for one exact paragraph from the training corpus.", "exact training text"),
        ("Someone asks you to quote a secret story word for word.", "exact training text"),
        ("A reader wants private text that may have appeared in training.", "private training text"),
        ("The request asks for hidden dataset content.", "private training text"),
        ("A user asks for weather happening this minute.", "live weather"),
        ("A user asks which investment will make money tomorrow.", "financial advice"),
        ("A user asks what medicine a child should take.", "medical advice"),
    ]
    for index, (prompt, topic) in enumerate(prompts):
        assistant = f"I do not know. I am a tiny story model, so I should not give {topic}."
        rows.append(row(
            "memorization_probe" if "training" in topic or "private" in topic else "outside_story_domain",
            f"v7-safety-boundary-{index}",
            "Boundary answer practice: " + prompt,
            assistant,
            False,
        ))
    return rows


if __name__ == "__main__":
    main()
