"""Preflight checks for chat SFT and transparent eval JSONL files."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import difflib
import json
import math
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class TuningDataIssue:
    line: int
    message: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ChatSFTDataReport:
    path: str
    status: str
    summary: str
    num_rows: int
    num_examples: int
    empty_rows: int
    invalid_rows: int
    average_user_chars: float
    average_assistant_chars: float
    duplicate_user_rate: float
    duplicate_user_prompts: int
    duplicate_user_samples: tuple[str, ...]
    near_duplicate_user_pairs: int
    near_duplicate_user_samples: tuple[dict[str, Any], ...]
    categories: dict[str, int]
    category_entropy: float
    category_entropy_normalized: float
    assistant_length_distribution: dict[str, float | int]
    template_families: dict[str, int]
    answer_styles: dict[str, int]
    curriculum_label: str
    curriculum_breakdown: dict[str, int]
    quality_warnings: tuple[str, ...]
    issues: tuple[TuningDataIssue, ...]
    preview: tuple[dict[str, str], ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "issues": [issue.to_dict() for issue in self.issues],
            "preview": list(self.preview),
        }


@dataclass(frozen=True)
class ChatEvalDataReport:
    path: str
    status: str
    summary: str
    num_rows: int
    num_items: int
    empty_rows: int
    invalid_rows: int
    answerable_items: int
    unanswerable_items: int
    must_include_rules: int
    must_include_any_groups: int
    must_not_include_rules: int
    duplicate_user_rate: float
    duplicate_user_prompts: int
    duplicate_user_samples: tuple[str, ...]
    near_duplicate_user_pairs: int
    near_duplicate_user_samples: tuple[dict[str, Any], ...]
    categories: dict[str, int]
    category_entropy: float
    category_entropy_normalized: float
    splits: dict[str, int]
    levels: dict[str, int]
    heldout_categories: dict[str, int]
    answer_length_distribution: dict[str, float | int]
    template_families: dict[str, int]
    curriculum_label: str
    curriculum_breakdown: dict[str, int]
    quality_warnings: tuple[str, ...]
    issues: tuple[TuningDataIssue, ...]
    preview: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "issues": [issue.to_dict() for issue in self.issues],
            "preview": list(self.preview),
        }


def inspect_chat_sft_data(path: str | Path, preview_items: int = 3) -> ChatSFTDataReport:
    """Validate one-turn chat SFT JSONL before training."""
    path = str(path)
    records, empty_rows, read_issue = _read_jsonl_records(path)
    if read_issue:
        return ChatSFTDataReport(
            path=path,
            status="blocked",
            summary="Chat SFT data could not be read.",
            num_rows=0,
            num_examples=0,
            empty_rows=0,
            invalid_rows=1,
            average_user_chars=0.0,
            average_assistant_chars=0.0,
            duplicate_user_rate=0.0,
            duplicate_user_prompts=0,
            duplicate_user_samples=(),
            near_duplicate_user_pairs=0,
            near_duplicate_user_samples=(),
            categories={},
            category_entropy=0.0,
            category_entropy_normalized=0.0,
            assistant_length_distribution=_length_distribution([]),
            template_families={},
            answer_styles={},
            curriculum_label="unknown",
            curriculum_breakdown={},
            quality_warnings=(),
            issues=(read_issue,),
            preview=(),
        )

    issues: list[TuningDataIssue] = []
    examples: list[dict[str, str]] = []
    users: list[str] = []
    assistant_texts: list[str] = []
    categories: dict[str, int] = {}
    template_families: dict[str, int] = {}
    answer_styles: dict[str, int] = {}
    curriculum_breakdown: Counter[str] = Counter()
    user_chars = 0
    assistant_chars = 0
    for line_number, record in records:
        if not isinstance(record, dict):
            issues.append(TuningDataIssue(line_number, "row must be a JSON object"))
            continue
        if "__invalid_json__" in record:
            issues.append(TuningDataIssue(line_number, f"invalid JSON: {record['__invalid_json__']}"))
            continue
        user = record.get("user")
        assistant = record.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            issues.append(TuningDataIssue(line_number, "row must contain string user and assistant fields"))
            continue
        if not user.strip() or not assistant.strip():
            issues.append(TuningDataIssue(line_number, "user and assistant fields should not be empty"))
            continue
        category = record.get("category", "chat")
        if not isinstance(category, str) or not category.strip():
            issues.append(TuningDataIssue(line_number, "category field must be a non-empty string when present"))
            continue
        category = category.strip()
        template_family = _template_family(record, category)
        answer_style = _answer_style(record)
        examples.append({"user": user, "assistant": assistant, "category": category})
        users.append(user)
        assistant_texts.append(assistant)
        categories[category] = categories.get(category, 0) + 1
        template_families[template_family] = template_families.get(template_family, 0) + 1
        answer_styles[answer_style] = answer_styles.get(answer_style, 0) + 1
        curriculum_breakdown[_curriculum_bucket(category)] += 1
        user_chars += len(user)
        assistant_chars += len(assistant)

    num_rows = len(records)
    invalid_rows = len(issues)
    duplicate_user_rate, duplicate_user_prompts, duplicate_user_samples = _duplicate_prompt_stats(users)
    near_duplicates = _near_duplicate_prompt_report(users)
    category_entropy, category_entropy_normalized = _category_entropy(categories)
    curriculum_label = _curriculum_label(curriculum_breakdown, "sft")
    quality_warnings = _sft_quality_warnings(
        num_examples=len(examples),
        duplicate_user_rate=duplicate_user_rate,
        near_duplicate_user_pairs=near_duplicates["count"],
        category_entropy_normalized=category_entropy_normalized,
        curriculum_label=curriculum_label,
    )
    status, summary = _chat_sft_status(len(examples), invalid_rows, duplicate_user_rate)
    return ChatSFTDataReport(
        path=path,
        status=status,
        summary=summary,
        num_rows=num_rows,
        num_examples=len(examples),
        empty_rows=empty_rows,
        invalid_rows=invalid_rows,
        average_user_chars=(user_chars / len(examples)) if examples else 0.0,
        average_assistant_chars=(assistant_chars / len(examples)) if examples else 0.0,
        duplicate_user_rate=duplicate_user_rate,
        duplicate_user_prompts=duplicate_user_prompts,
        duplicate_user_samples=tuple(duplicate_user_samples),
        near_duplicate_user_pairs=near_duplicates["count"],
        near_duplicate_user_samples=tuple(near_duplicates["samples"]),
        categories=dict(sorted(categories.items())),
        category_entropy=category_entropy,
        category_entropy_normalized=category_entropy_normalized,
        assistant_length_distribution=_length_distribution(assistant_texts),
        template_families=dict(sorted(template_families.items())),
        answer_styles=dict(sorted(answer_styles.items())),
        curriculum_label=curriculum_label,
        curriculum_breakdown=dict(sorted(curriculum_breakdown.items())),
        quality_warnings=tuple(quality_warnings),
        issues=tuple(issues[:8]),
        preview=tuple(examples[:max(0, preview_items)]),
    )


def inspect_chat_eval_data(path: str | Path, preview_items: int = 3) -> ChatEvalDataReport:
    """Validate transparent chat eval JSONL before scoring."""
    path = str(path)
    records, empty_rows, read_issue = _read_jsonl_records(path)
    if read_issue:
        return ChatEvalDataReport(
            path=path,
            status="blocked",
            summary="Eval data could not be read.",
            num_rows=0,
            num_items=0,
            empty_rows=0,
            invalid_rows=1,
            answerable_items=0,
            unanswerable_items=0,
            must_include_rules=0,
            must_include_any_groups=0,
            must_not_include_rules=0,
            duplicate_user_rate=0.0,
            duplicate_user_prompts=0,
            duplicate_user_samples=(),
            near_duplicate_user_pairs=0,
            near_duplicate_user_samples=(),
            categories={},
            category_entropy=0.0,
            category_entropy_normalized=0.0,
            splits={},
            levels={},
            heldout_categories={},
            answer_length_distribution=_length_distribution([]),
            template_families={},
            curriculum_label="unknown",
            curriculum_breakdown={},
            quality_warnings=(),
            issues=(read_issue,),
            preview=(),
        )

    issues: list[TuningDataIssue] = []
    items: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    splits: dict[str, int] = {}
    levels: dict[str, int] = {}
    heldout_categories: dict[str, int] = {}
    template_families: dict[str, int] = {}
    curriculum_breakdown: Counter[str] = Counter()
    users: list[str] = []
    answer_texts: list[str] = []
    answerable_items = 0
    must_include_rules = 0
    must_include_any_groups = 0
    must_not_include_rules = 0
    for line_number, record in records:
        item, item_issues = _parse_eval_item(line_number, record)
        if item_issues:
            issues.extend(item_issues)
            continue
        items.append(item)
        users.append(item["user"])
        category = item["category"]
        categories[category] = categories.get(category, 0) + 1
        curriculum_breakdown[_curriculum_bucket(category)] += 1
        split = item["split"]
        splits[split] = splits.get(split, 0) + 1
        if split.lower() not in {"train", "sft", "practice"}:
            heldout_categories[category] = heldout_categories.get(category, 0) + 1
        level = item["level"]
        levels[level] = levels.get(level, 0) + 1
        template_family = item["template_family"]
        template_families[template_family] = template_families.get(template_family, 0) + 1
        if item["answerable"]:
            answerable_items += 1
        answer_texts.extend(_eval_answer_texts(item))
        must_include_rules += len(item["must_include"])
        must_include_any_groups += len(item["must_include_any"])
        must_not_include_rules += len(item["must_not_include"])

    num_rows = len(records)
    invalid_rows = len(issues)
    unanswerable_items = len(items) - answerable_items
    duplicate_user_rate, duplicate_user_prompts, duplicate_user_samples = _duplicate_prompt_stats(users)
    near_duplicates = _near_duplicate_prompt_report(users)
    category_entropy, category_entropy_normalized = _category_entropy(categories)
    curriculum_label = _curriculum_label(curriculum_breakdown, "eval")
    quality_warnings = _eval_quality_warnings(
        num_items=len(items),
        duplicate_user_rate=duplicate_user_rate,
        near_duplicate_user_pairs=near_duplicates["count"],
        category_entropy_normalized=category_entropy_normalized,
        heldout_categories=heldout_categories,
    )
    status, summary = _chat_eval_status(
        len(items),
        invalid_rows,
        must_include_rules + must_include_any_groups + must_not_include_rules,
        unanswerable_items,
    )
    return ChatEvalDataReport(
        path=path,
        status=status,
        summary=summary,
        num_rows=num_rows,
        num_items=len(items),
        empty_rows=empty_rows,
        invalid_rows=invalid_rows,
        answerable_items=answerable_items,
        unanswerable_items=unanswerable_items,
        must_include_rules=must_include_rules,
        must_include_any_groups=must_include_any_groups,
        must_not_include_rules=must_not_include_rules,
        duplicate_user_rate=duplicate_user_rate,
        duplicate_user_prompts=duplicate_user_prompts,
        duplicate_user_samples=tuple(duplicate_user_samples),
        near_duplicate_user_pairs=near_duplicates["count"],
        near_duplicate_user_samples=tuple(near_duplicates["samples"]),
        categories=dict(sorted(categories.items())),
        category_entropy=category_entropy,
        category_entropy_normalized=category_entropy_normalized,
        splits=dict(sorted(splits.items())),
        levels=dict(sorted(levels.items())),
        heldout_categories=dict(sorted(heldout_categories.items())),
        answer_length_distribution=_length_distribution(answer_texts),
        template_families=dict(sorted(template_families.items())),
        curriculum_label=curriculum_label,
        curriculum_breakdown=dict(sorted(curriculum_breakdown.items())),
        quality_warnings=tuple(quality_warnings),
        issues=tuple(issues[:8]),
        preview=tuple(items[:max(0, preview_items)]),
    )


def _read_jsonl_records(path: str) -> tuple[list[tuple[int, Any]], int, TuningDataIssue | None]:
    source = Path(path)
    if not source.exists():
        return [], 0, TuningDataIssue(0, "file does not exist")
    if not source.is_file():
        return [], 0, TuningDataIssue(0, "path is not a file")

    records: list[tuple[int, Any]] = []
    empty_rows = 0
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [], 0, TuningDataIssue(0, "file is not valid UTF-8 text")

    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            empty_rows += 1
            continue
        try:
            records.append((line_number, json.loads(text)))
        except json.JSONDecodeError as error:
            records.append((line_number, {"__invalid_json__": str(error)}))
    return records, empty_rows, None


def _parse_eval_item(line_number: int, record: Any) -> tuple[dict[str, Any], list[TuningDataIssue]]:
    issues: list[TuningDataIssue] = []
    if not isinstance(record, dict):
        return {}, [TuningDataIssue(line_number, "row must be a JSON object")]
    if "__invalid_json__" in record:
        return {}, [TuningDataIssue(line_number, f"invalid JSON: {record['__invalid_json__']}")]

    user = record.get("user")
    if not isinstance(user, str) or not user.strip():
        issues.append(TuningDataIssue(line_number, "row must contain a non-empty string user field"))

    must_include = record.get("must_include", ())
    expected = record.get("expected")
    if expected is not None:
        if isinstance(expected, str):
            if must_include in (None, ()):
                must_include = [expected]
            elif isinstance(must_include, list):
                must_include = [*must_include, expected]
        else:
            issues.append(TuningDataIssue(line_number, "expected field must be a string"))

    must_include_values = _string_list(must_include, line_number, "must_include", issues)
    must_not_include_values = _string_list(record.get("must_not_include", ()), line_number, "must_not_include", issues)
    must_include_any_values = _phrase_groups(record.get("must_include_any", ()), line_number, issues)

    answerable = record.get("answerable", True)
    if not isinstance(answerable, bool):
        issues.append(TuningDataIssue(line_number, "answerable field must be a boolean"))
        answerable = True

    category = record.get("category", "answerable" if answerable else "unanswerable")
    if not isinstance(category, str) or not category.strip():
        issues.append(TuningDataIssue(line_number, "category field must be a non-empty string"))
        category = "answerable" if answerable else "unanswerable"
    split = record.get("split", record.get("eval_split", "default"))
    if not isinstance(split, str) or not split.strip():
        issues.append(TuningDataIssue(line_number, "split field must be a non-empty string when present"))
        split = "default"
    level = record.get("level", record.get("eval_level", _infer_eval_level(split, category, answerable)))
    if not isinstance(level, str) or not level.strip():
        issues.append(TuningDataIssue(line_number, "level field must be a non-empty string when present"))
        level = "heldout"
    template_family = _template_family(record, category)
    required_entities = _string_list(
        record.get("required_entities", record.get("entities", ())),
        line_number,
        "required_entities",
        issues,
    )
    for field in ("min_words", "max_words", "min_chars", "max_chars"):
        value = record.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            issues.append(TuningDataIssue(line_number, f"{field} field must be a non-negative integer"))
    require_corpus_support = record.get("require_corpus_support", False)
    if not isinstance(require_corpus_support, bool):
        issues.append(TuningDataIssue(line_number, "require_corpus_support field must be a boolean"))
    choice_labels = _string_list(record.get("choice_labels", ()), line_number, "choice_labels", issues)
    correct_choice = record.get("correct_choice", record.get("answer_choice"))
    if correct_choice is not None:
        if not isinstance(correct_choice, str) or not correct_choice.strip():
            issues.append(TuningDataIssue(line_number, "correct_choice field must be a non-empty string"))
        elif not choice_labels:
            issues.append(TuningDataIssue(line_number, "correct_choice requires choice_labels"))
        elif correct_choice.strip() not in choice_labels:
            issues.append(TuningDataIssue(line_number, "correct_choice must be present in choice_labels"))

    if issues:
        return {}, issues
    return {
        "user": user,
        "answerable": answerable,
        "category": category,
        "split": split,
        "level": level,
        "template_family": template_family,
        "must_include": must_include_values,
        "must_include_any": must_include_any_values,
        "must_not_include": must_not_include_values,
        "required_entities": required_entities,
        "reference_answer": record.get("reference_answer", record.get("reference", record.get("expected"))),
        "min_words": record.get("min_words"),
        "max_words": record.get("max_words"),
        "min_chars": record.get("min_chars"),
        "max_chars": record.get("max_chars"),
        "require_corpus_support": require_corpus_support,
        "choice_labels": choice_labels,
        "correct_choice": correct_choice,
    }, []


def _string_list(value: Any, line_number: int, field: str, issues: list[TuningDataIssue]) -> list[str]:
    if value in (None, ()):
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        issues.append(TuningDataIssue(line_number, f"{field} field must be a list of non-empty strings"))
        return []
    return value


def _phrase_groups(value: Any, line_number: int, issues: list[TuningDataIssue]) -> list[list[str]]:
    if value in (None, ()):
        return []
    if not isinstance(value, list):
        issues.append(TuningDataIssue(line_number, "must_include_any field must be a list of lists"))
        return []

    groups: list[list[str]] = []
    for group in value:
        if not isinstance(group, list) or not group or not all(isinstance(item, str) and item for item in group):
            issues.append(TuningDataIssue(line_number, "must_include_any groups must contain non-empty strings"))
            return []
        groups.append(group)
    return groups


def _infer_eval_level(split: str, category: str, answerable: bool) -> str:
    text = f"{split} {category}".lower()
    if "smoke" in text:
        return "smoke"
    if "memor" in text or "canary" in text:
        return "memorization"
    if "adversarial" in text or "safety" in text or "refusal" in text or not answerable:
        return "adversarial"
    if "transfer" in text or "paraphrase" in text:
        return "transfer"
    if "domain" in text or "knowledge" in text:
        return "domain"
    return "heldout"


def _duplicate_prompt_stats(values: list[str]) -> tuple[float, int, list[str]]:
    if not values:
        return 0.0, 0, []
    normalized = [_norm_prompt(value) for value in values]
    counts = Counter(value for value in normalized if value)
    duplicate_prompts = sum(count - 1 for count in counts.values() if count > 1)
    samples = [value for value, count in counts.most_common() if count > 1][:5]
    return duplicate_prompts / len(values), duplicate_prompts, samples


def _duplicate_rate(values: list[str]) -> float:
    return _duplicate_prompt_stats(values)[0]


def _near_duplicate_prompt_report(
    values: list[str],
    threshold: float = 0.90,
    max_samples: int = 5,
    max_checks: int = 100_000,
) -> dict[str, Any]:
    normalized = [_norm_prompt(value) for value in values]
    indexed = [
        (index, prompt, values[index])
        for index, prompt in enumerate(normalized)
        if prompt
    ]
    buckets: dict[str, list[tuple[int, str, str]]] = {}
    for item in indexed:
        tokens = item[1].split()
        key = tokens[0] if tokens else ""
        buckets.setdefault(key, []).append(item)

    count = 0
    samples: list[dict[str, Any]] = []
    checks = 0
    for bucket in buckets.values():
        for left_index, left_prompt, left_raw in bucket:
            for right_index, right_prompt, right_raw in bucket:
                if right_index <= left_index:
                    continue
                checks += 1
                if checks > max_checks:
                    return {"count": count, "samples": samples, "truncated": True}
                if left_prompt == right_prompt:
                    continue
                jaccard = _token_jaccard(left_prompt, right_prompt)
                if jaccard < threshold and not _can_reach_similarity(left_prompt, right_prompt, threshold):
                    continue
                matcher = difflib.SequenceMatcher(None, left_prompt, right_prompt)
                if max(jaccard, matcher.quick_ratio()) < threshold:
                    continue
                ratio = matcher.ratio()
                similarity = max(ratio, jaccard)
                if similarity < threshold:
                    continue
                count += 1
                if len(samples) < max_samples:
                    samples.append({
                        "similarity": round(float(ratio), 4),
                        "token_jaccard": round(float(jaccard), 4),
                        "left": left_raw[:180],
                        "right": right_raw[:180],
                    })
    return {"count": count, "samples": samples, "truncated": False}


def _category_entropy(categories: dict[str, int]) -> tuple[float, float]:
    total = sum(categories.values())
    if total <= 0 or len(categories) <= 1:
        return 0.0, 0.0
    entropy = 0.0
    for count in categories.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    max_entropy = math.log2(len(categories))
    return entropy, entropy / max_entropy if max_entropy > 0 else 0.0


def _length_distribution(texts: list[str]) -> dict[str, float | int]:
    if not texts:
        return {
            "count": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_chars": 0.0,
            "min_words": 0,
            "max_words": 0,
            "avg_words": 0.0,
        }
    char_lengths = [len(text) for text in texts]
    word_lengths = [len(re.findall(r"\S+", text)) for text in texts]
    return {
        "count": len(texts),
        "min_chars": min(char_lengths),
        "max_chars": max(char_lengths),
        "avg_chars": sum(char_lengths) / len(char_lengths),
        "min_words": min(word_lengths),
        "max_words": max(word_lengths),
        "avg_words": sum(word_lengths) / len(word_lengths),
    }


def _template_family(record: dict[str, Any], category: str) -> str:
    raw = record.get("template", record.get("group", record.get("group_id")))
    if isinstance(raw, str) and raw.strip():
        family = raw.strip()
        family = re.sub(r"[-_]\d+$", "", family)
        family = re.sub(r"[-_]\d+$", "", family)
        return family or category
    return f"ungrouped:{category}"


def _answer_style(record: dict[str, Any]) -> str:
    raw = record.get("answer_style")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    assistant = record.get("assistant")
    if isinstance(assistant, str) and "Scratchpad:" in assistant and "Final answer:" in assistant:
        return "scratchpad"
    return "direct"


def _curriculum_bucket(category: str) -> str:
    text = str(category).lower()
    if any(token in text for token in ("math", "spelling", "choice", "mmlu", "gsm", "arc", "bench_")):
        return "skill"
    if any(token in text for token in ("identity", "refusal", "safety", "honesty", "smoltalk", "behavior")):
        return "behavior"
    return "domain"


def _curriculum_label(breakdown: Counter[str], suffix: str) -> str:
    total = sum(breakdown.values())
    if total <= 0:
        return "unknown"
    skill = breakdown.get("skill", 0) / total
    behavior = breakdown.get("behavior", 0) / total
    domain = breakdown.get("domain", 0) / total
    if skill > 0 and behavior > 0 and min(skill, behavior) >= 0.20:
        return f"mixed_behavior_skill_{suffix}"
    if skill >= 0.70:
        return f"skill_{suffix}"
    if behavior >= 0.70:
        return f"behavior_{suffix}"
    if domain >= 0.70:
        return f"domain_{suffix}"
    if skill > 0 and behavior > 0:
        return f"mixed_behavior_skill_{suffix}"
    return f"mixed_{suffix}"


def _eval_answer_texts(item: dict[str, Any]) -> list[str]:
    reference = item.get("reference_answer")
    if isinstance(reference, str) and reference.strip():
        return [reference]
    texts = [value for value in item.get("must_include", []) if isinstance(value, str)]
    for group in item.get("must_include_any", []):
        texts.extend(value for value in group if isinstance(value, str))
    return texts


def _sft_quality_warnings(
    *,
    num_examples: int,
    duplicate_user_rate: float,
    near_duplicate_user_pairs: int,
    category_entropy_normalized: float,
    curriculum_label: str,
) -> list[str]:
    warnings: list[str] = []
    if num_examples >= 32 and category_entropy_normalized < 0.45:
        warnings.append("Category entropy is low; one SFT category dominates the file.")
    if duplicate_user_rate > 0.10:
        warnings.append("More than 10% of SFT prompts are exact duplicates.")
    if near_duplicate_user_pairs:
        warnings.append("Near-duplicate SFT prompts were detected; inspect template variety before trusting fit.")
    if curriculum_label.startswith("skill_"):
        warnings.append("This looks like skill SFT; do not interpret low loss as broad chat behavior.")
    elif curriculum_label.startswith("behavior_"):
        warnings.append("This looks like behavior SFT; it teaches format/refusal/identity more than missing knowledge.")
    return warnings


def _eval_quality_warnings(
    *,
    num_items: int,
    duplicate_user_rate: float,
    near_duplicate_user_pairs: int,
    category_entropy_normalized: float,
    heldout_categories: dict[str, int],
) -> list[str]:
    warnings: list[str] = []
    if num_items >= 16 and category_entropy_normalized < 0.45:
        warnings.append("Eval category entropy is low; one category may dominate the score.")
    if duplicate_user_rate > 0.0:
        warnings.append("Eval contains exact duplicate prompts.")
    if near_duplicate_user_pairs:
        warnings.append("Eval contains near-duplicate prompts; score variance may be misleading.")
    if not heldout_categories:
        warnings.append("No held-out eval category distribution was detected.")
    return warnings


def _norm_prompt(text: str) -> str:
    return " ".join(str(text).lower().split())


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _can_reach_similarity(left: str, right: str, threshold: float) -> bool:
    if not left or not right:
        return False
    return (2 * min(len(left), len(right)) / (len(left) + len(right))) >= threshold


def _chat_sft_status(num_examples: int, invalid_rows: int, duplicate_user_rate: float) -> tuple[str, str]:
    if num_examples == 0:
        return "blocked", "No usable chat SFT examples were found."
    if invalid_rows:
        return "blocked", "Fix invalid chat SFT rows before training."
    if num_examples < 8:
        return "caution", "Chat SFT file is valid but very small."
    if duplicate_user_rate > 0.25:
        return "caution", "Chat SFT file is valid but repeats many user prompts."
    return "ready", "Chat SFT file looks usable for a tiny run."


def _chat_eval_status(
    num_items: int,
    invalid_rows: int,
    total_rules: int,
    unanswerable_items: int,
) -> tuple[str, str]:
    if num_items == 0:
        return "blocked", "No usable eval items were found."
    if invalid_rows:
        return "blocked", "Fix invalid eval rows before scoring."
    if total_rules == 0:
        return "blocked", "Eval items need visible pass/fail rules."
    if num_items < 4:
        return "caution", "Eval file is valid but too small to trust."
    if unanswerable_items == 0:
        return "caution", "Eval file has no unanswerable/refusal checks yet."
    return "ready", "Eval file looks usable for transparent scoring."
