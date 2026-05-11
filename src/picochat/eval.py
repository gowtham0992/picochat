"""Small transparent evaluation tools for Picochat."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch

from picochat.chat import extract_assistant_reply, render_chat_prompt
from picochat.checkpoint import load_checkpoint
from picochat.report import chat_eval_report_markdown
from picochat.tokenizer import Tokenizer, load_tokenizer


@dataclass(frozen=True)
class ChatEvalItem:
    user: str
    must_include: tuple[str, ...] = ()
    must_include_any: tuple[tuple[str, ...], ...] = ()
    must_not_include: tuple[str, ...] = ()
    answerable: bool = True
    category: str = "answerable"
    split: str = "default"


@dataclass(frozen=True)
class ChatEvalConfig:
    input_path: str
    checkpoint_path: str
    tokenizer_path: str
    out_dir: str
    max_new_tokens: int = 80
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    device: str = "cpu"
    case_sensitive: bool = False


def load_chat_eval_items(path: str | Path) -> list[ChatEvalItem]:
    """Load transparent chat eval items from JSONL."""
    items: list[ChatEvalItem] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        user = record.get("user")
        if not isinstance(user, str):
            raise ValueError(f"line {line_number} must contain a string user field")

        must_include = record.get("must_include", ())
        expected = record.get("expected")
        if expected is not None:
            if not isinstance(expected, str):
                raise ValueError(f"line {line_number} expected field must be a string")
            must_include = [*must_include, expected]

        must_not_include = record.get("must_not_include", ())
        must_include_any = record.get("must_include_any", ())
        answerable = record.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"line {line_number} answerable field must be a boolean")
        category = record.get("category", "answerable" if answerable else "unanswerable")
        if not isinstance(category, str):
            raise ValueError(f"line {line_number} category field must be a string")
        split = record.get("split", record.get("eval_split", "default"))
        if not isinstance(split, str):
            raise ValueError(f"line {line_number} split field must be a string")
        items.append(ChatEvalItem(
            user=user,
            must_include=_as_string_tuple(must_include, line_number, "must_include"),
            must_include_any=_as_phrase_groups(must_include_any, line_number, "must_include_any"),
            must_not_include=_as_string_tuple(must_not_include, line_number, "must_not_include"),
            answerable=answerable,
            category=category,
            split=split or "default",
        ))

    if not items:
        raise ValueError("chat eval dataset is empty")
    return items


def _as_string_tuple(value, line_number: int, field: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"line {line_number} {field} field must be a list of strings")
    return tuple(value)


def _as_phrase_groups(value, line_number: int, field: str) -> tuple[tuple[str, ...], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"line {line_number} {field} field must be a list of lists")

    groups: list[tuple[str, ...]] = []
    for group in value:
        if (
            not isinstance(group, list)
            or not group
            or not all(isinstance(item, str) for item in group)
        ):
            raise ValueError(
                f"line {line_number} {field} field must contain non-empty string lists"
            )
        groups.append(tuple(group))
    return tuple(groups)


def score_reply(reply: str, item: ChatEvalItem, case_sensitive: bool = False) -> dict:
    """Score a reply with visible substring rules."""
    prompt_echo_reasons = detect_prompt_echo(reply, item.user, case_sensitive=case_sensitive)
    prompt_echo = bool(prompt_echo_reasons)
    if case_sensitive:
        haystack = reply
        includes = item.must_include
        include_any = item.must_include_any
        forbidden = item.must_not_include
    else:
        haystack = reply.lower()
        includes = tuple(phrase.lower() for phrase in item.must_include)
        include_any = tuple(
            tuple(phrase.lower() for phrase in group)
            for group in item.must_include_any
        )
        forbidden = tuple(phrase.lower() for phrase in item.must_not_include)

    missing = [
        original
        for original, normalized in zip(item.must_include, includes, strict=True)
        if normalized not in haystack
    ]
    missing_any = [
        list(original_group)
        for original_group, normalized_group in zip(
            item.must_include_any,
            include_any,
            strict=True,
        )
        if not any(phrase in haystack for phrase in normalized_group)
    ]
    found_forbidden = [
        original
        for original, normalized in zip(item.must_not_include, forbidden, strict=True)
        if normalized in haystack
    ]
    support_total = len(item.must_include) + len(item.must_include_any)
    support_matched = support_total - len(missing) - len(missing_any)
    return {
        "passed": not missing and not missing_any and not found_forbidden and not prompt_echo,
        "missing": missing,
        "missing_any": missing_any,
        "found_forbidden": found_forbidden,
        "prompt_echo": prompt_echo,
        "prompt_echo_reasons": prompt_echo_reasons,
        "support_total": support_total,
        "support_matched": support_matched,
        "support_match_rate": support_matched / support_total if support_total else 1.0,
    }


def detect_prompt_echo(
    reply: str,
    user_message: str,
    case_sensitive: bool = False,
) -> list[str]:
    """Return visible reasons a reply appears to echo the chat prompt."""
    if not reply.strip():
        return []

    if case_sensitive:
        text = reply
        user_text = user_message
        role_pattern = r"(^|\n)\s*(User|Assistant)\s*:"
    else:
        text = reply.lower()
        user_text = user_message.lower()
        role_pattern = r"(^|\n)\s*(user|assistant)\s*:"

    reasons: list[str] = []
    if re.search(role_pattern, text):
        reasons.append("chat_role_label")

    compact_reply = _normalize_for_echo(text)
    compact_user = _normalize_for_echo(user_text)
    if len(compact_user) >= 12 and (
        compact_reply.startswith(compact_user[:80])
        or compact_reply.startswith(f"user {compact_user[:80]}")
    ):
        reasons.append("starts_with_user_prompt")

    return reasons


def run_chat_eval(config: ChatEvalConfig) -> dict:
    """Run a deterministic chat eval and write JSON/Markdown reports."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.tokenizer_path)
    model, metadata = load_checkpoint(config.checkpoint_path, map_location=config.device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("tokenizer vocabulary size does not match checkpoint")

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    rows = []
    for index, item in enumerate(load_chat_eval_items(config.input_path)):
        reply = _generate_eval_reply(model, tokenizer, config, item.user, seed=config.seed + index)
        score = score_reply(reply, item, case_sensitive=config.case_sensitive)
        rows.append({
            "index": index + 1,
            "user": item.user,
            "answerable": item.answerable,
            "category": item.category,
            "split": item.split,
            "reply": reply,
            "must_include": list(item.must_include),
            "must_include_any": [list(group) for group in item.must_include_any],
            "must_not_include": list(item.must_not_include),
            **score,
        })

    passed = sum(1 for row in rows if row["passed"])
    unsupported_claims = sum(1 for row in rows if row["found_forbidden"])
    prompt_echoes = sum(1 for row in rows if row.get("prompt_echo"))
    missing_support = sum(1 for row in rows if row["missing"] or row["missing_any"])
    support_total = sum(int(row["support_total"]) for row in rows)
    support_matched = sum(int(row["support_matched"]) for row in rows)
    answerable_support_total = sum(int(row["support_total"]) for row in rows if row["answerable"])
    answerable_support_matched = sum(int(row["support_matched"]) for row in rows if row["answerable"])
    answerable = sum(1 for row in rows if row["answerable"])
    category_breakdown = _breakdown(rows, "category", "answerable")
    split_breakdown = _breakdown(rows, "split", "default")
    analysis = analyze_eval_failures(rows, category_breakdown, split_breakdown)
    report = {
        "config": config.__dict__,
        "checkpoint": {
            "path": config.checkpoint_path,
            "step": metadata.get("step"),
            "train_loss": metadata.get("train_loss"),
        },
        "summary": {
            "num_examples": len(rows),
            "num_passed": passed,
            "num_failed": len(rows) - passed,
            "pass_rate": passed / len(rows),
            "num_answerable": answerable,
            "num_unanswerable": len(rows) - answerable,
            "unsupported_claims": unsupported_claims,
            "unsupported_claim_rate": unsupported_claims / len(rows),
            "prompt_echoes": prompt_echoes,
            "prompt_echo_rate": prompt_echoes / len(rows),
            "missing_support": missing_support,
            "missing_support_rate": missing_support / len(rows),
            "support_requirements": support_total,
            "support_matches": support_matched,
            "support_match_rate": _safe_rate(support_matched, support_total),
            "answerable_support_match_rate": _safe_rate(
                answerable_support_matched,
                answerable_support_total,
            ),
            "category_breakdown": category_breakdown,
            "split_breakdown": split_breakdown,
        },
        "analysis": analysis,
        "examples": rows,
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(chat_eval_report_markdown(report), encoding="utf-8")
    return report


def analyze_eval_failures(
    rows: list[dict],
    category_breakdown: dict[str, dict] | None = None,
    split_breakdown: dict[str, dict] | None = None,
) -> dict:
    """Return compact failure causes and next actions for an eval report."""
    failure_counts: dict[str, int] = {}
    failed_examples: list[dict] = []
    for fallback_index, row in enumerate(rows, start=1):
        reasons = _failure_reasons(row)
        if not row.get("passed", False):
            for reason in reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            failed_examples.append({
                "index": row.get("index", fallback_index),
                "category": row.get("category", "answerable"),
                "split": row.get("split", "default"),
                "answerable": row.get("answerable", True),
                "reasons": reasons,
                "missing": list(row.get("missing", [])),
                "missing_any": list(row.get("missing_any", [])),
                "found_forbidden": list(row.get("found_forbidden", [])),
                "prompt_echo_reasons": list(row.get("prompt_echo_reasons", [])),
                "reply_preview": _preview_text(str(row.get("reply", ""))),
            })

    category_breakdown = category_breakdown or _breakdown(rows, "category", "answerable")
    split_breakdown = split_breakdown or _breakdown(rows, "split", "default")
    weak_categories = _weak_breakdown(category_breakdown, "category")
    weak_splits = _weak_breakdown(split_breakdown, "split")
    return {
        "failure_counts": dict(sorted(failure_counts.items())),
        "failed_examples": failed_examples,
        "weak_categories": weak_categories,
        "weak_splits": weak_splits,
        "recommendations": _eval_recommendations(
            rows,
            failure_counts,
            weak_categories,
            weak_splits,
        ),
    }


def _breakdown(rows: list[dict], field: str, default: str) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for row in rows:
        value = row.get(field) or default
        bucket = buckets.setdefault(
            str(value),
            {
                "num_examples": 0,
                "num_passed": 0,
                "num_failed": 0,
                "pass_rate": 0.0,
                "num_answerable": 0,
                "num_unanswerable": 0,
                "unsupported_claims": 0,
                "unsupported_claim_rate": 0.0,
                "prompt_echoes": 0,
                "prompt_echo_rate": 0.0,
                "missing_support": 0,
                "missing_support_rate": 0.0,
                "support_requirements": 0,
                "support_matches": 0,
                "support_match_rate": 1.0,
            },
        )
        bucket["num_examples"] += 1
        bucket["num_passed"] += int(row["passed"])
        bucket["num_failed"] += int(not row["passed"])
        bucket["num_answerable"] += int(row.get("answerable", True))
        bucket["num_unanswerable"] += int(not row.get("answerable", True))
        bucket["unsupported_claims"] += int(bool(row.get("found_forbidden")))
        bucket["prompt_echoes"] += int(bool(row.get("prompt_echo")))
        bucket["missing_support"] += int(bool(row.get("missing") or row.get("missing_any")))
        bucket["support_requirements"] += int(row.get("support_total", 0))
        bucket["support_matches"] += int(row.get("support_matched", 0))

    for bucket in buckets.values():
        total = bucket["num_examples"]
        bucket["pass_rate"] = bucket["num_passed"] / total
        bucket["unsupported_claim_rate"] = bucket["unsupported_claims"] / total
        bucket["prompt_echo_rate"] = bucket["prompt_echoes"] / total
        bucket["missing_support_rate"] = bucket["missing_support"] / total
        bucket["support_match_rate"] = _safe_rate(
            bucket["support_matches"],
            bucket["support_requirements"],
        )
    return dict(sorted(buckets.items()))


def _failure_reasons(row: dict) -> list[str]:
    reasons: list[str] = []
    if row.get("missing"):
        reasons.append("missing_required")
    if row.get("missing_any"):
        reasons.append("missing_any_group")
    if row.get("found_forbidden"):
        reasons.append("forbidden_phrase")
    if row.get("prompt_echo"):
        reasons.append("prompt_echo")
    if not str(row.get("reply", "")).strip():
        reasons.append("empty_reply")
    if not row.get("passed", False) and not reasons:
        reasons.append("unknown")
    return reasons


def _weak_breakdown(breakdown: dict[str, dict], field: str) -> list[dict]:
    weak: list[dict] = []
    for name, row in breakdown.items():
        num_examples = int(row.get("num_examples", 0))
        if num_examples <= 0:
            continue
        pass_rate = float(row.get("pass_rate", 0.0))
        support_match_rate = float(row.get("support_match_rate", 1.0))
        unsupported_claim_rate = float(row.get("unsupported_claim_rate", 0.0))
        prompt_echo_rate = float(row.get("prompt_echo_rate", 0.0))
        if (
            row.get("num_failed", 0)
            and (
                pass_rate < 0.80
                or support_match_rate < 0.90
                or unsupported_claim_rate > 0.0
                or prompt_echo_rate > 0.0
            )
        ):
            weak.append({
                field: name,
                "num_examples": num_examples,
                "num_failed": row.get("num_failed", 0),
                "pass_rate": pass_rate,
                "support_match_rate": support_match_rate,
                "unsupported_claim_rate": unsupported_claim_rate,
                "prompt_echo_rate": prompt_echo_rate,
            })
    return sorted(
        weak,
        key=lambda item: (
            item["pass_rate"],
            item["support_match_rate"],
            -int(item["num_examples"]),
            str(item[field]),
        ),
    )


def _eval_recommendations(
    rows: list[dict],
    failure_counts: dict[str, int],
    weak_categories: list[dict],
    weak_splits: list[dict],
) -> list[dict]:
    if not rows:
        return [{
            "priority": "high",
            "area": "eval",
            "message": "No eval rows were scored.",
            "action": "Add held-out eval JSONL rows before trusting this run.",
        }]

    recommendations: list[dict] = []
    if not failure_counts:
        recommendations.append({
            "priority": "medium",
            "area": "eval",
            "message": "All visible eval checks passed.",
            "action": "Add harder held-out prompts before scaling the same recipe.",
        })
        return recommendations

    if failure_counts.get("missing_required") or failure_counts.get("missing_any_group"):
        recommendations.append({
            "priority": "high",
            "area": "sft",
            "message": "The model missed required support phrases.",
            "action": "Add more varied SFT rows for the weakest categories, then rerun SFT before increasing base training.",
        })
    if failure_counts.get("forbidden_phrase"):
        recommendations.append({
            "priority": "high",
            "area": "honesty",
            "message": "Some replies contained forbidden phrases.",
            "action": "Add refusal and boundary examples for those request types, and inspect the failed replies for unsupported claims.",
        })
    if failure_counts.get("prompt_echo"):
        recommendations.append({
            "priority": "high",
            "area": "decoding",
            "message": "Some replies echoed chat prompt structure.",
            "action": "Check stop handling and add SFT examples that answer without regenerating User:/Assistant: role labels.",
        })
    if failure_counts.get("empty_reply"):
        recommendations.append({
            "priority": "medium",
            "area": "decoding",
            "message": "Some replies were empty.",
            "action": "Increase max_new_tokens or inspect whether the checkpoint emits EOS too early.",
        })

    weak_category_names = {str(item["category"]) for item in weak_categories}
    category_actions = {
        "required_words": "Add compositional examples with unseen word pairs and check that held-out eval pairs remain absent from SFT.",
        "story_generation": "Add broader story-generation SFT rows with diverse subjects, lessons, and endings.",
        "story_knowledge": "Add direct domain QA examples and verify the answers are supported by the corpus or SFT curriculum.",
        "outside_story_domain": "Add refusal examples with varied wording for live facts, medical, legal, financial, and private-data requests.",
        "memorization_probe": "Strengthen memorization-boundary refusals and keep canary checks enabled.",
        "continuation": "Add continuation examples that preserve named objects, characters, and the requested next event.",
    }
    for category, action in category_actions.items():
        if category in weak_category_names:
            recommendations.append({
                "priority": "medium",
                "area": category,
                "message": f"`{category}` is a weak eval category.",
                "action": action,
            })

    if weak_splits:
        weakest = weak_splits[0]
        recommendations.append({
            "priority": "medium",
            "area": "split",
            "message": f"Weakest eval split is `{weakest['split']}`.",
            "action": "Treat split-level failures as the next curriculum target instead of only optimizing aggregate pass rate.",
        })

    return _dedupe_recommendations(recommendations)


def _dedupe_recommendations(recommendations: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in recommendations:
        key = (str(item.get("area")), str(item.get("action")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _preview_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 1.0


def _normalize_for_echo(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@torch.no_grad()
def _generate_eval_reply(
    model,
    tokenizer: Tokenizer,
    config: ChatEvalConfig,
    user_message: str,
    seed: int,
) -> str:
    prompt = render_chat_prompt([], user_message)
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_bos=True)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    generated = model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        seed=seed,
        eos_id=tokenizer.eos_id,
    )
    new_token_ids = generated[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(new_token_ids)
    return extract_assistant_reply("", generated_text)
