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
        "examples": rows,
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(chat_eval_report_markdown(report), encoding="utf-8")
    return report


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
        seed=seed,
        eos_id=tokenizer.eos_id,
    )
    new_token_ids = generated[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(new_token_ids)
    return extract_assistant_reply("", generated_text)
