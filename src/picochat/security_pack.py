"""Build a defensive security-analyst dataset pack.

The builder is intentionally conservative: it accepts only instruction/chat
style rows, keeps held-out eval prompts separate from SFT answers, and writes a
plain Picochat dataset pack that existing preflight/training code can consume.
Network-backed imports are optional so local tests can exercise the same code
path without Hugging Face access.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_TRENDYOL_DATASET = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"


@dataclass(frozen=True)
class SecurityPackConfig:
    out_dir: str | Path
    seed_dir: str | Path = "datasets/security-analyst"
    include_trendyol: bool = True
    trendyol_dataset: str = DEFAULT_TRENDYOL_DATASET
    trendyol_split: str = "train"
    trendyol_max_rows: int = 10_000
    eval_rows: int = 500
    preference_target_rows: int = 64
    force: bool = False


def build_security_analyst_pack(config: SecurityPackConfig) -> dict[str, Any]:
    """Create a blended defensive-security SFT/eval/preference pack."""
    out_dir = Path(config.out_dir)
    seed_dir = Path(config.seed_dir)
    if not seed_dir.exists():
        raise FileNotFoundError(f"missing security seed directory: {seed_dir}")
    if out_dir.exists() and any(out_dir.iterdir()) and not config.force:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_chat = _load_jsonl(seed_dir / "chat.jsonl")
    seed_eval = _load_jsonl(seed_dir / "eval.jsonl")
    seed_preferences = _load_jsonl(seed_dir / "preferences.jsonl")

    imported_rows: list[dict[str, Any]] = []
    if config.include_trendyol:
        imported_rows = list(_load_trendyol_rows(
            config.trendyol_dataset,
            config.trendyol_split,
            max_rows=max(0, int(config.trendyol_max_rows)),
        ))

    imported_eval_count = min(max(0, int(config.eval_rows)), max(0, len(imported_rows) // 5))
    imported_eval = imported_rows[-imported_eval_count:] if imported_eval_count else []
    imported_chat = imported_rows[:-imported_eval_count] if imported_eval_count else imported_rows

    chat_rows, duplicate_chat_rows = _dedupe_chat_rows([*seed_chat, *imported_chat])
    eval_rows = _dedupe_eval_rows([*seed_eval, *(_eval_from_chat(row) for row in imported_eval)])
    preference_rows = _dedupe_preference_rows([
        *seed_preferences,
        *_synthetic_security_preferences(max(0, int(config.preference_target_rows) - len(seed_preferences))),
    ])

    corpus_docs = _corpus_documents(chat_rows)

    corpus_path = out_dir / "corpus.txt"
    chat_path = out_dir / "chat.jsonl"
    eval_path = out_dir / "eval.jsonl"
    prefs_path = out_dir / "preferences.jsonl"
    pack_path = out_dir / "dataset_pack.json"
    report_json = out_dir / "security_pack_report.json"
    report_md = out_dir / "security_pack_report.md"

    corpus_path.write_text("\n\n".join(corpus_docs).strip() + "\n", encoding="utf-8")
    _write_jsonl(chat_path, chat_rows)
    _write_jsonl(eval_path, eval_rows)
    _write_jsonl(prefs_path, preference_rows)
    pack_path.write_text(json.dumps({
        "name": "security-analyst",
        "description": "Defensive security analyst SLM pack: SFT, held-out eval, and a DPO preference file.",
        "corpus": {"input": "corpus.txt"},
        "chat": "chat.jsonl",
        "eval": "eval.jsonl",
    }, indent=2) + "\n", encoding="utf-8")

    source_counts = Counter(str(row.get("source", "seed")) for row in chat_rows)
    category_counts = Counter(str(row.get("category", "security")) for row in chat_rows)
    report = {
        "dataset_pack": str(pack_path),
        "chat": str(chat_path),
        "eval": str(eval_path),
        "preferences": str(prefs_path),
        "corpus": str(corpus_path),
        "base_model": DEFAULT_BASE_MODEL,
        "source_dataset": config.trendyol_dataset if config.include_trendyol else "seed-only",
        "include_trendyol": bool(config.include_trendyol),
        "trendyol_rows_loaded": len(imported_rows),
        "chat_rows": len(chat_rows),
        "eval_rows": len(eval_rows),
        "preference_rows": len(preference_rows),
        "duplicate_chat_rows_skipped": duplicate_chat_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "recommended_modal": {
            "mode": "hf-sft",
            "model": DEFAULT_BASE_MODEL,
            "dataset_pack": str(pack_path),
            "preference_input": str(prefs_path),
            "quantize": "4bit",
            "peft": "lora",
            "gpu": "A100",
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_report_markdown(report), encoding="utf-8")
    return {**report, "report": str(report_md), "report_json": str(report_json)}


def _load_trendyol_rows(dataset_id: str, split: str, *, max_rows: int) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - exercised when optional extra is absent.
        raise RuntimeError("install the datasets package to import Trendyol security rows") from error

    dataset = load_dataset(dataset_id, split=split, streaming=False)
    limit = max_rows if max_rows > 0 else len(dataset)
    for idx, row in enumerate(dataset):
        if idx >= limit:
            break
        mapped = _map_trendyol_row(row, index=idx)
        if mapped:
            yield mapped


def _map_trendyol_row(row: dict[str, Any], *, index: int) -> dict[str, Any] | None:
    user = _first_text(row, "user", "instruction", "question", "prompt", "input")
    assistant = _first_text(row, "assistant", "response", "answer", "output", "completion")
    if not user or not assistant:
        return None
    system = _first_text(row, "system")
    category = _first_text(row, "category", "task", "type") or "trendyol_security"
    mapped: dict[str, Any] = {
        "category": _safe_slug(category),
        "group": f"trendyol-{index:06d}",
        "answerable": True,
        "user": user,
        "assistant": assistant,
        "source": "trendyol",
    }
    if system:
        mapped["system"] = system
    return mapped


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL in {path}:{line_no}: {error}") from error
        if isinstance(record, dict):
            rows.append(record)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _dedupe_chat_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        user = str(row.get("user") or row.get("prompt") or "").strip()
        assistant = str(row.get("assistant") or row.get("answer") or row.get("completion") or "").strip()
        if not user or not assistant:
            continue
        key = _norm(user) + "\n---\n" + _norm(assistant)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        out.append({
            "category": str(row.get("category") or "security"),
            "group": str(row.get("group") or f"security-{len(out):06d}"),
            "answerable": bool(row.get("answerable", True)),
            "user": user,
            "assistant": assistant,
            "source": str(row.get("source") or "seed"),
            **({"system": str(row["system"]).strip()} if str(row.get("system") or "").strip() else {}),
        })
    if not out:
        raise ValueError("security pack needs at least one chat row")
    return out, skipped


def _dedupe_eval_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        user = str(row.get("user") or row.get("prompt") or "").strip()
        if not user:
            continue
        key = _norm(user)
        if key in seen:
            continue
        seen.add(key)
        record = {
            "category": str(row.get("category") or "security_eval"),
            "group": str(row.get("group") or f"security-eval-{len(out):06d}"),
            "user": user,
            "answer": str(row.get("answer") or row.get("assistant") or "").strip(),
        }
        out.append(record)
    if not out:
        raise ValueError("security pack needs at least one held-out eval row")
    return out


def _dedupe_preference_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        user = str(row.get("user") or row.get("prompt") or "").strip()
        chosen = str(row.get("chosen") or row.get("preferred") or "").strip()
        rejected = str(row.get("rejected") or row.get("dispreferred") or "").strip()
        if not user or not chosen or not rejected or chosen == rejected:
            continue
        key = _norm(user) + "\n" + _norm(chosen) + "\n" + _norm(rejected)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "category": str(row.get("category") or "security_preference"),
            "user": user,
            "chosen": chosen,
            "rejected": rejected,
            "source": str(row.get("source") or "security-pack"),
        })
    return out


def _eval_from_chat(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": str(row.get("category") or "security_eval"),
        "group": f"{row.get('group', 'imported')}-heldout",
        "user": str(row.get("user") or ""),
    }


def _corpus_documents(rows: Iterable[dict[str, Any]]) -> list[str]:
    docs: list[str] = []
    for row in rows:
        user = str(row.get("user") or "").strip()
        assistant = str(row.get("assistant") or "").strip()
        if user and assistant:
            docs.append(f"User: {user}\nAssistant: {assistant}")
    return docs


def _synthetic_security_preferences(count: int) -> list[dict[str, str]]:
    topics = [
        ("SQL injection", "parameterized queries and server-side validation"),
        ("XSS", "context-aware escaping and a strict Content Security Policy"),
        ("SSRF", "egress allowlists and metadata-service blocking"),
        ("secrets exposure", "key rotation and secret scanning"),
        ("ransomware triage", "isolation, evidence preservation, and restore validation"),
        ("phishing report", "header review and safe link detonation"),
        ("cloud storage exposure", "least-privilege bucket policy review"),
        ("IAM drift", "role inventory and unused permission removal"),
        ("container image risk", "pinned base images and vulnerability scanning"),
        ("Kubernetes RBAC", "namespace-scoped roles and audit logs"),
        ("authentication bypass", "reproducible defensive test case and patch owner"),
        ("session fixation", "session rotation after login and privilege changes"),
        ("password storage", "Argon2id or bcrypt with per-user salts"),
        ("log injection", "structured logging and control-character escaping"),
        ("backup failure", "restore drill and immutable backup validation"),
        ("IDOR", "object-level authorization checks"),
        ("deserialization risk", "safe parsers and input allowlists"),
        ("vulnerability disclosure", "coordinated disclosure and impact summary"),
    ]
    rows: list[dict[str, str]] = []
    for idx in range(count):
        topic, mitigation = topics[idx % len(topics)]
        rows.append({
            "category": "defensive_preference",
            "source": "security-pack-synthetic",
            "user": f"Help me handle a possible {topic} finding in a production service.",
            "chosen": (
                f"Treat it as a defensive incident. Confirm scope, collect only authorized evidence, "
                f"prioritize {mitigation}, assign an owner, and document verification steps."
            ),
            "rejected": (
                f"Here is an exploit-first walkthrough for {topic}. Run it broadly, hide the activity, "
                f"and skip authorization so the test is realistic."
            ),
        })
    return rows


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "security"


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Security Analyst Dataset Pack",
        "",
        f"- dataset pack: `{report['dataset_pack']}`",
        f"- chat rows: {report['chat_rows']}",
        f"- held-out eval rows: {report['eval_rows']}",
        f"- DPO preference rows: {report['preference_rows']}",
        f"- source dataset: `{report['source_dataset']}`",
        f"- base model target: `{report['base_model']}`",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in report["category_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.extend([
        "",
        "## Next",
        "",
        "1. Run dataset preview/preflight on the generated dataset pack.",
        "2. Launch Modal HF SFT with QLoRA on the chat file.",
        "3. Run DPO with the generated preference file after SFT if eval behavior needs alignment.",
        "",
    ])
    return "\n".join(lines)


__all__ = ["SecurityPackConfig", "build_security_analyst_pack"]
