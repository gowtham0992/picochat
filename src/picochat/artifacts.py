"""Packaging helpers for interrupted and completed Picochat runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile
from typing import Any


DEFAULT_BUNDLE_SMALL_FILES = (
    "tokenizer.json",
    "preflight.json",
    "preflight.md",
    "summary.json",
    "summary.md",
)
DEFAULT_BUNDLE_DIRS = (
    "base/best_checkpoint",
    "base/resume_checkpoint",
    "sft/best_checkpoint",
    "sft/resume_checkpoint",
    "honesty",
    "eval",
    "sft_fit",
    "external_eval",
)
LARGE_RUN_FILES = (
    "corpus.txt",
    "corpus_manifest.json",
)
TOKEN_SHARD_DIRS = (
    "base/token_shards",
    "base/shards",
)


@dataclass(frozen=True)
class RunBundleConfig:
    run_dir: str | Path
    out_path: str | Path | None = None
    logs_dir: str | Path | None = None
    include_corpus: bool = False
    include_token_shards: bool = False
    strict: bool = False


def create_run_bundle(config: RunBundleConfig) -> dict[str, Any]:
    """Create a tar.gz bundle for a finished or interrupted run.

    The default bundle is intentionally checkpoint-first: it includes resumable
    and best checkpoints plus reports, but excludes giant corpus and token-shard
    files unless explicitly requested.
    """
    run_dir = Path(config.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    out_path = Path(config.out_path) if config.out_path else Path(f"{run_dir.name}-bundle.tgz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[Path] = []
    missing_expected: list[str] = []
    excluded_large: list[str] = []

    for relative in DEFAULT_BUNDLE_SMALL_FILES:
        _collect_path(run_dir / relative, entries, missing_expected, relative)
    for relative in DEFAULT_BUNDLE_DIRS:
        _collect_path(run_dir / relative, entries, missing_expected, relative)

    for relative in LARGE_RUN_FILES:
        candidate = run_dir / relative
        if config.include_corpus:
            _collect_path(candidate, entries, missing_expected, relative)
        elif candidate.exists():
            excluded_large.append(relative)

    for relative in TOKEN_SHARD_DIRS:
        candidate = run_dir / relative
        if config.include_token_shards:
            _collect_path(candidate, entries, missing_expected, relative)
        elif candidate.exists():
            excluded_large.append(relative)

    if config.logs_dir:
        logs_dir = Path(config.logs_dir)
        _collect_path(logs_dir, entries, missing_expected, str(logs_dir))

    checkpoint_entries = [
        item for item in entries
        if item.name == "model.pt" or item.name == "training_state.pt"
    ]
    if config.strict and not checkpoint_entries:
        raise FileNotFoundError(
            f"no checkpoint payload found under {run_dir}; expected best_checkpoint or resume_checkpoint"
        )

    entries = _dedupe_paths(entries)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "bundle": str(out_path),
        "include_corpus": config.include_corpus,
        "include_token_shards": config.include_token_shards,
        "logs_dir": str(config.logs_dir) if config.logs_dir else None,
        "included_files": [str(path) for path in entries if path.is_file()],
        "included_file_count": sum(1 for path in entries if path.is_file()),
        "missing_expected": missing_expected,
        "excluded_large": excluded_large,
        "resume_hint": (
            "Use base/resume_checkpoint for interrupted base training and "
            "sft/resume_checkpoint for interrupted SFT. Rebuild or recopy the original "
            "dataset/corpus when resuming if this bundle excludes corpus.txt."
        ),
    }

    manifest_json = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_md = out_path.with_suffix(out_path.suffix + ".manifest.md")
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_md.write_text(_bundle_manifest_markdown(manifest), encoding="utf-8")

    with tarfile.open(out_path, "w:gz") as archive:
        for path in entries:
            if path == out_path or path in (manifest_json, manifest_md):
                continue
            archive.add(path, arcname=_archive_name(path, run_dir))
        archive.add(manifest_json, arcname=manifest_json.name)
        archive.add(manifest_md, arcname=manifest_md.name)

    manifest["manifest_json"] = str(manifest_json)
    manifest["manifest_md"] = str(manifest_md)
    manifest["bundle_bytes"] = out_path.stat().st_size
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_md.write_text(_bundle_manifest_markdown(manifest), encoding="utf-8")
    return manifest


def _collect_path(path: Path, entries: list[Path], missing: list[str], label: str) -> None:
    if not path.exists():
        missing.append(label)
        return
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            entries.append(child)
    else:
        entries.append(path)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        key = path.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _archive_name(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir.parent))
    except ValueError:
        pass
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(Path(path.parent.name) / path.name)


def _bundle_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Picochat Run Bundle",
        "",
        f"- Run: `{manifest['run_dir']}`",
        f"- Bundle: `{manifest['bundle']}`",
        f"- Files included: {manifest['included_file_count']}",
        f"- Include corpus: `{manifest['include_corpus']}`",
        f"- Include token shards: `{manifest['include_token_shards']}`",
    ]
    if manifest.get("bundle_bytes") is not None:
        lines.append(f"- Bundle bytes: {manifest['bundle_bytes']}")
    if manifest.get("manifest_json"):
        lines.append(f"- Manifest JSON: `{manifest['manifest_json']}`")
    lines.extend([
        "",
        "## Resume",
        "",
        str(manifest.get("resume_hint") or ""),
        "",
    ])
    excluded = manifest.get("excluded_large") or []
    if excluded:
        lines.extend([
            "## Excluded Large Files",
            "",
            *[f"- `{item}`" for item in excluded],
            "",
        ])
    missing = manifest.get("missing_expected") or []
    if missing:
        lines.extend([
            "## Missing Optional Artifacts",
            "",
            *[f"- `{item}`" for item in missing],
            "",
        ])
    return "\n".join(lines)
