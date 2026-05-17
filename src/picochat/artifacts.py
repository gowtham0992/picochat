"""Packaging helpers for interrupted and completed Picochat runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from pathlib import PurePosixPath
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


def inspect_run_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Inspect a copied run bundle without extracting model weights.

    This intentionally reads only tar metadata plus lightweight JSON metadata
    files. It does not load model.pt or training_state.pt, so it is safe to run
    on large copied archives before deciding whether to extract or resume.
    """
    path = Path(bundle_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"bundle does not exist: {path}")

    members: list[str] = []
    metadata_by_checkpoint: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] | None = _load_sidecar_manifest(path)
    embedded_manifest: dict[str, Any] | None = None

    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            members.append(member.name)
            if member.name.endswith(".manifest.json") and embedded_manifest is None:
                embedded_manifest = _read_tar_json(archive, member)
            if member.name.endswith("/metadata.json"):
                checkpoint_root = member.name.removesuffix("/metadata.json")
                if checkpoint_root.endswith("_checkpoint"):
                    metadata = _read_tar_json(archive, member)
                    if metadata is not None:
                        metadata_by_checkpoint[checkpoint_root] = metadata

    member_set = set(members)
    if manifest is None:
        manifest = embedded_manifest
    checkpoints = [
        _checkpoint_summary(root, metadata, member_set)
        for root, metadata in sorted(metadata_by_checkpoint.items())
    ]
    roots = sorted({_run_root_from_member(name) for name in members if _run_root_from_member(name)})
    has_corpus = any(name.endswith("/corpus.txt") for name in members)
    has_manifest = manifest is not None
    report = {
        "bundle": str(path),
        "bundle_bytes": path.stat().st_size,
        "manifest_found": has_manifest,
        "run_roots": roots,
        "included_file_count": len(members),
        "checkpoints": checkpoints,
        "resume_capable_checkpoints": [
            item for item in checkpoints
            if item["has_model"] and item["has_training_state"]
        ],
        "has_corpus": has_corpus,
        "has_tokenizer": any(name.endswith("/tokenizer.json") for name in members),
        "has_preflight": any(name.endswith("/preflight.md") or name.endswith("/preflight.json") for name in members),
        "has_summary": any(name.endswith("/summary.md") or name.endswith("/summary.json") for name in members),
        "embedded_manifest_found": embedded_manifest is not None,
        "sidecar_manifest_found": _sidecar_manifest_path(path).exists(),
        "manifest": manifest,
    }
    report["resume_hints"] = _bundle_resume_hints(report)
    return report


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


def bundle_inspection_markdown(report: dict[str, Any]) -> str:
    """Render a copied-bundle inspection as concise Markdown."""
    lines = [
        "# Picochat Bundle Inspection",
        "",
        f"- Bundle: `{report['bundle']}`",
        f"- Bundle bytes: {report['bundle_bytes']}",
        f"- Manifest found: `{report['manifest_found']}`",
        f"- Files scanned: {report['included_file_count']}",
        f"- Corpus included: `{report['has_corpus']}`",
        f"- Tokenizer included: `{report['has_tokenizer']}`",
        f"- Preflight included: `{report['has_preflight']}`",
        f"- Summary included: `{report['has_summary']}`",
        "",
    ]
    roots = report.get("run_roots") or []
    if roots:
        lines.extend(["## Run Roots", "", *[f"- `{root}`" for root in roots], ""])

    checkpoints = report.get("checkpoints") or []
    lines.extend(["## Checkpoints", ""])
    if checkpoints:
        lines.append("| Path | Phase | Kind | Step | Model | Training State |")
        lines.append("| --- | --- | --- | ---: | --- | --- |")
        for item in checkpoints:
            lines.append(
                f"| `{item['path']}` | `{item['phase']}` | `{item['checkpoint_kind']}` | "
                f"{item['step']} | `{item['has_model']}` | `{item['has_training_state']}` |"
            )
    else:
        lines.append("No checkpoint metadata found.")
    lines.append("")

    hints = report.get("resume_hints") or []
    lines.extend(["## Resume Hints", ""])
    if hints:
        lines.extend(f"- {hint}" for hint in hints)
    else:
        lines.append("- No resumable checkpoint was found in this bundle.")
    lines.append("")
    return "\n".join(lines)


def _sidecar_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _load_sidecar_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = _sidecar_manifest_path(path)
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_tar_json(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any] | None:
    handle = archive.extractfile(member)
    if handle is None:
        return None
    try:
        return json.loads(handle.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _run_root_from_member(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if len(parts) < 2:
        return None
    if parts[0] == "runs" and len(parts) >= 3:
        if parts[2] in {"base", "sft", "honesty", "eval", "sft_fit", "external_eval"}:
            return str(PurePosixPath(parts[0]) / parts[1])
        if parts[2] in {"tokenizer.json", "preflight.json", "preflight.md", "summary.json", "summary.md"}:
            return str(PurePosixPath(parts[0]) / parts[1])
    if parts[1] in {"base", "sft", "honesty", "eval", "sft_fit", "external_eval"}:
        return parts[0]
    if parts[1] in {"tokenizer.json", "preflight.json", "preflight.md", "summary.json", "summary.md"}:
        return parts[0]
    return None


def _checkpoint_summary(root: str, metadata: dict[str, Any], member_set: set[str]) -> dict[str, Any]:
    parts = PurePosixPath(root).parts
    phase = parts[-2] if len(parts) >= 2 else "unknown"
    config = metadata.get("model_config") if isinstance(metadata.get("model_config"), dict) else {}
    checkpoint_kind = metadata.get("checkpoint_kind") or (parts[-1] if parts else "unknown")
    has_training_state = f"{root}/training_state.pt" in member_set
    return {
        "path": root,
        "phase": phase,
        "checkpoint_dir": parts[-1] if parts else root,
        "checkpoint_kind": str(checkpoint_kind),
        "step": int(metadata.get("step", 0) or 0),
        "train_loss": metadata.get("train_loss"),
        "has_model": f"{root}/model.pt" in member_set,
        "has_training_state": has_training_state,
        "metadata_has_training_state": bool(metadata.get("has_training_state")),
        "n_layer": config.get("n_layer"),
        "n_embd": config.get("n_embd"),
        "n_head": config.get("n_head"),
        "n_kv_head": config.get("n_kv_head"),
        "context_size": config.get("context_size"),
        "vocab_size": config.get("vocab_size"),
    }


def _bundle_resume_hints(report: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    resume_checkpoints = report.get("resume_capable_checkpoints") or []
    by_phase = {item.get("phase"): item for item in resume_checkpoints}
    if "base" in by_phase:
        path = by_phase["base"]["path"]
        hints.append(
            f"After extracting the bundle, rerun the same `run tiny` command with "
            f"`--base-resume-from {path}`."
        )
    if "sft" in by_phase:
        path = by_phase["sft"]["path"]
        base_path = by_phase.get("base", {}).get("path", "<base/resume_checkpoint>")
        hints.append(
            f"For an interrupted SFT phase, pass both `--base-resume-from {base_path}` "
            f"and `--sft-resume-from {path}`."
        )
    if not report.get("has_corpus"):
        hints.append(
            "This bundle excludes `corpus.txt`; rebuild or recopy the same dataset pack "
            "before resuming so the fingerprint guard can validate the run."
        )
    return hints
