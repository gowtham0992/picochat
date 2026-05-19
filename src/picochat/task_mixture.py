"""Task-mixture SFT/eval packs for staged Picochat capability work.

Benchmark packs answer "can we produce a held-out local eval?". Task mixtures
answer a slightly different question: which curriculum should be used for a
given stage after base pretraining? Keeping that distinction explicit prevents
first-release runs from accidentally claiming broad math/spelling skill while
still giving research runs an explicit capability curriculum.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from picochat.benchmark_pack import (
    BENCHMARK_SKILL_ANSWER_STYLES,
    BENCHMARK_SOURCES,
    _assert_no_prompt_overlap,
    _build_benchmark_eval_result,
    _build_benchmark_sft_result,
    _combined_source_status,
    _contamination_report,
    _curriculum_stage_counts,
    _normalize_skill_answer_style,
    _normalize_source,
    _unique_rows,
    _write_jsonl,
)
from picochat.dataset_pack import load_dataset_pack, update_dataset_pack_tuning_paths
from picochat.tuning_data import inspect_chat_eval_data, inspect_chat_sft_data


TASK_MIXTURE_PROFILES = ("release", "capability", "balanced", "benchmark")
DEFAULT_TASK_MIXTURE_SFT_ROWS = 1600
DEFAULT_TASK_MIXTURE_EVAL_ROWS = 320


@dataclass(frozen=True)
class TaskMixtureComponent:
    name: str
    benchmark_profile: str
    train_weight: float
    eval_weight: float
    source: str | None = None
    skill_answer_style: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskMixturePackReport:
    dataset_pack: str
    chat_output_path: str
    eval_output_path: str
    report_path: str
    profile: str
    sft_rows: int
    eval_rows: int
    components: tuple[TaskMixtureComponent, ...]
    chat_component_counts: dict[str, int]
    eval_component_counts: dict[str, int]
    chat_categories: dict[str, int]
    eval_categories: dict[str, int]
    chat_stages: dict[str, int]
    eval_stages: dict[str, int]
    eval_splits: dict[str, int]
    source_mode: str
    source_status: str
    source_datasets: dict[str, int]
    fallback_reason: str | None
    contamination: dict[str, Any]
    promoted_to_pack: bool
    pack_chat_input: str | None
    pack_eval_input: str | None
    source_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "components": [component.to_dict() for component in self.components],
            "source_notes": list(self.source_notes),
        }


TASK_MIXTURE_COMPONENTS: dict[str, tuple[TaskMixtureComponent, ...]] = {
    "release": (
        TaskMixtureComponent(
            name="release_behavior",
            benchmark_profile="release_behavior",
            train_weight=1.0,
            eval_weight=1.0,
            source="offline",
            skill_answer_style="direct",
            note="Identity and refusal only; use this when the model should not claim math/spelling skill.",
        ),
    ),
    "capability": (
        TaskMixtureComponent(
            name="weak_skills",
            benchmark_profile="weak_skills",
            train_weight=0.72,
            eval_weight=0.72,
            source="offline",
            skill_answer_style="scratchpad",
            note="Over-sampled arithmetic and spelling drills with short work traces.",
        ),
        TaskMixtureComponent(
            name="behavior_anchor",
            benchmark_profile="release_behavior",
            train_weight=0.28,
            eval_weight=0.28,
            source="offline",
            skill_answer_style="direct",
            note="Keeps release identity and refusal behavior present during capability tuning.",
        ),
    ),
    "balanced": (
        TaskMixtureComponent(
            name="release_behavior",
            benchmark_profile="release_behavior",
            train_weight=0.25,
            eval_weight=0.25,
            source="offline",
            skill_answer_style="direct",
            note="Public-release identity and boundary behavior.",
        ),
        TaskMixtureComponent(
            name="weak_skills",
            benchmark_profile="weak_skills",
            train_weight=0.45,
            eval_weight=0.45,
            source="offline",
            skill_answer_style="scratchpad",
            note="Arithmetic and spelling transfer curriculum.",
        ),
        TaskMixtureComponent(
            name="benchmark",
            benchmark_profile="full",
            train_weight=0.30,
            eval_weight=0.30,
            source=None,
            skill_answer_style=None,
            note="General benchmark rows; uses --source for optional HF-backed rows.",
        ),
    ),
    "benchmark": (
        TaskMixtureComponent(
            name="benchmark",
            benchmark_profile="full",
            train_weight=1.0,
            eval_weight=1.0,
            source=None,
            skill_answer_style=None,
            note="Existing full benchmark curriculum as a named task mixture.",
        ),
    ),
}


def generate_task_mixture_pack(
    dataset_pack: str | Path,
    out_dir: str | Path | None = None,
    chat_out: str | Path | None = None,
    eval_out: str | Path | None = None,
    sft_rows: int = DEFAULT_TASK_MIXTURE_SFT_ROWS,
    eval_rows: int = DEFAULT_TASK_MIXTURE_EVAL_ROWS,
    seed: int = 42,
    source: str = "offline",
    profile: str = "capability",
    skill_answer_style: str = "scratchpad",
    force: bool = False,
    promote_to_pack: bool = True,
) -> TaskMixturePackReport:
    """Write a staged task-mixture SFT/eval pair and optionally promote it."""
    if sft_rows < 32:
        raise ValueError("sft_rows must be at least 32")
    if eval_rows < 16:
        raise ValueError("eval_rows must be at least 16")
    source = _normalize_source(source)
    profile = _normalize_task_mixture_profile(profile)
    skill_answer_style = _normalize_skill_answer_style(skill_answer_style)

    pack = load_dataset_pack(dataset_pack)
    pack_dir = Path(pack.path).parent
    output_dir = Path(out_dir) if out_dir is not None else pack_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    chat_path = _output_path(chat_out, output_dir / f"chat_task_mixture_{profile}.jsonl")
    eval_path = _output_path(eval_out, output_dir / f"eval_task_mixture_{profile}.jsonl")
    report_path = output_dir / f"task_mixture_{profile}.md"

    existing = [path for path in (chat_path, eval_path, report_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing task mixture file(s): {names}")

    components = TASK_MIXTURE_COMPONENTS[profile]
    chat_rows, chat_meta = _build_component_rows(
        components=components,
        total_rows=sft_rows,
        seed=seed,
        split_seed_offset=0,
        eval_rows=False,
        source=source,
        default_skill_answer_style=skill_answer_style,
        mixture_profile=profile,
    )
    eval_items, eval_meta = _build_component_rows(
        components=components,
        total_rows=eval_rows,
        seed=seed,
        split_seed_offset=100_000,
        eval_rows=True,
        source=source,
        default_skill_answer_style=skill_answer_style,
        mixture_profile=profile,
    )
    _assert_no_prompt_overlap(chat_rows, eval_items)
    contamination = _contamination_report(chat_rows, eval_items)

    chat_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(chat_path, chat_rows)
    _write_jsonl(eval_path, eval_items)

    chat_report = inspect_chat_sft_data(chat_path, preview_items=0)
    eval_report = inspect_chat_eval_data(eval_path, preview_items=0)
    if chat_report.status == "blocked":
        raise ValueError(f"generated task-mixture SFT data failed validation: {chat_report.summary}")
    if eval_report.status == "blocked":
        raise ValueError(f"generated task-mixture eval data failed validation: {eval_report.summary}")

    promoted_pack = None
    if promote_to_pack:
        promoted_pack = update_dataset_pack_tuning_paths(
            pack.path,
            chat_input=str(chat_path),
            eval_input=str(eval_path),
        )

    source_status = _merge_statuses(chat_meta["source_statuses"] + eval_meta["source_statuses"])
    fallback_reasons = [
        reason for reason in (chat_meta["fallback_reasons"] + eval_meta["fallback_reasons"]) if reason
    ]
    report = TaskMixturePackReport(
        dataset_pack=pack.path,
        chat_output_path=str(chat_path),
        eval_output_path=str(eval_path),
        report_path=str(report_path),
        profile=profile,
        sft_rows=len(chat_rows),
        eval_rows=len(eval_items),
        components=components,
        chat_component_counts=dict(sorted(Counter(row["mixture_component"] for row in chat_rows).items())),
        eval_component_counts=dict(sorted(Counter(row["mixture_component"] for row in eval_items).items())),
        chat_categories=dict(sorted(Counter(row["category"] for row in chat_rows).items())),
        eval_categories=dict(sorted(Counter(row["category"] for row in eval_items).items())),
        chat_stages=_curriculum_stage_counts(chat_rows),
        eval_stages=_curriculum_stage_counts(eval_items),
        eval_splits=dict(sorted(Counter(row.get("split", "heldout") for row in eval_items).items())),
        source_mode=source,
        source_status=source_status,
        source_datasets=dict(sorted((chat_meta["source_datasets"] + eval_meta["source_datasets"]).items())),
        fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
        contamination=contamination,
        promoted_to_pack=promoted_pack is not None,
        pack_chat_input=promoted_pack.chat_input if promoted_pack else None,
        pack_eval_input=promoted_pack.eval_input if promoted_pack else None,
        source_notes=tuple(dict.fromkeys((
            "ClimbMix or the selected corpus remains the base pretraining data.",
            "Task-mixture rows are supervised objectives only; no eval prompt is copied into SFT.",
            "Use release for first public model claims; use capability or balanced for research sweeps.",
            "Capability and balanced mixtures intentionally teach arithmetic/spelling patterns before measuring them.",
            "Synthetic rows use separate train/eval templates and held-out word pools.",
            "Rows include mixture_component metadata so reports can expose which lane helped or failed.",
            *chat_meta["source_notes"],
            *eval_meta["source_notes"],
        ))),
    )
    report_path.write_text(task_mixture_markdown(report), encoding="utf-8")
    return report


def task_mixture_markdown(report: TaskMixturePackReport) -> str:
    component_rows = "\n".join(
        "| "
        + " | ".join((
            component.name,
            component.benchmark_profile,
            f"{component.train_weight:.2f}",
            f"{component.eval_weight:.2f}",
            component.source or report.source_mode,
            component.skill_answer_style or "default",
            component.note,
        ))
        + " |"
        for component in report.components
    )
    chat_component_lines = "\n".join(f"- {name}: {count}" for name, count in report.chat_component_counts.items())
    eval_component_lines = "\n".join(f"- {name}: {count}" for name, count in report.eval_component_counts.items())
    chat_category_lines = "\n".join(f"- {name}: {count}" for name, count in report.chat_categories.items())
    eval_category_lines = "\n".join(f"- {name}: {count}" for name, count in report.eval_categories.items())
    chat_stage_lines = "\n".join(f"- {name}: {count}" for name, count in report.chat_stages.items())
    eval_stage_lines = "\n".join(f"- {name}: {count}" for name, count in report.eval_stages.items())
    eval_split_lines = "\n".join(f"- {name}: {count}" for name, count in report.eval_splits.items())
    source_dataset_lines = "\n".join(f"- {name}: {count}" for name, count in report.source_datasets.items())
    source_note_lines = "\n".join(f"- {note}" for note in report.source_notes)
    contamination = report.contamination
    return f"""# Task Mixture Pack

Dataset pack: `{report.dataset_pack}`

Profile: `{report.profile}`

Chat SFT: `{report.chat_output_path}`

Eval: `{report.eval_output_path}`

Rows: {report.sft_rows} SFT, {report.eval_rows} eval

Promoted to dataset pack: `{report.promoted_to_pack}`

## Components

| Component | Benchmark profile | Train weight | Eval weight | Source | Answer style | Note |
| --- | --- | ---: | ---: | --- | --- | --- |
{component_rows}

## Component Counts

Chat:
{chat_component_lines}

Eval:
{eval_component_lines}

## Categories

Chat:
{chat_category_lines}

Eval:
{eval_category_lines}

## Curriculum Stages

Chat:
{chat_stage_lines or "- none"}

Eval:
{eval_stage_lines or "- none"}

Eval splits:
{eval_split_lines or "- none"}

## Source

- Mode: `{report.source_mode}`
- Status: `{report.source_status}`
- Fallback reason: `{report.fallback_reason or "none"}`

Datasets:
{source_dataset_lines or "- none"}

Notes:
{source_note_lines}

## Contamination

- Status: `{contamination["status"]}`
- Exact prompt overlaps: {contamination["exact_prompt_overlaps"]}
- Near prompt overlaps: {contamination["near_prompt_overlaps"]}
- Answer overlaps: {contamination["answer_overlaps"]}
"""


def _build_component_rows(
    *,
    components: tuple[TaskMixtureComponent, ...],
    total_rows: int,
    seed: int,
    split_seed_offset: int,
    eval_rows: bool,
    source: str,
    default_skill_answer_style: str,
    mixture_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = _component_quotas(
        total_rows,
        [(component.name, component.eval_weight if eval_rows else component.train_weight) for component in components],
    )
    rows: list[dict[str, Any]] = []
    source_statuses: list[str] = []
    fallback_reasons: list[str | None] = []
    source_datasets: Counter[str] = Counter()
    source_notes: list[str] = []
    for index, component in enumerate(components):
        target = quotas.get(component.name, 0)
        if target <= 0:
            continue
        component_source = _normalize_source(component.source or source)
        answer_style = _normalize_skill_answer_style(component.skill_answer_style or default_skill_answer_style)
        builder = _build_benchmark_eval_result if eval_rows else _build_benchmark_sft_result
        requested = max(target, min(target * 4, target + 256))
        result = builder(
            requested,
            seed=seed + split_seed_offset + (index * 10_000),
            source=component_source,
            profile=component.benchmark_profile,
            skill_answer_style=answer_style,
        )
        tagged = [_tag_row(row, component, mixture_profile) for row in result.rows]
        chosen = _unique_rows(tagged, rows, limit=target)
        if len(chosen) < target:
            raise RuntimeError(
                f"could not generate enough unique {mixture_profile}/{component.name} rows: "
                f"{len(chosen)}/{target}"
            )
        rows.extend(chosen)
        source_statuses.append(result.source_status)
        fallback_reasons.append(result.fallback_reason)
        source_datasets.update(
            str(row.get("source_dataset", f"picochat_{component.benchmark_profile}"))
            for row in chosen
        )
        source_notes.extend(result.source_notes)
    if len(rows) != total_rows:
        raise RuntimeError(f"task mixture built {len(rows)} rows, expected {total_rows}")
    return rows, {
        "source_statuses": source_statuses,
        "fallback_reasons": fallback_reasons,
        "source_datasets": source_datasets,
        "source_notes": source_notes,
    }


def _tag_row(row: dict[str, Any], component: TaskMixtureComponent, mixture_profile: str) -> dict[str, Any]:
    tagged = dict(row)
    tagged["mixture_profile"] = mixture_profile
    tagged["mixture_component"] = component.name
    tagged["mixture_benchmark_profile"] = component.benchmark_profile
    return tagged


def _component_quotas(total: int, weights: list[tuple[str, float]]) -> dict[str, int]:
    positive = [(name, float(weight)) for name, weight in weights if float(weight) > 0]
    if not positive:
        raise ValueError("task mixture must include at least one positive component weight")
    weight_sum = sum(weight for _, weight in positive)
    raw = [(name, total * weight / weight_sum) for name, weight in positive]
    quotas = {name: int(value) for name, value in raw}
    remainder = total - sum(quotas.values())
    ranked = sorted(raw, key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True)
    for index in range(remainder):
        quotas[ranked[index % len(ranked)][0]] += 1
    return quotas


def _normalize_task_mixture_profile(profile: str) -> str:
    normalized = str(profile or "capability").strip().lower()
    if normalized not in TASK_MIXTURE_PROFILES:
        raise ValueError(f"profile must be one of {', '.join(TASK_MIXTURE_PROFILES)}")
    return normalized


def _merge_statuses(statuses: list[str]) -> str:
    if not statuses:
        return "unknown"
    merged = statuses[0]
    for status in statuses[1:]:
        merged = _combined_source_status(merged, status)
    return merged


def _output_path(path: str | Path | None, default: Path) -> Path:
    if path is None or not str(path).strip():
        return default
    return Path(path)
