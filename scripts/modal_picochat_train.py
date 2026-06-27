"""Run a Picochat training recipe on Modal.

This script is intentionally thin: the local dashboard remains the control
plane, while Modal supplies the remote GPU process and a persistent /runs
volume for artifacts.

Example:
    modal run scripts/modal_picochat_train.py \
      --repo-url https://github.com/gowtham0992/picochat.git \
      --branch develop \
      --run-name security-smollm3-3b-qlora-v1 \
      --scale h100-100m \
      --gpu A100 \
      --mode hf-sft \
      --dataset-pack runs/security-analyst-pack-v1/dataset_pack.json \
      --hf-model HuggingFaceTB/SmolLM3-3B \
      --hf-sft-steps 3000 \
      --hf-batch-size 1 \
      --hf-grad-accum-steps 4 \
      --hf-dataset karpathy/climbmix-400b-shuffle \
      --hf-max-rows 800000
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "picochat-external-train"
REPO_DIR = Path("/workspace/picochat")
RUNS_DIR = Path("/runs")
IMPORT_DIR = RUNS_DIR / "imported-pack"
DEFAULT_VOLUME_NAME = "picochat-runs"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate",
        "bitsandbytes",
        "datasets",
        "numpy",
        "peft",
        "safetensors",
        "tokenizers",
        "torch",
        "tqdm",
        "transformers",
        "trl",
    )
)

runs_volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _clone_or_update(repo_url: str, branch: str) -> None:
    if REPO_DIR.exists():
        _run(["git", "-C", str(REPO_DIR), "fetch", "origin", branch, "--depth", "1"])
        _run(["git", "-C", str(REPO_DIR), "checkout", "FETCH_HEAD"])
        return
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(REPO_DIR)])


def _install_repo() -> None:
    _run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=REPO_DIR)


def _import_hf_dataset(
    hf_dataset: str,
    hf_split: str,
    hf_text_column: str,
    hf_max_rows: int,
    hf_shards: int,
) -> Path:
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    if hf_dataset == "karpathy/climbmix-400b-shuffle":
        _run(
            [
                sys.executable,
                "-m",
                "picochat.cli",
                "data",
                "climbmix-import",
                "--out-dir",
                str(IMPORT_DIR),
                "--shards",
                str(hf_shards),
                "--max-rows",
                str(hf_max_rows),
                "--document-shard-rows",
                "1000",
                "--force",
            ],
            cwd=REPO_DIR,
        )
        return IMPORT_DIR / "dataset_pack.json"
    _run(
        [
            sys.executable,
            "-m",
            "picochat.cli",
            "data",
            "hf-import",
            "--dataset",
            hf_dataset,
            "--split",
            hf_split,
            "--text-column",
            hf_text_column,
            "--max-rows",
            str(hf_max_rows),
            "--out",
            str(IMPORT_DIR / "corpus.txt"),
            "--report",
            str(IMPORT_DIR / "import_report.json"),
            "--pack-out",
            str(IMPORT_DIR),
            "--pack-name",
            "modal-import",
            "--pack-force",
        ],
        cwd=REPO_DIR,
    )
    return IMPORT_DIR / "dataset_pack.json"


def _resolve_dataset_pack(
    dataset_pack: str,
    hf_dataset: str,
    hf_split: str,
    hf_text_column: str,
    hf_max_rows: int,
    hf_shards: int,
) -> Path:
    if dataset_pack:
        candidate = Path(dataset_pack)
        if not candidate.is_absolute():
            candidate = REPO_DIR / candidate
        if candidate.exists():
            return candidate
        print(f"dataset pack not found on Modal image: {candidate}; importing from HF instead", flush=True)
    return _import_hf_dataset(hf_dataset, hf_split, hf_text_column, hf_max_rows, hf_shards)


def _build_security_pack_on_modal(hf_max_rows: int) -> Path:
    out_dir = RUNS_DIR / "security-analyst-pack"
    seed_dir = REPO_DIR / "datasets" / "security-analyst"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"missing security seed directory in cloned repo: {seed_dir}. "
            "Commit datasets/security-analyst before launching this Modal recipe."
        )
    _run(
        [
            sys.executable,
            "-m",
            "picochat.cli",
            "data",
            "security-pack",
            "--source",
            "trendyol",
            "--out-dir",
            str(out_dir),
            "--seed-dir",
            str(seed_dir),
            "--trendyol-max-rows",
            str(hf_max_rows),
            "--eval-rows",
            "500",
            "--preference-rows",
            "128",
            "--force",
        ],
        cwd=REPO_DIR,
    )
    return out_dir / "dataset_pack.json"


def _resolve_hf_sft_dataset_pack(dataset_pack: str, hf_max_rows: int) -> Path:
    if dataset_pack:
        candidate = Path(dataset_pack)
        if not candidate.is_absolute():
            candidate = REPO_DIR / candidate
        if candidate.exists():
            return candidate
        print(
            f"HF-SFT dataset pack not found on Modal image: {candidate}; rebuilding the security pack on Modal",
            flush=True,
        )
    return _build_security_pack_on_modal(hf_max_rows)


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_DIR / path
    return path


def _chat_input_from_pack(pack_path: Path) -> Path:
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    chat_ref = payload.get("chat") or payload.get("chat_input")
    if not chat_ref:
        raise ValueError(f"dataset pack has no chat input: {pack_path}")
    chat_path = Path(str(chat_ref))
    if not chat_path.is_absolute():
        chat_path = pack_path.parent / chat_path
    if not chat_path.is_file():
        raise FileNotFoundError(f"missing chat data from dataset pack: {chat_path}")
    return chat_path


def _run_native_training(
    pack_path: Path,
    run_name: str,
    scale: str,
    base_steps: str,
    sft_steps: str,
) -> Path:
    out_dir = RUNS_DIR / run_name
    cmd = [
        sys.executable,
        "-m",
        "picochat.cli",
        "run",
        "tiny",
        "--out-dir",
        str(out_dir),
        "--dataset-pack",
        str(pack_path),
        "--scale",
        scale,
        "--device",
        "cuda",
    ]
    if base_steps:
        cmd.extend(["--base-steps", str(base_steps)])
    if sft_steps:
        cmd.extend(["--sft-steps", str(sft_steps)])
    _run(cmd, cwd=REPO_DIR)
    return out_dir


def _run_hf_sft_training(
    pack_path: Path,
    run_name: str,
    hf_model: str,
    hf_sft_steps: int,
    hf_batch_size: int,
    hf_grad_accum_steps: int,
    hf_eval_batches: int,
    hf_log_every: int,
    hf_learning_rate: float,
    hf_max_length: int,
    hf_lora_rank: int,
    hf_lora_alpha: float,
    hf_quantize: str,
    preference_input: str,
    run_dpo: bool,
    dpo_steps: int,
    dpo_beta: float,
) -> Path:
    out_dir = RUNS_DIR / run_name
    chat_input = _chat_input_from_pack(pack_path)
    cmd = [
        sys.executable,
        "-m",
        "picochat.cli",
        "train",
        "hf-sft",
        "--model",
        hf_model,
        "--input",
        str(chat_input),
        "--out-dir",
        str(out_dir),
        "--max-steps",
        str(hf_sft_steps),
        "--batch-size",
        str(hf_batch_size),
        "--grad-accum-steps",
        str(hf_grad_accum_steps),
        "--learning-rate",
        str(hf_learning_rate),
        "--max-length",
        str(hf_max_length),
        "--eval-batches",
        str(hf_eval_batches),
        "--log-every",
        str(hf_log_every),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--peft",
        "lora",
        "--quantize",
        hf_quantize,
        "--lora-rank",
        str(hf_lora_rank),
        "--lora-alpha",
        str(hf_lora_alpha),
        "--gradient-checkpointing",
    ]
    _run(cmd, cwd=REPO_DIR)

    if run_dpo:
        prefs = _resolve_repo_path(preference_input) if preference_input else pack_path.parent / "preferences.jsonl"
        if not prefs.is_file():
            fallback_prefs = pack_path.parent / "preferences.jsonl"
            if fallback_prefs.is_file():
                prefs = fallback_prefs
            else:
                print(f"DPO preference file not found: {prefs}; skipping DPO", flush=True)
                return out_dir
        dpo_dir = out_dir / "dpo"
        _run(
            [
                sys.executable,
                "-m",
                "picochat.cli",
                "train",
                "hf-dpo",
                "--model",
                str(out_dir / "final_model"),
                "--input",
                str(prefs),
                "--out-dir",
                str(dpo_dir),
                "--max-steps",
                str(dpo_steps),
                "--beta",
                str(dpo_beta),
                "--device",
                "cuda",
            ],
            cwd=REPO_DIR,
        )
    return out_dir


@app.function(image=image, gpu="A100", volumes={str(RUNS_DIR): runs_volume}, timeout=8 * 60 * 60)
def train_remote(
    repo_url: str,
    branch: str,
    dataset_pack: str,
    hf_dataset: str,
    hf_split: str,
    hf_text_column: str,
    hf_max_rows: int,
    hf_shards: int,
    run_name: str,
    scale: str,
    base_steps: str,
    sft_steps: str,
    mode: str,
    hf_model: str,
    hf_sft_steps: int,
    hf_batch_size: int,
    hf_grad_accum_steps: int,
    hf_eval_batches: int,
    hf_log_every: int,
    hf_learning_rate: float,
    hf_max_length: int,
    hf_lora_rank: int,
    hf_lora_alpha: float,
    hf_quantize: str,
    preference_input: str,
    run_dpo: bool,
    dpo_steps: int,
    dpo_beta: float,
) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _clone_or_update(repo_url, branch)
    _install_repo()
    mode = (mode or "native").strip().lower()
    if mode == "hf-sft":
        pack_path = _resolve_hf_sft_dataset_pack(dataset_pack, hf_max_rows)
        out_dir = _run_hf_sft_training(
            pack_path,
            run_name,
            hf_model,
            hf_sft_steps,
            hf_batch_size,
            hf_grad_accum_steps,
            hf_eval_batches,
            hf_log_every,
            hf_learning_rate,
            hf_max_length,
            hf_lora_rank,
            hf_lora_alpha,
            hf_quantize,
            preference_input,
            run_dpo,
            dpo_steps,
            dpo_beta,
        )
    elif mode == "native":
        pack_path = _resolve_dataset_pack(dataset_pack, hf_dataset, hf_split, hf_text_column, hf_max_rows, hf_shards)
        out_dir = _run_native_training(pack_path, run_name, scale, base_steps, sft_steps)
    else:
        raise ValueError("mode must be native or hf-sft")
    try:
        runs_volume.commit()
    except Exception as exc:  # pragma: no cover - depends on Modal runtime behavior.
        print(f"volume commit warning: {exc}", flush=True)
    return str(out_dir)


@app.local_entrypoint()
def main(
    repo_url: str = "https://github.com/gowtham0992/picochat.git",
    branch: str = "develop",
    run_name: str = "picochat-modal-100m-v1",
    scale: str = "h100-100m",
    dataset_pack: str = "",
    hf_dataset: str = "karpathy/climbmix-400b-shuffle",
    hf_split: str = "train",
    hf_text_column: str = "text",
    hf_max_rows: int = 800000,
    hf_shards: int = 170,
    gpu: str = "A100",
    timeout_hours: int = 12,
    volume_name: str = DEFAULT_VOLUME_NAME,
    secret_name: str = "",
    base_steps: str = "",
    sft_steps: str = "",
    mode: str = "native",
    hf_model: str = "HuggingFaceTB/SmolLM3-3B",
    hf_sft_steps: int = 3000,
    hf_batch_size: int = 1,
    hf_grad_accum_steps: int = 4,
    hf_eval_batches: int = 20,
    hf_log_every: int = 25,
    hf_learning_rate: float = 2e-5,
    hf_max_length: int = 1024,
    hf_lora_rank: int = 16,
    hf_lora_alpha: float = 32.0,
    hf_quantize: str = "4bit",
    preference_input: str = "",
    run_dpo: bool = False,
    dpo_steps: int = 100,
    dpo_beta: float = 0.1,
) -> None:
    options: dict[str, object] = {"gpu": gpu, "timeout": int(timeout_hours) * 60 * 60}
    if volume_name != DEFAULT_VOLUME_NAME:
        volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        options["volumes"] = {str(RUNS_DIR): volume}
    if secret_name:
        options["secrets"] = [modal.Secret.from_name(secret_name)]
    remote = train_remote.with_options(**options)
    out_dir = remote.remote(
        repo_url,
        branch,
        dataset_pack,
        hf_dataset,
        hf_split,
        hf_text_column,
        hf_max_rows,
        hf_shards,
        run_name,
        scale,
        base_steps,
        sft_steps,
        mode,
        hf_model,
        hf_sft_steps,
        hf_batch_size,
        hf_grad_accum_steps,
        hf_eval_batches,
        hf_log_every,
        hf_learning_rate,
        hf_max_length,
        hf_lora_rank,
        hf_lora_alpha,
        hf_quantize,
        preference_input,
        run_dpo,
        dpo_steps,
        dpo_beta,
    )
    print(f"Picochat Modal run complete: {out_dir}")
