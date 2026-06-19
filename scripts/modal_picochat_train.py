"""Run a Picochat training recipe on Modal.

This script is intentionally thin: the local dashboard remains the control
plane, while Modal supplies the remote GPU process and a persistent /runs
volume for artifacts.

Example:
    modal run scripts/modal_picochat_train.py \
      --repo-url https://github.com/gowtham0992/picochat.git \
      --branch develop \
      --run-name picochat-modal-100m-v1 \
      --scale h100-100m \
      --gpu A100 \
      --hf-dataset karpathy/climbmix-400b-shuffle \
      --hf-max-rows 800000
"""

from __future__ import annotations

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
        "datasets",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "tqdm",
        "transformers",
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
) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _clone_or_update(repo_url, branch)
    _install_repo()
    pack_path = _resolve_dataset_pack(dataset_pack, hf_dataset, hf_split, hf_text_column, hf_max_rows, hf_shards)
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
    timeout_hours: int = 8,
    volume_name: str = DEFAULT_VOLUME_NAME,
    secret_name: str = "",
    base_steps: str = "",
    sft_steps: str = "",
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
    )
    print(f"Picochat Modal run complete: {out_dir}")
