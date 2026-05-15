"""Export Picochat checkpoints as HuggingFace-style release folders."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import torch
import torch.nn as nn

from picochat.checkpoint import load_checkpoint
from picochat.tokenizer import load_tokenizer


@dataclass(frozen=True)
class HFExportConfig:
    checkpoint_path: str
    tokenizer_path: str
    out_dir: str
    model_name: str = "picochat"
    base_model: bool = True
    license_name: str = "unknown"
    dataset_summary: str = "Not provided."
    eval_summary: str = "Not provided."
    dynamic_int8: bool = False


def export_hf_checkpoint(config: HFExportConfig) -> dict[str, Any]:
    """Write a HF-style model folder plus release manifest and model card."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, metadata = load_checkpoint(config.checkpoint_path, map_location="cpu")
    tokenizer = load_tokenizer(config.tokenizer_path)

    weights_path = out_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), weights_path)
    tokenizer_path = out_dir / "tokenizer.json"
    shutil.copyfile(config.tokenizer_path, tokenizer_path)

    quantized_files: dict[str, str] = {}
    if config.dynamic_int8:
        _ensure_quantized_engine()
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model.eval(),
            {nn.Linear},
            dtype=torch.qint8,
        )
        quantized_path = out_dir / "pytorch_model.dynamic_int8.bin"
        torch.save(quantized_model.state_dict(), quantized_path)
        quantized_files["dynamic_int8"] = quantized_path.name

    hf_config = {
        "model_type": "picochat",
        "architectures": ["TinyGPT"],
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.context_size,
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "n_layer": model.config.n_layer,
        "dropout": model.config.dropout,
        "norm_type": model.config.norm_type,
        "position_encoding": model.config.position_encoding,
        "activation": model.config.activation,
        "rope_base": model.config.rope_base,
        "logit_softcap": model.config.logit_softcap,
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
        "pad_token_id": tokenizer.pad_id,
        "unk_token_id": tokenizer.unk_id,
        "picochat_model_config": model.config.to_dict(),
        "picochat_tokenizer_type": tokenizer.tokenizer_type,
        "picochat_checkpoint_metadata": metadata,
    }
    (out_dir / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")

    tokenizer_config = {
        "tokenizer_class": "PicochatTokenizer",
        "model_max_length": model.config.context_size,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "picochat_tokenizer_type": tokenizer.tokenizer_type,
        "native_tokenizer_file": "tokenizer.json",
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "format": "picochat-hf-style",
        "model_name": config.model_name,
        "base_model": config.base_model,
        "checkpoint_path": str(config.checkpoint_path),
        "tokenizer_path": str(config.tokenizer_path),
        "files": {
            "weights": "pytorch_model.bin",
            "config": "config.json",
            "tokenizer": "tokenizer.json",
            "tokenizer_config": "tokenizer_config.json",
            "model_card": "README.md",
            "serving_manifest": "serving_manifest.json",
            **quantized_files,
        },
        "limitations": [
            "This is a HF-style release folder for Picochat's custom TinyGPT architecture.",
            "Generic Transformers/vLLM/TGI loading needs a Picochat adapter or custom model code.",
        ],
    }
    serving_manifest = {
        "runtime": "picochat",
        "architecture": "TinyGPT",
        "checkpoint_format": "picochat-hf-style",
        "supports_kv_cache": True,
        "max_context_tokens": model.config.context_size,
        "tokenizer": {
            "type": tokenizer.tokenizer_type,
            "file": "tokenizer.json",
            "vocab_size": len(tokenizer),
        },
        "entrypoints": {
            "generate": "pico generate --checkpoint <checkpoint_dir> --tokenizer <tokenizer.json>",
            "chat": "pico chat --checkpoint <checkpoint_dir> --tokenizer <tokenizer.json>",
        },
        "artifacts": {
            "fp32_weights": "pytorch_model.bin",
            "dynamic_int8_weights": quantized_files.get("dynamic_int8"),
        },
        "limitations": [
            "Dynamic int8 weights are for Picochat/PyTorch CPU serving experiments.",
            "Load dynamic int8 by constructing TinyGPT, applying torch dynamic quantization to Linear layers, then loading the quantized state dict.",
            "This export does not create GGUF, TensorRT-LLM, vLLM, or TGI-native artifacts.",
        ],
    }
    (out_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "serving_manifest.json").write_text(
        json.dumps(serving_manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_model_card(config, model, tokenizer, metadata), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "files": manifest["files"],
        "manifest": str(out_dir / "release_manifest.json"),
        "serving_manifest": str(out_dir / "serving_manifest.json"),
        "model_card": str(out_dir / "README.md"),
        "num_parameters": model.num_parameters(),
        "tokenizer_type": tokenizer.tokenizer_type,
        "vocab_size": len(tokenizer),
        "dynamic_int8": config.dynamic_int8,
    }


def _model_card(config: HFExportConfig, model, tokenizer, metadata: dict) -> str:
    model_kind = "base model" if config.base_model else "fine-tuned model"
    return "\n".join([
        f"# {config.model_name}",
        "",
        f"This is a Picochat {model_kind} export for the custom `TinyGPT` architecture.",
        "",
        "## Intended Use",
        "",
        "Use this checkpoint as a transparent Picochat model artifact. For domain work, "
        "continue training or SFT with Picochat tooling, then re-export a derived release.",
        "",
        "## Architecture",
        "",
        f"- Parameters: {model.num_parameters():,}",
        f"- Layers: {model.config.n_layer}",
        f"- Embedding size: {model.config.n_embd}",
        f"- Attention heads: {model.config.n_head}",
        f"- Context size: {model.config.context_size}",
        f"- Tokenizer: `{tokenizer.tokenizer_type}` with {len(tokenizer)} tokens",
        "",
        "## Training Data",
        "",
        config.dataset_summary,
        "",
        "## Evaluation",
        "",
        config.eval_summary,
        "",
        "## License",
        "",
        config.license_name,
        "",
        "## Checkpoint Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
        "## Loading Note",
        "",
        "This folder follows common HuggingFace file names, but Picochat currently uses "
        "a custom architecture and tokenizer. Standard serving stacks require an adapter "
        "before they can load it directly.",
        "",
        "## Serving",
        "",
        "Picochat generation uses KV-cache decoding when the prompt plus requested "
        "completion fits inside the configured context window. See `serving_manifest.json` "
        "for runtime artifacts and limitations.",
        "",
    ])


def _ensure_quantized_engine() -> None:
    if torch.backends.quantized.engine != "none":
        return
    supported_engines = [
        engine
        for engine in torch.backends.quantized.supported_engines
        if engine != "none"
    ]
    if not supported_engines:
        raise RuntimeError("PyTorch dynamic quantization is unavailable: no quantized backend engine")
    torch.backends.quantized.engine = supported_engines[0]
