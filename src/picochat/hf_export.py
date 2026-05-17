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
    safetensors: bool = True
    transformers_adapter: bool = True


def export_hf_checkpoint(config: HFExportConfig) -> dict[str, Any]:
    """Write a HF-style model folder plus release manifest and model card."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, metadata = load_checkpoint(config.checkpoint_path, map_location="cpu")
    tokenizer = load_tokenizer(config.tokenizer_path)

    weights_path = out_dir / "pytorch_model.bin"
    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)
    tokenizer_path = out_dir / "tokenizer.json"
    shutil.copyfile(config.tokenizer_path, tokenizer_path)

    files: dict[str, str] = {
        "weights": "pytorch_model.bin",
        "config": "config.json",
        "tokenizer": "tokenizer.json",
        "tokenizer_config": "tokenizer_config.json",
        "model_card": "README.md",
        "serving_manifest": "serving_manifest.json",
        "requirements": "requirements.txt",
    }
    safetensors_error = None
    if config.safetensors:
        saved_safetensors, safetensors_error = _try_write_safetensors(
            state_dict,
            out_dir / "model.safetensors",
        )
        if saved_safetensors:
            files["weights_safetensors"] = "model.safetensors"

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
        files.update(quantized_files)

    hf_config = {
        "model_type": "picochat",
        "architectures": ["PicochatForCausalLM" if config.transformers_adapter else "TinyGPT"],
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.context_size,
        "context_size": model.config.context_size,
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "n_kv_head": model.config.n_kv_head,
        "n_layer": model.config.n_layer,
        "dropout": model.config.dropout,
        "norm_type": model.config.norm_type,
        "position_encoding": model.config.position_encoding,
        "activation": model.config.activation,
        "rope_base": model.config.rope_base,
        "logit_softcap": model.config.logit_softcap,
        "initializer_range": model.config.initializer_range,
        "gradient_checkpointing": model.config.gradient_checkpointing,
        "tie_embeddings": model.config.tie_embeddings,
        "qk_norm": model.config.qk_norm,
        "attn_backend": model.config.attn_backend,
        "parallel_residual": model.config.parallel_residual,
        "linear_bias": model.config.linear_bias,
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
        "pad_token_id": tokenizer.pad_id,
        "unk_token_id": tokenizer.unk_id,
        "picochat_model_config": model.config.to_dict(),
        "picochat_tokenizer_type": tokenizer.tokenizer_type,
        "picochat_checkpoint_metadata": metadata,
    }
    if config.transformers_adapter:
        hf_config["auto_map"] = {
            "AutoConfig": "configuration_picochat.PicochatConfig",
            "AutoModelForCausalLM": "modeling_picochat.PicochatForCausalLM",
        }
        files.update(_write_transformers_adapter(out_dir))
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
        "tokenizer_file": "tokenizer.json",
    }
    if config.transformers_adapter:
        tokenizer_config["auto_map"] = {
            "AutoTokenizer": ["tokenization_picochat.PicochatTokenizer", None],
        }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )
    (out_dir / "requirements.txt").write_text(_requirements_text(config), encoding="utf-8")

    manifest = {
        "format": "picochat-hf-style",
        "model_name": config.model_name,
        "base_model": config.base_model,
        "checkpoint_path": str(config.checkpoint_path),
        "tokenizer_path": str(config.tokenizer_path),
        "files": files,
        "safetensors": files.get("weights_safetensors") is not None,
        "safetensors_error": safetensors_error,
        "transformers_adapter": config.transformers_adapter,
        "limitations": [
            "This is a HF-style release folder for Picochat's custom TinyGPT architecture.",
            "Transformers loading requires trust_remote_code=True and the Picochat package installed.",
            "The Transformers adapter supports padded attention masks by compacting padded rows; high-throughput serving still needs a native vLLM/TGI/llama.cpp adapter.",
            "vLLM/TGI/llama.cpp still require native adapters or conversion work.",
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
            "safetensors_weights": files.get("weights_safetensors"),
            "dynamic_int8_weights": quantized_files.get("dynamic_int8"),
        },
        "transformers": {
            "adapter": config.transformers_adapter,
            "requires_trust_remote_code": config.transformers_adapter,
            "requires_picochat_package": config.transformers_adapter,
            "supports_padded_attention_mask": True,
        },
        "limitations": [
            "Dynamic int8 weights are for Picochat/PyTorch CPU serving experiments.",
            "Load dynamic int8 by constructing TinyGPT, applying torch dynamic quantization to Linear layers, then loading the quantized state dict.",
            "The Transformers adapter compacts padded rows and disables KV-cache for padded batches; use a native serving adapter for high-throughput padded batch decoding.",
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
        "safetensors": files.get("weights_safetensors") is not None,
        "safetensors_error": safetensors_error,
        "transformers_adapter": config.transformers_adapter,
    }


def _try_write_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> tuple[bool, str | None]:
    try:
        from safetensors.torch import save_file
    except ImportError:
        return False, "safetensors is not installed; install picochat[hf] or safetensors>=0.4"
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
    }
    save_file(tensors, str(path), metadata={"format": "pt"})
    return True, None


def _write_transformers_adapter(out_dir: Path) -> dict[str, str]:
    files = {
        "configuration_picochat.py": _configuration_adapter_text(),
        "modeling_picochat.py": _modeling_adapter_text(),
        "tokenization_picochat.py": _tokenization_adapter_text(),
    }
    for filename, text in files.items():
        (out_dir / filename).write_text(text, encoding="utf-8")
    return {
        "transformers_config_adapter": "configuration_picochat.py",
        "transformers_model_adapter": "modeling_picochat.py",
        "transformers_tokenizer_adapter": "tokenization_picochat.py",
    }


def _requirements_text(config: HFExportConfig) -> str:
    requirements = [
        "torch>=2.2",
        "transformers>=4.40",
        "picochat>=0.1.0",
    ]
    if config.safetensors:
        requirements.append("safetensors>=0.4")
    return "\n".join(requirements) + "\n"


def _configuration_adapter_text() -> str:
    return '''"""Transformers configuration for Picochat exports."""

from __future__ import annotations

from transformers import PretrainedConfig


class PicochatConfig(PretrainedConfig):
    model_type = "picochat"

    def __init__(
        self,
        vocab_size=260,
        context_size=64,
        max_position_embeddings=None,
        n_embd=128,
        n_head=4,
        n_kv_head=None,
        n_layer=2,
        dropout=0.0,
        norm_type="layernorm",
        position_encoding="learned",
        activation="gelu",
        rope_base=10000.0,
        logit_softcap=0.0,
        initializer_range=0.02,
        gradient_checkpointing=False,
        tie_embeddings=False,
        qk_norm=False,
        attn_backend="auto",
        parallel_residual=False,
        linear_bias=True,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        unk_token_id=3,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.context_size = context_size if context_size is not None else max_position_embeddings
        self.max_position_embeddings = self.context_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.norm_type = norm_type
        self.position_encoding = position_encoding
        self.activation = activation
        self.rope_base = rope_base
        self.logit_softcap = logit_softcap
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.tie_embeddings = tie_embeddings
        self.qk_norm = qk_norm
        self.attn_backend = attn_backend
        self.parallel_residual = parallel_residual
        self.linear_bias = linear_bias
        self.use_cache = use_cache

    def to_picochat_config(self):
        from picochat.model import GPTConfig

        return GPTConfig(
            vocab_size=self.vocab_size,
            context_size=self.context_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            n_layer=self.n_layer,
            dropout=self.dropout,
            norm_type=self.norm_type,
            position_encoding=self.position_encoding,
            activation=self.activation,
            rope_base=self.rope_base,
            logit_softcap=self.logit_softcap,
            initializer_range=self.initializer_range,
            gradient_checkpointing=self.gradient_checkpointing,
            tie_embeddings=self.tie_embeddings,
            qk_norm=self.qk_norm,
            attn_backend=self.attn_backend,
            parallel_residual=self.parallel_residual,
            linear_bias=self.linear_bias,
        )
'''


def _modeling_adapter_text() -> str:
    return '''"""Transformers model shim for Picochat exports.

This adapter keeps the exported folder small by reusing the installed Picochat
implementation. Install the release requirements, then load with
trust_remote_code=True.
"""

from __future__ import annotations

import torch

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from picochat.model import TinyGPT

from .configuration_picochat import PicochatConfig


class PicochatForCausalLM(PreTrainedModel):
    config_class = PicochatConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True

    def __init__(self, config: PicochatConfig):
        super().__init__(config)
        base = TinyGPT(config.to_picochat_config())
        self.token_embedding = base.token_embedding
        self.position_embedding = base.position_embedding
        self.blocks = base.blocks
        self.ln_f = base.ln_f
        self.lm_head = base.lm_head

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required")
        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))
        if labels is not None:
            use_cache = False
        has_padding = attention_mask is not None and bool((attention_mask == 0).any())
        if has_padding:
            if past_key_values is not None:
                raise ValueError("padded attention masks with past_key_values are not supported; pass use_cache=False")
            logits, loss = self._forward_padded(input_ids, attention_mask, labels)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
            )
        targets = self._shift_labels(labels, attention_mask) if labels is not None else None
        output = TinyGPT.forward(
            self,
            input_ids,
            targets=targets,
            past_kv=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            logits, loss, present = output
        else:
            logits, loss = output
            present = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present,
        )

    @staticmethod
    def _shift_labels(labels, attention_mask=None):
        if labels is None:
            return None
        ignore = -100
        targets = torch.cat(
            [labels[:, 1:], labels.new_full((labels.size(0), 1), ignore)],
            dim=1,
        )
        if attention_mask is not None:
            mask = attention_mask.to(device=labels.device).bool()
            if mask.shape != labels.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            next_mask = torch.cat(
                [mask[:, 1:], mask.new_zeros((mask.size(0), 1))],
                dim=1,
            )
            targets = targets.masked_fill(~mask | ~next_mask, ignore)
        return targets

    def _forward_padded(self, input_ids, attention_mask, labels=None):
        mask = attention_mask.to(device=input_ids.device).bool()
        if mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        records = []
        weighted_loss = None
        total_targets = input_ids.new_tensor(0, dtype=torch.long)
        for batch_index in range(input_ids.size(0)):
            active = mask[batch_index]
            if not bool(active.any()):
                raise ValueError("attention_mask must leave at least one token per row")
            positions = torch.nonzero(active, as_tuple=False).flatten()
            row_input = input_ids[batch_index, positions].unsqueeze(0)
            row_labels = labels[batch_index, positions].unsqueeze(0) if labels is not None else None
            row_targets = self._shift_labels(row_labels) if row_labels is not None else None
            row_logits, row_loss = TinyGPT.forward(
                self,
                row_input,
                targets=row_targets,
                use_cache=False,
            )
            records.append((batch_index, positions, row_logits))
            if row_targets is not None:
                target_count = (row_targets != -100).sum()
                if int(target_count.item()) > 0:
                    weighted = row_loss * target_count.to(device=row_loss.device, dtype=row_loss.dtype)
                    weighted_loss = weighted if weighted_loss is None else weighted_loss + weighted
                    total_targets = total_targets + target_count.to(device=total_targets.device)

        vocab_size = records[0][2].size(-1)
        logits = records[0][2].new_zeros(input_ids.size(0), input_ids.size(1), vocab_size)
        for batch_index, positions, row_logits in records:
            logits[batch_index, positions, :] = row_logits.squeeze(0)

        loss = None
        if labels is not None:
            if weighted_loss is None or int(total_targets.item()) == 0:
                loss = logits.sum() * 0.0
            else:
                loss = weighted_loss / total_targets.to(device=weighted_loss.device, dtype=weighted_loss.dtype)
        return logits, loss

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        has_padding = attention_mask is not None and bool((attention_mask == 0).any())
        if past_key_values is not None and has_padding:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "use_cache": False,
            }
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if past_key_values is None:
            return None
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
'''


def _tokenization_adapter_text() -> str:
    return '''"""Transformers tokenizer shim for Picochat tokenizer.json files."""

from __future__ import annotations

import os
import shutil

from transformers import PreTrainedTokenizer

from picochat.tokenizer import load_tokenizer


class PicochatTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file=None, **kwargs):
        if tokenizer_file is None:
            tokenizer_file = kwargs.pop("vocab_file", None)
        if tokenizer_file is None:
            tokenizer_file = kwargs.pop("native_tokenizer_file", None)
        if tokenizer_file is None:
            raise ValueError("tokenizer_file is required")
        self.tokenizer_file = tokenizer_file
        self.native_tokenizer = load_tokenizer(tokenizer_file)
        super().__init__(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.native_tokenizer)

    def get_vocab(self):
        return dict(self.native_tokenizer.token_to_id)

    def _tokenize(self, text, **kwargs):
        return [
            self.native_tokenizer.id_to_token[token_id]
            for token_id in self.native_tokenizer.encode(text)
        ]

    def _convert_token_to_id(self, token):
        return self.native_tokenizer.token_to_id.get(token, self.native_tokenizer.unk_id)

    def _convert_id_to_token(self, index):
        return self.native_tokenizer.id_to_token.get(int(index), "<unk>")

    def convert_tokens_to_string(self, tokens):
        ids = [self._convert_token_to_id(token) for token in tokens]
        return self.native_tokenizer.decode(ids, skip_special=True)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.native_tokenizer.bos_id, *token_ids_0, self.native_tokenizer.eos_id]
        return [
            self.native_tokenizer.bos_id,
            *token_ids_0,
            self.native_tokenizer.eos_id,
            *token_ids_1,
            self.native_tokenizer.eos_id,
        ]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        filename = "tokenizer.json" if filename_prefix is None else f"{filename_prefix}-tokenizer.json"
        out_path = os.path.join(save_directory, filename)
        shutil.copyfile(self.tokenizer_file, out_path)
        return (out_path,)
'''


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
        "This folder includes a Transformers `trust_remote_code` adapter when adapter "
        "export is enabled. Install `requirements.txt`, then load with "
        "`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`. The adapter "
        "depends on the Picochat package so the release uses the same audited model code "
        "as training.",
        "",
        "## Serving",
        "",
        "Picochat generation uses KV-cache decoding when the prompt plus requested "
        "completion fits inside the configured context window. See `serving_manifest.json` "
        "for runtime artifacts and limitations. The Transformers adapter supports padded "
        "attention masks by compacting padded rows, but high-throughput production serving "
        "still needs native runtime adapters before it should be presented as production-ready.",
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
