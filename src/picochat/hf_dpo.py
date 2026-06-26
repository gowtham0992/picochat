"""Direct Preference Optimization for fine-tuned Hugging Face models.

A second-stage aligner that runs *after* ``train hf-sft``: it nudges the model
toward preferred answers and away from rejected ones, using TRL's DPOTrainer
with a fresh LoRA adapter. The SFT adapter is merged into the base first, so the
DPO reference model is the SFT model (DPO refines what SFT produced).

Requires the ``hf`` + ``dpo`` extras (transformers, peft, trl).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HFDPOConfig:
    model: str            # an hf-sft run's final_model/ (LoRA adapter dir) or a full model
    input_path: str       # preference JSONL with user / chosen / rejected
    out_dir: str
    max_steps: int = 50
    learning_rate: float = 5e-6
    beta: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    device: str = "auto"
    seed: int = 42
    log_every: int = 10


def load_preference_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read user/chosen/rejected rows into TRL's conversational DPO format."""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        user = record.get("user") or record.get("prompt")
        chosen = record.get("chosen") or record.get("preferred") or record.get("winner")
        rejected = record.get("rejected") or record.get("dispreferred") or record.get("loser")
        if not (user and chosen and rejected):
            continue
        if str(chosen).strip() == str(rejected).strip():
            continue
        rows.append({
            "prompt": [{"role": "user", "content": str(user)}],
            "chosen": [{"role": "assistant", "content": str(chosen)}],
            "rejected": [{"role": "assistant", "content": str(rejected)}],
        })
    if not rows:
        raise ValueError(f"no usable preference rows in {path} (need user + chosen + rejected with chosen != rejected)")
    return rows


def _load_sft_model(model_path: str):
    """Load the model to align. If it's a LoRA adapter dir, merge the adapter into
    the base so DPO starts from (and references) the SFT weights."""
    from transformers import AutoModelForCausalLM

    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        base_name = json.loads(adapter_cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if base_name:
            base = AutoModelForCausalLM.from_pretrained(base_name)
            return PeftModel.from_pretrained(base, model_path).merge_and_unload()
    return AutoModelForCausalLM.from_pretrained(model_path)


def train_hf_dpo(config: HFDPOConfig) -> dict:
    from transformers import AutoTokenizer, set_seed
    from peft import LoraConfig
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer
    from datasets import Dataset

    from picochat.device import resolve_device

    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_sft_model(config.model)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    rows = load_preference_rows(config.input_path)
    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[m.strip() for m in config.lora_target_modules.split(",") if m.strip()],
        task_type="CAUSAL_LM",
    )
    args = TRLDPOConfig(
        output_dir=str(out_dir / "_trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_length=config.max_length,
        logging_steps=config.log_every,
        save_strategy="no",
        report_to=[],
        seed=config.seed,
        use_cpu=(device.type == "cpu"),
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(out_dir / "final_model"))
    tokenizer.save_pretrained(str(out_dir / "final_model"))

    losses: list[dict[str, float | int]] = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            losses.append({
                "step": int(entry.get("step", len(losses) + 1)),
                "train_loss": float(entry["loss"]),
                "lr": float(entry.get("learning_rate", 0.0)),
            })
    final_loss = float(result.training_loss) if result is not None and result.training_loss is not None else None
    report = {
        "model": config.model,
        "input": config.input_path,
        "out_dir": str(out_dir),
        "stage": "dpo",
        "beta": config.beta,
        "num_pairs": len(rows),
        "final_loss": final_loss,
        "losses": losses,
        "peft": {"mode": "lora", "rank": config.lora_rank, "alpha": config.lora_alpha},
    }
    (out_dir / "hf_dpo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(
        f"# HF DPO alignment\n\n- base SFT model: {config.model}\n- preference pairs: {len(rows)}\n"
        f"- beta: {config.beta}\n- final loss: {final_loss}\n",
        encoding="utf-8",
    )
    (out_dir / "done.txt").write_text("ok\n", encoding="utf-8")
    return report
