---
layout: default
title: Training Paths
---

# Training Paths

Picochat separates three workflows that are often mixed together in small-model
projects.

## Path A: Train From Scratch

Use this when the goal is to create a Picochat-native model.

```bash
picochat run tiny \
  --dataset-pack runs/my-pack/dataset_pack.json \
  --out-dir runs/my-scratch-model
```

This path owns the full factory:

- dataset import and corpus reports
- tokenizer training
- base pretraining
- SFT and optional DPO
- transparent eval
- honesty checks
- release gate
- Picochat checkpoint, bundle, registry, and serving artifacts

This is the right path for the 100M/1B proof runs.

## Path B: Fine-Tune an Existing Hugging Face Model

Use this when the goal is to start from an existing model such as SmolLM, Qwen,
or Llama and adapt it to a task.

```bash
picochat train hf-sft \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --input runs/my-pack/chat_benchmark.jsonl \
  --out-dir runs/smollm-hf-sft-v1 \
  --max-steps 200 \
  --max-length 1024 \
  --device cuda \
  --precision bf16 \
  --gradient-checkpointing
```

This path uses Picochat chat JSONL and assistant-only loss masking, but it
writes Hugging Face model folders:

- `final_model/`
- `best_model/`
- `hf_sft_report.json`
- `report.md`

It does not train a Picochat-native base model and does not claim a release gate
by itself. Use it for hackathons, adapter experiments, and quick task models
where pretraining from zero is not the objective.

## Path C: Evaluate, Gate, and Serve

Use this after a native Picochat run when the goal is evidence.

```bash
picochat eval chat \
  --checkpoint runs/my-scratch-model/sft/checkpoint \
  --tokenizer runs/my-scratch-model/tokenizer.json \
  --input runs/my-pack/eval_benchmark.jsonl \
  --out-dir runs/my-scratch-model/eval
```

Native release evidence should include:

- preflight report
- training summary
- SFT fit report
- held-out eval report
- external benchmark report where relevant
- honesty/contamination report
- release card or model registry entry

Existing Hugging Face models can still use Picochat-generated SFT/eval data,
but direct HF release gating is intentionally separate until Picochat has a full
HF eval bridge.
