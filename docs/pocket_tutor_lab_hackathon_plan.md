# Pocket Tutor Lab Hackathon Plan

Pocket Tutor Lab is a Backyard AI project: a private 5-minute tutor for one
learner who needs short arithmetic, spelling, and reading practice.

This document is planning and practice scaffolding. Rebuild the official app,
dataset, model, Space, trace, and report during the hackathon window.

## Judging Fit

- Specific real problem: one learner needs short daily practice.
- Person actually used it: capture one real session during the event period.
- Honest small-model fit: deterministic code grades answers; the model gives
  hints, explanations, variants, and parent summaries.
- Gradio polish: card-based custom UI, not a generic chatbot.

## Model Choice

Practice with `Qwen/Qwen2.5-1.5B-Instruct`.

Why:

- 1.54B parameters, comfortably below the 32B limit.
- Apache-2.0 license.
- Strong instruction following, math, structured output, and JSON behavior.
- Official GGUF artifacts exist for the llama.cpp bonus path.

Backup models:

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct`

## Product Loop

```text
learner profile -> 5-card round -> answer -> check -> hint/explain
-> mistake bank -> next round -> parent/tutor summary -> evidence panel
```

Deterministic code should grade:

- arithmetic exact answer
- spelling normalized exact answer
- multiple-choice exact answer

The fine-tuned model should handle:

- short hints without revealing the answer
- kind feedback after deterministic grading
- similar-card generation
- reading explanations
- parent/tutor summaries

## Practice Pack

Build practice rows:

```bash
python tools/build_pocket_tutor_practice.py \
  --out-dir runs/pocket-tutor-practice-pack-v1 \
  --train-rows 360 \
  --eval-rows 120
```

Practice LoRA fine-tune:

```bash
picochat train hf-sft \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --input runs/pocket-tutor-practice-pack-v1/pocket_tutor_train_messages.jsonl \
  --out-dir runs/qwen-pocket-tutor-practice-lora-v1 \
  --max-steps 200 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --learning-rate 0.00002 \
  --lr-warmup-steps 20 \
  --max-length 1024 \
  --device cuda \
  --precision bf16 \
  --gradient-checkpointing \
  --peft lora \
  --lora-rank 16 \
  --lora-alpha 32
```

On Apple Silicon, use this only as a tiny smoke test:

```bash
picochat train hf-sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input runs/pocket-tutor-practice-pack-v1/pocket_tutor_train_messages.jsonl \
  --out-dir runs/qwen-pocket-tutor-mps-smoke-v1 \
  --max-steps 10 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-length 512 \
  --device mps \
  --precision fp32 \
  --peft lora
```

## Bakeoff Checks

Before the official window, compare base vs practice fine-tune on:

- valid JSON rate
- hint-without-answer compliance
- arithmetic feedback quality after deterministic grading
- spelling feedback quality after deterministic grading
- parent summary usefulness
- latency on Space-like hardware

Do not claim the practice model in the final submission. During the hackathon,
rebuild the official data and model with the proven recipe.

