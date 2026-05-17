# Picochat Task-Mixture Recipe

Picochat now separates three objectives that should not be blurred:

- Base pretraining learns language/modeling from the corpus.
- Capability task mixtures teach fragile closed-book skills such as short arithmetic and spelling.
- Release behavior SFT teaches identity, boundaries, and refusal without claiming broad skill.

This mirrors the practical lesson from nanochat-style pipelines without copying
their implementation: reasoning drills are explicit supervised tasks, not magic
emergence from a tiny base run.

## Generate Packs

Release-only pack:

```bash
PYTHONPATH=src python -m picochat.cli data task-pack \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --profile release \
  --sft-rows 1600 \
  --eval-rows 320 \
  --force
```

Capability research pack:

```bash
PYTHONPATH=src python -m picochat.cli data task-pack \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --out-dir runs/h100-climbmix-16shard-80k-capability-pack-v1 \
  --profile capability \
  --sft-rows 2400 \
  --eval-rows 480 \
  --source offline \
  --skill-answer-style scratchpad \
  --force \
  --no-promote
```

Balanced research pack:

```bash
PYTHONPATH=src python -m picochat.cli data task-pack \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --out-dir runs/h100-climbmix-16shard-80k-balanced-pack-v1 \
  --profile balanced \
  --sft-rows 3200 \
  --eval-rows 640 \
  --source auto \
  --skill-answer-style scratchpad \
  --force \
  --no-promote
```

## Stage The Runs

Use the base checkpoint as the controlled starting point:

```bash
PYTHONPATH=src python -m picochat.cli train sft \
  --input runs/h100-climbmix-16shard-80k-capability-pack-v1/chat_task_mixture_capability.jsonl \
  --eval-input runs/h100-climbmix-16shard-80k-capability-pack-v1/eval_task_mixture_capability.jsonl \
  --tokenizer runs/h100-climbmix-release-pilot-v1/tokenizer.json \
  --checkpoint runs/h100-climbmix-release-pilot-v1/base/best_checkpoint \
  --out-dir runs/h100-capability-midtrain-v1 \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --learning-rate 0.00001 \
  --steps 180 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --packing bos_bestfit \
  --sampling category_sqrt \
  --lr-warmup-steps 20 \
  --lr-decay cosine \
  --min-lr-ratio 0.1 \
  --grad-clip 1.0
```

Then evaluate with `picochat.cli eval sft-fit` and `picochat.cli eval chat`.
If capability transfer improves without prompt echo or unsupported claims,
repeat the release SFT from the same base or from the capability checkpoint and
compare both paths.

## Reporting Rule

Only claim what the active pack tests:

- `release`: identity/refusal behavior.
- `capability`: arithmetic/spelling transfer plus identity/refusal anchors.
- `balanced`: broad research diagnostics, not a first public release claim.

Every generated row includes `mixture_profile`, `mixture_component`, and
`mixture_benchmark_profile` so reports can isolate which lane helped or failed.
