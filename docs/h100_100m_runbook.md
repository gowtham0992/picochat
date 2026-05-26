---
layout: default
title: 100M Public Proof Runbook
---

# 100M Public Proof Runbook

This is the recommended one-rented-GPU path. Use it before spending on a 1B
run. The goal is not a frontier assistant; the goal is a public, honest,
reproducible 100M-class Picochat model with a complete evidence bundle.

## Target

Use the `h100-100m` preset:

- about 107M parameters
- 512 context
- 16 layers, 768 embedding
- GQA 12:4
- RMSNorm, RoPE, SwiGLU, QK norm, tied embeddings, parallel residual
- BF16, PyTorch SDPA FlashAttention, torch.compile
- 33,000 base steps
- 8 batch x 16 grad accumulation = 128 sequences per step
- about 2.16B planned base-training tokens
- `skill_release` gate after SFT/eval

## Dataset Decision

Use a bounded local pack derived from
[HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus):

- `fineweb-edu-dedup` for deduplicated educational web text.
- `cosmopedia-v2` for synthetic textbook/blog/story style explanations.

This is a better first 100M public proof than raw ClimbMix alone because the
source was curated for small language models. Do not try to consume the full
upstream corpus on a one-GPU run. Instead, import a bounded local pack, verify
the preflight replay/coverage checks, and train every local token shard in that
pack under the fixed 100M budget.

## 1. Setup

```bash
git clone https://github.com/gowtham0992/picochat.git
cd picochat
git checkout develop

sudo apt-get update
sudo apt-get install -y python3.10-venv python3.10-dev build-essential
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[hf,dev]"
```

Install a CUDA-compatible PyTorch wheel for the rented host. If FA3 is not
clean on that machine, use `--attn-backend flash`.

## 2. Sanity

```bash
mkdir -p logs runs

PYTHONUNBUFFERED=1 picochat sanity preh100 \
  --out-dir runs/h100-100m-sanity-v1 \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend flash \
  --include-compile \
  --capacity-scale h100-100m \
  2>&1 | tee logs/sanity-h100-100m.log
```

Do not continue until sanity passes.

## 3. Import SmolLM Sources

Start with a bounded local pack. These row counts are intentionally moderate;
after import, use preflight to decide whether the pack is too small or too
large for the fixed 100M budget.

```bash
mkdir -p runs/smollm-100m-sources logs

PYTHONUNBUFFERED=1 picochat data hf-import \
  --dataset HuggingFaceTB/smollm-corpus \
  --config fineweb-edu-dedup \
  --split train \
  --text-column text \
  --out runs/smollm-100m-sources/fineweb-edu-dedup/corpus.txt \
  --documents-dir runs/smollm-100m-sources/fineweb-edu-dedup/documents \
  --report runs/smollm-100m-sources/fineweb-edu-dedup/hf_import_report.json \
  --max-rows 800000 \
  --min-chars 200 \
  --document-shard-rows 1000 \
  2>&1 | tee logs/import-smollm-fineweb-edu-dedup.log

PYTHONUNBUFFERED=1 picochat data hf-import \
  --dataset HuggingFaceTB/smollm-corpus \
  --config cosmopedia-v2 \
  --split train \
  --text-column text \
  --out runs/smollm-100m-sources/cosmopedia-v2/corpus.txt \
  --documents-dir runs/smollm-100m-sources/cosmopedia-v2/documents \
  --report runs/smollm-100m-sources/cosmopedia-v2/hf_import_report.json \
  --max-rows 200000 \
  --min-chars 300 \
  --document-shard-rows 1000 \
  2>&1 | tee logs/import-smollm-cosmopedia-v2.log
```

If import size is too small, increase FineWeb-Edu rows first. If it is too
large for the instance disk or setup time, lower Cosmopedia rows first.

## 4. Create the Dataset Pack

```bash
mkdir -p runs/smollm-100m-pack-v1

cat > runs/smollm-100m-pack-v1/dataset_pack.json <<'JSON'
{
  "name": "smollm-100m-public-v1",
  "description": "Bounded SmolLM-Corpus pack for the Picochat 100M public proof run.",
  "corpus": {"recipe": "corpus_recipe.json"},
  "chat": "chat.jsonl",
  "eval": "eval.jsonl"
}
JSON

cat > runs/smollm-100m-pack-v1/corpus_recipe.json <<'JSON'
{
  "name": "smollm-100m-public-v1",
  "description": "FineWeb-Edu-Dedup plus Cosmopedia v2, imported into local document shards.",
  "sources": [
    {
      "path": "../smollm-100m-sources/fineweb-edu-dedup/documents",
      "label": "fineweb-edu-dedup"
    },
    {
      "path": "../smollm-100m-sources/cosmopedia-v2/documents",
      "label": "cosmopedia-v2"
    }
  ],
  "exclude": [
    "**/.DS_Store",
    "**/.git/**",
    "**/__pycache__/**"
  ]
}
JSON

touch runs/smollm-100m-pack-v1/chat.jsonl
touch runs/smollm-100m-pack-v1/eval.jsonl
```

Preview before creating SFT/eval:

```bash
picochat data preview \
  --dataset-pack runs/smollm-100m-pack-v1/dataset_pack.json \
  2>&1 | tee logs/preview-smollm-100m-pack-v1.log
```

## 5. Generate Release SFT/Eval

```bash
PYTHONUNBUFFERED=1 picochat data task-pack \
  --dataset-pack runs/smollm-100m-pack-v1/dataset_pack.json \
  --profile release_skills \
  --sft-rows 2800 \
  --eval-rows 700 \
  --skill-answer-style scratchpad \
  --source offline \
  --force \
  2>&1 | tee logs/task-pack-smollm-100m-release-skills.log
```

The pack must include identity, refusal, choice, arithmetic, and spelling
categories before a `skill_release` run is allowed.

## 6. Preflight

```bash
PYTHONUNBUFFERED=1 picochat run tiny \
  --out-dir runs/h100-smollm-100m-public-v1 \
  --dataset-pack runs/smollm-100m-pack-v1/dataset_pack.json \
  --scale h100-100m \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend flash \
  --long-run-gate-profile skill_release \
  --preflight-only \
  2>&1 | tee logs/preflight-h100-smollm-100m-public-v1.log
```

Continue only if these checks pass:

- `release_token_budget`
- `compute_optimal_horizon`
- `corpus_model_fit`
- `base_exposure`
- `sft_category_balance`
- `eval_skill_release_coverage`
- `attention_backend_runtime`
- contamination/honesty checks

If `corpus_model_fit` or `base_exposure` blocks, the pack is too small. Import
more FineWeb-Edu-Dedup rows, regenerate the preview, and rerun preflight.

## 7. Dry Run

```bash
PYTHONUNBUFFERED=1 picochat run tiny \
  --out-dir runs/h100-smollm-100m-public-dryrun-v1 \
  --dataset-pack runs/smollm-100m-pack-v1/dataset_pack.json \
  --scale h100-100m \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend flash \
  --base-steps 100 \
  --sft-steps 1 \
  --long-run-gate-profile research \
  2>&1 | tee logs/dryrun-h100-smollm-100m-public-v1.log
```

The dry run should reach training quickly, write checkpoints, and show finite
loss. It is not a release run; it exists to catch CUDA, tokenizer, compile,
shard, and checkpoint issues before the full burn.

## 8. Full Run

```bash
PYTHONUNBUFFERED=1 picochat run tiny \
  --out-dir runs/h100-smollm-100m-public-v1 \
  --dataset-pack runs/smollm-100m-pack-v1/dataset_pack.json \
  --scale h100-100m \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend flash \
  --long-run-gate-profile skill_release \
  2>&1 | tee logs/train-h100-smollm-100m-public-v1.log
```

Expected behavior:

- token shard build happens before step 1
- batch sampling should be `permutation` if the imported corpus stays under the
  permutation threshold
- validation BPB should drop smoothly
- no loss spike warnings above threshold
- SFT val loss should be monitored during SFT
- final release gate may still block if skills or honesty fail

## 9. Bundle and Publish Evidence

```bash
picochat run bundle \
  --run-dir runs/h100-smollm-100m-public-v1 \
  --out h100-smollm-100m-public-v1.tgz \
  --logs-dir logs \
  --strict

picochat export hf \
  --checkpoint runs/h100-smollm-100m-public-v1/sft/checkpoint \
  --tokenizer runs/h100-smollm-100m-public-v1/tokenizer.json \
  --out-dir exports/picochat-100m-smollm-public-v1 \
  --model-name picochat-100m-smollm-public-v1 \
  --license mit \
  --dataset-summary "Bounded HuggingFaceTB/smollm-corpus fineweb-edu-dedup + cosmopedia-v2 pack." \
  --eval-summary "Publish only with summary.md, gate report, eval report, and honesty report."
```

Only push to Hugging Face after the evidence bundle is complete:

```bash
picochat export hf \
  --checkpoint runs/h100-smollm-100m-public-v1/sft/checkpoint \
  --tokenizer runs/h100-smollm-100m-public-v1/tokenizer.json \
  --out-dir exports/picochat-100m-smollm-public-v1 \
  --model-name picochat-100m-smollm-public-v1 \
  --license mit \
  --dataset-summary "Bounded HuggingFaceTB/smollm-corpus fineweb-edu-dedup + cosmopedia-v2 pack." \
  --eval-summary "See attached release evidence." \
  --push-to-hub \
  --repo-id gowtham0992/picochat-100m-smollm-public-v1
```

Do not publish a checkpoint as a model-quality claim unless the model card links
the preflight, training summary, eval report, release gate, and honesty report.
