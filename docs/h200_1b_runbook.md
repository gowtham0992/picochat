---
layout: default
title: 8xH200 1B Runbook
---

# 8xH200 1B Runbook

This is the paid-compute path for a 1B-class Picochat run. Do not skip sanity,
preflight, or the DDP dry run.

The target scale is `h200-1b-ddp8`:

- about 1.12B parameters
- 2048 context
- 24 layers
- 2048 embedding
- GQA 16:4
- BF16
- FlashAttention/FA3 when available
- torch.compile
- gradient checkpointing
- 8-GPU DDP
- about 22.4B planned base-training tokens

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

Install a CUDA-compatible PyTorch wheel for the host. For FA3, use the exact
wheel/kernel combination that passes sanity on the instance image. If FA3 is
not clean on that host, use `--attn-backend flash` for PyTorch SDPA
FlashAttention rather than forcing a broken optional kernel.

Verify GPUs:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), round(p.total_memory / 1024**3, 1), torch.cuda.get_device_capability(i))
PY
```

## 2. Sanity

```bash
mkdir -p logs runs

PYTHONUNBUFFERED=1 picochat sanity preh100 \
  --out-dir runs/h200-ddp8-sanity-v1 \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend fa3 \
  --include-compile \
  --capacity-scale h200-1b-ddp8 \
  2>&1 | tee logs/sanity-h200-ddp8.log
```

`--capacity-scale` instantiates the target 1B model and runs one
forward/backward pass on the GPU so memory headroom is measured before the
paid DDP launch.

If FA3 fails because the optional kernel is not installed or compatible, rerun
with:

```bash
--attn-backend flash
```

Do not continue until sanity passes.

## 3. Import ClimbMix

Use the Scale Up page for generated commands, or run:

```bash
PYTHONUNBUFFERED=1 picochat data climbmix-import \
  --out-dir runs/climbmix-cuda \
  --shards 2048 \
  --max-rows 10000000 \
  --min-chars 100 \
  --document-shard-rows 1000 \
  --force \
  2>&1 | tee logs/import-climbmix.log
```

The import must produce a corpus manifest with document-boundary metadata.

## 4. Generate Release Skills Pack

```bash
PYTHONUNBUFFERED=1 picochat data benchmark-pack \
  --dataset-pack runs/climbmix-cuda/dataset_pack.json \
  --sft-rows 1600 \
  --eval-rows 320 \
  --profile release_skills \
  --skill-answer-style scratchpad \
  --source offline \
  --force \
  2>&1 | tee logs/benchmark-pack-cuda.log
```

If offline templates cannot produce enough unique rows, lower row counts or use
an approved external data source. Do not reuse eval prompts as SFT rows.

## 5. Preflight

```bash
PYTHONUNBUFFERED=1 picochat run tiny \
  --out-dir runs/h200-1b-release-preflight \
  --dataset-pack runs/climbmix-cuda/dataset_pack.json \
  --scale h200-1b-ddp8 \
  --device cuda \
  --ddp \
  --ddp-world-size 8 \
  --long-run-gate-profile skill_release \
  --preflight-only \
  2>&1 | tee logs/preflight-h200-1b-release.log
```

Required pass signals:

- `release_token_budget`
- `compute_optimal_horizon`
- `corpus_model_fit`
- `base_exposure`
- `sft_category_balance`
- `eval_skill_release_coverage`
- `attention_backend_runtime`
- no corpus/eval contamination blocks

## 6. 100-Step DDP Dry Run

This deliberately uses the `research` gate because release token-budget gates
should block a 100-step run.

```bash
OMP_NUM_THREADS=1 \
PICOCHAT_DDP_TIMEOUT_MINUTES=120 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=8 \
  -m picochat.cli run tiny \
  --out-dir runs/h200-1b-ddp8-dryrun \
  --dataset-pack runs/climbmix-cuda/dataset_pack.json \
  --scale h200-1b-ddp8 \
  --device cuda \
  --ddp \
  --ddp-world-size 8 \
  --base-steps 100 \
  --sft-steps 1 \
  --long-run-gate-profile research \
  2>&1 | tee logs/dryrun-h200-1b-ddp8.log
```

Watch for:

- all 8 ranks active
- no NCCL timeout
- memory headroom
- loss decreasing or at least behaving sanely
- no repeated spike warnings above the threshold
- `ddp_control.json` reaching setup completion

Optional: kill around step 50, resume from the checkpoint, and verify loss
continuation.

## 7. Full Run

Only launch this after sanity, preflight, and dry run pass.

```bash
OMP_NUM_THREADS=1 \
PICOCHAT_DDP_TIMEOUT_MINUTES=120 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=8 \
  -m picochat.cli run tiny \
  --out-dir runs/h200-1b-release \
  --dataset-pack runs/climbmix-cuda/dataset_pack.json \
  --scale h200-1b-ddp8 \
  --device cuda \
  --ddp \
  --ddp-world-size 8 \
  --long-run-gate-profile skill_release \
  2>&1 | tee logs/train-h200-1b-release.log
```

After the run:

1. inspect `summary.md`
2. inspect the long-run gate in `summary.json`
3. inspect honesty reports
4. run or attach external benchmark results
5. export or bundle only if the release gate is acceptable

## 8. Handoff to Domain Teams

Give downstream teams:

- base checkpoint or HF export
- tokenizer
- model config
- preflight report
- honesty report
- release gate status
- known limitations
- exact training data recipe

Domain fine-tuning works best when the base model already has enough language
and reasoning structure to adapt. The 1B run is a foundation experiment, not a
guarantee that every downstream domain will work without additional data and
evaluation.
