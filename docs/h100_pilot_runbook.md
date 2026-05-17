# Picochat H100 Pilot Runbook

This is the current single-GPU H100 path for a serious pilot. It favors a
modern architecture, explicit FlashAttention verification, auditable artifacts,
and conservative SFT learning rates.

## 1. Instance Setup

Use a CUDA build compatible with the host driver. On hosts reporting CUDA 12.2
through `nvidia-smi`, the cu121 PyTorch wheel is a safe default.

```bash
git clone https://github.com/gowtham0992/picochat.git
cd picochat
git checkout develop

sudo apt-get update
sudo apt-get install -y python3.10-venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
python -m pip install -e ".[hf,dev]"
```

Verify CUDA before spending time:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("mem_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY
```

## 2. Pre-H100 Sanity

Run this before a long job. It validates bf16 backward, forced flash attention,
KV cache equivalence, resume fingerprint rejection, sharded loading, HF export,
and `torch.compile`.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli sanity preh100 \
  --out-dir runs/h100-sanity-v1 \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend flash \
  --include-compile
```

Do not continue if this fails.

The sanity output should include `modern_init_loss: pass`. For an 8k tokenizer,
a fresh model should start near `log(vocab)` loss, roughly 9.0. If the first
base-training line shows a loss in the hundreds, stop: that indicates a model
initialization or checkpoint compatibility bug, not learning.

## 3. Import ClimbMix Data

Start with 16 shards / 80k rows. This is large enough to expose throughput and
SFT behavior without turning the first instance into a blind spend.

```bash
mkdir -p logs

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli data climbmix-import \
  --out-dir runs/h100-climbmix-16shard-80k-pack-v1 \
  --shards 16 \
  --max-rows 80000 \
  --min-chars 100 \
  --force 2>&1 | tee logs/import-h100.log

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli data benchmark-pack \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --sft-rows 1600 \
  --eval-rows 320 \
  --profile release_behavior \
  --skill-answer-style direct \
  --source offline \
  --force 2>&1 | tee logs/benchmark-pack-h100.log
```

## 4. Preflight The Pilot

This recipe uses hf BPE, regex pretokenization, RoPE/RMSNorm/SwiGLU, GQA, tied
embeddings, QK norm, optional parallel residual, bf16, `torch.compile`,
FlashAttention, high matmul precision, and sharded base-token data. The same
defaults are also available as `--scale h100-pilot`; the expanded command below
keeps every important knob visible for auditability.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-16shard-80k-modern-pilot-v1 \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend flash \
  --torch-compile \
  --tokenizer-type hf_bpe \
  --tokenizer-vocab-size 8192 \
  --bpe-pretokenizer regex \
  --context-size 512 \
  --n-embd 384 \
  --n-head 8 \
  --n-kv-head 2 \
  --n-layer 8 \
  --norm-type rmsnorm \
  --position-encoding rope \
  --activation swiglu \
  --tie-embeddings \
  --qk-norm \
  --parallel-residual \
  --base-steps 5000 \
  --sft-steps 120 \
  --base-batch-size 8 \
  --base-grad-accum-steps 16 \
  --base-dataset-mode sharded \
  --base-shard-token-size 1000000 \
  --base-shard-cache-size 2 \
  --sft-batch-size 8 \
  --sft-grad-accum-steps 4 \
  --base-learning-rate 0.0001 \
  --sft-learning-rate 0.00001 \
  --base-lr-warmup-steps 500 \
  --sft-lr-warmup-steps 40 \
  --base-lr-decay cosine \
  --sft-lr-decay cosine \
  --base-min-lr-ratio 0.1 \
  --sft-min-lr-ratio 0.1 \
  --base-grad-clip 1.0 \
  --sft-grad-clip 1.0 \
  --sft-packing bos_bestfit \
  --sft-sampling category_sqrt \
  --eval-max-new-tokens 120 \
  --long-run-gate-profile first_release \
  --loss-spike-rollback \
  --auto-lr-scaling \
  --preflight-only 2>&1 | tee logs/preflight-h100-pilot.log
```

The `--auto-lr-scaling` flag makes the effective SFT LR `0.00002` at SFT
effective batch 32. That is intentionally much lower than the earlier
`0.0002` effective SFT LR that overfit within a few dozen H100 steps. The SFT
step count is also intentionally short because release-behavior packs are
compact; if exact fit is high but held-out eval is weak, improve the identity
curriculum instead of replaying the same rows longer.

The pilot pack uses `--profile release_behavior`: only identity and
refusal/boundary rows are used for first-release SFT and held-out eval. Keep
math, spelling, and choice as separate diagnostic sweeps until the narrow
release behavior gate is healthy. Pass `--long-run-gate-profile first_release`
on the train command when evaluating this release pack; it still reports all
categories present in the eval file, but it approves only the first releasable
closed-book behaviors instead of silently turning hard skill failures into a
product claim.

Optional comparable benchmarks can be attached directly to the run. Picochat
will convert ARC/MMLU-style JSONL/JSON/CSV into internal choice-eval JSONL,
score them with the same logprob evaluator, and record each result in
`summary.json` and `summary.md`.

```bash
# Append one or more of these to the preflight/run commands when files exist:
--external-eval arc-easy=/path/to/arc_easy.jsonl \
--external-eval mmlu-mini=/path/to/mmlu_mini.csv \
--external-eval-format auto \
--external-eval-max-rows 200
```

## 5. Run The Pilot

Only remove `--preflight-only` after the preflight is clean or only warning on
known LR-scaling notes.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-16shard-80k-modern-pilot-v1 \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --attn-backend flash \
  --torch-compile \
  --tokenizer-type hf_bpe \
  --tokenizer-vocab-size 8192 \
  --bpe-pretokenizer regex \
  --context-size 512 \
  --n-embd 384 \
  --n-head 8 \
  --n-kv-head 2 \
  --n-layer 8 \
  --norm-type rmsnorm \
  --position-encoding rope \
  --activation swiglu \
  --tie-embeddings \
  --qk-norm \
  --parallel-residual \
  --base-steps 5000 \
  --sft-steps 120 \
  --base-batch-size 8 \
  --base-grad-accum-steps 16 \
  --base-dataset-mode sharded \
  --base-shard-token-size 1000000 \
  --base-shard-cache-size 2 \
  --sft-batch-size 8 \
  --sft-grad-accum-steps 4 \
  --base-learning-rate 0.0001 \
  --sft-learning-rate 0.00001 \
  --base-lr-warmup-steps 500 \
  --sft-lr-warmup-steps 40 \
  --base-lr-decay cosine \
  --sft-lr-decay cosine \
  --base-min-lr-ratio 0.1 \
  --sft-min-lr-ratio 0.1 \
  --base-grad-clip 1.0 \
  --sft-grad-clip 1.0 \
  --sft-packing bos_bestfit \
  --sft-sampling category_sqrt \
  --eval-max-new-tokens 120 \
  --long-run-gate-profile first_release \
  --loss-spike-rollback \
  --auto-lr-scaling 2>&1 | tee logs/train-h100-pilot.log
```

## 6. If SFT Misses, Sweep The Release Behavior Lane

If the base BPB is healthy but first-release SFT fit is below the gate, do not
scale the model yet and do not mix math/spelling into the release checkpoint.
Sweep a narrow release-behavior SFT schedule from the best base checkpoint with
the same CUDA runtime settings. This isolates behavior tuning from base
learning.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli train sft-sweep \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --tokenizer runs/h100-climbmix-16shard-80k-modern-pilot-v1/tokenizer.json \
  --checkpoint runs/h100-climbmix-16shard-80k-modern-pilot-v1/base/best_checkpoint \
  --out-dir runs/h100-release-behavior-sft-sweep-v1 \
  --device cuda \
  --precision bf16 \
  --matmul-precision high \
  --learning-rates 0.000005,0.00001,0.00002,0.00004 \
  --steps 120,240,400 \
  --samplings category_sqrt,category_balanced \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --packing bos_bestfit \
  --lr-warmup-steps 40 \
  --lr-decay cosine \
  --min-lr-ratio 0.1 \
  --grad-clip 1.0 \
  --early-stop-patience 4 \
  --fit-max-rows 1000 \
  --eval-max-new-tokens 120 \
  --support-corpus runs/h100-climbmix-16shard-80k-modern-pilot-v1/corpus.txt \
  2>&1 | tee logs/sft-sweep-h100-release-behavior-v1.log
```

Promote only a candidate that clears the first-release gate. Keep weak-skills
sweeps separate, because they are diagnostics for future capability work, not a
license to claim the first chat release can do arithmetic or spelling reliably.

## 7. Monitor And Package

Use another SSH pane:

```bash
watch -n 2 nvidia-smi
tail -f logs/train-h100-pilot.log
```

The eval stages print an immediate `eval 0000/N` line and ETA-bearing progress
rows. If a stage is silent, inspect `ps`, `nvidia-smi`, and run directory
timestamps before terminating.

During base training, the first loss should be near the tokenizer log-vocab
baseline and then fall quickly. For an 8,192-token vocabulary, seeing the first
base loss around 8-10 is normal; seeing hundreds is a stop condition.

Before terminating the instance:

```bash
tar -czf h100-climbmix-modern-pilot-artifacts.tgz \
  runs/h100-sanity-v1 \
  runs/h100-climbmix-16shard-80k-pack-v1 \
  runs/h100-climbmix-16shard-80k-modern-pilot-v1 \
  logs
```

For model release tests, export the best SFT checkpoint after the run:

```bash
PYTHONPATH=src python -m picochat.cli export hf \
  --checkpoint runs/h100-climbmix-16shard-80k-modern-pilot-v1/sft/best_checkpoint \
  --tokenizer runs/h100-climbmix-16shard-80k-modern-pilot-v1/tokenizer.json \
  --out-dir runs/h100-climbmix-16shard-80k-modern-pilot-v1/hf-release \
  --model-name picochat-climbmix-h100-pilot \
  --license mit \
  --dataset-summary "ClimbMix 16-shard 80k-row pilot pack." \
  --eval-summary "See summary.md and eval/report.md in this run artifact."
```
