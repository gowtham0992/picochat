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
sudo apt-get install -y python3.10-venv python3.10-dev build-essential
python3 -m venv .venv
source .venv/bin/activate
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PICOCHAT_DDP_TIMEOUT_MINUTES=120

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

`--attn-backend flash` uses PyTorch SDPA FlashAttention. If the host has a
compatible `flash-attn` wheel installed, you can sanity-check the external
kernel path with `--attn-backend external_flash`. Keep the PyTorch SDPA path as
the portable default unless the external package passes sanity on that exact
machine image.

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
  --document-shard-rows 1000 \
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

For capability research, generate a separate task-mixture pack instead of
overloading the first-release pack. `release` is identity/refusal only,
`capability` mixes scratchpad arithmetic/spelling with release behavior, and
`balanced` adds general benchmark rows. Use these as explicit midtraining or
diagnostic packs; keep first-release model claims tied to `release_behavior` or
the `release` task mixture.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli data task-pack \
  --dataset-pack runs/h100-climbmix-16shard-80k-pack-v1/dataset_pack.json \
  --out-dir runs/h100-climbmix-16shard-80k-capability-pack-v1 \
  --sft-rows 2400 \
  --eval-rows 480 \
  --profile capability \
  --source offline \
  --skill-answer-style scratchpad \
  --force \
  --no-promote 2>&1 | tee logs/task-pack-capability-h100.log
```

A staged research sequence should train from the base checkpoint into the
capability task mixture, then run a narrow release SFT from the same base or
from the capability checkpoint depending on the claim being tested. Do not mix
those results in reports: capability rows are for math/spelling transfer
diagnostics, while release rows are for a conservative public SLM identity and
boundary claim.

## 4. Preflight The Pilot

This recipe uses hf BPE, regex pretokenization, RoPE/RMSNorm/SwiGLU, GQA, tied
embeddings, QK norm, optional parallel residual, bf16, `torch.compile`,
FlashAttention, high matmul precision, and sharded base-token data. The same
defaults are also available as `--scale h100-pilot`; the expanded command below
keeps every important knob visible for auditability.

For very large corpora, `--base-dataset-mode packed` is now available as the
more auditable disk path: it holds out complete source documents first, then
writes BOS-bestfit packed rows. `sharded` remains the fastest conservative
default and preserves BOS/EOS document boundaries when a corpus manifest exists,
but validates by token shard rather than by complete document.

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
  --sft-steps 180 \
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
  --sft-lr-warmup-steps 20 \
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
`0.0002` effective SFT LR that overfit within a few dozen H100 steps. The
release-behavior transfer sweep selected 180 SFT steps with a short warmup:
enough to clear first-release eval without returning to the high-exposure
700-step replay loop.

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
  --sft-steps 180 \
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
  --sft-lr-warmup-steps 20 \
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

## 6. Scale To 100M

After the 16-shard pilot proves the runtime and release-behavior SFT lane, the
next single-GPU scale target is the `h100-100m` preset. This is the compact
form of the 100M recipe used by the Scale Up screen: 768 width, 16 layers, GQA,
SwiGLU, FlashAttention, sharded base data, bf16, high matmul precision, and
first-release SFT defaults. The H100 presets also disable transformer linear
biases to match modern decoder-only LM practice while keeping legacy/local
scales backwards compatible.

The default H100 presets use PyTorch SDPA FlashAttention via
`--attn-backend flash`. If a Hopper box has the optional FA3 `kernels` package
installed and `sanity preh100 --attn-backend fa3` passes, you can run the same
recipe with `--attn-backend fa3`. Do not silently fall back on a paid run:
forced `fa3` should fail if the kernel is unavailable.

Before launching, generate an explicit scale plan. This is the reproducible
bridge between a target size, dataset token budget, global batch, and step
count. It is useful when changing GPU count, context size, or target model size
instead of hand-editing the run command.

```bash
PYTHONPATH=src python -m picochat.cli scale plan \
  --target-params 100m \
  --depth 16 \
  --dataset-tokens 667m \
  --world-size 1 \
  --out runs/h100-100m-scale-plan.md
```

For an 8-GPU box, use the same command with `--world-size 8` and compare the
recommended steps and LR candidates before deciding whether to use the
conservative `h100-100m-ddp8` preset or a fresh optimizer experiment.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli data climbmix-import \
  --out-dir runs/h100-climbmix-170shard-800k-pack-v1 \
  --shards 170 \
  --max-rows 800000 \
  --min-chars 100 \
  --document-shard-rows 1000 \
  --force 2>&1 | tee logs/import-h100-100m.log

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli data benchmark-pack \
  --dataset-pack runs/h100-climbmix-170shard-800k-pack-v1/dataset_pack.json \
  --sft-rows 1600 \
  --eval-rows 320 \
  --profile release_behavior \
  --skill-answer-style direct \
  --source offline \
  --force 2>&1 | tee logs/benchmark-pack-h100-100m.log

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-100m-release-v1 \
  --dataset-pack runs/h100-climbmix-170shard-800k-pack-v1/dataset_pack.json \
  --scale h100-100m \
  --device cuda \
  --long-run-gate-profile first_release \
  --preflight-only 2>&1 | tee logs/preflight-h100-100m.log
```

If preflight is clean or only warning on the reviewed token-shard validation
and LR-scaling tradeoffs, remove `--preflight-only`. Sharded base data
preserves source document BOS/EOS boundaries when `corpus_manifest.json` is
present, but validation is still a held-out token-shard split rather than a
complete-document split:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-100m-release-v1 \
  --dataset-pack runs/h100-climbmix-170shard-800k-pack-v1/dataset_pack.json \
  --scale h100-100m \
  --device cuda \
  --long-run-gate-profile first_release \
  2>&1 | tee logs/train-h100-100m.log
```

Do not call this a general reasoning model unless external and internal
diagnostic benchmarks support that claim. This lane is for a stronger base
checkpoint plus first-release identity/refusal behavior.

### 6A. 8-GPU Variant

For an 8x H100/B200 box, do not reuse the single-GPU 33k-step command. The
global base batch and SFT example batch are both 8x larger, so the same budget
needs about one eighth the steps in both phases. Run preflight once with a
simulated DDP world size:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-100m-ddp8-release-v1 \
  --dataset-pack runs/h100-climbmix-170shard-800k-pack-v1/dataset_pack.json \
  --scale h100-100m-ddp8 \
  --device cuda \
  --ddp \
  --ddp-world-size 8 \
  --long-run-gate-profile first_release \
  --preflight-only 2>&1 | tee logs/preflight-h100-100m-ddp8.log
```

If the only warnings are the reviewed token-shard validation tradeoff and LR
notes, launch the actual distributed run with `torchrun`:

```bash
OMP_NUM_THREADS=1 PICOCHAT_DDP_TIMEOUT_MINUTES=120 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=8 -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-100m-ddp8-release-v1 \
  --dataset-pack runs/h100-climbmix-170shard-800k-pack-v1/dataset_pack.json \
  --scale h100-100m-ddp8 \
  --device cuda \
  --ddp \
  --ddp-world-size 8 \
  --long-run-gate-profile first_release \
  2>&1 | tee logs/train-h100-100m-ddp8.log
```

This preset uses explicit learning rates instead of `--auto-lr-scaling`, because
auto scaling includes DDP world size and can silently turn a stable single-GPU
LR into a different optimizer experiment.

## 7. If SFT Misses, Sweep The Release Behavior Lane

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

For domain adaptation sweeps where the base checkpoint is expensive and you
want lightweight adapter artifacts, add `--peft lora --lora-rank 8
--lora-alpha 16 --lora-targets attn_qkv,attn_proj` to the SFT sweep. Picochat
will train only the adapter weights, save adapter files, and still write merged
full checkpoints for the existing eval/export path.

## 8. Monitor And Package

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

Training reports include tokens/sec, estimated FLOP/s, and MFU when Picochat
can identify the GPU family. Set `PICOCHAT_PEAK_TFLOPS=<per-run TFLOP/s>` to
override the built-in H100/H200/B200/A100 reference when benchmarking a new
instance type or a multi-GPU box.

For sharded base data, setup now prints `base data: token shard build ...`
progress while token shards are being written. That stage is CPU/tokenizer
work; the GPU starts doing real work after `base data: ready ... mode=sharded`
and the first `step 0001/...` row.

Before terminating the instance, package the run with Picochat's artifact
bundler instead of hand-writing a `tar` file list. By default this keeps the
copy small: checkpoints, tokenizer, progress files, summaries, eval reports,
and logs are included; `corpus.txt`, `corpus_manifest.json`, and token shards
are excluded unless you pass `--include-corpus` or `--include-token-shards`.

```bash
PYTHONPATH=src python -m picochat.cli run bundle \
  --run-dir runs/h100-climbmix-16shard-80k-modern-pilot-v1 \
  --out h100-climbmix-modern-pilot-artifacts.tgz \
  --logs-dir logs \
  --strict
```

After copying a bundle back to your workstation, inspect it before extracting
or resuming. This reads checkpoint metadata only; it does not load model
weights.

```bash
PYTHONPATH=src python -m picochat.cli run inspect-bundle \
  --bundle h100-climbmix-modern-pilot-artifacts.tgz
```

For an interrupted 100M run, package the partial checkpoint the same way and
resume later from `base/resume_checkpoint` after recreating or copying the
same dataset/corpus.

To resume the top-level `run tiny` pipeline after an interrupted base phase,
rerun the same command and add `--base-resume-from`. Keep the dataset pack,
scale, tokenizer settings, seed, split/shard settings, and optimizer schedule
identical; Picochat validates the training fingerprint and refuses mismatched
corpus/tokenizer/model settings.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-100m-release-v1 \
  --dataset-pack runs/h100-climbmix-170shard-800k-pack-v1/dataset_pack.json \
  --scale h100-100m \
  --device cuda \
  --long-run-gate-profile first_release \
  --base-resume-from runs/h100-climbmix-100m-release-v1/base/resume_checkpoint \
  2>&1 | tee logs/train-h100-100m-resume.log
```

If only chat SFT was interrupted after base training completed, pass both
resume flags. The base resume checkpoint should already be at the planned
base step, so the base phase validates and exits without spending another full
base run before SFT resumes.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/h100-climbmix-100m-release-v1 \
  --dataset-pack runs/h100-climbmix-170shard-800k-pack-v1/dataset_pack.json \
  --scale h100-100m \
  --device cuda \
  --long-run-gate-profile first_release \
  --base-resume-from runs/h100-climbmix-100m-release-v1/base/resume_checkpoint \
  --sft-resume-from runs/h100-climbmix-100m-release-v1/sft/resume_checkpoint \
  2>&1 | tee logs/train-h100-100m-sft-resume.log
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
