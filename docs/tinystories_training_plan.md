# TinyStories Training Plan

This plan is the current Picochat path for turning the local TinyStories sample
into an honest tiny story model.

## Goal

Train a small decoder-only language model from `runs/tinystories-1k/documents`
and measure whether it can write simple child-friendly stories without copying
training text or pretending to know outside facts.

Good progress means:

- lower base validation loss
- higher TinyStories eval pass rate
- lower SFT train/validation gap
- low memorization overlap
- low or zero canary hits
- improving validation BPB when tokenizer/model settings change
- readable generated samples

## Current Pack

Use the v5 pack for current runs:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v5.json
```

The pack points to:

- corpus: `runs/tinystories-1k/documents`
- SFT: `examples/tinystories_chat_v5.jsonl`
- eval: `examples/tinystories_eval_v5.jsonl`

The corpus rows are separate files so document-level validation can hold out
complete stories.

v5 is prompt-conditioned. SFT teaches the model to copy requested subjects,
lessons, and required words into a short scaffold before writing the story.
Eval is split into `prompt_conditioned`, `transfer`, `knowledge`, `refusal`,
and `safety`, so a failed run shows whether the bottleneck is prompt binding,
generalization, simple story knowledge, or refusal behavior.

Use the 10k pack after importing a larger local TinyStories sample:

```bash
.venv/bin/python -m picochat.cli data hf-import \
  --dataset roneneldan/TinyStories \
  --split train \
  --text-column text \
  --out runs/tinystories-10k/corpus.txt \
  --documents-dir runs/tinystories-10k/documents \
  --report runs/tinystories-10k/import_report.json \
  --max-rows 10000 \
  --min-chars 100

PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v5_10k.json
```

The 10k pack uses the same SFT/eval files as v5, but gives base pretraining
about ten times more story text before chat SFT.

## Near-Term Experiments

Run these as separate run folders and compare `summary.md` files.

### Baseline

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v5-pico-v1 --dataset-pack examples/tinystories_dataset_pack_v5.json --scale pico --split-mode document
```

### Larger Corpus Baseline

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v5-10k-pico-v1 --dataset-pack examples/tinystories_dataset_pack_v5_10k.json --scale pico --base-steps 3000 --sft-steps 1200 --split-mode document
```

### Bigger Tiny Model

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v5-128x4-v1 --dataset-pack examples/tinystories_dataset_pack_v5_10k.json --scale pico --tokenizer-vocab-size 1024 --n-embd 128 --n-layer 4 --n-head 4 --base-steps 12000 --sft-steps 1200 --base-lr-decay none --sft-lr-decay none --base-lr-warmup-steps 0 --sft-lr-warmup-steps 0 --base-grad-clip 0 --sft-grad-clip 0 --split-mode document
```

## Tokenizer Comparisons

Picochat keeps the character tokenizer as the educational baseline. This is useful for learning
because every character maps to a visible token id, but it is inefficient:
`puppy` becomes five tokens instead of one or two subword tokens.

The byte tokenizer is available with `--tokenizer-type byte`, and the
dependency-free BPE tokenizer is available with `--tokenizer-type bpe`. Both are
comparison tools, not automatic replacements for the char baseline.

1. Keep the current char tokenizer as the educational baseline.
2. Run the same TinyStories pack with byte and BPE tokenizers.
3. Keep all other run settings the same.
4. Compare validation loss, eval pass rate, training time, and sample quality.

Do not replace the char tokenizer until a comparison run proves the new
tokenizer helps.

## Current Finding

The previous BPE and larger-model runs lowered validation BPB but did not
reliably improve phrase-based eval. That means the bottleneck is not only raw
LM loss. Picochat now checks three things separately:

1. More base data with the 10k TinyStories pack.
2. Prompt binding with v5 scaffolded SFT examples.
3. Split-level eval, so prompt-conditioned failures and transfer failures are
   visible separately.

Compare runs using both pass rate and support match rate. A useful next run
should first improve `prompt_conditioned` support match before we expect broad
`transfer` gains.

## Longer Guarded Runs

Before running overnight or on a larger TinyStories import, keep the run
guarded. Use document split, best-validation checkpoints, BPB, train-only
canaries, and early stopping:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-guarded-v1 --dataset-pack examples/tinystories_dataset_pack_v5_10k.json --scale pico --base-steps 12000 --sft-steps 1200 --base-max-minutes 60 --sft-max-minutes 15 --base-early-stop-patience 6 --sft-early-stop-patience 4 --canary-count 3 --split-mode document
```

Interpretation rules:

- If train loss falls but validation BPB stops improving, the run is probably done.
- If the final checkpoint is worse than `best_checkpoint`, use `best_checkpoint`.
- If canary hits appear, treat the run as memorization-risk.
- If estimated train epochs get very high while eval stays flat, stop scaling
  steps and change tokenizer/model/data instead.

## What Not To Do

- Do not train on `examples/tinystories_eval_v5.jsonl`.
- Do not tune only until the 10 eval rows pass.
- Do not call the result a general assistant.
- Do not judge the model only by one generated sample.

This is a small TinyStories language model. It should be described as a tiny
story model trained from scratch, with visible limits and reports.
