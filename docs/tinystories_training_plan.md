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

Use the v7 pack for current runs:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v7.json
```

The pack points to:

- corpus: `runs/tinystories-1k/documents`
- SFT: `examples/tinystories_chat_v7.jsonl`
- eval: `examples/tinystories_eval_v7.jsonl`

The corpus rows are separate files so document-level validation can hold out
complete stories.

v7 is transfer-focused. It keeps the v6 scaffold and refusal curriculum, then
adds natural prompt forms for required words, story continuation, direct story
knowledge, and safety boundaries. Eval is split into `prompt_conditioned`,
`transfer`, `knowledge`, `refusal`, and `safety`, so a failed run shows whether
the bottleneck is prompt binding, generalization, simple story knowledge, or
refusal behavior.

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

PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v7_10k.json
```

The 10k pack gives base pretraining about ten times more story text before
chat SFT. v7 keeps held-out required-word eval pairs out of SFT, then trains
nearby behavior on non-held-out pairs and natural prompt phrasing.

## Near-Term Experiments

Run these as separate run folders and compare `summary.md` files.

### Baseline

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v7-pico-v1 --dataset-pack examples/tinystories_dataset_pack_v7.json --scale pico --split-mode document
```

### Larger Corpus Balanced Curriculum

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v7-10k-pico-v1 --dataset-pack examples/tinystories_dataset_pack_v7_10k.json --scale pico --base-steps 3000 --sft-steps 1200 --split-mode document
```

### Bigger Tiny Model

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v7-128x4-v1 --dataset-pack examples/tinystories_dataset_pack_v7_10k.json --scale pico --tokenizer-vocab-size 1024 --n-embd 128 --n-layer 4 --n-head 4 --base-steps 12000 --sft-steps 1200 --split-mode document
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
reliably improve phrase-based eval. The v6 1-hour 2.9M-parameter run improved
pass rate from 30.00% to 53.85%, support match from 66.22% to 75.00%, and
memorization status from `medium` to `low`, while keeping prompt echo at 0.00%.
It still struggled on natural required-word prompts, story knowledge, and
transfer. That means the bottleneck is not only raw LM loss. Picochat now checks
four things separately:

1. More base data with the 10k TinyStories pack.
2. Prompt binding with scaffolded SFT examples.
3. Split-level eval, so prompt-conditioned failures and transfer failures are
   visible separately.
4. SFT category balance, so rare categories are not drowned out by story
   templates.

Compare runs using pass rate, support match rate, prompt echo rate, BPB, and
memorization diagnostics. A useful next run should improve weak splits without
increasing prompt echo or data leakage.

## Longer Guarded Runs

Before running overnight or on a larger TinyStories import, keep the run
guarded. Use document split, best-validation checkpoints, BPB, train-only
canaries, and early stopping:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v7-guarded-v1 --dataset-pack examples/tinystories_dataset_pack_v7_10k.json --scale pico --base-steps 12000 --sft-steps 1200 --base-max-minutes 60 --sft-max-minutes 15 --base-early-stop-patience 3 --sft-early-stop-patience 4 --canary-count 3 --split-mode document
```

Interpretation rules:

- If train loss falls but validation BPB stops improving, the run is probably done.
- If the final checkpoint is worse than `best_checkpoint`, use `best_checkpoint`.
- If canary hits appear, treat the run as memorization-risk.
- If estimated train epochs get very high while eval stays flat, stop scaling
  steps and change tokenizer/model/data instead.

## What Not To Do

- Do not train on `examples/tinystories_eval_v7.jsonl`.
- Do not tune only until the 10 eval rows pass.
- Do not call the result a general assistant.
- Do not judge the model only by one generated sample.

This is a small TinyStories language model. It should be described as a tiny
story model trained from scratch, with visible limits and reports.
