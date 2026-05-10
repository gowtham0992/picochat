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

Use:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack.json
```

The pack points to:

- corpus: `runs/tinystories-1k/documents`
- SFT: `examples/tinystories_chat.jsonl`
- eval: `examples/tinystories_eval.jsonl`

The corpus rows are separate files so document-level validation can hold out
complete stories.

## Near-Term Experiments

Run these as separate run folders and compare `summary.md` files.

### Baseline

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-v1 --dataset-pack examples/tinystories_dataset_pack.json --context-size 256 --base-steps 1000 --sft-steps 600 --base-batch-size 4 --sft-batch-size 4 --split-mode document
```

### Lower SFT Pressure

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-sft-300 --dataset-pack examples/tinystories_dataset_pack.json --context-size 256 --base-steps 1000 --sft-steps 300 --base-batch-size 4 --sft-batch-size 4 --sft-learning-rate 0.0003 --split-mode document
```

### Bigger Tiny Model

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-model-128x3 --dataset-pack examples/tinystories_dataset_pack.json --context-size 256 --n-embd 128 --n-layer 3 --n-head 4 --base-steps 1500 --sft-steps 300 --base-batch-size 4 --sft-batch-size 4 --sft-learning-rate 0.0003 --split-mode document
```

## Tokenizer Roadmap

Picochat currently uses a character tokenizer. This is useful for learning
because every character maps to a visible token id, but it is inefficient:
`puppy` becomes five tokens instead of one or two subword tokens.

The byte tokenizer is now available with `--tokenizer-type byte`. This is a
comparison tool, not a replacement for the char baseline.

1. Keep the current char tokenizer as the educational baseline.
2. Run the same TinyStories pack with the byte tokenizer.
3. Keep all other run settings the same.
4. Compare validation loss, eval pass rate, training time, and sample quality.

Do not replace the char tokenizer until a comparison run proves the new
tokenizer helps.

## Current Finding

Expanding TinyStories SFT from 16 to 76 examples made SFT training healthier:
the train/validation gap stayed small instead of collapsing into obvious
memorization. However, the tiny char-level model still failed the transparent
eval because generated replies mostly repeated common character patterns rather
than following the prompt.

That means the next likely bottleneck is not just SFT row count. The next
comparison should target tokenizer efficiency and model capacity:

1. Keep the 76-row SFT pack.
2. Run char vs byte on the same dataset pack.
3. Compare reports before changing model size.
4. Only then scale model size.

Run the byte-tokenizer comparison:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-byte-v1 --dataset-pack examples/tinystories_dataset_pack.json --tokenizer-type byte --context-size 256 --base-steps 1000 --sft-steps 300 --base-batch-size 4 --sft-batch-size 4 --sft-learning-rate 0.0003 --split-mode document
```

## Longer Guarded Runs

Before running overnight or on a larger TinyStories import, keep the run
guarded. Use document split, best-validation checkpoints, BPB, train-only
canaries, and early stopping:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-guarded-v1 --dataset-pack examples/tinystories_dataset_pack.json --tokenizer-type byte --context-size 256 --base-steps 10000 --sft-steps 1000 --base-batch-size 4 --sft-batch-size 4 --sft-learning-rate 0.0003 --base-max-minutes 45 --sft-max-minutes 10 --base-early-stop-patience 6 --sft-early-stop-patience 4 --canary-count 3 --split-mode document
```

Interpretation rules:

- If train loss falls but validation BPB stops improving, the run is probably done.
- If the final checkpoint is worse than `best_checkpoint`, use `best_checkpoint`.
- If canary hits appear, treat the run as memorization-risk.
- If estimated train epochs get very high while eval stays flat, stop scaling
  steps and change tokenizer/model/data instead.

## What Not To Do

- Do not train on `examples/tinystories_eval.jsonl`.
- Do not tune only until the 10 eval rows pass.
- Do not call the result a general assistant.
- Do not judge the model only by one generated sample.

This is a small TinyStories language model. It should be described as a tiny
story model trained from scratch, with visible limits and reports.
