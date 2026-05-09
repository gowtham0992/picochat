# Picochat Pipeline Guide

Picochat is a tiny local LLM factory. The goal is not to pretend this is a
large assistant. The goal is to make each step of language-model training small
enough to inspect, rerun, and explain.

The pipeline is:

```text
dataset -> tokenizer -> base pretraining -> chat SFT -> eval -> chat -> report
```

Each stage writes artifacts to a run folder so the next stage has visible
inputs instead of hidden state.

## 1. Dataset

Purpose: turn local source files into one normalized training corpus.

Input examples:

- `examples/tiny_corpus.txt`
- a folder of `.txt`, `.md`, `.jsonl`, `.csv`, or `.py` files
- a dataset pack such as `examples/tiny_dataset_pack.json`

Output artifacts:

- `corpus.txt`
- `corpus_manifest.json`
- `corpus_report.md`

What to inspect:

- how many files were included or skipped
- duplicate-line rate
- empty-line rate
- source quality scores
- whether files were filtered by `--min-score`

Important idea: a tiny model cannot learn what the dataset does not contain.
If the corpus is tiny, repeated, noisy, or off-topic, the model will mostly
memorize that data.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tiny_dataset_pack.json
```

## 2. Tokenizer

Purpose: convert text into token IDs the model can read.

Picochat currently starts with a character tokenizer because it is easy to
inspect. That makes it slower and less efficient than modern BPE tokenizers,
but it is ideal for learning the mechanics.

Input:

- `corpus.txt`

Output:

- `tokenizer.json`

What to inspect:

- vocab size
- special tokens
- how a string maps to IDs
- whether important characters are missing

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli tok train --input runs/manual/corpus.txt --out runs/manual/tokenizer.json
```

## 3. Base Pretraining

Purpose: train a decoder-only transformer to predict the next token.

This is the actual language-model training stage. The model sees token windows
from the corpus and learns next-token patterns.

Inputs:

- `corpus.txt`
- `tokenizer.json`

Output artifacts:

- `base/checkpoint/`
- `base/train_report.json`
- `base/report.md`
- `base/sample.txt`

What to inspect:

- train loss
- validation loss
- final train/validation gap
- best validation step
- generated base sample

Important idea: decreasing train loss only means the model is fitting the
training windows. Validation loss tells you whether that fit is carrying over
to held-out windows.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli train base --corpus runs/manual/corpus.txt --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/base --context-size 128 --max-steps 300
```

## 4. Chat SFT

Purpose: teach the base model a chat response format using supervised examples.

SFT does not create new knowledge by itself. It teaches behavior found in the
chat JSONL rows.

Inputs:

- chat JSONL with `user` and `assistant` fields
- `tokenizer.json`
- base checkpoint

Output artifacts:

- `sft/checkpoint/`
- `sft/sft_report.json`
- `sft/report.md`
- `sft/sample.txt`

What to inspect:

- number of usable chat examples
- truncated examples
- supervised answer tokens
- SFT train/validation gap
- whether validation loss diverged while train loss fell

Important idea: on very small chat files, SFT can quickly memorize exact
answers. That is why Picochat reports a `memorization-risk` diagnostic instead
of hiding the gap.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli train sft --input examples/tiny_chat.jsonl --tokenizer runs/manual/tokenizer.json --checkpoint runs/manual/base/checkpoint --out-dir runs/manual/sft --max-steps 600
```

## 5. Eval

Purpose: score generated replies with transparent rules.

Picochat evals are intentionally simple. Each item can define required phrases,
any-of phrase groups, forbidden phrases, and whether the question is answerable.

Inputs:

- eval JSONL
- SFT checkpoint
- tokenizer

Output artifacts:

- `eval/eval_report.json`
- `eval/report.md`

What to inspect:

- pass rate
- unsupported claim rate
- missing support rate
- matched and missing phrases
- forbidden phrases found in replies

Important idea: this is not semantic truth evaluation. It is an inspectable
measurement for a tiny model, especially for whether it makes unsupported
answers when it should refuse.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli eval chat --input examples/tiny_eval.jsonl --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/eval
```

## 6. Chat And Generation

Purpose: sample text from a checkpoint and inspect behavior manually.

Inputs:

- base or SFT checkpoint
- tokenizer
- prompt

What to inspect:

- exact prompt formatting
- temperature
- top-k
- seed
- repeated or collapsed output

Important idea: generation is a sample, not a proof. Use it alongside reports
and evals.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli chat --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json
```

## 7. Report

Purpose: make the run explainable after it finishes.

Output artifacts:

- `summary.json`
- `summary.md`
- stage-level Markdown reports

What to inspect:

- run settings
- artifact paths
- final losses
- loss diagnostics
- eval summary
- generated samples

Important idea: reports are part of the experiment, not decoration. If a result
cannot be traced back to the data, model settings, checkpoint, and eval rules,
it is not useful yet.

## How To Read A Tiny Run

1. Start with `corpus_report.md`.
2. Check tokenizer stats and token examples.
3. Read base loss diagnostics.
4. Read SFT loss diagnostics.
5. Check eval pass/fail details.
6. Compare generated samples with eval results.
7. Only then increase data size, context length, steps, or model size.

The point of Picochat is controlled learning. Make one change, rerun, compare
artifacts, and keep the explanation honest.
