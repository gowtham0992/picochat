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
- a small imported Hugging Face sample written to a local `corpus.txt`

Output artifacts:

- `corpus.txt`
- `corpus_manifest.json`
- `corpus_report.md`

What to inspect:

- how many files were included or skipped
- duplicate-document rate
- duplicate-line rate
- empty-line rate
- source quality scores
- whether files were filtered by `--min-score`
- document spans used for whole-document validation holdout

Important idea: a tiny model cannot learn what the dataset does not contain.
If the corpus is tiny, repeated, noisy, or off-topic, the model will mostly
memorize that data.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tiny_dataset_pack.json
```

Hugging Face import is an intake helper before this stage. It streams or loads
rows from a dataset split, writes a local text file, and then hands control
back to the normal Picochat corpus preview:

```bash
PYTHONPATH=src python -m picochat.cli data hf-import --dataset HuggingFaceFW/fineweb-edu --split train --text-column text --max-rows 1000 --out runs/fineweb-edu-sample/corpus.txt
PYTHONPATH=src python -m picochat.cli data preview --input runs/fineweb-edu-sample/corpus.txt
```

Once the corpus exists, generate an eval starter from the same text:

```bash
PYTHONPATH=src python -m picochat.cli data eval-starter --input runs/fineweb-edu-sample/corpus.txt --out runs/fineweb-edu-sample/eval_starter.jsonl --max-items 40
```

Important idea: generated eval rows are scaffolding. They are useful because
they force the first benchmark to reference the loaded corpus, but they still
need human review before the score means anything.

## 2. Tokenizer

Purpose: convert text into token IDs the model can read.

Picochat starts with a character tokenizer because it is easy to inspect. It
also supports a byte tokenizer for UTF-8 coverage experiments and a small
dependency-free BPE tokenizer for learned merge experiments. These are simpler
than production tokenizers, but they keep the mechanics visible.

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

To compare BPE against the educational baseline:

```bash
PYTHONPATH=src python -m picochat.cli tok train --input runs/manual/corpus.txt --out runs/manual/tokenizer-bpe.json --type bpe --vocab-size 512 --min-freq 2
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
- `base/best_checkpoint/`
- `base/train_report.json`
- `base/report.md`
- `base/sample.txt`
- `base/canary_probe.txt` when train-only canaries are enabled

What to inspect:

- train loss
- validation loss
- validation BPB, which compares tokenizers more fairly than plain loss
- estimated tokens seen and dataset passes
- stop reason: max steps, time budget, or early stop
- final train/validation gap
- best validation step
- recommended checkpoint step
- split mode: random token windows or held-out complete documents
- memorization diagnostics: train copy rate, held-out overlap, copied spans, canary hits
- generated base sample

Important idea: decreasing train loss only means the model is fitting the
training windows. Validation loss tells you whether that fit is carrying over
to held-out windows. When `run tiny` has a corpus manifest with multiple
documents, Picochat prefers document-level holdout so validation text comes
from complete unseen sources. That is a stronger signal than splitting random
windows from the same document.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli train base --corpus runs/manual/corpus.txt --tokenizer runs/manual/tokenizer.json --corpus-manifest runs/manual/corpus_manifest.json --split-mode document --out-dir runs/manual/base --context-size 128 --max-steps 300
```

For longer runs, add guardrails:

```bash
PYTHONPATH=src python -m picochat.cli train base --corpus runs/manual/corpus.txt --tokenizer runs/manual/tokenizer.json --corpus-manifest runs/manual/corpus_manifest.json --split-mode document --out-dir runs/manual/base --context-size 128 --max-steps 10000 --max-minutes 45 --early-stop-patience 3 --canary-count 3
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
- `sft/checkpoint/adapter_model.pt` and `adapter_config.json` when using LoRA
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
PYTHONPATH=src python -m picochat.cli train sft --input examples/tiny_chat.jsonl --tokenizer runs/manual/tokenizer.json --checkpoint runs/manual/base/best_checkpoint --out-dir runs/manual/sft --max-steps 600 --early-stop-patience 6
```

LoRA command for lightweight domain adapters:

```bash
PYTHONPATH=src python -m picochat.cli train sft --input examples/tiny_chat.jsonl --tokenizer runs/manual/tokenizer.json --checkpoint runs/manual/base/best_checkpoint --out-dir runs/manual/sft-lora --peft lora --lora-rank 8 --lora-alpha 16 --lora-targets attn_qkv,attn_proj --max-steps 600 --early-stop-patience 6
```

LoRA is for adapting an existing base checkpoint to a domain or behavior style.
It does not replace base pretraining or make the base model know facts it never
saw.

## 5. Eval

Purpose: score generated replies with transparent rules.

Picochat evals are intentionally simple. Each item can define required phrases,
any-of phrase groups, forbidden phrases, required entities, length bounds, a
reference answer, corpus-support requirements, and whether the question is
answerable.

Inputs:

- eval JSONL
- SFT checkpoint
- tokenizer
- optional support corpus for corpus-overlap diagnostics

Output artifacts:

- `eval/eval_report.json`
- `eval/report.md`

What to inspect:

- pass rate
- unsupported claim rate
- eval ladder breakdown
- prompt echo rate
- missing support rate
- missing entity rate
- length violation rate
- corpus support failure rate
- token-F1 and ROUGE-L-style reference overlap
- repetition diagnostics
- failure analysis and recommendations
- failure clusters and weak eval levels
- matched and missing phrases
- matched and missing entities
- forbidden phrases found in replies

Important idea: this is not semantic truth evaluation. It is an inspectable
measurement for a tiny model, especially for whether it makes unsupported
answers when it should refuse or echoes the prompt instead of answering.

Useful command:

```bash
PYTHONPATH=src python -m picochat.cli eval chat --input examples/tiny_eval.jsonl --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/eval
PYTHONPATH=src python -m picochat.cli eval chat --input examples/tiny_eval.jsonl --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/eval --support-corpus runs/manual/corpus.txt
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
- top-p
- repetition penalty
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
4. Check base memorization diagnostics and copied-span rates.
5. Read SFT loss diagnostics.
6. Check eval pass/fail details.
7. Check the eval ladder: smoke failures mean wiring is broken, held-out
   failures mean weak recall/generalization, transfer failures mean brittle
   behavior, adversarial failures mean weak refusal, and memorization-probe
   failures mean the model may be copying.
8. Compare generated samples with eval results.
9. Only then increase data size, context length, steps, or model size.

The point of Picochat is controlled learning. Make one change, rerun, compare
artifacts, and keep the explanation honest.
