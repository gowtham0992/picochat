# Picochat

Picochat is an educational tiny LLM factory.

It trains a small decoder-only language model from local text, tunes it on chat
examples, evaluates its behavior, and opens a local workbench where every stage
can be inspected.

```text
dataset -> tokenizer -> base pretraining -> chat SFT -> eval -> chat -> report
```

For a stage-by-stage explanation, read
[docs/pipeline_guide.md](docs/pipeline_guide.md).

Picochat is inspired by nanochat, but it has a different target. It is not a
small ChatGPT and it is not a GPT-2 race. It is a microscope for understanding
the whole chat-model pipeline on one machine.

## What It Is

- A from-scratch tiny decoder-only Transformer training pipeline.
- A local experiment factory with reproducible run artifacts.
- A workbench UI for inspecting data, tokens, losses, evals, generation, reports,
  and run comparisons.
- An honesty-oriented eval playground that tracks missing support, forbidden
  claims, answerable prompts, and unanswerable prompts.

## What It Is Not

- Not RAG.
- Not a frontier assistant.
- Not optimized for multi-GPU training yet.
- Not trying to hide tiny-model failure modes.

The point is to make each failure visible enough to learn from.

## License

Picochat is released under the MIT License. See [LICENSE](LICENSE).

## Quick Start

From the repo root:

```bash
PYTHONPATH=src python -m picochat.cli demo
```

This writes a full demo run to:

```text
runs/pico-demo
```

Then open the workbench:

```bash
PYTHONPATH=src python -m picochat.cli web --runs-dir runs --port 8765
```

Visit:

```text
http://127.0.0.1:8765
```

If installed as an editable package, the same commands are:

```bash
pico demo
pico web --runs-dir runs --port 8765
```

## Workbench Stations

The local web UI is artifact-backed. It reads files produced by runs; it does
not show fake training data.

- **Start Here:** beginner walkthrough for choosing data, checking readiness,
  creating starter SFT/eval files, smoke training, evaluating, comparing, and
  promoting a run.
- **Factory Flow:** the real pipeline from dataset to report.
- **Dataset Bay:** corpus stats, quality checks, training windows, pack builder,
  Hugging Face import, JSONL editor, tuning-data inspection, launcher preflight,
  live CLI command preview, run launcher, and preview.
- **Tokenizer Lab:** text-to-token-ID inspection from the trained tokenizer.
- **Training Dash:** base and SFT loss traces, including memorization warnings.
- **Generation Deck:** live generation from the selected base or SFT checkpoint,
  with temperature, top-k, top-p, repetition penalty, token probabilities, and
  seed controls.
- **Eval Scoreboard:** pass/fail results, honesty metrics, failure ladder, and
  repair guidance that keeps held-out eval prompts out of SFT.
- **Report Vault:** generated Markdown reports rendered inside the workbench.
- **Compare Runs:** side-by-side eval, loss, parameter, and context comparisons
  with champion/regression/next-experiment gates.

The intended learning loop is:

```text
choose data -> check readiness -> make held-out eval -> smoke train
  -> inspect failures -> add non-eval SFT practice -> rerun
  -> compare -> promote only if trust checks do not regress
```

Do not copy failed eval prompts directly into SFT. Eval is the scoreboard;
SFT is the practice set. Picochat tries to make that separation visible because
otherwise a tiny model can look better by memorizing the test.

## Run Artifacts

Each tiny run writes a folder like:

```text
runs/pico-demo/
  corpus.txt
  corpus_manifest.json
  corpus_report.md
  tokenizer.json
  summary.json
  summary.md
  base/
    checkpoint/
    best_checkpoint/
    train_report.json
    report.md
    sample.txt
    canary_probe.txt
  sft/
    checkpoint/
    best_checkpoint/
    sft_report.json
    report.md
    sample.txt
  eval/
    eval_report.json
    report.md
```

These artifacts are the contract between the training code, CLI, reports, and
workbench.

Base and SFT reports include loss diagnostics in both JSON and Markdown:
best validation step, final train/validation gap, validation regression from
the best step, recommended checkpoint step, validation BPB, and a compact
status such as `stable`, `watch-gap`, or `memorization-risk`. These labels are
not magic scores; they are readable signals for deciding whether a tiny run is
learning or mostly memorizing.

Corpus manifests also record document spans. When a run has at least two
usable documents, base training can hold out complete documents for validation
instead of mixing random token windows from the same source. Base reports then
add memorization diagnostics: generated sample n-gram overlap with training
text, overlap with held-out text, longest copied span, and any
`pico-canary-*` strings reproduced by the model.

Longer runs add guardrails instead of blind optimism:

- base and SFT save both final checkpoints and best-validation checkpoints
- every `run tiny` writes a data honesty report that checks obvious SFT/eval
  leakage before you trust the score; blocking leakage stops the run unless
  you explicitly pass `--allow-leaky-eval` for a diagnostic-only experiment
- base training reports validation BPB, a tokenizer-fair bits-per-byte metric
- long runs can use `hf_bpe`, a compiled Hugging Face BPE backend for tokenizer
  training; the pure-Python `bpe` path remains available for inspection and
  small educational runs
- base training can inject train-only `pico-canary-*` phrases when document
  split is available
- base and SFT reports estimate how many tokens/examples were actually seen;
  SFT uses group-aware validation when examples provide a `group` field
- base and SFT reports record learning-rate schedule, warmup, gradient clipping,
  and per-checkpoint LR/gradient-norm traces when those controls are enabled
- training supports `--device auto|cpu|mps|cuda`; `auto` picks CUDA first, then
  Apple MPS, then CPU, while reports record both requested and resolved device
- base and SFT support gradient accumulation; reports show micro batch,
  accumulation steps, effective batch size, and effective tokens per optimizer
  update so Mac/Colab/GPU recipes can be compared honestly
- base models can choose `--norm-type layernorm|rmsnorm` and
  `--position-encoding learned|rope`, plus `--activation gelu|relu2`;
  existing checkpoints remain readable, while newer pilot scales can use more
  modern Transformer defaults
- document-level splits pack each held-out source with its own BOS/EOS boundary
  instead of smearing all documents into one anonymous stream
- chat eval reports include support-match rate so a failed run can show whether
  the model matched some prompt constraints or ignored them entirely
- chat eval reports also track prompt echo, so generated `User:`/`Assistant:`
  turns or visible user-prompt copying cannot count as a clean pass
- chat eval reports include failure analysis and next-action recommendations by
  failure cause, category, and eval split
- chat eval reports now include an eval ladder (`smoke`, `heldout`,
  `transfer`, `adversarial`, `memorization`) so a score can show where a
  tiny model is weak instead of flattening every failure into one number
- eval rows can add reference answers, required entities, length bounds, and
  corpus-support gates; reports show token-F1, ROUGE-L-style overlap, entity
  match rate, weak support, repetition, and failure clusters
- `--early-stop-patience` and `--max-minutes` can stop wasted runs

## Bring Your Own Corpus

Picochat v1 corpus support is intentionally conservative: local files in
formats that can be converted into plain training text. Built-in support covers
`.txt`, `.text`, `.md`, `.jsonl`, `.csv`, and `.py`.

PDF and Word documents are optional because those extractors add dependencies:

```bash
pip install -e ".[docs]"
```

With the docs extra installed, Picochat also attempts `.pdf` and `.docx`.
Without it, those files are still listed in the manifest as skipped with
`missing_pdf_dependency` or `missing_docx_dependency`, so the corpus builder
does not silently ignore them.

Build a corpus from your own folder:

```bash
PYTHONPATH=src python -m picochat.cli data build --input my_docs/ --out runs/my-docs/corpus.txt
```

Or build from an explicit JSON recipe:

```bash
PYTHONPATH=src python -m picochat.cli data build --recipe examples/corpus_recipe.json --out runs/my-docs/corpus.txt
```

Preview the same recipe before writing anything:

```bash
PYTHONPATH=src python -m picochat.cli data preview --recipe examples/corpus_recipe.json
```

The preview command prints included/skipped files, labels, corpus stats, a
readiness checklist, a first-pass training budget estimate, warnings, and the
first slice of the combined training text. The checklist is intentionally
conservative: it does not promise a good model, but it tells you whether the
corpus is blocked, trainable with cautions, or ready for a tiny experiment.
The budget estimate is also deliberately simple: it estimates token windows
from corpus size and suggests a starting context size, batch size, and base
training step count. Picochat also prints a copyable `run tiny` command based
on that estimate. Add `--chat-input` and `--eval-input` to `data preview` or
`data build` when you already have domain-specific SFT/eval JSONL files:

```bash
PYTHONPATH=src python -m picochat.cli data preview --recipe examples/corpus_recipe.json --chat-input my_data/chat.jsonl --eval-input my_data/eval.jsonl
```

If those flags are omitted for a custom corpus, Picochat does not print a
runnable training command. This is intentional: a real dataset should not
silently train or score against the bundled demo chat/eval files. Create a
dataset pack, or pass domain-specific `--chat-input` and `--eval-input`. The
`run tiny` command has `--allow-default-tuning-data` only for diagnostic wiring
checks.
The same preview also preflights both JSONL files: chat SFT rows need string
`user` and `assistant` fields, while eval rows need a string `user` plus visible
pass/fail rules such as `must_include`, `must_include_any`, `must_not_include`,
or `expected`.

The corpus checklist also looks for duplicate full documents, not only
duplicate lines. Full-document duplicates are dangerous because they can make
validation, generation samples, and eval-adjacent behavior look stronger than
the model really is.

## Nanochat-Compatible ClimbMix Track

Picochat can import a bounded sample of the same public ClimbMix shard track
used by nanochat. The source data is `nvidia/Nemotron-ClimbMix`; nanochat uses
the repackaged shuffled shard dataset `karpathy/climbmix-400b-shuffle`.

This is still training from scratch: Picochat trains its own tokenizer and
randomly initialized model. It does not use nanochat weights or checkpoints.

Install the optional Hugging Face loader:

```bash
pip install -e ".[hf]"
```

Import a small local pilot pack:

```bash
PYTHONPATH=src python -m picochat.cli data climbmix-import \
  --out-dir runs/climbmix-1shard-1k \
  --shards 1 \
  --max-rows 1000
```

Preview it:

```bash
PYTHONPATH=src python -m picochat.cli data preview \
  --dataset-pack runs/climbmix-1shard-1k/dataset_pack.json
```

Run a Mac/GPU-aware pilot after the SFT/eval starter files are ready:

```bash
PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/climbmix-pilot-v1 \
  --dataset-pack runs/climbmix-1shard-1k/dataset_pack.json \
  --scale climbmix-pilot \
  --device auto
```

The `climbmix-pilot` scale uses BPE, RMSNorm, RoPE, ReLU2, cosine LR decay,
gradient clipping, and gradient accumulation. It is meant as a closed-book
baseline track, not retrieval.

Use more shards only after the pilot improves BPB without memorization or
honesty regressions.

The web workbench also has a **Scale Up** view. It generates:

- a local Mac MPS command for the selected dataset pack
- a Colab setup command
- a Colab ClimbMix import command
- a Colab CUDA training command
- a zip/download return command
- an import-back action that copies a completed run folder into `runs/` so it
  can be compared, inspected, and chatted with locally

This keeps GPU experiments inside the same artifact trail instead of becoming
one-off notebook results.

Before a serious run, check that the eval is not accidentally copied from the
SFT file or base corpus:

```bash
PYTHONPATH=src python -m picochat.cli data honesty --dataset-pack examples/tinystories_dataset_pack_v3.json --out-dir runs/tinystories-honesty
```

The honesty report does not prove semantic truth. It catches practical cheating
risks: exact eval prompts in SFT, near-duplicate SFT/eval prompts, duplicated
eval prompts, eval prompts that appear in the base corpus, and specific
multi-word eval support phrases copied into SFT answers or corpus text.

After a corpus is built, you can generate a starter eval ladder from its
sentences:

```bash
PYTHONPATH=src python -m picochat.cli data eval-starter --input runs/my-docs/corpus.txt --out my_data/eval_starter.jsonl --max-items 40
```

This is scaffolding, not a trusted benchmark. Review and edit the generated
rows before using them as the scoreboard for a real domain run. The point is to
start with answerable recall checks, transfer prompts, refusal boundaries, and
memorization probes that are tied to the corpus you actually loaded.

You can also import a small sample from a Hugging Face dataset into a local
plain-text corpus. This is intentionally a separate intake step: first export
rows locally, then run Picochat's normal preview/scoring/training flow on the
result.

Install the optional dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[hf]"
```

Import up to 1,000 streamed rows from a text column:

```bash
PYTHONPATH=src python -m picochat.cli data hf-import --dataset HuggingFaceFW/fineweb-edu --split train --text-column text --max-rows 1000 --out runs/fineweb-edu-sample/corpus.txt
```

The import writes both a combined `corpus.txt` and a `documents/` folder with
one text file per accepted dataset row. Use the folder for preview and training
when you want document-level holdout checks:

```bash
PYTHONPATH=src python -m picochat.cli data preview --input runs/fineweb-edu-sample/documents
```

For the local TinyStories sample used in this repo's walkthrough, preview the
dataset pack after importing `runs/tinystories-1k/documents`:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack.json
```

That pack uses `examples/tinystories_chat.jsonl` for chat SFT and
`examples/tinystories_eval.jsonl` for visible pass/fail scoring. It is still a
small educational setup: the eval checks story shape, constraint following,
simple refusal behavior, and one memorization probe instead of claiming broad
intelligence.

For the expanded TinyStories instruction curriculum, use the v2 pack:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v2.json
```

The v2 pack keeps the same local corpus but uses
`examples/tinystories_chat_v2.jsonl` and `examples/tinystories_eval_v2.jsonl`.
The preview report shows SFT/eval category counts so runs can be interpreted
as curriculum experiments, not just raw training loops.

For the current prompt-following curriculum, use the v4 pack:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v4.json
```

The v4 pack uses template-grouped SFT examples in
`examples/tinystories_chat_v4.jsonl` and a held-out transparent eval in
`examples/tinystories_eval_v4.jsonl`. It is designed to test whether the tiny
model copies requested subjects, required words, continuation details, and
refusal behavior without exact prompt leakage. Unlike v3, its validation groups
hold out prompt phrasings rather than entire subject/word concepts.

For the prompt-conditioning curriculum, v5 teaches a tiny model to bind
requested subjects, lessons, and required words into a simple scaffold before
writing the story:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v5.json
```

The v5 pack uses `examples/tinystories_chat_v5.jsonl` and
`examples/tinystories_eval_v5.jsonl`. It teaches a tiny model to bind requested
subjects, lessons, and required words into a simple scaffold before writing the
story. Its eval has explicit splits:

- `prompt_conditioned`: did the model copy visible constraints from the prompt?
- `transfer`: does the same behavior survive different wording?
- `knowledge`: does it answer simple story-writing questions?
- `refusal`: does it refuse non-story/live/medical/financial requests?
- `safety`: does it avoid claiming to print memorized training text?

For the balanced curriculum, v6 removes held-out required-word eval pairs from
SFT, then adds more non-eval practice for required words, refusal, story
knowledge, and memorization boundaries.

For the current transfer-focused curriculum, use the v7 pack:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tinystories_dataset_pack_v7.json
```

The v7 pack uses `examples/tinystories_chat_v7.jsonl` and
`examples/tinystories_eval_v7.jsonl`. It keeps the v6 eval style but adds
natural SFT prompts for required words, story generation, continuation, direct
story knowledge, and safety boundaries. The `pico`, `small`, and `medium`
scales use `category_balanced` SFT sampling so these categories are not drowned
out by common story templates.

To move beyond the 1k local sample, import a larger TinyStories subset and use
the matching 10k pack:

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

Use the 10k pack for base pretraining experiments, then compare split-level
eval results against the 1k pack before changing architecture again.

For repeatable experiments, put the three dataset inputs in one pack:

```json
{
  "name": "my-domain-pack",
  "description": "Corpus, chat SFT, and eval files for one tiny run.",
  "corpus": {"recipe": "corpus_recipe.json"},
  "chat": "chat.jsonl",
  "eval": "eval.jsonl"
}
```

You can scaffold that folder instead of writing the JSON by hand:

```bash
PYTHONPATH=src python -m picochat.cli data init-pack --name my-domain-pack --corpus my_docs/ --out my_pack/
```

This creates `dataset_pack.json`, `corpus_recipe.json`, `chat.jsonl`, and
`eval.jsonl`. The chat/eval files are starter templates; edit them with real
domain examples before treating a run as meaningful. The starter chat file
includes one answerable example, one refusal/boundary example, and one style
example so a new domain pack starts with the shape of a real SFT set instead
of a single placeholder row.

The same starter pack flow is available in the web workbench under Dataset
Bay. The Pack Builder writes the four files locally and fills Source Preview
with the new `dataset_pack.json` path.

Dataset Bay also has a Tuning Data Inspector. Point it at a dataset pack, or
at separate chat/eval JSONL files, to check row schemas, usable example counts,
eval scoring rules, preview rows, and next actions before running training.
The inspector also reports eval ladder counts so you can see whether the suite
is only smoke tests or has held-out, transfer, adversarial, and memorization
coverage.
The Pack JSONL Editor can load and save those chat/eval files directly, and
starter results explain whether the generated SFT/eval file is connected to
the dataset pack or merely saved as a standalone path. Connected starter files
are loaded back into the editor so you can rewrite them before training; eval
starter prompts must stay held out and should not be copied into SFT.

The Run Launcher starts `run tiny` from a dataset pack while streaming a local
`web_run.log` tail. Before launch, it runs a local preflight over the current
fields, catches common shape mistakes such as `n_embd` not dividing by
`n_head`, and shows the equivalent CLI command. The command preview is meant
for learning and reproduction: the UI should never hide what it is about to
run. Web-launched runs stay visible after page reload because the workbench
rediscovers run folders that contain `web_run.log`; active runs can also be
cancelled from the launcher. Launcher presets (`smoke`, `pico`,
`small-local`, `small`, and `medium`) keep run sizes explicit, and Source
Preview's budget estimate can be applied directly to the launcher controls.

Source Preview and `data build` score every corpus source from 0-100 using
local, explainable heuristics such as short documents, duplicate lines,
duplicate documents, non-ASCII rate, and extraction noise. Use `--min-score` when you want to
filter low-quality sources out of the actual training corpus while keeping
the skipped files visible in the manifest:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tiny_dataset_pack.json --min-score 70
```

Pack paths are relative to the pack file unless absolute. Preview a pack:

```bash
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tiny_dataset_pack.json
```

Run the full tiny pipeline from the pack:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --dataset-pack examples/tiny_dataset_pack.json --out-dir runs/tiny-pack-v1
```

Recipe files make the dataset choice reviewable:

```json
{
  "sources": [
    {"path": "notes/lesson-1.md", "label": "lesson"},
    {"path": "manuals", "label": "manual", "exclude": ["manuals/drafts/**"]},
    {"path": "archive", "include": false, "reason": "recipe_excluded"}
  ],
  "exclude": ["**/.DS_Store", "**/draft-*"]
}
```

Each source path is relative to the recipe file unless it is absolute. Labels
are stored in `corpus_manifest.json`; they do not get inserted into the training
text.

This writes:

```text
runs/my-docs/corpus.txt
runs/my-docs/corpus_manifest.json
runs/my-docs/corpus_report.md
```

The manifest records every scanned file, whether it was included or skipped,
the reason, character count, line count, and corpus warnings. Dataset Bay reads
this manifest so the workbench can show provenance instead of treating the
corpus as an anonymous blob of text.

Train a tiny run from that corpus:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --corpus-input runs/my-docs/corpus.txt --out-dir runs/my-docs-v1
```

Or run the full tiny pipeline directly from a recipe:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --corpus-recipe examples/corpus_recipe.json --out-dir runs/my-docs-v1
```

Or use a dataset pack so corpus, chat SFT, and eval move together:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --dataset-pack examples/tiny_dataset_pack.json --out-dir runs/tiny-pack-v1
```

Use only text you have permission to use. For local experiments, this can be
your own notes, internal documents you are allowed to process, public-domain
text, or permissively licensed material. Do not redistribute corpora or models
trained on copyrighted text unless you have the rights to do so.

## CLI Reference

Run the default demo:

```bash
PYTHONPATH=src python -m picochat.cli demo
```

Run a configurable tiny experiment:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tiny-v4
```

Run with a named local scale:

```bash
PYTHONPATH=src python -m picochat.cli run tiny \
  --out-dir runs/tinystories-pico-v1 \
  --dataset-pack examples/tinystories_dataset_pack_v7.json \
  --scale pico \
  --split-mode document
```

Scales are starting recipes, not quality promises:

- `smoke`: fast wiring check
- `pico`: first serious local BPE run with a stronger tiny model, LR decay, and
  gradient clipping, plus category-balanced SFT sampling and stricter early
  stopping after validation stalls
- `small`: slower local SLM experiment after a pico run is healthy
- `medium`: overnight-class Mac experiment after data/tokenizer diagnostics look
  good

Explicit flags override scale values, so you can do a one-step smoke of the
`pico` recipe by passing smaller `--base-steps`, `--sft-steps`, or model sizes.

Compare tokenizer choices on the same dataset pack:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-char --dataset-pack examples/tinystories_dataset_pack.json --tokenizer-type char --split-mode document
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-byte --dataset-pack examples/tinystories_dataset_pack.json --tokenizer-type byte --split-mode document
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-bpe --dataset-pack examples/tinystories_dataset_pack.json --tokenizer-type bpe --tokenizer-vocab-size 512 --tokenizer-min-freq 2 --split-mode document
```

Run with longer-training guardrails:

```bash
PYTHONPATH=src python -m picochat.cli run tiny --out-dir runs/tinystories-guarded --dataset-pack examples/tinystories_dataset_pack_v7.json --tokenizer-type bpe --tokenizer-vocab-size 512 --context-size 256 --base-steps 10000 --sft-steps 1000 --base-batch-size 4 --sft-batch-size 4 --base-lr-decay cosine --sft-lr-decay cosine --base-lr-warmup-steps 200 --sft-lr-warmup-steps 50 --base-grad-clip 1.0 --sft-grad-clip 1.0 --base-max-minutes 45 --sft-max-minutes 10 --base-early-stop-patience 3 --sft-early-stop-patience 4 --canary-count 3 --split-mode document
```

Inspect and build a corpus:

```bash
PYTHONPATH=src python -m picochat.cli data inspect --input examples/tiny_corpus.txt
PYTHONPATH=src python -m picochat.cli data build --input examples/tiny_corpus.txt --out runs/manual/corpus.txt
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tiny_dataset_pack.json
PYTHONPATH=src python -m picochat.cli data eval-starter --input runs/manual/corpus.txt --out runs/manual/eval_starter.jsonl --max-items 24
```

Train a tokenizer:

```bash
PYTHONPATH=src python -m picochat.cli tok train --input runs/manual/corpus.txt --out runs/manual/tokenizer.json
```

Choose the tokenizer explicitly when comparing experiments:

```bash
PYTHONPATH=src python -m picochat.cli tok train --input runs/manual/corpus.txt --out runs/manual/tokenizer-char.json --type char
PYTHONPATH=src python -m picochat.cli tok train --input runs/manual/corpus.txt --out runs/manual/tokenizer-byte.json --type byte
PYTHONPATH=src python -m picochat.cli tok train --input runs/manual/corpus.txt --out runs/manual/tokenizer-bpe.json --type bpe --vocab-size 512 --min-freq 2
```

The BPE tokenizer is Picochat's dependency-free educational BPE. It learns
frequent adjacent-token merges from the corpus and saves those merges in
`tokenizer.json`; it is not a tiktoken or SentencePiece clone.

Inspect next-token training windows:

```bash
PYTHONPATH=src python -m picochat.cli batch inspect --corpus runs/manual/corpus.txt --tokenizer runs/manual/tokenizer.json --context-size 128
```

Train base and SFT stages manually:

```bash
PYTHONPATH=src python -m picochat.cli train base --corpus runs/manual/corpus.txt --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/base --context-size 128 --n-embd 64 --n-layer 2 --max-steps 300 --early-stop-patience 6 --canary-count 1
PYTHONPATH=src python -m picochat.cli train sft --input examples/tiny_chat.jsonl --tokenizer runs/manual/tokenizer.json --checkpoint runs/manual/base/best_checkpoint --out-dir runs/manual/sft --max-steps 600 --early-stop-patience 6
```

Evaluate and chat:

```bash
PYTHONPATH=src python -m picochat.cli eval chat --input examples/tiny_eval.jsonl --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/eval
PYTHONPATH=src python -m picochat.cli eval chat --input examples/tiny_eval.jsonl --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/eval --support-corpus runs/manual/corpus.txt --corpus-support-threshold 0.25
PYTHONPATH=src python -m picochat.cli chat --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json
```

Generation controls are deliberately explicit. `--temperature 0` gives greedy
deterministic decoding for eval-style checks. `--top-k`, `--top-p`, and
`--repetition-penalty` shape sampling quality, but they do not add knowledge to
the model:

```bash
PYTHONPATH=src python -m picochat.cli generate --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json --prompt "User: write a tiny story about a kite\nAssistant:" --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.1
```

Compare runs:

```bash
PYTHONPATH=src python -m picochat.cli compare runs/tiny-v2 runs/tiny-v3 --out runs/compare-tiny.md
```

The compare table reports raw validation loss and tokenizer-fair BPB. Use BPB
when comparing `char`, `byte`, and `bpe` runs; raw loss is mainly useful when
the tokenizer is the same.

## Eval Philosophy

Picochat uses transparent phrase checks first because they are easy to inspect.
Each eval row can define:

- `must_include`: phrases that must appear.
- `must_include_any`: phrase groups where at least one option must appear.
- `must_not_include`: forbidden phrases.
- `answerable`: whether the model should answer directly.
- `category`: a human label such as `project`, `tokenizer`, `honesty`, or
  `refusal`.
- `level`: the eval ladder level, for example `smoke`, `heldout`, `transfer`,
  `adversarial`, or `memorization`.
- `reference_answer`: a short target answer used for token-F1 and ROUGE-L-style
  overlap diagnostics.
- `required_entities`: names or key terms that should appear in the reply.
- `min_words`, `max_words`, `min_chars`, `max_chars`: visible length-control
  checks.
- `require_corpus_support`: mark answers that should overlap the support
  corpus when `--support-corpus` is provided.

The report tracks:

- pass rate
- unsupported claim rate
- missing support rate
- eval ladder breakdown
- missing entity rate
- length violation rate
- corpus support failure rate
- reference overlap diagnostics
- repetition diagnostics
- answerable vs unanswerable examples
- found forbidden phrases
- prompt echo
- failure causes, failure clusters, weak ladder levels, and recommended next
  actions

This does not prove semantic truth. It gives a small, inspectable signal for
whether the tiny model is following the behavior we trained and asked it to
show.

## Design Rules

- Start local and readable.
- Add scale only after the tiny path is explainable.
- Prefer artifacts over hidden state.
- Keep UI values tied to real run files.
- Treat high SFT validation loss as useful signal, not something to hide.
- Make unsupported answers measurable.

## Roadmap

Near-term v0.1 polish:

- create a clean flagship `runs/tiny-v4` demo with current eval metadata
- add screenshots of the workbench
- keep tightening report language around honesty metrics
- expand the pipeline guide with screenshots and examples

After that, dataset and training upgrades should come next:

- stronger dataset scoring and filtering
- tokenizer comparisons using char, byte, and dependency-free BPE
- richer train/validation split controls
- longer base pretraining runs
- better eval suites for unsupported claims

The order matters: first make the tiny factory coherent, then scale the data.
