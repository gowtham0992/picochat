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

- **Factory Flow:** the real pipeline from dataset to report.
- **Dataset Bay:** corpus stats, quality checks, training windows, pack builder,
  JSONL editor, tuning-data inspection, run launcher, and preview.
- **Tokenizer Lab:** character-to-token-ID inspection from the trained tokenizer.
- **Training Dash:** base and SFT loss traces, including memorization warnings.
- **Generation Deck:** live generation from the selected base or SFT checkpoint.
- **Eval Scoreboard:** pass/fail results plus honesty metrics.
- **Report Vault:** generated Markdown reports rendered inside the workbench.
- **Compare Runs:** side-by-side eval, loss, parameter, and context comparisons.

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
    train_report.json
    report.md
    sample.txt
  sft/
    checkpoint/
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
the best step, and a compact status such as `stable`, `watch-gap`, or
`memorization-risk`. These labels are not magic scores; they are readable
signals for deciding whether a tiny run is learning or mostly memorizing.

Corpus manifests also record document spans. When a run has at least two
usable documents, base training can hold out complete documents for validation
instead of mixing random token windows from the same source. Base reports then
add memorization diagnostics: generated sample n-gram overlap with training
text, overlap with held-out text, longest copied span, and any
`pico-canary-*` strings reproduced by the model.

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
The budget estimate is also deliberately simple: it assumes the current
character tokenizer and suggests a starting context size, batch size, and base
training step count. Picochat also prints a copyable `run tiny` command based
on that estimate. Add `--chat-input` and `--eval-input` to `data preview` or
`data build` when you already have domain-specific SFT/eval JSONL files:

```bash
PYTHONPATH=src python -m picochat.cli data preview --recipe examples/corpus_recipe.json --chat-input my_data/chat.jsonl --eval-input my_data/eval.jsonl
```

If those flags are omitted, the suggested command keeps the default chat/eval
files visible so you remember to replace them before a real domain-specific run.
The same preview also preflights both JSONL files: chat SFT rows need string
`user` and `assistant` fields, while eval rows need a string `user` plus visible
pass/fail rules such as `must_include`, `must_include_any`, `must_not_include`,
or `expected`.

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
domain examples before treating a run as meaningful.

The same starter pack flow is available in the web workbench under Dataset
Bay. The Pack Builder writes the four files locally and fills Source Preview
with the new `dataset_pack.json` path.

Dataset Bay also has a Tuning Data Inspector. Point it at a dataset pack, or
at separate chat/eval JSONL files, to check row schemas, usable example counts,
eval scoring rules, preview rows, and next actions before running training.
The Pack JSONL Editor can load and save those chat/eval files directly, and
the Run Launcher starts `run tiny` from a dataset pack while streaming a local
`web_run.log` tail. Web-launched runs stay visible after page reload because
the workbench rediscovers run folders that contain `web_run.log`; active runs
can also be cancelled from the launcher. Launcher presets (`smoke`, `tiny`,
and `small-local`) keep run sizes explicit, and Source Preview's budget
estimate can be applied directly to the launcher controls.

Source Preview and `data build` score every corpus source from 0-100 using
local, explainable heuristics such as short documents, duplicate lines,
non-ASCII rate, and extraction noise. Use `--min-score` when you want to
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

Inspect and build a corpus:

```bash
PYTHONPATH=src python -m picochat.cli data inspect --input examples/tiny_corpus.txt
PYTHONPATH=src python -m picochat.cli data build --input examples/tiny_corpus.txt --out runs/manual/corpus.txt
PYTHONPATH=src python -m picochat.cli data preview --dataset-pack examples/tiny_dataset_pack.json
```

Train a tokenizer:

```bash
PYTHONPATH=src python -m picochat.cli tok train --input runs/manual/corpus.txt --out runs/manual/tokenizer.json
```

Inspect next-token training windows:

```bash
PYTHONPATH=src python -m picochat.cli batch inspect --corpus runs/manual/corpus.txt --tokenizer runs/manual/tokenizer.json --context-size 128
```

Train base and SFT stages manually:

```bash
PYTHONPATH=src python -m picochat.cli train base --corpus runs/manual/corpus.txt --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/base --context-size 128 --n-embd 64 --n-layer 2 --max-steps 300
PYTHONPATH=src python -m picochat.cli train sft --input examples/tiny_chat.jsonl --tokenizer runs/manual/tokenizer.json --checkpoint runs/manual/base/checkpoint --out-dir runs/manual/sft --max-steps 600
```

Evaluate and chat:

```bash
PYTHONPATH=src python -m picochat.cli eval chat --input examples/tiny_eval.jsonl --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json --out-dir runs/manual/eval
PYTHONPATH=src python -m picochat.cli chat --checkpoint runs/manual/sft/checkpoint --tokenizer runs/manual/tokenizer.json
```

Compare runs:

```bash
PYTHONPATH=src python -m picochat.cli compare runs/tiny-v2 runs/tiny-v3 --out runs/compare-tiny.md
```

## Eval Philosophy

Picochat uses transparent phrase checks first because they are easy to inspect.
Each eval row can define:

- `must_include`: phrases that must appear.
- `must_include_any`: phrase groups where at least one option must appear.
- `must_not_include`: forbidden phrases.
- `answerable`: whether the model should answer directly.
- `category`: a human label such as `project`, `tokenizer`, `honesty`, or
  `refusal`.

The report tracks:

- pass rate
- unsupported claim rate
- missing support rate
- answerable vs unanswerable examples
- found forbidden phrases

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
- larger tokenizer option, likely BPE
- richer train/validation split controls
- longer base pretraining runs
- better eval suites for unsupported claims

The order matters: first make the tiny factory coherent, then scale the data.
