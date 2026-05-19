# Contamination and Honesty Checks

Picochat assumes small-model evals are easy to fool unless the pipeline makes
cheating visible.

The honesty system is designed to answer four questions:

1. Did eval prompts leak into SFT?
2. Did eval answers leak into SFT?
3. Does the base corpus contain eval prompts or support phrases?
4. Is the generated answer copying the prompt, corpus, or training rows?

## Checked Before Training

Preflight and data honesty inspection check:

- exact prompt overlaps between SFT and eval
- near prompt overlaps between SFT and eval
- answer overlaps
- corpus prompt hits
- support phrase hits in corpus
- SFT/eval category coverage
- eval split coverage
- unanswerable/boundary checks

Release profiles block if corpus contamination is detected.

## Checked During and After Runs

Training and reports track:

- train/validation gap
- validation BPB
- train-only canary reproduction
- generated n-gram overlap
- prompt echo
- missing support
- forbidden claims
- unsupported answers
- refusal behavior

The post-run gate uses those signals to block release when needed.

## Why Regex Stream Scanning Exists

Large ClimbMix corpora are too big for naive phrase matching. Picochat uses a
compiled word-boundary regex matcher over normalized stream chunks for corpus
phrase scans. This avoids an expensive phrase-by-phrase inner loop and makes
preflight practical on multi-billion-character corpora.

## What Honesty Does Not Prove

Passing honesty checks does not prove the model is smart. It only means the
score is less likely to be inflated by obvious leakage.

A clean score can still be weak if:

- the base corpus is low quality
- the model is undertrained
- SFT only taught format
- eval is too narrow
- external benchmarks are missing

Honesty checks are necessary, not sufficient.

## Operator Rule

Do not copy failed eval prompts into SFT. Eval is the scoreboard; SFT is the
practice set. Add new practice rows that teach the underlying behavior without
duplicating the held-out scoring item.
