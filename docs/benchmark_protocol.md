---
layout: default
title: Benchmark Protocol
---

# Benchmark Protocol

Picochat should earn model claims with reproducible evidence, not screenshots
or a single cherry-picked prompt. Use this protocol for public results.

## Minimum Public Result

A credible result bundle includes:

- exact git commit
- dataset pack path and import report
- tokenizer hash
- model scale and parameter count
- planned training tokens and token/parameter ratio
- preflight report
- honesty/contamination report
- base validation curve
- SFT fit and held-out SFT fit
- visible eval report
- external benchmark report
- release gate status
- generated model card or HF export manifest

## Baselines

Every public claim should compare against at least two baselines:

1. **Random or majority baseline** for choice-heavy evals.
2. **Previous Picochat run** with the same data/eval split.

When compute allows, add a third baseline:

3. **Reference small model** evaluated through the same prompt and scoring
   harness.

Do not compare a Picochat model scored with closed-book prompts against a
retrieval-assisted baseline unless the report says retrieval was used.

## Recommended Eval Stack

For a 1B release-style run:

- internal release-skills eval: identity, refusal, choice, math, spelling
- held-out SFT fit diagnostic
- ARC-Easy or ARC-Challenge subset
- MMLU subset when available
- a small adversarial prompt-echo/refusal set
- contamination scan against base corpus, SFT, eval, and support phrases

## Required Tables

Public reports should show:

| Section | Required fields |
| --- | --- |
| Model | params, layers, context, tokenizer, attention backend |
| Data | source, documents, estimated tokens, duplicate rates |
| Budget | planned tokens, tokens/parameter, epochs, GPU type |
| Eval | pass rate, confidence interval, category breakdown |
| Honesty | exact prompt hits, near prompt hits, corpus phrase hits |
| Release | approved/blocked/warn plus blocking issues |

## Failure Reporting

A blocked run is still useful if it says why it failed. Publish the weakest
categories and the gate issues instead of hiding them.

Examples:

- "Math failed at 18%; release threshold is 30%."
- "External ARC subset scored below threshold."
- "Prompt echo exceeded release gate."
- "Corpus/eval contamination found; rerun with cleaned eval."

## Command Skeleton

```bash
pico leaderboard runs/run-a runs/run-b --out reports/leaderboard.md

pico eval external \
  --input external/arc_easy_validation.jsonl \
  --format arc \
  --benchmark-name arc_easy \
  --checkpoint runs/<run>/sft/checkpoint \
  --tokenizer runs/<run>/tokenizer.json \
  --out-dir runs/<run>/external_eval/arc_easy \
  --device cuda \
  --precision bf16 \
  --max-new-tokens 1
```

## lm-eval-harness Bridge

Picochat's native eval reports are intentionally transparent, but public model
claims should also be easy to score with common benchmark tooling. Export a
checkpoint first:

```bash
pico export hf \
  --checkpoint runs/<run>/sft/checkpoint \
  --tokenizer runs/<run>/tokenizer.json \
  --out-dir exports/<run>
```

Then write a reproducible EleutherAI `lm-eval-harness` command:

```bash
pico eval lm-harness \
  --model-path exports/<run> \
  --tasks arc_easy,hellaswag \
  --out-dir reports/<run>/lm_eval \
  --device cuda:0 \
  --batch-size auto \
  --model-arg dtype=bfloat16 \
  --dry-run
```

When `lm-eval` is installed, remove `--dry-run` to execute the command. The
bridge writes `lm_eval_command.json` so benchmark reports can quote the exact
adapter, task list, model path, and device settings.

The goal is not to make every number high. The goal is to make every number
auditable.
