---
layout: default
title: Release Gates
---

# Release Gates

Picochat separates three ideas that are often blurred together:

- **training completed**
- **eval produced a score**
- **the model is releasable**

A model can complete training and still be blocked from release.

## Preflight Gate

Preflight runs before training. Its job is to stop obviously bad or dishonest
long runs before GPU spend.

Release-profile preflight can block on:

- missing corpus, SFT, or eval files
- too few documents
- duplicate or near-duplicate data pressure
- excessive corpus replay
- insufficient token/parameter budget
- planned tokens below target budget
- invalid DDP launch shape
- missing release-skill SFT/eval categories
- missing document boundary tokens for sharded/packed data
- incompatible attention backend/device/precision
- eval contamination signals

For release profiles, token budget has two checks:

- `target_param_data_ratio >= 20.0`
- `planned_to_target_ratio >= 0.90`

This means a 1B run cannot quietly claim a 20:1 recipe while only scheduling
8:1 worth of training tokens.

## Post-Run Gate

The post-run gate runs after base training, SFT, fit diagnostic, eval, and
external benchmark scoring.

It checks:

- preflight status
- honesty report status
- SFT fit rate
- held-out SFT fit rate
- visible eval pass rate
- per-skill release eval rates
- external benchmark presence
- refusal behavior
- prompt echo
- contamination issues

The result is one of:

- `approved`
- `warn`
- `blocked`

## Skill Release Profile

The `skill_release` profile is intentionally conservative about claims. It
requires coverage for:

- identity
- refusal/boundary behavior
- multiple-choice handling
- arithmetic
- spelling

The default thresholds are deliberately honest for a small model:

| Skill | Gate |
| --- | ---: |
| Identity | 60% |
| Refusal | 75% |
| Choice | 50% |
| Math | 30% |
| Spelling | 40% |

Low thresholds are not victory claims. They are tripwires. If the model cannot
clear them, the release should say so.

## SFT Fit Is Not Enough

SFT fit asks: did the model learn the supervised practice rows?

Held-out SFT asks: did that behavior transfer to held-out rows from the same
curriculum?

Visible eval asks: did the model answer the actual scoring tasks?

All three matter. High SFT fit with low held-out/eval performance usually means
the SFT stage taught format or replay, not robust behavior.

## External Benchmarks

Release profiles require at least one external benchmark attachment before
serious claims. Picochat can score ARC/MMLU-style choice rows through the same
transparent evaluator path.

External benchmarks do not make the model good. They reduce the chance that an
internal benchmark pack is accidentally too narrow.

## Fix Categories, Not Overall Score

When release fails, do not chase a single aggregate number. Fix the weakest
category:

- add non-eval SFT practice rows for that category
- improve base data if the skill requires knowledge the base never learned
- rerun SFT/eval
- compare against the previous run
- keep eval prompts out of SFT

The goal is a model whose claims match its evidence.
