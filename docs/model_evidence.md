---
layout: default
title: Picochat Model Evidence
---

# Model Evidence

Picochat's public claims should be tied to artifacts, not screenshots or a
single impressive prompt. This page is the scoreboard for what exists today,
what is pending, and what must be published before a model is called releasable.

## Current Public Status

| Model artifact | Status | What it means |
| --- | --- | --- |
| Tiny local demo | Ready | Exercises the end-to-end factory on a local toy run. Useful for installation and UI checks only. |
| 100M H100/H200 pilot | Exercised, not yet published | The path has run on rented GPUs, but no public model card and benchmark bundle are published yet. |
| 1B `h200-1b-ddp8` model | Pending | The recipe and gates exist. The model is not claimed until the full run and release gate pass. |
| Hugging Face checkpoint | Pending | The export/publish path exists through `picochat export hf --push-to-hub`, but no official model repo is claimed yet. |

## Required Release Bundle

A public Picochat model should ship as a bundle, not just a weight file:

| Required artifact | Why it matters |
| --- | --- |
| Git commit | Reproduces the exact training code. |
| Dataset pack and import report | Shows the corpus source, filters, and document counts. |
| Preflight report | Proves token budget, replay risk, split integrity, and coverage were checked before training. |
| Training curves | Shows base/SFT loss, validation BPB, best checkpoint, and stop reason. |
| Honesty report | Shows prompt overlap, corpus/eval contamination, duplicate prompts, and support phrase hits. |
| Internal eval report | Shows identity, refusal, choice, arithmetic, and spelling category scores. |
| External benchmark report | Gives at least one common benchmark anchor such as ARC or MMLU subset. |
| Release gate result | Shows approved, warn, or blocked with explicit issues. |
| HF export manifest and model card | Makes the checkpoint reviewable and loadable by other users. |

## Planned Hardware Lanes

These are run plans, not model-quality claims. Real throughput and cost should
come from dry-run logs on the exact GPU host.

| Lane | Approx params | Planned tokens | Hardware | Purpose |
| --- | ---: | ---: | --- | --- |
| Local demo | tiny | tiny | CPU or single GPU | Verify install, UI, export, and serving smoke paths. |
| 100M pilot | ~100M | 1B-2B class | 1x H100/H200 | Validate data path, SFT/eval pack, loss behavior, and release-gate plumbing. |
| 1B release candidate | ~1.12B | ~22.4B | 8x H100/H200 DDP | First serious closed-book SLM result with 20:1 token/parameter budget. |

## Readiness Gates

Before spending on a full 1B run:

1. `picochat sanity preh100` passes on the target GPU host.
2. The dataset pack passes preflight with no token-budget, replay, coverage, or contamination blocks.
3. A 100-step 8-GPU DDP dry run completes and resumes from checkpoint.
4. The release-skills SFT/eval pack contains identity, refusal, choice, math, and spelling categories.
5. At least one external benchmark file is ready for post-run scoring.

After training:

1. Publish the model only if the long-run gate says approved.
2. If the gate blocks, publish it as a blocked research run, not a release model.
3. Report the weakest category instead of hiding it.
4. Include bad generations as well as good generations in the model card.

## Comparison Positioning

Picochat is not trying to replace every training framework. Its focus is the
audit trail around small-model training.

| Area | Picochat position |
| --- | --- |
| From-scratch training | Built in, from corpus to tokenizer to base model. |
| Release gates | Core feature; preflight and post-run gates can block claims. |
| Contamination checks | Core feature across corpus, SFT, eval, and support phrases. |
| Production serving | Local smoke server only; vLLM/TGI/llama.cpp adapters are future work. |
| Large-scale distributed training | 8-GPU DDP path is the mainline; FSDP is experimental. |
| Public model proof | Pending first published model and benchmark bundle. |

The fastest path to credibility is a small public model with complete evidence,
then a larger 1B run with the same release bundle discipline.
