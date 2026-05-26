---
layout: default
title: Model Registry
---

# Model Registry

Picochat runs write many useful artifacts. The model registry turns those run
folders into a single auditable table for teams deciding which checkpoint is
safe to inspect, export, or deploy.

## Build a Registry

From explicit run directories:

```bash
picochat registry runs/run-a runs/run-b \
  --out reports/model_registry.md \
  --json-out reports/model_registry.json
```

Or discover every run with `summary.json` under a run bank:

```bash
picochat registry --runs-dir runs \
  --out reports/model_registry.md \
  --json-out reports/model_registry.json
```

The registry includes:

- gate status and profile
- parameter count and planned token budget
- tokens per parameter
- eval and SFT fit rates
- held-out SFT fit when available
- external benchmark count
- honesty and preflight status
- best/resume checkpoint paths
- tokenizer path

## Write a Release Card

For a single run:

```bash
picochat registry runs/h200-1b-release \
  --release-card reports/h200-1b-release-card.md
```

The release card is a compact artifact for handoff. It points to required
evidence: preflight, honesty, eval, external benchmarks, checkpoints, tokenizer,
and gate issues.

## Product Rule

The registry is an index, not an approval system. A row with status `approved`
is useful only when the linked evidence exists and the run's model card/export
matches the same checkpoint. A blocked run should stay visible so failures are
not lost between experiments.
