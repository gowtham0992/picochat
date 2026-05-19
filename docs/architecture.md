---
layout: default
title: Picochat Architecture
---

# Picochat Architecture

Picochat is a small-language-model factory with one explicit contract: every
stage writes artifacts that later stages can verify.

```text
dataset pack
  -> corpus.txt + corpus_manifest.json
  -> tokenizer.json
  -> base checkpoint + base report
  -> SFT checkpoint + SFT report
  -> eval report + external benchmark reports
  -> summary.json + summary.md + release gate
```

## Training Stack

The current 1B-class scale is `h200-1b-ddp8`.

| Component | Choice |
| --- | --- |
| Architecture | Decoder-only Transformer |
| Layers | 24 |
| Context | 2048 |
| Embedding | 2048 |
| Attention | GQA, 16 query heads, 4 KV heads |
| Position | RoPE |
| Norm | RMSNorm |
| MLP | SwiGLU |
| Residual | Parallel residual |
| Stabilizers | QK norm, scaled residual init |
| Embeddings | Tied token embeddings / LM head |
| Precision | BF16 |
| Runtime | CUDA, torch.compile, DDP |
| Memory | Per-block gradient checkpointing |

The scale targets roughly 1.12B parameters and 22.4B planned base-training
tokens, about 20 tokens per parameter.

## DDP Strategy

Picochat uses plain DDP, not FSDP or tensor parallelism, for the 1B path.

That is not the most memory-efficient possible setup, but it is simpler,
debuggable, and fits comfortably on 8xH100/H200 for this model size. The next
scaling step can add FSDP, but the 1B launch should avoid introducing another
distributed system unless needed.

DDP safeguards:

- `static_graph=True`
- `gradient_as_bucket_view=True`
- `broadcast_buffers=False`
- rank-aware batch streams
- rank-0 setup coordination through `ddp_control.json`
- setup heartbeat every 30 seconds during long CPU setup phases
- worker timeout with explicit rank-0 log guidance

## Dataset Modes

Picochat supports three base data modes:

- `memory`: simplest path, useful for tiny and local runs.
- `sharded`: disk token shards, fastest long-run path.
- `packed`: document holdout first, then BOS-bestfit packing.

For long release profiles, sharded/packed mode must have document boundary
tokens available. This prevents a run from validating on partial fragments of
the same source document it trained on.

## Checkpoint and Resume Safety

Checkpoints are written crash-safely:

1. write into a unique temporary directory
2. atomically swap with `os.replace`
3. keep `.previous` as rollback protection

Resume safety includes:

- training fingerprint
- corpus SHA256
- tokenizer SHA256
- corpus manifest SHA256
- model config
- optimizer/scaler state
- batcher position
- Python/Torch/CUDA RNG state

The sanity suite includes resume-loss determinism checks so a short run can
prove that resume is not silently changing the training stream.

## Web Workbench

The workbench reads real run artifacts. It exposes:

- launch readiness
- full preflight output
- loss/BPB curves
- release readiness
- SFT fit and held-out fit
- token budget
- external benchmark status
- Scale Up commands
- remote DDP dry-run commands
- paid GPU launch confirmation

The dashboard is an operator surface, not a marketing page. If a gate blocks,
the UI should show what failed and what to fix.
