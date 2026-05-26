---
layout: default
title: Deployment
---

# Deployment

Picochat includes a local OpenAI-compatible server and a Docker path for smoke
testing trained checkpoints. This is not a high-throughput production serving
stack yet; vLLM, TGI, TensorRT-LLM, and llama.cpp adapters remain future work.

## Local Workbench Container

Build and run the dashboard:

```bash
docker compose up --build picochat-web
```

Then open:

```text
http://127.0.0.1:8765
```

The container mounts `./runs` into `/workspace/runs`, so local run artifacts
stay outside the image.

## Local OpenAI-Compatible API

After a demo or training run writes a checkpoint:

```bash
export PICOCHAT_API_KEY="replace-me"

pico serve \
  --checkpoint runs/pico-demo/sft/checkpoint \
  --tokenizer runs/pico-demo/tokenizer.json \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key-env PICOCHAT_API_KEY
```

The Docker smoke path is also available:

```bash
PICOCHAT_CHECKPOINT=/workspace/runs/pico-demo/sft/checkpoint \
PICOCHAT_TOKENIZER=/workspace/runs/pico-demo/tokenizer.json \
docker compose --profile serve up --build picochat-serve
```

Then call:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $PICOCHAT_API_KEY" \
  -d '{"model":"picochat","messages":[{"role":"user","content":"What is Picochat?"}],"max_tokens":80}'
```

Streaming response framing is available for local integrations:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $PICOCHAT_API_KEY" \
  -d '{"model":"picochat","stream":true,"messages":[{"role":"user","content":"Stream one sentence."}],"max_tokens":80}'
```

## Release Rule

Do not deploy a checkpoint as a product claim until the run includes:

- preflight report
- data honesty report
- SFT fit and held-out SFT fit
- visible eval report
- at least one external benchmark for release profiles
- long-run release gate status

The server can load an experimental model for smoke tests, but serving does not
turn an experimental run into a release.

## Current Limits

- native `pico serve` is single-process PyTorch serving
- optional bearer-token auth is available, but there is no HTTPS, queueing, or multi-tenant isolation
- no paged attention or continuous batching
- streaming is OpenAI-style SSE response framing, not token-by-token decoding
- no native vLLM/TGI/GGUF artifact yet

Those limits are explicit so users do not confuse the local server with a
production inference fleet.
