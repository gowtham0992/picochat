# Contributing to Picochat

Picochat is an honest small-language-model training factory. Contributions
should make training, evaluation, release, or deployment more inspectable.

## Development Setup

```bash
git clone https://github.com/gowtham0992/picochat.git
cd picochat
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,hf,monitor]"
PYTHONPATH=src pytest -q
```

## Contribution Standards

- Keep generated runs, checkpoints, logs, model weights, and private notes out
  of git.
- Add tests for behavior changes in `src/picochat`.
- Do not weaken honesty, contamination, or release gates to make a score look
  better.
- If a change affects paid GPU launch behavior, update the runbook and add a
  test that proves the preflight or launcher behavior.
- If a feature is experimental, label it as such in CLI help, docs, and gate
  output.

## Pull Request Checklist

- `PYTHONPATH=src pytest -q` passes locally or the PR explains why it could not
  be run.
- Public docs are updated for user-facing commands or workflow changes.
- No generated artifacts are included.
- New release claims include a reproducible eval command and an honesty report.

## Scope

Good first areas:

- external benchmark converters
- `lm-eval-harness` task/report integration
- dashboard clarity
- deployment adapters
- run report/model card quality
- DDP/FSDP safety checks

Avoid adding opaque benchmark shortcuts, hidden retrieval, or eval data in SFT.
