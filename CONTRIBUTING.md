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

The dashboard is a React + Vite app in `frontend/`, built into
`src/picochat/web_assets/react/` and served at `/react/`:

```bash
npm ci
npm run frontend:check   # tsc typecheck
npm run frontend:build   # rebuild the served bundle
npm run frontend:dev     # hot-reload dev server (proxies /api to :8765)
```

Commit the rebuilt `src/picochat/web_assets/react/` along with any frontend
source change — CI verifies the committed bundle matches `frontend/`.

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
- `ruff check src tests` passes (lint runs in CI).
- Frontend changes include a rebuilt `src/picochat/web_assets/react/`, and
  `npm run frontend:check` passes.
- Public docs are updated for user-facing commands or workflow changes.
- No generated artifacts are included.
- New release claims include a reproducible eval command and an honesty report.

## Licensing and provenance

Picochat is MIT licensed. By submitting a contribution you certify the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO): you
wrote the change or otherwise have the right to submit it under the project
license, and you agree it is provided under the MIT License. Sign off each
commit with `git commit -s` (adds a `Signed-off-by:` trailer). Do not contribute
code, weights, or data you are not licensed to redistribute.

## Scope

Good first areas:

- external benchmark converters
- `lm-eval-harness` task/report integration
- dashboard clarity
- deployment adapters
- run report/model card quality
- DDP/FSDP safety checks

Avoid adding opaque benchmark shortcuts, hidden retrieval, or eval data in SFT.
