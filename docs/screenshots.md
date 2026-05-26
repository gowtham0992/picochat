---
layout: default
title: Screenshot Capture Guide
---

# Screenshot Capture Guide

Use real screenshots for public posts and outreach. The SVGs in `docs/assets/`
are product previews, not evidence of a completed 1B run.

## Local Workbench

Start the web UI:

```bash
picochat web --runs-dir runs --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Recommended captures:

1. Home page
2. Workbench with a selected run
3. Release Readiness panel
4. Training Dash loss curves
5. Eval Scoreboard
6. Scale Up command generator
7. Preflight job output card

Save final screenshots under:

```text
docs/screenshots/
```

Current public screenshots:

```text
workbench-release-readiness.jpg
release-readiness-panel.jpg
training-dash-loss-curves.jpg
scale-up-commands.jpg
```

Additional useful screenshots before a public launch:

```text
home.jpg
eval-scoreboard.jpg
preflight-checklist.jpg
```

## Screenshot Rules

- Do not show API keys, SSH hostnames, private paths, or paid provider account
  details.
- Do not show a run as approved unless the actual gate says approved.
- Prefer a real completed tiny/pilot run over a fake demo state.
- If a screenshot is from a dry run or pilot, label it that way in captions.
- Keep failure states if they teach something. Picochat is about honest gates,
  not perfect-looking dashboards.
