---
layout: default
title: Screenshots
---

# Screenshots

Current dashboard screenshots (real renders of the React workbench, captured
against a live `picochat web` server):

| File | View |
| --- | --- |
| `onboarding.png` | First-run "Get started" screen |
| `dashboard-overview.png` | Run overview: release gate, KPIs, loss, architecture |
| `leaderboard.png` | Leaderboard + side-by-side run compare |
| `playground.png` | Chat playground with serve + export panels |
| `cloud.png` | Cloud training (Modal / Colab / Lambda) |
| `evaluation.png` | Eval scoreboard with report links |
| `dataset.png` | Dataset import and pack tools |

## Recapturing

The screenshots are generated headlessly so they stay current with the UI.
Start a server with the runs you want to feature:

```bash
picochat web --runs-dir runs --port 8765
```

Each dashboard section is reachable by URL hash (`#overview`, `#compare`,
`#playground`, `#cloud`, …), so a headless browser can capture each one:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --hide-scrollbars --window-size=1440,940 \
  --virtual-time-budget=12000 \
  --screenshot=docs/screenshots/leaderboard.png \
  "http://127.0.0.1:8765/#compare"
```

For the first-run screen, point at an empty runs directory
(`--runs-dir /tmp/empty`) and capture `#overview`.

## Rules

- Do not show API keys, tokens, SSH hostnames, or private paths.
- Do not show a run as approved unless the gate actually says approved.
- Prefer real completed runs over a faked demo state; keep honest failure
  states if they teach something — Picochat is about honest gates, not
  perfect-looking dashboards.
