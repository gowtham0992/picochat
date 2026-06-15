# Picochat Product UI Spec

This spec captures the current target for the Picochat web app. The source of truth is the left-sidebar, stage-based interface shown in the June 2026 mockups.

## Product Shape

Picochat is a local SLM factory dashboard. The UI should answer three questions in this order:

1. Where is this run in the pipeline?
2. Is the run healthy?
3. What is blocking the next handoff?

The app should feel like a production console, not a marketing page and not a notebook. Dark mode is first-class. Light mode must remain readable but does not drive the visual identity.

## Shell

- Persistent left sidebar.
- Main top bar with breadcrumb, status pill, and page actions.
- Dense cards with one clear value per card.
- Stage-specific pages instead of large generic tabs.
- Run blockers must be visible without hovering or opening drawers.

## Navigation

Primary:

- Dashboard
- Runs
- Data packs

Pipeline:

- Preflight
- Train
- Eval
- Honesty
- Release gate
- Handoff

System:

- Settings

## Page Responsibilities

### Dashboard

Purpose: overview of all runs and system health.

Shows:

- Total runs
- Passing runs
- Warnings needing attention
- Average recent loss
- Recent run list
- Pipeline health summary
- Recent training-loss chart

### Runs

Purpose: current run command center.

Shows:

- Current run title and status
- Primary blocker/warning banner
- Pipeline progress track
- Model/steps/loss/throughput cards
- Run history list

### Data Packs

Purpose: dataset registry and contamination summary.

Shows:

- Pack count
- Total tokens
- Languages
- Average quality
- Pack list with status
- Contamination scan summary

### Preflight

Purpose: launch-readiness checks before compute spend.

Shows:

- Checks run
- Passed count
- Duration
- Data hash
- Environment checks
- Data checks
- Preflight log

### Train

Purpose: live or historical training health.

Shows:

- Steps
- Train loss
- Validation loss
- Throughput
- Loss curve
- Hyperparameters
- Hardware

### Eval

Purpose: benchmark and regression results.

Shows:

- Overall score
- Key benchmark scores
- Dangerous calls
- Benchmark table
- Regression deltas
- Eval configuration

### Honesty

Purpose: honesty, hallucination, refusal, and calibration evidence.

Shows:

- Honesty score
- Hallucination rate
- Refusal rate
- Calibration
- Honesty dimensions
- Failure analysis
- Sample failures

### Release Gate

Purpose: block or approve release/handoff.

Shows:

- Blocking warning banner
- Quality checks
- Gate thresholds
- Repair queue

### Handoff

Purpose: final artifact readiness.

Shows:

- Locked/ready banner
- Run passport
- Handoff packet
- Eval reports
- Checkpoints
- Artifact checklist

### Settings

Purpose: product-level thresholds and integrations.

Shows:

- Gate thresholds
- Default hyperparameters
- Notifications
- Integrations

## Visual Rules

- No nested card clutter.
- No giant hero panels inside the application.
- No hover-only critical information.
- No raw JSON on primary screens.
- Warnings must say exactly what is wrong.
- Stage icons are small and repeated; status color carries meaning.
- Cards should fit a laptop viewport without excessive scrolling.

## Interaction Rules

- Clicking sidebar items switches stage pages.
- Clicking a pipeline stage opens the corresponding stage page.
- Primary page actions may reuse existing panels or no-op with status text until backend actions are wired.
- Existing guided builder and launch internals may remain available behind legacy DOM, but the product shell is the default user-facing surface.
