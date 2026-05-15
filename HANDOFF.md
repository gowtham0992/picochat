# Picochat Handoff

Date: 2026-05-14
Branch: `develop`
Workspace: `/Users/g/Documents/andrej/picochat`

This file exists so a fresh Codex session can continue the Picochat work without losing the long project context.

## North Star

Picochat is an honest, educational, from-scratch tiny SLM factory.

The goal is not to wrap an existing model and call it ours. The goal is to let a user bring a dataset, build a corpus, train a tokenizer, pretrain a decoder-only transformer, run chat SFT, evaluate honestly, compare runs, and understand every stage through CLI and UI.

Current long-term target:

1. Prove the full local/Mac workflow on small data.
2. Prove a stronger Colab/GPU workflow.
3. Only then spend one serious 8xH100 run on the best tested recipe.
4. Keep the model closed-book for the flagship claim. Retrieval/RAG can exist later as a separate optional mode, but it must not inflate closed-book eval.

The user cares a lot about not cheating:

- No eval leakage.
- No hidden demo data in real eval.
- No support corpus used during generation for closed-book scores.
- No hardcoded dashboard numbers.
- No fake confidence from synthetic rows.
- Treat BPB, SFT fit, category-level eval, contamination, and memorization checks as evidence.

## How The Project Started

The user first studied Andrej Karpathy's `nanochat` and wanted a smaller project called `picochat`.

Early positioning:

- Not "a smaller ChatGPT."
- More like "a tiny LLM factory I can fully explain."
- Educational first, but real enough to train a small decoder-only language model.
- UI should make the pipeline understandable: dataset -> tokenizer -> base pretraining -> chat SFT -> eval -> chat/report.

Important discussion points:

- Picochat is not RAG by default.
- RAG-style grounded answers are useful, but the current research goal is closed-book model training.
- A domain SLM trained on coffee shop data will not magically answer facts not in the data. It needs either a good general base, domain continued pretraining/SFT, or retrieval.
- The user wants to eventually email Karpathy, so the project has to be honest and technically defensible.

## Current Product Shape

Picochat now has both CLI and web UI.

The UI evolved from a bright CRT terminal concept into a calmer retro/paper workbench. Important UI ideas:

- `Home`
- `Guide Me`
- `Workbench`
- `Beginner` / `Research` modes
- `Classic` / `Paper` theme
- HF import flow
- local documents flow
- dataset pack flow
- starter SFT/eval creation
- JSONL editor
- readiness checks
- run launcher
- run bank/archive
- run progress/log tailing
- factory flow
- reports/compare/eval/leaderboard

Recent UI issues that were fixed or partly fixed:

- Text contrast and font-size consistency.
- Workbench layout needed left/right run bank instead of huge vertical scroll.
- Archive could show stale run history after archive.
- HF import split handling.
- Guide flow needed clearer "what do I do next?" onboarding.

Remaining UI/product gap:

- Needs an unmistakable "Ready For Long Run" flow with hard warnings and gates.
- Needs Colab/H100 export/import lane in UI.
- Needs beginner guidance that explains when SFT/eval starter files are scaffolds, not real benchmark proof.

## Current Technical Capabilities

Core pipeline exists:

- Corpus building from local docs, HF imports, dataset packs, and recipes.
- Quality scoring and filtering.
- Honesty/leakage checks.
- Tokenizers: char, byte, BPE.
- Decoder-only transformer model.
- Configurable model size, heads, layers, context size.
- Learned/RoPE positions depending on current code config.
- LayerNorm/RMSNorm style configuration exists in recent runs.
- GELU/ReLU-squared style activation knobs exist in recent runs.
- Base pretraining.
- Chat SFT with assistant-only masking.
- Category-aware SFT sampling: uniform, category-balanced, category-sqrt.
- Grad accumulation.
- Cosine LR/warmup/min LR.
- Gradient clipping.
- AdamW and Muon/EMA-related knobs added in the project.
- Device selection: CPU, MPS, CUDA/auto depending on environment.
- Eval with phrase/choice/scoring-style checks.
- Support-corpus diagnostics for eval reports, not closed-book generation.
- SFT fit diagnostic.
- Leaderboard and compare.
- Run reports and summaries.
- HF benchmark pack generation with offline fallback.
- Skills corpus generation for synthetic math/spelling/choice pretraining.

Important files/directories:

- `src/picochat/cli.py`
- `src/picochat/run.py`
- `src/picochat/train.py`
- `src/picochat/sft.py`
- `src/picochat/model.py`
- `src/picochat/tokenizer.py`
- `src/picochat/data.py`
- `src/picochat/eval.py`
- `src/picochat/benchmark_pack.py`
- `src/picochat/sft_sweep.py`
- `src/picochat/web.py`
- `src/picochat/web_assets/`
- `tests/`
- `runs/`
- `knowledge/summary_sft_eval_h100.md`
- `/Users/g/Documents/andrej/pico-path-to-production.md`

## Repo State At Handoff

At the time this file was created:

- Branch: `develop`
- Git status showed `?? knowledge/`
- `knowledge/summary_sft_eval_h100.md` is an untracked research note unless committed later.

Check status in the new session:

```bash
cd /Users/g/Documents/andrej/picochat
git status --short --branch
```

## Dataset Path We Are Focused On

Current main dataset direction:

- `nvidia/Nemotron-ClimbMix`
- Start locally with 1 shard and small row counts.
- Use this as a serious base-data direction, because it is closer to nanochat's data story than TinyStories.

Important local imported pack:

- `runs/climbmix-1shard-5k/dataset_pack.json`
- `runs/climbmix-1shard-5k/documents`
- `runs/climbmix-1shard-5k/chat_benchmark.jsonl`
- `runs/climbmix-1shard-5k/eval_benchmark.jsonl`
- `runs/climbmix-1shard-5k/skills_recipe.json`
- `runs/climbmix-1shard-5k/skills_corpus.txt`

The user has been driving the flow through the web UI where possible.

## Key Run History

TinyStories phase:

- Early tiny-char/byte/BPE runs proved the full factory loop.
- TinyStories eventually reached a good narrow-domain run around `runs/tinystories-v7-10k-192x6-3h-v1`.
- A richer eval reported around `37/52 = 71.15%` at one point.
- This proved Picochat can train a small model inside a narrow distribution.
- It did not prove general SLM ability.

SmolLM/Cosmopedia phase:

- Tried SmolLM/Cosmopedia-style small imports.
- Smoke/medium runs revealed SFT/eval scaffolding problems.
- Many failures were not base-model proof; they were "bad or tiny SFT/eval" proof.

ClimbMix phase:

1. `runs/climbmix-1shard-1k-local-v1`
   - Rough first local/MPS run.
   - Eval: `1/80 = 1.25%`
   - Confirmed MPS was active using `powermetrics`: GPU residency ~100%.

2. `runs/climbmix-1shard-5k-hfbench-v1`
   - 5k ClimbMix, 4L x 128-ish, MPS.
   - Eval: `17/80 = 21.25%`
   - Base BPB around `2.5368`.
   - This was the first meaningful ClimbMix bump.

3. `runs/climbmix-1shard-5k-192x6-4h-adamw-v1`
   - Main overnight-ish Mac run.
   - Config shape:
     - context 512
     - 192 embedding
     - 6 layers
     - 6 heads
     - BPE vocab 1024
     - MPS
     - AdamW
     - base steps 30000
     - SFT steps 300
   - Base:
     - Step 30000/30000
     - Base val `2.5816`
     - Base BPB `1.6341`
   - SFT:
     - Early stopped around step 150.
     - SFT val `2.0222`
     - SFT BPB `1.6922`
   - Eval:
     - `14/80 = 17.50%`
   - This is still the best base BPB/closed-book baseline in compare.
   - But it did not beat the earlier 5k hfbench eval, so SFT/eval remains the bottleneck.

4. SFT-only sweeps from the 4h base:
   - `runs/climbmix-1shard-5k-192x6-sft-lr5e5-v1`
     - Eval: `15/80 = 18.75%`
   - `runs/climbmix-1shard-5k-192x6-sft-sweep-v1`
     - Best held-out eval: `category-sqrt-lr5e-5-steps1000`, about `19%`
   - `runs/climbmix-1shard-5k-192x6-behavior-sft-sweep-v1`
     - Best held-out eval: `25%`
     - Best SFT fit: `66.60%`
   - `runs/climbmix-1shard-5k-192x6-behavior-sft-sweep-v2`
     - Best SFT fit: `72.80%`
     - Best eval: about `21%`
   - Interpretation: behavior SFT can fit better now, but held-out eval still does not scale cleanly.

5. Skills corpus runs:
   - `data skills-corpus` created `110000` rows:
     - `skills_choice: 10000`
     - `skills_math: 50000`
     - `skills_spelling: 50000`
   - `runs/climbmix-1shard-5k-192x6-skills-base12k-v1`
     - Base BPB `1.6369`
     - SFT BPB `0.2074`
     - Eval `20/200 = 10.00%`
     - Compare says worse than 4h baseline.
   - `runs/climbmix-1shard-5k-192x6-skills-clean-smoke-v1`
     - Base BPB `2.1037`
     - SFT BPB `0.3144`
     - SFT fit `35.60%`
     - Eval `31/200 = 15.50%`
   - Interpretation: skills corpus approach needs better curriculum/eval alignment before a long run.

## Current Honest Assessment

What is good:

- Picochat is a real from-scratch training pipeline.
- It trains actual decoder-only transformer checkpoints.
- It is not RAG by default.
- It has real tokenizer/base/SFT/eval/report stages.
- It has MPS/GPU running on Mac.
- The UI is useful and close to a product-style workbench.
- The project has much better honesty infrastructure than at the start.
- We have learned that TinyStories works as a narrow benchmark and ClimbMix is harder but more serious.

What is still weak:

- Model quality is still low for general SLM behavior.
- SFT/eval is the biggest bottleneck.
- The benchmark/SFT curriculum is still not polished enough.
- Synthetic skills rows can make loss look good while eval remains weak.
- General benchmark scores are low.
- No formal external benchmark story yet.
- No production Colab/H100 lane fully wired.
- No single "approved long-run config" should be trusted yet.

Do not claim "world-class SLM" yet.

Correct claim today:

> Picochat is an honest educational tiny SLM factory with a working closed-book training pipeline, transparent metrics, and an improving local research loop. It is not yet a strong general SLM.

## Research Notes Already Captured

See:

- `knowledge/summary_sft_eval_h100.md`

Key research takeaways:

- LIMA: SFT teaches behavior/style, not missing knowledge.
- Chinchilla: token/parameter budget should gate expensive runs.
- Phi/Textbooks: high-quality, diverse, self-contained data matters a lot for small models.
- TinyStories: small models need task-appropriate eval and memorization/novelty checks.
- Tulu 3: keep post-training manifest and data provenance.
- Contamination work: add stronger overlap gates before any big benchmark claim.

Nanochat lessons we care about:

- It uses much broader real SFT mixtures.
- It uses BOS/best-fit style packing.
- It centers benchmark score against random baselines.
- It has a stronger data/scale story and target param/data ratio.
- Copying lessons is fine; copying code/claims blindly is not.

## Immediate Next Engineering Work

The next session should not start with another blind long run.

Start with implementation/gating work:

1. Add SFT BOS-bestfit packing.
   - Current SFT has assistant-only masking but does not pack examples efficiently.
   - Need a `--sft-packing bos_bestfit` option.
   - Keep default `separate` until verified.
   - Report packing efficiency, padded tokens, source examples, packed sequences.

2. Fix benchmark-pack uniqueness/diversity ceiling.
   - `benchmark-pack --profile behavior --sft-rows 1000 --source offline` failed at `908/1000`.
   - The generator needs either more templates, more source pools, or clear max-row warnings.
   - It should fail early with actionable output or auto-lower safely.

3. Add/strengthen SFT/eval quality reports.
   - Category entropy.
   - Prompt duplicates and near duplicates.
   - Answer length distribution.
   - Template family split.
   - Held-out category distribution.
   - Explicit "behavior SFT" vs "skill SFT" labeling.

4. Add stronger contamination/novelty matrix.
   - Base corpus vs SFT.
   - Base corpus vs eval.
   - SFT vs eval.
   - Generated answers vs SFT/corpus nearest neighbor.
   - Report max n-gram overlap.

5. Add "Ready For Long Run" checklist.
   - CLI and UI.
   - Blocks or warns if:
     - SFT fit is below threshold.
     - eval leakage exists.
     - corpus epochs are too high.
     - token/parameter ratio is poor.
     - benchmark pack has too few rows or too many duplicates.
     - support-corpus scoring is being mistaken for closed-book generation.

6. Add training-budget planner.
   - params
   - corpus tokens
   - effective batch tokens
   - planned tokens
   - estimated epochs
   - token/parameter ratio
   - runtime estimate
   - danger flags

7. Wire Colab/H100 production lane later.
   - Export run bundle.
   - Setup commands/notebook.
   - Import completed run back into workbench.
   - Compare against local runs.

## Production Plan File

The user's broader production plan is:

- `/Users/g/Documents/andrej/pico-path-to-production.md`

Important themes from that file:

- Compute-optimal training horizon.
- BOS-aligned document packing.
- LR/batch scaling.
- `torch.compile`.
- Streaming dataloader for large corpora.
- Larger tokenizer support.
- HF checkpoint export.
- Multi-GPU/DDP later.
- Domain one-command workflow.
- BPB as headline metric.

The plan is ambitious. Do not promise production readiness until the gates above are implemented and tested.

## Commands Useful In A New Session

Start:

```bash
cd /Users/g/Documents/andrej/picochat
source .venv/bin/activate
git pull origin develop
git status --short --branch
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Compare current important runs:

```bash
PYTHONPATH=src python -m picochat.cli compare \
  runs/climbmix-1shard-5k-192x6-4h-adamw-v1 \
  runs/climbmix-1shard-5k-192x6-skills-base12k-v1 \
  runs/climbmix-1shard-5k-192x6-skills-clean-smoke-v1
```

Read research note:

```bash
sed -n '1,260p' knowledge/summary_sft_eval_h100.md
```

Read production plan:

```bash
sed -n '1,260p' /Users/g/Documents/andrej/pico-path-to-production.md
```

## Suggested Next Commit

Recommended next commit theme:

> Add SFT packing and long-run readiness groundwork.

Scope:

- `src/picochat/sft.py`
- `src/picochat/cli.py`
- `src/picochat/run.py`
- `src/picochat/sft_sweep.py`
- tests in `tests/test_sft.py`
- optionally report/UI surfacing if small.

Do this before another overnight run.

## User Preferences

- The user wants a research-scientist style, but honest and practical.
- They want to understand the process, not just receive commands.
- They often prefer UI-driven instructions, but CLI is fine when the UI is not wired yet.
- They like step-by-step guidance.
- They dislike fake confidence.
- They want commits/pushes after meaningful fixes when asked.
- They are aiming for a project that could impress Karpathy, but only if it is technically honest.

## Do Not Forget

- Closed-book flagship mode means no retrieval at generation time.
- Support corpus is allowed for diagnostics/reporting, not for inflating generation.
- TinyStories success does not imply ClimbMix/general success.
- Better SFT loss can still mean worse held-out eval.
- More data/longer runs are useful only after readiness gates pass.
- 8xH100 should be used only after local/Colab ablations show a stable recipe.
